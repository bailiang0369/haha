"""一键跑完 pipeline: download -> features -> train -> backtest.

用法:
    python main.py download          # 拉 60d 1m K 线 (可加 --days 10)
    python main.py features          # 从已下载 K 线生成特征并保存
    python main.py train             # 训练模型并保存
    python main.py backtest          # 用已训练模型在测试集上回测
    python main.py all               # 整条流水线一口气跑完
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

import config
from src import fetcher, features as feat, model as mod, backtest as bt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("main")


def _cache_path(days: int) -> Path:
    return config.DATA_DIR / f"{config.SYMBOL}_1m_klines_{days}d.parquet"


def _load_or_fetch_klines(days: int) -> pd.DataFrame:
    path = _cache_path(days)
    if path.exists():
        log.info(f"using cached {path}")
        return pd.read_parquet(path)
    return fetcher.download_klines(days=days, out_path=path)


# ---------------------------------------------------------------------------
# 各阶段
# ---------------------------------------------------------------------------
def stage_download(args) -> pd.DataFrame:
    log.info("=== stage: download klines ===")
    df = _load_or_fetch_klines(args.days)
    log.info(f"klines shape={df.shape}")
    return df


def stage_features(klines: pd.DataFrame, args) -> tuple[pd.DataFrame, pd.Series]:
    log.info("=== stage: build features ===")
    # 逐笔/逐档: 若用户已缓存, 可在此加载, 否则先只用 K 线特征
    feat_mat = feat.build_feature_matrix(klines, trades=None, depth_snapshots=None)
    log.info(f"feature matrix shape={feat_mat.shape} n_cols={feat_mat.shape[1]}")
    feat_path = config.DATA_DIR / "feature_matrix.parquet"
    feat_mat.to_parquet(feat_path)
    log.info(f"feature matrix saved -> {feat_path}")
    return feat_mat, klines["close"]


def stage_train(feat_mat: pd.DataFrame, close_series: pd.Series, args) -> None:
    log.info("=== stage: train ===")
    y = mod.make_labels(feat_mat, close_series)
    log.info(f"label dist:\n{pd.Series(y).value_counts().sort_index().rename({0:'DOWN',1:'NEUTRAL',2:'UP'})}")
    X_train, X_test, y_train, y_test = mod.split_ts(feat_mat, y)
    log.info(f"train={X_train.shape} test={X_test.shape}")

    clf = mod.train(X_train, y_train)
    metrics = mod.evaluate(clf, X_test, y_test)
    log.info(f"test metrics: {metrics}")

    imp = mod.feature_importance(clf, X_train, top_n=15)
    log.info("top features:\n" + tabulate(imp, headers="keys", showindex=False))
    mod.save_model(clf)


def stage_backtest(feat_mat: pd.DataFrame, close_series: pd.Series, args) -> None:
    log.info("=== stage: backtest ===")
    clf = mod.load_model()

    # 切出测试集段 (与 train 时的 split 保持一致)
    n = len(feat_mat)
    cut = int(n * (1 - config.TEST_RATIO))
    test_df = feat_mat.iloc[cut:]

    X_test = test_df.fillna(0.0)
    proba = clf.predict_proba(X_test)      # shape (n, 3): DOWN, NEUTRAL, UP
    pred = proba.argmax(axis=1)
    conf_up = proba[:, 2]
    conf_down = proba[:, 0]

    preds = pd.DataFrame({
        "pred": pred,
        "conf_up": conf_up,
        "conf_down": conf_down,
    }, index=test_df.index)

    result = bt.run(preds, close_series, initial=config.INITIAL_CAPITAL)
    log.info("=== backtest stats ===")
    log.info(tabulate([(k, v) for k, v in result.stats.items()],
                      headers=["metric", "value"], showindex=False))

    # 保存交易明细
    trades_df = pd.DataFrame([t.__dict__ for t in result.trades])
    if not trades_df.empty:
        trades_df.to_parquet(config.DATA_DIR / "backtest_trades.parquet")
        log.info(f"trades saved -> backtest_trades.parquet ({len(trades_df)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BTC event contract predict & backtest")
    parser.add_argument("cmd", choices=["download", "features", "train", "backtest", "all"])
    parser.add_argument("--days", type=int, default=60, help="下载最近 N 天 1m K 线")
    parser.add_argument("--days-trades", type=int, default=3, help="下载最近 N 天逐笔 (需 API key)")
    args = parser.parse_args()

    if args.cmd == "download":
        stage_download(args)
        return

    # 后面阶段都需要 K 线
    klines = stage_download(args)

    if args.cmd == "features":
        stage_features(klines, args)
        return

    if args.cmd == "train":
        feat_mat, close = stage_features(klines, args)
        stage_train(feat_mat, close, args)
        return

    if args.cmd == "backtest":
        feat_mat, close = stage_features(klines, args)
        stage_backtest(feat_mat, close, args)
        return

    if args.cmd == "all":
        feat_mat, close = stage_features(klines, args)
        stage_train(feat_mat, close, args)
        stage_backtest(feat_mat, close, args)
        return

    log.error(f"unknown cmd {args.cmd}")
    sys.exit(1)


if __name__ == "__main__":
    t0 = time.time()
    main()
    log.info(f"done in {time.time() - t0:.1f}s")
