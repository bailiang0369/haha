#!/usr/bin/env python3
"""
完整流水线 — 信号检验 + ML Walk-Forward
========================================
运行: python run_pipeline.py
需要: data/klines_btc/*.csv, data/klines_eth/*.csv
"""
import polars as pl
import numpy as np
import json
import os
import sys
import gc
from scipy.stats import spearmanr, norm

sys.path.insert(0, "/workspace")
from feature_pipeline import (
    load_aggtrades, load_klines,
    compute_features,
    variance_ratio_test, bootstrap_feature_ic,
    walkforward_lgbm,
)

RESULT_DIR = "/workspace/results"
os.makedirs(RESULT_DIR, exist_ok=True)

# 要用到的特征列 (排除 target, 原始 OHLC 等)
EXCLUDE_COLS = {
    "open_time", "open", "high", "low", "close", "volume", "quote_volume",
    "trade_count", "taker_buy_base", "taker_buy_quote", "ignore",
    "ts", "hour", "dow", "date", "close_time", "prev_final_update_id",
    "future_ret", "label", "ret_1m", "taker_sell_base", "total_qty",
    "active_buy_qty", "active_sell_qty", "open_raw", "close_raw",
    "high_raw", "low_raw", "volume_raw", "illiq_raw", "amplitude",
}


def run_asset(asset: str, klines_dir: str, horizon_min: int = 15):
    print(f"\n{'='*60}")
    print(f"  {asset}  (horizon={horizon_min}min)")
    print(f"{'='*60}")

    # ── Step 1: 加载 ──
    print("  [1/5] Loading klines...")
    klines = load_klines(klines_dir)
    print(f"        {len(klines):,} rows, {len(klines['date'].unique())} days")

    # ── Step 2: 特征 ──
    print(f"  [2/5] Computing features... (horizon={horizon_min}min)")
    feats = compute_features(klines, horizon_min=horizon_min)
    del klines; gc.collect()

    # 二分类 label
    feats = feats.with_columns(
        (pl.col("future_ret") > 0).cast(pl.Int32).alias("label")
    )

    feat_cols = [c for c in feats.columns if c not in EXCLUDE_COLS]
    print(f"        {len(feat_cols)} features: {feat_cols}")

    # ── Step 3: Variance Ratio Test ──
    print("  [3/5] Variance Ratio Test (随机游走检验)...")
    ret_arr = feats["ret_1m"].to_numpy()
    vr = variance_ratio_test(ret_arr, lag=5)
    print(f"        VR(5) = {vr['vr']:.4f}, z = {vr['z']:.2f}, p = {vr['p']:.4f}")
    if vr["p"] < 0.01:
        print(f"        ✅ p<0.01: 拒绝随机游走, 信号存在!")
    elif vr["p"] < 0.05:
        print(f"        ⚠️  p<0.05: 边缘显著")
    else:
        print(f"        💀 p>=0.05: 不能拒绝随机游走, 信号不存在!")

    # ── Step 4: Bootstrap IC ──
    print("  [4/5] Bootstrap IC (每个特征的显著性)...")
    ic_results = bootstrap_feature_ic(feats, feat_cols, "future_ret", n_boot=500)
    print(f"        Top 15 by IC:")
    for r in ic_results.head(15).iter_rows(named=True):
        crosses = "💀跨0" if r.get("ci_crosses_zero") else "✅不跨0"
        print(f"          {r['feature']:30s} IC={r['ic_full']:+.4f} "
              f"boot95%CI=[{r['ic_lo']:+.4f}, {r['ic_hi']:+.4f}] {crosses}")

    n_sig = len(ic_results.filter(~pl.col("ci_crosses_zero")))
    print(f"        {n_sig}/{len(feat_cols)} 特征 bootstrap CI 不跨0")

    # ── Step 5: Walk-Forward ML ──
    print("  [5/5] Walk-Forward LightGBM (expanding window)...")
    wf = walkforward_lgbm(feats, feat_cols, "future_ret", "label",
                          train_window=30, test_window=1, min_train_samples=1000)

    if not wf.get("valid"):
        print(f"        ❌ {wf.get('msg')}")
        return None

    print(f"        Tested {wf['n_days_tested']} days, {wf['n_samples']} samples")
    print(f"        AUC  = {wf['auc']:.4f}")
    print(f"        Acc  = {wf['accuracy']:.4f}")
    print(f"        Prec = {wf['precision']:.4f}")
    print(f"        Rec  = {wf['recall']:.4f}")

    print(f"        Top 10 feature importances (avg over all test days):")
    for r in wf["avg_importance"].head(10).iter_rows(named=True):
        print(f"          {r['feature']:30s} importance={r['avg_importance']:.1f}")

    print(f"        Daily acc distribution:")
    daily_accs = [d["acc"] for d in wf["daily_metrics"]]
    print(f"          mean={np.mean(daily_accs):.4f}, median={np.median(daily_accs):.4f}, "
          f"std={np.std(daily_accs):.4f}, min={min(daily_accs):.4f}, max={max(daily_accs):.4f}")

    # 保存
    feats.write_parquet(f"{RESULT_DIR}/features_{asset}.parquet")
    ic_results.write_parquet(f"{RESULT_DIR}/ic_{asset}.parquet")

    # daily metrics
    pl.DataFrame(wf["daily_metrics"]).write_csv(f"{RESULT_DIR}/daily_{asset}.csv")

    # importances
    wf["avg_importance"].write_csv(f"{RESULT_DIR}/importance_{asset}.csv")

    # 摘要
    summary = {
        "asset": asset,
        "horizon_min": horizon_min,
        "n_features": len(feat_cols),
        "n_days": len(feats["date"].unique()),
        "vr_test": vr,
        "n_sig_features_bootstrap": n_sig,
        "wf_auc": wf["auc"],
        "wf_accuracy": wf["accuracy"],
        "wf_precision": wf["precision"],
        "wf_recall": wf["recall"],
        "wf_n_days_tested": wf["n_days_tested"],
        "wf_n_samples": wf["n_samples"],
        "daily_acc_mean": float(np.mean(daily_accs)),
        "daily_acc_median": float(np.median(daily_accs)),
        "daily_acc_std": float(np.std(daily_accs)),
        "daily_acc_min": float(min(daily_accs)),
        "daily_acc_max": float(max(daily_accs)),
    }

    print(f"\n  📁 结果已保存到 {RESULT_DIR}/")
    return summary


def main():
    print("=" * 60)
    print("  🧪 完整流水线: VR Test → Bootstrap IC → WF LightGBM")
    print("=" * 60)

    all_summaries = []

    assets = [
        ("BTCUSDT", "/workspace/data/klines_btc"),
        ("ETHUSDT", "/workspace/data/klines_eth"),
    ]

    for asset, kdir in assets:
        if not os.path.exists(kdir):
            print(f"⚠️ Skip {asset}: {kdir} 不存在")
            continue
        s = run_asset(asset, kdir, horizon_min=15)
        if s:
            all_summaries.append(s)

    if all_summaries:
        with open(f"{RESULT_DIR}/pipeline_summary.json", "w") as f:
            json.dump(all_summaries, f, indent=2, default=str)

        print(f"\n{'='*60}")
        print("  📊 最终汇总")
        print(f"{'='*60}")
        for s in all_summaries:
            print(f"\n  {s['asset']}:")
            print(f"    VR p={s['vr_test']['p']:.4f}, 显著特征={s['n_sig_features_bootstrap']}")
            print(f"    WF AUC={s['wf_auc']:.4f}, Acc={s['wf_accuracy']:.4f}")
            print(f"    Daily acc: median={s['daily_acc_median']:.4f}, std={s['daily_acc_std']:.4f}")

    print("\n✅ 完成")


if __name__ == "__main__":
    main()
