#!/usr/bin/env python3
"""
LightGBM Baseline v2 — 正确的评估流程
  1. Label: future return (用分布分位数确定 threshold, 均衡类别)
  2. 模型: 二分类 (UP vs DOWN)
  3. 评估: 先全量 accuracy, 再按 confidence threshold 过滤看高置信度准确率
"""
import argparse, sys, time
from pathlib import Path
import polars as pl
import numpy as np

# ============== 1. 加载 ==============
def load_all(snap_dir: Path, market: str, prefix: str) -> pl.DataFrame:
    files = sorted((snap_dir / market).glob("*.parquet"))
    dfs = []
    for f in files:
        df = pl.read_parquet(f).with_columns(
            pl.col("*").name.map(lambda c: f"{prefix}_{c}" if c not in ("ts_ms","ts") else c)
        )
        dfs.append(df)
    return pl.concat(dfs)

def merge_features(spot: pl.DataFrame, fut: pl.DataFrame) -> pl.DataFrame:
    """backward asof join: fut t 时刻对齐 spot 不晚于 t 的最近快照 (避免前瞻泄露)."""
    return fut.sort("ts_ms").join_asof(
        spot.sort("ts_ms"), on="ts_ms", strategy="backward", tolerance=5000
    )

# ============== 2. Label (按分位数均衡) ==============
def make_label_balanced(df: pl.DataFrame, horizon_s: int = 300, pct: float = 0.25) -> pl.DataFrame:
    """
    用 future return 分布的分位数做 threshold, 让 label 均衡
      horizon_s: 预测未来多少秒 (default 5min=300s)
      pct: 取 top/bottom pct 作为 UP/DOWN, 中间丢 (default 25%)
    """
    df = df.sort("ts_ms")
    step = horizon_s // 10
    mid = df["fut_mid"].to_numpy()
    n = len(mid)
    future_ret = np.full(n, np.nan)
    for i in range(n - step):
        if mid[i] > 0 and not np.isnan(mid[i+step]):
            future_ret[i] = (mid[i+step] - mid[i]) / mid[i]

    # 训练集上算分位数 threshold (避免前瞻)
    ts_ms_arr = df["ts_ms"].to_numpy()
    train_end = 1790380800000
    ret_train = future_ret[(ts_ms_arr < train_end) & ~np.isnan(future_ret)]
    thr_up   = np.quantile(ret_train, 1 - pct)
    thr_down = np.quantile(ret_train, pct)

    label = np.full(n, np.nan)  # NaN = 过滤掉
    label[future_ret > thr_up]   = 1   # UP
    label[future_ret < thr_down] = 0   # DOWN

    print(f"  Future ret 分布 (train):")
    print(f"    thr_down (bottom {pct*100:.0f}%): {thr_down*10000:.1f} bps")
    print(f"    thr_up   (top    {pct*100:.0f}%): {thr_up*10000:.1f} bps")
    print(f"    中间区域 (过滤掉): {thr_down*10000:.1f} ~ {thr_up*10000:.1f} bps")

    return df.with_columns([
        pl.Series("label", label),
        pl.Series("future_ret", future_ret),
    ])

# ============== 3. 特征清理 ==============
def get_feature_cols(df: pl.DataFrame) -> list[str]:
    """丢弃 list 列 + ts + label + 垃圾特征 (*_right, ts_ms 时间戳)"""
    drop_contains = ["_prices", "_qtys", "ts", "label", "future_ret", "ts_ms_right", "_right"]
    return [c for c in df.columns if not any(k in c for k in drop_contains)]

# ============== 4. 训练 + 评估 ==============
def train_and_eval():
    snap_dir = Path("/workspace/snapshots")

    # 加载
    print("📂 加载快照...")
    spot = load_all(snap_dir, "binance_spot", "spot")
    fut  = load_all(snap_dir, "binance_futures", "fut")
    print(f"  现货 {spot.height:,}  合约 {fut.height:,}")

    # 合并
    merged = merge_features(spot, fut)

    # Label (分位数均衡)
    print("\n🏷️  Label 生成 (balanced, 25%/25% 分位数)...")
    merged = make_label_balanced(merged, horizon_s=300, pct=0.25)

    # 只保留有 label 的样本 (过滤掉中间区域)
    has_label = merged.filter(pl.col("label").is_in([0, 1]))
    no_label  = merged.filter(~pl.col("label").is_in([0, 1]))
    print(f"  有 label 样本: {has_label.height:,}  (UP+DOWN, 中间 {no_label.height:,} 已过滤)")

    # 切分 (时间切!)
    train_end = 1790380800000  # 2026-09-26 UTC
    train = has_label.filter(pl.col("ts_ms") < train_end)
    val   = has_label.filter(pl.col("ts_ms") >= train_end)

    # 也保留全量 val (包含中间区域) 用来评估 confidence 过滤
    val_full = merged.filter(pl.col("ts_ms") >= train_end)

    print(f"\n✂️  Train: {train.height:,} (9/4-9/25)  Val: {val.height:,} (9/26-10/3)")

    # 特征
    feature_cols = get_feature_cols(train)
    print(f"\n📊 特征数: {len(feature_cols)}")

    X_train = train.select(feature_cols).to_numpy()
    y_train = train["label"].to_numpy().astype(int)
    X_val   = val.select(feature_cols).to_numpy()
    y_val   = val["label"].to_numpy().astype(int)

    # 全量 val 也要 predict (包含中间区域的样本)
    X_val_full = val_full.select(feature_cols).to_numpy()
    y_val_full_ret = val_full["future_ret"].to_numpy()  # 真实 return

    # 类别权重
    classes, counts = np.unique(y_train, return_counts=True)
    cw = {int(c): len(y_train) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sample_weight = np.array([cw[int(y)] for y in y_train])
    print(f"  类别权重: {cw}")

    # 训练
    print(f"\n🚀 LightGBM 二分类训练...")
    import lightgbm as lgb
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
    }
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight, feature_name=feature_cols)
    val_data   = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_cols)

    model = lgb.train(
        params, train_data,
        num_boost_round=3000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )

    # ====== 评估 ======
    print(f"\n{'='*60}")
    print(f"📊 EVALUATION")
    print(f"{'='*60}")

    # 有 label 的 val 上预测
    proba_val = model.predict(X_val)  # P(UP)
    pred_val  = (proba_val > 0.5).astype(int)

    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
    acc  = accuracy_score(y_val, pred_val)
    auc  = roc_auc_score(y_val, proba_val)
    prec = precision_score(y_val, pred_val)
    rec  = recall_score(y_val, pred_val)
    f1   = f1_score(y_val, pred_val)

    print(f"\n【全量 Val (有 label 样本)】n={len(y_val):,}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  (Baseline 猜多数: {max(np.mean(y_val), 1-np.mean(y_val)):.4f})")

    # ====== Confidence Filter 关键部分 ======
    print(f"\n【Confidence Filter — 只看高置信度预测】")
    print(f"{'Threshold':>12s}  {'Kept':>10s}  {'Pct':>7s}  {'Accuracy':>10s}  {'Prec_UP':>10s}  {'Prec_DOWN':>10s}")
    print(f"{'-'*65}")

    for thr in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        # |proba - 0.5| * 2 > threshold → 置信度够高
        conf = np.abs(proba_val - 0.5) * 2  # range [0, 1]
        mask = conf > thr
        if mask.sum() < 50:
            continue
        proba_f = proba_val[mask]
        y_f = y_val[mask]
        pred_f = (proba_f > 0.5).astype(int)
        acc_f = accuracy_score(y_f, pred_f)
        prec_up = precision_score(y_f, pred_f, pos_label=1, zero_division=0)
        prec_dn = precision_score(y_f, pred_f, pos_label=0, zero_division=0)
        print(f"  > {thr:.2f}      {mask.sum():>8,}  {mask.sum()/len(y_val)*100:>6.1f}%  {acc_f:>10.4f}  {prec_up:>10.4f}  {prec_dn:>10.4f}")

    # ====== 全量 Val (包含中间区域) 上做同样的分析 ======
    print(f"\n【全量 Val (包含中间区域, n={len(y_val_full_ret):,})】")
    proba_full = model.predict(X_val_full)
    print(f"{'Threshold':>12s}  {'Kept':>10s}  {'Pct':>7s}  {'Acc_vs_UP/DOWN':>16s}  {'Avg Ret':>10s}")
    print(f"{'-'*58}")
    for thr in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        conf = np.abs(proba_full - 0.5) * 2
        mask = conf > thr
        if mask.sum() < 50:
            continue
        proba_f = proba_full[mask]
        ret_f = y_val_full_ret[mask]
        pred_f = (proba_f > 0.5).astype(int)
        # 只在真实 return 有信号的样本上算 accuracy
        has_signal = ~np.isnan(ret_f)
        if has_signal.sum() < 50:
            continue
        acc_f = accuracy_score((ret_f[has_signal] > 0).astype(int), pred_f[has_signal])
        avg_ret = np.mean(ret_f[pred_f==1]) - np.mean(ret_f[pred_f==0])  # long-short 期望收益差
        print(f"  > {thr:.2f}      {mask.sum():>8,}  {mask.sum()/len(proba_full)*100:>6.1f}%  {acc_f:>16.4f}  {avg_ret*10000:>9.1f}bps")

    # ====== 特征重要性 ======
    importances = model.feature_importance(importance_type="gain")
    idx = np.argsort(importances)[::-1][:15]
    print(f"\n⭐ Top 15 Features (gain):")
    for i in idx:
        print(f"  {feature_cols[i]:35s}  {importances[i]:>10.0f}")

    # 保存
    out = Path("/workspace/models")
    out.mkdir(exist_ok=True)
    model.save_model(str(out / "lgb_v2.txt"))
    print(f"\n💾 模型: {out / 'lgb_v2.txt'}")

if __name__ == "__main__":
    train_and_eval()
