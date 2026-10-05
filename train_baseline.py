#!/usr/bin/env python3
"""
Label 对齐 + LightGBM Baseline
  用 10s 快照特征预测合约未来 5 分钟涨跌
"""
import argparse, sys, time
from pathlib import Path
import polars as pl
import numpy as np
from datetime import datetime

def load_all(snap_dir: Path, market: str, prefix: str) -> pl.DataFrame:
    """加载一个市场全部快照 → 加前缀"""
    files = sorted((snap_dir / market).glob("*.parquet"))
    dfs = []
    for f in files:
        df = pl.read_parquet(f).with_columns([
            pl.col("*").name.map(lambda c: f"{prefix}_{c}" if c not in ("ts_ms","ts") else c)
        ])
        dfs.append(df)
    return pl.concat(dfs)

def merge_features(spot: pl.DataFrame, fut: pl.DataFrame) -> pl.DataFrame:
    """按时间戳 join 现货 + 合约"""
    # 用 asof join (时间戳不完全对齐)
    return fut.sort("ts_ms").join_asof(
        spot.sort("ts_ms"), on="ts_ms", strategy="nearest", tolerance=5000
    )

def make_label(df: pl.DataFrame, horizon_s: int = 300, threshold_bps: float = 10.0) -> pl.DataFrame:
    """
    Label: future mid return over [t, t+horizon_s]
      UP   if ret > +threshold_bps/10000
      DOWN if ret < -threshold_bps/10000
      FLAT otherwise
    """
    df = df.sort("ts_ms")
    step = horizon_s // 10  # 10s 采样, 5min = 30 步
    mid = df["fut_mid"].to_numpy()
    n = len(mid)
    future_ret = np.full(n, np.nan)
    for i in range(n - step):
        if not np.isnan(mid[i]) and not np.isnan(mid[i+step]) and mid[i] > 0:
            future_ret[i] = (mid[i+step] - mid[i]) / mid[i]

    thr = threshold_bps / 10000
    label = np.full(n, -1, dtype=np.int32)
    label[future_ret > thr] = 1   # UP
    label[future_ret < -thr] = 0  # DOWN
    label[np.abs(future_ret) <= thr] = 2  # FLAT

    return df.with_columns([
        pl.Series("label", label),
        pl.Series("future_ret", future_ret),
    ])

def train():
    snap_dir = Path("/workspace/snapshots")

    print("📂 加载快照...")
    t0 = time.time()
    spot = load_all(snap_dir, "binance_spot", "spot")
    fut  = load_all(snap_dir, "binance_futures", "fut")
    print(f"  现货 {spot.height:,}  合约 {fut.height:,}  ({time.time()-t0:.1f}s)")

    print("🔗 合并 + Label...")
    merged = merge_features(spot, fut)
    merged = make_label(merged, horizon_s=300, threshold_bps=10.0)
    print(f"  合并后 {merged.height:,}")

    # Label 分布
    lbl = merged.filter(pl.col("label").is_in([0,1,2]))
    if lbl.height > 0:
        print(f"  Label 分布 (threshold ±10bps, horizon 5min):")
        for v, name in [(1,"UP"),(0,"DOWN"),(2,"FLAT")]:
            cnt = lbl.filter(pl.col("label")==v).height
            print(f"    {name}: {cnt:,} ({cnt/lbl.height*100:.1f}%)")

    # 特征: 丢弃 list 列 + ts + ts_ms + label + target
    drop_cols = [
        c for c in merged.columns
        if any(k in c for k in ["_prices", "_qtys"]) or c in ("ts","label","future_ret")
    ]
    feature_cols = [c for c in merged.columns if c not in drop_cols]
    print(f"\n📊 特征数: {len(feature_cols)}")
    print(f"  {feature_cols[:10]} ...")

    # Train/Val split (时间切分, 避免前瞻)
    train_end = 1790380800000
    train = merged.filter(pl.col("ts_ms") < train_end).filter(pl.col("label").is_in([0,1,2]))
    val   = merged.filter(pl.col("ts_ms") >= train_end).filter(pl.col("label").is_in([0,1,2]))

    print(f"\n✂️  Train: {train.height:,}  9/4-9/25")
    print(f"    Val:   {val.height:,}  9/26-10/3")

    X_train = train.select(feature_cols).to_numpy()
    y_train = train["label"].to_numpy()
    X_val   = val.select(feature_cols).to_numpy()
    y_val   = val["label"].to_numpy()

    # 类别权重
    classes, counts = np.unique(y_train, return_counts=True)
    cw = {int(c): len(y_train) / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    sample_weight = np.array([cw[int(y)] for y in y_train])

    print(f"\n🚀 LightGBM 训练...")
    import lightgbm as lgb
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
    }

    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight, feature_name=feature_cols)
    val_data   = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_cols)

    model = lgb.train(
        params, train_data,
        num_boost_round=2000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
    )

    # 评估
    pred_proba = model.predict(X_val)
    pred_cls = pred_proba.argmax(axis=1)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
    acc = accuracy_score(y_val, pred_cls)
    ll  = log_loss(y_val, pred_proba, labels=[0,1,2])

    print(f"\n{'='*50}")
    print(f"📊 Val Results:")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Log Loss:    {ll:.4f}")
    print(f"{'='*50}")
    print("\nConfusion Matrix (Val):")
    cm = confusion_matrix(y_val, pred_cls, labels=[0,1,2])
    print("            Pred-DOWN  Pred-UP  Pred-FLAT")
    for i, row in enumerate(cm):
        names = ["True-DOWN", "True-UP", "True-FLAT"]
        print(f"  {names[i]:10s}  {row[0]:>8d}  {row[1]:>7d}  {row[2]:>8d}")

    print("\nClassification Report:")
    print(classification_report(y_val, pred_cls, labels=[0,1,2], target_names=["DOWN","UP","FLAT"]))

    # 只看方向 (binary: UP vs DOWN, 忽略 FLAT)
    mask = y_val != 2
    if mask.sum() > 100:
        acc_bin = accuracy_score(y_val[mask], pred_cls[mask])
        print(f"\n🎯 方向预测 (UP vs DOWN only, 忽略 FLAT):")
        print(f"  Accuracy: {acc_bin:.4f}  ({mask.sum()} samples)")

    # 特征重要性 Top 20
    importances = model.feature_importance(importance_type="gain")
    idx = np.argsort(importances)[::-1][:20]
    print(f"\n⭐ Top 20 Feature Importance (gain):")
    for i in idx:
        print(f"  {feature_cols[i]:35s}  {importances[i]:>10.0f}")

    # 保存
    import joblib
    out_dir = Path("/workspace/models")
    out_dir.mkdir(exist_ok=True)
    model_path = out_dir / "lgb_baseline.txt"
    model.save_model(str(model_path))
    print(f"\n💾 模型保存: {model_path}")

if __name__ == "__main__":
    train()
