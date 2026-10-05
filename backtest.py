#!/usr/bin/env python3
import time
from pathlib import Path
from datetime import datetime
import polars as pl
import numpy as np
import lightgbm as lgb

def load_all(snap_dir, market, prefix):
    files = sorted((snap_dir / market).glob("*.parquet"))
    dfs = []
    for f in files:
        df = pl.read_parquet(f).with_columns(
            pl.col("*").name.map(lambda c: f"{prefix}_{c}" if c not in ("ts_ms","ts") else c)
        )
        dfs.append(df)
    return pl.concat(dfs).sort("ts_ms")

def merge_features(spot, fut):
    return fut.sort("ts_ms").join_asof(spot.sort("ts_ms"), on="ts_ms", strategy="backward", tolerance=5000)

def make_future_ret(df, horizon_s=300):
    mid = df["fut_mid"].to_numpy()
    step = horizon_s // 10
    ret = np.full(len(mid), np.nan)
    for i in range(len(mid) - step):
        if mid[i] > 0 and not np.isnan(mid[i+step]):
            ret[i] = (mid[i+step] - mid[i]) / mid[i]
    return ret

def get_feature_cols(df):
    drop = ["_prices", "_qtys", "ts", "label", "future_ret", "_right"]
    return [c for c in df.columns if not any(k in c for k in drop)]

def walk_forward(df, future_ret, feat_cols, train_days=7, test_days=1, horizon_s=300, cost_bps=4.0):
    ts = df["ts_ms"].to_numpy()
    day_starts = sorted(set(ts // 86400000))
    n_days = len(day_starts)
    cost = cost_bps / 10000
    all_trades = []
    step = horizon_s // 10
    i = train_days  # 前 train_days 天训练, 从第 train_days 天开始测

    while i + test_days <= n_days:
        train_lo = day_starts[i-train_days] * 86400000
        train_hi = day_starts[i] * 86400000
        test_lo  = day_starts[i] * 86400000
        test_hi  = day_starts[i+test_days-1] * 86400000 + 86400000

        train_mask = (ts >= train_lo) & (ts < train_hi)
        test_mask  = (ts >= test_lo)  & (ts < test_hi)

        train_y = future_ret[train_mask]
        train_has = ~np.isnan(train_y)
        if train_has.sum() < 1000:
            i += test_days; continue

        train_idx = np.where(train_mask)[0][train_has]
        X_train = df.select(feat_cols).to_numpy()[train_idx]
        y_train = train_y[train_has]

        model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63,
                                   min_child_samples=200, verbose=-1, random_state=42)
        model.fit(X_train, y_train)

        # 训练集预测 → 确定 threshold (top/bottom 10%)
        train_pred = model.predict(X_train)
        thr_long  = np.quantile(train_pred, 0.90)
        thr_short = np.quantile(train_pred, 0.10)

        test_idx = np.where(test_mask)[0]
        if len(test_idx) == 0:
            i += test_days; continue

        X_test = df.select(feat_cols).to_numpy()[test_idx]
        pred_ret = model.predict(X_test)
        real_ret = future_ret[test_idx]

        pos_end = 0
        for j in range(len(test_idx)):
            gi = test_idx[j]
            if gi < pos_end: continue
            if np.isnan(real_ret[j]): continue
            pred = pred_ret[j]
            entry_mid = mid_arr[gi]
            if gi + step >= len(df): continue

            if pred > thr_long:
                exit_mid = mid_arr[gi + step]
                gross = (exit_mid - entry_mid) / entry_mid
                all_trades.append((ts[gi], "LONG", entry_mid, exit_mid, gross, gross - cost))
                pos_end = gi + step
            elif pred < thr_short:
                exit_mid = mid_arr[gi + step]
                gross = (entry_mid - exit_mid) / entry_mid
                all_trades.append((ts[gi], "SHORT", entry_mid, exit_mid, gross, gross - cost))
                pos_end = gi + step

        i += test_days
    return all_trades

def main():
    snap_dir = Path("/workspace/snapshots")
    print("📂 加载快照...")
    t0 = time.time()
    spot = load_all(snap_dir, "binance_spot", "spot")
    fut  = load_all(snap_dir, "binance_futures", "fut")
    print(f"  现货 {spot.height:,}  合约 {fut.height:,} ({time.time()-t0:.1f}s)")

    print("🔗 backward asof join + future_ret...")
    merged = merge_features(spot, fut)
    future_ret = make_future_ret(merged, horizon_s=300)
mid_arr = merged["fut_mid"].to_numpy()
    feat_cols = get_feature_cols(merged)
    print(f"  特征数: {len(feat_cols)}")

    print("\n🚶 Walk-Forward (7d训练/1d测试, 5min horizon, 4bps cost)...")
    trades = walk_forward(merged, future_ret, feat_cols)

    if not trades:
        print("❌ 没有交易"); return

    nets = np.array([t[5] for t in trades])
    grosses = np.array([t[4] for t in trades])

    print("\n" + "=" * 60)
    print("  WALK-FORWARD BACKTEST (L2 Only, 5min, 4bps)")
    print("=" * 60)
    print(f"\n总交易:     {len(trades)}")
    print(f"胜率:       {(nets > 0).mean()*100:.1f}%")
    print(f"平均净收益: {nets.mean()*10000:.2f} bps")
    print(f"平均毛收益: {grosses.mean()*10000:.2f} bps")
    print(f"累计净收益: {nets.sum()*100:.2f}%")
    if nets.std() > 0:
        print(f"Sharpe:     {nets.mean()/nets.std()*np.sqrt(365*288):.2f}")
    cum = np.cumsum(nets)
    print(f"最大回撤:   {cum.min()*100:.2f}%")

    # 每天
    daily = {}
    for t in trades:
        day = t[0] // 86400000
        daily.setdefault(day, []).append(t[5])
    print(f"\n每日净收益:")
    for day, rets in sorted(daily.items()):
        dt = datetime.utcfromtimestamp(day * 86400 / 1000).strftime("%m-%d")
        total = sum(rets) * 100
        win = (np.array(rets) > 0).mean() * 100
        print(f"  {dt}: {total:+.2f}%  ({len(rets):>3d}笔, 胜率 {win:.0f}%)")

    # 只用动量的对照组
    print("\n" + "=" * 60)
    print("  对照组: 只用 mid_ret_10s 追涨 (同样 walk-forward)")
    print("=" * 60)
    mom_col = fut.columns[fut.columns.index("mid_ret_10s")]
    ret_10s = merged[mom_col].to_numpy()
    thr = 0.001
    mom_trades = []
    step = 30
    ts = merged["ts_ms"].to_numpy()
    day_starts = sorted(set(ts // 86400000))
    pos_end = 0
    for day in day_starts[7:]:  # 跳过前 7 天 warmup
        day_lo = day * 86400000
        day_hi = (day + 1) * 86400000
        mask = (ts >= day_lo) & (ts < day_hi)
        for i in np.where(mask)[0]:
            if i < pos_end: continue
            if np.isnan(ret_10s[i]) or np.isnan(future_ret[i]): continue
            if i + step >= len(merged): continue
            if ret_10s[i] > thr:
                mid = merged["fut_mid"].to_numpy()
                gross = (mid[i+step] - mid[i]) / mid[i]
                mom_trades.append((ts[i], "LONG", gross, gross - 0.0004))
                pos_end = i + step
            elif ret_10s[i] < -thr:
                mid = merged["fut_mid"].to_numpy()
                gross = (mid[i] - mid[i+step]) / mid[i]
                mom_trades.append((ts[i], "SHORT", gross, gross - 0.0004))
                pos_end = i + step

    mnets = np.array([t[3] for t in mom_trades])
    print(f"\n总交易: {len(mom_trades)}  胜率 {(mnets>0).mean()*100:.1f}%  累计 {mnets.sum()*100:+.2f}%")

if __name__ == "__main__":
    main()
