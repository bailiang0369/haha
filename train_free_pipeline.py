"""
最简免费 Pipeline v2 — 修复版
币安现货 1s K 线 → 10s 重采样 → ret_1bar + ret_6bar → LightGBM → walk-forward
"""
import os, glob, time, argparse, json, sys
import numpy as np
import polars as pl
import lightgbm as lgb
from datetime import datetime, timezone
from sklearn.metrics import roc_auc_score

def log(msg): print(msg, flush=True)

def load_and_resample(data_dir, target_sec=10):
    """加载 1s parquet → 重采样到 target_sec 粒度, 返回 (ts_ms, mid)"""
    files = sorted(glob.glob(f"{data_dir}/*.parquet"))
    log(f"加载 {len(files)} 个 parquet...")
    
    dfs = []
    for f in files:
        try: dfs.append(pl.read_parquet(f))
        except: pass
    
    df = pl.concat(dfs).sort("ts_ms")
    log(f"  1s 原始: {len(df):,} 行")
    
    # 重采样到 target_sec 粒度: 按时间桶分组, 取每组最后一个 close
    df = df.with_columns(
        (pl.col("ts_ms") // (target_sec * 1_000_000) * (target_sec * 1_000_000)).alias("bucket")
    )
    # 每个 bucket 取最后一行 (即该时间段的 close)
    df_10s = df.group_by("bucket").agg([
        pl.col("ts_ms").last().alias("ts_ms"),
        pl.col("mid").last().alias("mid"),
    ]).sort("ts_ms")
    
    ts = df_10s["ts_ms"].to_numpy()
    mid = df_10s["mid"].to_numpy()
    log(f"  {target_sec}s 重采样: {len(ts):,} 行, 范围: {len(ts)*target_sec/86400:.1f} 天")
    
    return ts, mid

def compute_features(mid, horizon_min=5, label_quantile=0.25):
    """
    10s 粒度下:
      ret_1  = mid[t]/mid[t-1] - 1  (10s 动量)
      ret_6  = mid[t]/mid[t-6] - 1  (60s = 1min 动量)
      future_ret step = horizon_min * 6  (如 5min = 30 步)
    """
    n = len(mid)
    ret_1 = np.full(n, np.nan); ret_1[1:] = mid[1:] / mid[:-1] - 1
    ret_6 = np.full(n, np.nan); ret_6[6:] = mid[6:] / mid[:-6] - 1
    
    step = horizon_min * 6  # 10s bar
    future_ret = np.full(n, np.nan)
    for i in range(n - step):
        future_ret[i] = (mid[i + step] - mid[i]) / mid[i]
    
    log(f"  step={step} ({horizon_min}min), labeled≈{((~np.isnan(future_ret)) & (~np.isnan(ret_1))).sum():,}")
    return ret_1, ret_6, future_ret, step

def walk_forward(ts, ret_1, ret_6, future_ret,
                 train_days=7, horizon_min=5, label_quantile=0.25):
    """Walk-forward"""
    step = horizon_min * 6
    has = ~np.isnan(future_ret) & ~np.isnan(ret_1) & ~np.isnan(ret_6)
    
    # 用第一个 train_days 算 balanced threshold
    day_start = ts // 86400000
    unique_days = sorted(set(day_start))
    first_week = (day_start >= unique_days[0]) & (day_start < unique_days[train_days])
    thr = abs(np.quantile(future_ret[has & first_week], label_quantile))
    labeled = has & (np.abs(future_ret) > thr)
    
    l_ts = ts[labeled]; l_fr = future_ret[labeled]
    l_r1 = ret_1[labeled]; l_r6 = ret_6[labeled]
    
    unique_days = sorted(set(l_ts // 86400000))
    log(f"  labeled={len(l_ts):,}, thr={thr*10000:.2f}bps, 共 {len(unique_days)} 天")
    
    all_p, all_y, all_t = [], [], []
    per_day = {}
    n_models = len(unique_days) - train_days
    log(f"  walk-forward: train={train_days}d → test=1d, 共 {n_models} 个模型")
    
    for day_i in range(train_days, len(unique_days)):
        tl = unique_days[day_i - train_days] * 86400000000
        th = unique_days[day_i] * 86400000000
        ql = unique_days[day_i] * 86400000000
        qh = (unique_days[day_i] + 1) * 86400000000
        
        tm = (l_ts >= tl) & (l_ts < th)
        qm = (l_ts >= ql) & (l_ts < qh)
        
        if tm.sum() < 5000 or qm.sum() < 200:
            continue
        
        X_tr = np.column_stack([l_r1[tm], l_r6[tm]])
        y_tr = (l_fr[tm] > 0).astype(int)
        X_te = np.column_stack([l_r1[qm], l_r6[qm]])
        y_te = (l_fr[qm] > 0).astype(int)
        
        m = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1,
                                num_leaves=31, min_child_samples=500,
                                verbose=-1, random_state=42, n_jobs=-1)
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_te)[:, 1]
        
        all_p.extend(proba)
        all_y.extend(y_te)
        all_t.extend(l_ts[qm])
        
        d = unique_days[day_i]
        acc = ((proba > 0.5).astype(int) == y_te).mean()
        per_day[int(d)] = round(float(acc*100), 2)
        
        if (day_i - train_days + 1) % 30 == 0:
            elapsed = time.time() - t0
            log(f"    进度 {day_i - train_days + 1}/{n_models} ({(day_i - train_days + 1)*100//n_models}%) 耗时 {elapsed:.0f}s")
    
    return np.array(all_p), np.array(all_y), np.array(all_t), thr, per_day

def apply_risk_control(all_p, all_y, all_t, conf_thr=0.7,
                       loss_breaker_n=3, loss_breaker_pause_min=15):
    conf = np.abs(all_p - 0.5) * 2
    correct = (all_p > 0.5).astype(int) == all_y
    m_conf = conf >= conf_thr
    
    streak = 0; pause_until = 0
    pause_ms = loss_breaker_pause_min * 60 * 1000
    executed = np.zeros(len(all_p), dtype=bool)
    
    conf_idx = np.where(m_conf)[0]
    for i in conf_idx:
        t = all_t[i]
        if t < pause_until:
            continue
        executed[i] = True
        if correct[i]: streak = 0
        else:
            streak += 1
            if streak >= loss_breaker_n:
                pause_until = t + pause_ms
                streak = 0
    
    return executed, correct, conf, m_conf

def main():
    global t0; t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/workspace/data/spot_1s")
    ap.add_argument("--horizon-min", type=int, default=5)
    ap.add_argument("--train-days", type=int, default=7)
    args = ap.parse_args()
    
    log(f"=== 最简免费 Pipeline: {args.data_dir} ===")
    log(f"  horizon={args.horizon_min}min, train={args.train_days}d")
    
    # 1. 加载 + 重采样
    log("\n[1] 加载 + 10s 重采样...")
    ts, mid = load_and_resample(args.data_dir, target_sec=10)
    
    # 2. 特征
    log("\n[2] 特征计算...")
    ret_1, ret_6, future_ret, step = compute_features(mid, args.horizon_min)
    
    # 3. Walk-forward
    log("\n[3] Walk-forward 训练...")
    all_p, all_y, all_t, thr, per_day = walk_forward(
        ts, ret_1, ret_6, future_ret,
        train_days=args.train_days, horizon_min=args.horizon_min)
    
    log(f"\n训练完成! 耗时 {time.time()-t0:.1f}s")
    
    # 4. 评估
    log("\n" + "="*70)
    log(f"  RESULTS: 现货 1s K 线 → 10s 重采样 → ret_10s+ret_1m → pred {args.horizon_min}min")
    log("="*70)
    
    auc = roc_auc_score(all_y, all_p)
    correct_nocb = (all_p > 0.5).astype(int) == all_y
    
    log(f"\nAUC = {auc:.4f}")
    log(f"无风控全量:  {correct_nocb.mean()*100:.1f}% ({len(all_p):,} 笔)")
    
    log(f"\n{'conf':>5s}  {'n':>8s}  {'acc':>7s}  {'cb_n':>8s}  {'cb_acc':>7s}  {'cb_min100':>9s}  {'每笔期望':>9s}")
    log("="*65)
    
    results = []
    payout = 0.8
    for c in [0.0, 0.3, 0.5, 0.7, 0.8]:
        executed, correct_arr, conf, m_conf = apply_risk_control(
            all_p, all_y, all_t, conf_thr=c)
        cb_n = executed.sum()
        if cb_n < 100: continue
        cb_acc = correct_arr[executed].mean()*100
        arr = correct_arr[executed].astype(float)
        min100 = min(arr[i:i+100].mean() for i in range(len(arr)-99))*100 if len(arr) >= 100 else cb_acc
        exp = cb_acc/100 * payout - (1 - cb_acc/100)
        
        log(f"{c:>5.1f}  {m_conf.sum():>8,}  {correct_nocb[m_conf].mean()*100:>6.1f}%  {cb_n:>8,}  {cb_acc:>6.1f}%  {min100:>8.1f}%  {exp*100:>8.1f} cent")
        results.append({"conf": c, "n_conf_thr": int(m_conf.sum()), "acc_nocb": round(correct_nocb[m_conf].mean()*100,1),
                        "cb_n": int(cb_n), "cb_acc": round(cb_acc,1), "cb_min100": round(min100,1), "exp_cent": round(exp*100,1)})
    
    # 按天
    log(f"\n按天准确率 (无风控, {len(per_day)} 天):")
    for d, acc in sorted(per_day.items()):
        dt = datetime.fromtimestamp(d*86400, tz=timezone.utc).strftime("%Y-%m-%d")
        flag = "⚠️" if acc < 55 else ("✅" if acc > 70 else "  ")
        log(f"  {flag} {dt}: {acc:.1f}%")
    
    # 保存
    out = {
        "auc": round(float(auc), 4), "thr_bps": round(float(thr*10000), 2),
        "total_predictions": len(all_p), "n_test_days": len(per_day),
        "results_by_conf": results, "per_day_acc": per_day,
        "data_range_ms": [int(ts[0]), int(ts[-1])],
        "data_range": [datetime.fromtimestamp(ts[0]/1000, tz=timezone.utc).isoformat(),
                       datetime.fromtimestamp(ts[-1]/1000, tz=timezone.utc).isoformat()],
        "horizon_min": args.horizon_min, "train_days": args.train_days,
        "risk_control": {"conf_thr": 0.7, "loss_breaker_n": 3, "loss_breaker_pause_min": 15},
    }
    os.makedirs("/workspace/results", exist_ok=True)
    fname = f"/workspace/results/free_spot_{args.horizon_min}min.json"
    with open(fname, "w") as f: json.dump(out, f, indent=2, default=str)
    log(f"\n✅ 结果: {fname}, 总耗时 {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
