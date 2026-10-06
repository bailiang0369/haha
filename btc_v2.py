#!/usr/bin/env python3
"""
BTCUSDT Binary Options Strategy — V2 (单币种专注版)
===================================================
- 16天 aggTrades 10s bars
- 多周期动量 + LogReg
- 严格 Walk-Forward: 每天重选小时/波动率阈值/重新训练
- Regime Monitor: 滑窗检测 + 自动暂停
- Min100 诊断: 找出到底哪段时间崩了
"""
import polars as pl
import numpy as np
import glob
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = "/workspace/data/aggtrades"
HORIZON = 60       # 10min at 10s
BARS_PER_DAY = 8640
TRAIN_DAYS = 5     # 训练窗口
TEST_DAYS_PER_ROUND = 1  # 每天滚动测试

def load_data():
    files = sorted(glob.glob(f"{DATA_DIR}/*.csv"))
    print(f"加载 {len(files)} 天 aggTrades...")
    all_bars = []
    for f in files:
        df = pl.read_csv(f, has_header=False,
                         new_columns=["agg_id","price","qty","first_id","last_id","ts_us","is_buyer_maker","is_best"])
        df = df.with_columns([(pl.col("ts_us") // 10_000_000 * 10_000_000).alias("bucket")])
        bar = df.group_by("bucket").agg([
            pl.col("price").last().alias("close"),
            pl.col("qty").sum().alias("vol"),
            (pl.col("qty") * (~pl.col("is_buyer_maker")).cast(int)).sum().alias("sell_vol"),
        ]).sort("bucket")
        all_bars.append(bar)
    bars = pl.concat(all_bars).sort("bucket")
    c = bars["close"].to_numpy().astype(np.float64)
    vol = bars["vol"].to_numpy().astype(np.float64)
    sell_vol = bars["sell_vol"].to_numpy().astype(np.float64)
    bucket = bars["bucket"].to_numpy()
    hour = (bucket % 86400000000) // 3600000000
    n = len(c)
    print(f"  {n:,} bars = {n/BARS_PER_DAY:.1f}d | close=[{c.min():.1f}, {c.max():.1f}]")
    return c, vol, sell_vol, hour, n

def build_all_features(c, vol, sell_vol, n):
    """构造所有特征 + future label."""
    ret1 = c[1:]/c[:-1]-1  # 1 bar return
    
    feats = {}
    # 1. 多周期动量 (都用反转方向，权重预期为负)
    for w, name in [(60,"ret_10m"), (120,"ret_20m"), (180,"ret_30m"), (360,"ret_60m")]:
        f = np.full(n, np.nan); f[w:] = c[w:]/c[:-w]-1; feats[name] = f
    
    # 2. 波动率
    sig = np.full(n, np.nan)
    for i in range(60, n): sig[i] = np.std(ret1[i-60:i])
    feats["sigma"] = sig
    
    # 3. 最近N根bar的胜率 (buy pressure proxy: close>open的bar比例)
    bar_sign = np.sign(ret1); bar_sign = np.concatenate([[0], bar_sign])
    for w, name in [(60,"up_ratio_10m"), (180,"up_ratio_30m")]:
        f = np.full(n, np.nan)
        for i in range(w, n): f[i] = (bar_sign[i-w:i] > 0).mean()
        feats[name] = f
    
    # 4. Volume imbalance
    buy = vol - sell_vol
    total = vol + 1e-9
    feats["vol_imb"] = np.where(total > 0, buy/total, 0.5)
    
    # 5. 价格位置: 相对过去120m高低点的位置 (0=最低, 1=最高)
    f = np.full(n, np.nan)
    for i in range(720, n):
        win = c[i-720:i]
        hi, lo = win.max(), win.min()
        f[i] = (c[i] - lo) / (hi - lo + 1e-9)
    feats["price_pos_2h"] = f
    
    # 6. Future label
    future_ret = np.full(n, np.nan)
    future_ret[:-HORIZON] = c[HORIZON:]/c[:-HORIZON]-1
    y = (future_ret > 0).astype(float)
    
    feat_names = ["ret_10m","ret_20m","ret_30m","ret_60m","sigma",
                  "up_ratio_10m","up_ratio_30m","vol_imb","price_pos_2h"]
    X = np.column_stack([feats[k] for k in feat_names])
    
    return X, y, feat_names

def run_walk_forward():
    c, vol, sell_vol, hour, n = load_data()
    X, y, feat_names = build_all_features(c, vol, sell_vol, n)
    n_days = n // BARS_PER_DAY
    
    print(f"\n特征: {feat_names}")
    print(f"Walk-Forward: {TRAIN_DAYS}d train → 1d test, 共 {n_days-TRAIN_DAYS} 轮\n")
    
    # === Walk-Forward ===
    all_rows = []  # (pred, true, prob, day_idx, hour, sigma)
    
    for test_day in range(TRAIN_DAYS, n_days):
        tr_s = (test_day - TRAIN_DAYS) * BARS_PER_DAY
        tr_e = test_day * BARS_PER_DAY
        te_s = test_day * BARS_PER_DAY
        te_e = min((test_day + 1) * BARS_PER_DAY, n)
        
        # 训练集有效范围
        tr_mask = slice(tr_s + 400, tr_e - HORIZON)
        te_mask = slice(te_s + 400, te_e - HORIZON)
        
        X_tr = X[tr_mask]; y_tr = y[tr_mask]
        h_tr = hour[tr_mask]; s_tr = X[tr_mask, feat_names.index("sigma")]
        
        # 清除训练集NaN
        keep = ~np.isnan(X_tr).any(axis=1) & ~np.isnan(y_tr)
        X_tr, y_tr = X_tr[keep], y_tr[keep]
        h_tr, s_tr = h_tr[keep], s_tr[keep]
        
        if len(X_tr) < 500: continue
        
        # === 在训练集上选 Top8 小时 ===
        # 先用简单的ret_60m反转算每个小时的acc
        idx_60 = feat_names.index("ret_60m")
        f60_tr = X_tr[:, idx_60]
        h_acc = {}
        for h in range(24):
            m = h_tr == h
            if m.sum() < 20: continue
            acc_h = ((-f60_tr[m] > 0).astype(float) == y_tr[m]).mean()
            h_acc[h] = acc_h
        top_h_set = set(sorted(h_acc, key=h_acc.get, reverse=True)[:8])
        
        # === 在训练集上选 sigma 阈值 (top 30%) ===
        sig_thr = np.quantile(s_tr, 0.7)
        
        # === 训练 LogReg (只用过滤后的训练样本) ===
        keep2 = np.isin(h_tr, list(top_h_set)) & (s_tr >= sig_thr)
        X_tr_f, y_tr_f = X_tr[keep2], y_tr[keep2]
        
        if len(X_tr_f) < 100: continue
        
        mu = X_tr_f.mean(axis=0); sd = X_tr_f.std(axis=0) + 1e-8
        X_tr_s = (X_tr_f - mu) / sd
        
        lr = LogisticRegression(C=0.3, max_iter=2000, solver="lbfgs")
        lr.fit(X_tr_s, y_tr_f)
        
        # === 测试集 ===
        X_te = X[te_mask]; y_te = y[te_mask]
        h_te = hour[te_mask]; s_te = X[te_mask, feat_names.index("sigma")]
        
        keep = ~np.isnan(X_te).any(axis=1) & ~np.isnan(y_te)
        X_te, y_te = X_te[keep], y_te[keep]
        h_te, s_te = h_te[keep], s_te[keep]
        
        keep2 = np.isin(h_te, list(top_h_set)) & (s_te >= sig_thr)
        X_te_f, y_te_f = X_te[keep2], y_te[keep2]
        h_te_f, s_te_f = h_te[keep2], s_te[keep2]
        
        if len(X_te_f) < 10: continue
        
        X_te_s = (X_te_f - mu) / sd
        prob = lr.predict_proba(X_te_s)[:, 1]
        pred = (prob > 0.5).astype(int)
        
        auc = roc_auc_score(y_te_f, prob)
        acc = (pred == y_te_f).mean()
        
        day_date = f"2026-09-{(14+test_day)%30+1:02d}" if test_day < 16 else f"2026-10-{(test_day-15):02d}"
        print(f"  Day {test_day+1:2d} (UTC{te_s//BARS_PER_DAY:2d}h→): n={len(y_te_f):4d} acc={acc*100:5.1f}% AUC={auc:.4f} | TopH={sorted(top_h_set)} sig>={sig_thr:.6f}")
        
        for p, t, pr, hh, ss in zip(pred, y_te_f, prob, h_te_f, s_te_f):
            all_rows.append({"pred": int(p), "true": int(t), "prob": float(pr),
                             "day": test_day, "hour": int(hh), "sigma": float(ss)})
    
    # === 汇总 ===
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD 汇总")
    print(f"{'='*70}")
    
    arr = np.array([r["pred"] == r["true"] for r in all_rows]).astype(float)
    probs = np.array([r["prob"] for r in all_rows])
    preds = np.array([r["pred"] for r in all_rows])
    trues = np.array([r["true"] for r in all_rows])
    
    n_total = len(arr)
    acc = arr.mean()
    auc = roc_auc_score(trues, probs)
    
    # min100 诊断
    if n_total >= 100:
        min100_val = 1.0; min100_start = 0
        for i in range(n_total - 99):
            win = arr[i:i+100].mean()
            if win < min100_val:
                min100_val = win; min100_start = i
        min100 = min100_val * 100
        print(f"\n  Total n={n_total:,} | Base acc={acc*100:.1f}% | AUC={auc:.4f}")
        print(f"  min100={min100:.1f}% (起点: row {min100_start})")
        
        # 打印最差100笔的详情
        print(f"\n  🔍 最差100笔窗口诊断 (rows {min100_start}~{min100_start+100}):")
        win_rows = all_rows[min100_start:min100_start+100]
        win_days = [r["day"] for r in win_rows]
        win_hours = [r["hour"] for r in win_rows]
        unique_days = sorted(set(win_days))
        print(f"    涉及天数: {len(unique_days)} 天 (day {unique_days})")
        # 每小时统计
        for h in sorted(set(win_hours)):
            m = [r for r in win_rows if r["hour"] == h]
            correct = sum(r["pred"] == r["true"] for r in m)
            print(f"    UTC{h:02d}: {len(m)}笔 正确{correct} ({correct/len(m)*100:.0f}%)")
    else:
        print(f"  n={n_total} 太少")
    
    # === 置信度过滤 ===
    print(f"\n  --- 置信度过滤 ---")
    for pt in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]:
        m = (probs > 0.5 + pt) | (probs < 0.5 - pt)
        if m.sum() < 50: continue
        a = arr[m]
        acc_m = a.mean()
        if len(a) >= 100:
            min100_m = min(a[i:i+100].mean() for i in range(len(a)-99))*100
        else:
            min100_m = 0
        exp = acc_m * 0.8 - (1 - acc_m)  # 0.8 payout
        print(f"  |p-0.5|>{pt:.2f}: n={m.sum():4d} acc={acc_m*100:.1f}% min100={min100_m:.1f}% exp={exp*100:.1f}c")
    
    # === Regime Monitor ===
    print(f"\n  --- + Regime Monitor (W=30, pause<0.47) ---")
    W, THR = 30, 0.47
    recent, executed = [], []
    for cv in arr:
        should = (np.mean(recent[-W:]) >= THR) if len(recent) >= W else True
        if should: executed.append(cv)
        recent.append(float(cv))
    
    executed = np.array(executed)
    acc_e = executed.mean()
    if len(executed) >= 100:
        min100_e = min(executed[i:i+100].mean() for i in range(len(executed)-99))*100
    else:
        min100_e = 0
    exp_e = acc_e * 0.8 - (1 - acc_e)
    print(f"  n={len(executed):4d} acc={acc_e*100:.1f}% min100={min100_e:.1f}% exp={exp_e*100:.1f}c")
    
    # shuffle 验证
    np.random.seed(42); shuf = arr.copy(); np.random.shuffle(shuf)
    recent2, exec2 = [], []
    for cv in shuf:
        should = (np.mean(recent2[-W:]) >= THR) if len(recent2) >= W else True
        if should: exec2.append(cv)
        recent2.append(float(cv))
    exec2 = np.array(exec2)
    acc_s = exec2.mean()
    min100_s = min(exec2[i:i+100].mean() for i in range(len(exec2)-99))*100
    print(f"  Shuffled: n={len(exec2):4d} acc={acc_s*100:.1f}% min100={min100_s:.1f}%")
    print(f"  ✅ 通过" if min100_e - min100_s > 5 else f"  ⚠️ 未通过shuffle检查")
    
    # === 逐天表现 ===
    print(f"\n  --- 逐天表现 ---")
    daily = {}
    for r in all_rows:
        d = r["day"]
        if d not in daily: daily[d] = []
        daily[d].append(float(r["pred"] == r["true"]))
    for d in sorted(daily):
        a = np.mean(daily[d])
        flag = "💀" if a < 0.45 else "⚠️" if a < 0.52 else "✅" if a > 0.6 else "  "
        print(f"    Day {d+1:2d}: n={len(daily[d]):4d} acc={a*100:5.1f}% {flag}")
    
    return all_rows

if __name__ == "__main__":
    run_walk_forward()
