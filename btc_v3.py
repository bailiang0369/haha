#!/usr/bin/env python3
"""诊断失败日 + 尝试更稳的配置"""
import polars as pl
import numpy as np
import glob
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

DATA_DIR = "/workspace/data/aggtrades"
HORIZON = 60
BARS_PER_DAY = 8640

def load_data():
    files = sorted(glob.glob(f"{DATA_DIR}/*.csv"))
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
    return c, vol, sell_vol, hour, len(c)

def diagnose_failure_days():
    c, vol, sell_vol, hour, n = load_data()
    n_days = n // BARS_PER_DAY
    
    print("="*70)
    print("失败日行情诊断")
    print("="*70)
    print(f"{'Day':>4} {'日期':>12} {'Open':>9} {'Close':>9} {'Return':>8} {'MaxDD':>8} {'MaxUp':>8} {'Sigma':>9} {'Trend?':>7}")
    
    for d in range(n_days):
        s, e = d*BARS_PER_DAY, (d+1)*BARS_PER_DAY
        day_c = c[s:e]
        day_ret = day_c[-1]/day_c[0]-1
        # max drawdown
        peak = np.maximum.accumulate(day_c)
        dd = (day_c/peak - 1).min()
        # max up
        trough = np.minimum.accumulate(day_c)
        mu = (day_c/trough - 1).max()
        # sigma
        r1 = day_c[1:]/day_c[:-1]-1
        sig = np.std(r1)
        
        # 趋势检测: 60m动量方向
        day_ret_60 = day_c[60:]/day_c[:-60]-1
        # 如果整天大部分ret_60同方向 = 趋势市
        frac_up = (day_ret_60 > 0).mean()
        trend = "📈UP" if frac_up > 0.6 else "📉DN" if frac_up < 0.4 else "震荡"
        
        # 失败日标记
        marker = " 💀" if d+1 in [7, 11, 13, 16] else ""
        
        print(f"{d+1:4d} 9/{14+d:02d}   {day_c[0]:9.1f} {day_c[-1]:9.1f} {day_ret*100:7.2f}% {dd*100:7.2f}% {mu*100:7.2f}% {sig*10000:7.1f}bps {trend:>7}{marker}")
    
    print("\n💀 失败日特征: 大趋势市 + 高波动率")

def build_features(c, vol, sell_vol, n):
    ret1 = c[1:]/c[:-1]-1
    feats = {}
    for w, name in [(60,"ret_10m"), (120,"ret_20m"), (180,"ret_30m"), (360,"ret_60m")]:
        f = np.full(n, np.nan); f[w:] = c[w:]/c[:-w]-1; feats[name] = f
    sig = np.full(n, np.nan)
    for i in range(60, n): sig[i] = np.std(ret1[i-60:i])
    feats["sigma"] = sig
    bar_sign = np.sign(ret1); bar_sign = np.concatenate([[0], bar_sign])
    for w, name in [(60,"up_ratio_10m"), (180,"up_ratio_30m")]:
        f = np.full(n, np.nan)
        for i in range(w, n): f[i] = (bar_sign[i-w:i] > 0).mean()
        feats[name] = f
    buy = vol - sell_vol
    feats["vol_imb"] = np.where(vol > 0, buy/(vol+1e-9), 0.5)
    f = np.full(n, np.nan)
    for i in range(720, n):
        win = c[i-720:i]; hi, lo = win.max(), win.min()
        f[i] = (c[i] - lo) / (hi - lo + 1e-9)
    feats["price_pos_2h"] = f
    
    # ============ 新特征: 趋势强度检测 ============
    # 如果ret_60m和ret_10m同方向 = 趋势
    f = np.full(n, np.nan)
    for i in range(360, n):
        r60 = c[i]/c[i-360]-1
        r10 = c[i]/c[i-60]-1
        f[i] = r60 * r10  # 同号=正(趋势), 异号=负(反转)
    feats["trend_strength"] = f
    
    # ============ 新特征: 波动率z-score ============
    # sigma相对过去24h的位置
    f = np.full(n, np.nan)
    for i in range(60*24, n):
        win_sig = sig[i-60*24:i]
        mu_s = np.nanmean(win_sig); sd_s = np.nanstd(win_sig)+1e-9
        f[i] = (sig[i] - mu_s) / sd_s
    feats["sigma_zscore"] = f
    
    future_ret = np.full(n, np.nan)
    future_ret[:-HORIZON] = c[HORIZON:]/c[:-HORIZON]-1
    y = (future_ret > 0).astype(float)
    
    feat_names = ["ret_10m","ret_20m","ret_30m","ret_60m","sigma","sigma_zscore",
                  "up_ratio_10m","up_ratio_30m","vol_imb","price_pos_2h","trend_strength"]
    X = np.column_stack([feats[k] for k in feat_names])
    return X, y, feat_names, feats

def run_robust_wf():
    c, vol, sell_vol, hour, n = load_data()
    X, y, feat_names, feats = build_features(c, vol, sell_vol, n)
    n_days = n // BARS_PER_DAY
    
    print("\n" + "="*70)
    print("稳健 Walk-Forward: 固定小时池 + 10d train + XGBoost")
    print("="*70)
    
    # === 用前10天选固定小时池 ===
    TR_INIT = 10
    idx_60 = feat_names.index("ret_60m")
    tr_s, tr_e = 0, TR_INIT * BARS_PER_DAY
    tr_hour = hour[tr_s+400:tr_e-HORIZON]
    tr_f60 = feats["ret_60m"][tr_s+400:tr_e-HORIZON]
    tr_y = y[tr_s+400:tr_e-HORIZON]
    valid = ~np.isnan(tr_f60) & ~np.isnan(tr_y)
    h_acc = {}
    for h in range(24):
        m = valid & (tr_hour == h)
        if m.sum() < 50: continue
        acc_h = ((-tr_f60[m] > 0).astype(float) == tr_y[m]).mean()
        h_acc[h] = acc_h
    fixed_top_h = set(sorted(h_acc, key=h_acc.get, reverse=True)[:8])
    print(f"固定Top8小时: {sorted(fixed_top_h)}")
    for h in sorted(fixed_top_h):
        print(f"  UTC{h:02d}: acc={h_acc[h]*100:.1f}%")
    
    all_preds, all_trues, all_probs, all_correct = [], [], [], []
    
    for test_day in range(TR_INIT, n_days):
        tr_s = max(0, (test_day - 10) * BARS_PER_DAY)
        tr_e = test_day * BARS_PER_DAY
        te_s = test_day * BARS_PER_DAY
        te_e = min((test_day + 1) * BARS_PER_DAY, n)
        
        tr_mask = slice(tr_s + 400, tr_e - HORIZON)
        te_mask = slice(te_s + 400, te_e - HORIZON)
        
        X_tr = X[tr_mask]; y_tr = y[tr_mask]
        h_tr = hour[tr_mask]; s_tr = X[tr_mask, feat_names.index("sigma")]
        
        keep = ~np.isnan(X_tr).any(axis=1) & ~np.isnan(y_tr) & np.isin(h_tr, list(fixed_top_h))
        X_tr, y_tr = X_tr[keep], y_tr[keep]
        s_tr = s_tr[keep]
        
        if len(X_tr) < 300: continue
        sig_thr = np.quantile(s_tr, 0.8)  # top 20%
        
        keep2 = s_tr >= sig_thr
        X_tr_f, y_tr_f = X_tr[keep2], y_tr[keep2]
        
        if len(X_tr_f) < 100: continue
        
        # 标准化 + XGB
        mu = X_tr_f.mean(axis=0); sd = X_tr_f.std(axis=0)+1e-8
        X_tr_s = (X_tr_f - mu)/sd
        
        model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=1.0,
                                  objective="binary:logistic", random_state=42, verbosity=0)
        model.fit(X_tr_s, y_tr_f)
        
        X_te = X[te_mask]; y_te = y[te_mask]
        h_te = hour[te_mask]; s_te = X[te_mask, feat_names.index("sigma")]
        
        keep = ~np.isnan(X_te).any(axis=1) & ~np.isnan(y_te) & np.isin(h_te, list(fixed_top_h))
        X_te, y_te = X_te[keep], y_te[keep]
        s_te = s_te[keep]
        
        keep2 = s_te >= sig_thr
        X_te_f, y_te_f = X_te[keep2], y_te[keep2]
        
        if len(X_te_f) < 10: continue
        
        X_te_s = (X_te_f - mu)/sd
        prob = model.predict_proba(X_te_s)[:,1]
        pred = (prob > 0.5).astype(int)
        
        all_preds.extend(pred.tolist())
        all_trues.extend(y_te_f.tolist())
        all_probs.extend(prob.tolist())
        all_correct.extend((pred == y_te_f).tolist())
        
        auc = roc_auc_score(y_te_f, prob)
        acc = (pred == y_te_f).mean()
        flag = "💀" if acc < 0.45 else "⚠️" if acc < 0.52 else "✅" if acc > 0.6 else "  "
        print(f"  Day{test_day+1:2d}: n={len(y_te_f):4d} acc={acc*100:5.1f}% AUC={auc:.4f} {flag}")
    
    arr = np.array(all_correct).astype(float)
    probs = np.array(all_probs)
    
    acc = arr.mean()
    print(f"\n{'='*70}")
    print(f"Base: n={len(arr):,} acc={acc*100:.1f}%")
    
    if len(arr) >= 100:
        min100 = min(arr[i:i+100].mean() for i in range(len(arr)-99))*100
        print(f"min100={min100:.1f}%")
    
    # + Regime Monitor
    W, THR = 30, 0.47
    recent, executed = [], []
    for cv in arr:
        should = (np.mean(recent[-W:]) >= THR) if len(recent) >= W else True
        if should: executed.append(cv)
        recent.append(float(cv))
    executed = np.array(executed)
    acc_e = executed.mean()
    min100_e = min(executed[i:i+100].mean() for i in range(len(executed)-99))*100 if len(executed)>=100 else 0
    exp = acc_e*0.8 - (1-acc_e)
    print(f"\n+Regime Mon (W={W}, thr={THR}): n={len(executed):,} acc={acc_e*100:.1f}% min100={min100_e:.1f}% exp={exp*100:.1f}c")
    
    # shuffle
    np.random.seed(42); shuf = arr.copy(); np.random.shuffle(shuf)
    recent2, exec2 = [], []
    for cv in shuf:
        should = (np.mean(recent2[-W:]) >= THR) if len(recent2) >= W else True
        if should: exec2.append(cv)
        recent2.append(float(cv))
    exec2 = np.array(exec2)
    min100_s = min(exec2[i:i+100].mean() for i in range(len(exec2)-99))*100
    print(f"Shuffled: n={len(exec2):,} min100={min100_s:.1f}%")
    print(f"✅ 通过 shuffle" if min100_e - min100_s > 5 else "⚠️ 未通过")
    
    # 不同阈值测试
    print(f"\n  --- 不同配置对比 ---")
    for W2, THR2 in [(20, 0.45), (30, 0.47), (50, 0.49), (40, 0.48)]:
        recent, exec = [], []
        for cv in arr:
            should = (np.mean(recent[-W2:]) >= THR2) if len(recent) >= W2 else True
            if should: exec.append(cv)
            recent.append(float(cv))
        exec = np.array(exec)
        a = exec.mean()
        m = min(exec[i:i+100].mean() for i in range(len(exec)-99))*100 if len(exec)>=100 else 0
        e = a*0.8 - (1-a)
        print(f"  W={W2:2d} thr={THR2:.2f}: n={len(exec):4d} acc={a*100:.1f}% min100={m:.1f}% exp={e*100:.1f}c")
    
    return arr, probs

if __name__ == "__main__":
    diagnose_failure_days()
    run_robust_wf()
