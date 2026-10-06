#!/usr/bin/env python3
"""BTC策略全面网格搜索 - 29天数据"""
import polars as pl
import numpy as np
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = "/workspace/data/aggtrades"
HORIZON = 60
BARS_PER_DAY = 8640

def load():
    files = sorted(glob.glob(f"{DATA_DIR}/*.csv"))
    print(f"加载 {len(files)} 天...")
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

def features(c, vol, sell_vol, n):
    ret1 = c[1:]/c[:-1]-1
    feats = {}
    for w, name in [(60,"ret_10m"), (120,"ret_20m"), (180,"ret_30m"), (360,"ret_60m"), (720,"ret_120m")]:
        f = np.full(n, np.nan); f[w:] = c[w:]/c[:-w]-1; feats[name] = f
    sig = np.full(n, np.nan)
    for i in range(60, n): sig[i] = np.std(ret1[i-60:i])
    feats["sigma"] = sig
    bar_sign = np.sign(ret1); bar_sign = np.concatenate([[0], bar_sign])
    for w, name in [(60,"up_ratio_10m"), (180,"up_ratio_30m"), (360,"up_ratio_60m")]:
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
    # 趋势强度
    f = np.full(n, np.nan)
    for i in range(360, n):
        r60 = c[i]/c[i-360]-1; r10 = c[i]/c[i-60]-1
        f[i] = r60 * r10
    feats["trend_strength"] = f
    # sigma z-score
    f = np.full(n, np.nan)
    for i in range(60*24, n):
        win_sig = sig[i-60*24:i]
        f[i] = (sig[i] - np.nanmean(win_sig)) / (np.nanstd(win_sig)+1e-9)
    feats["sigma_zscore"] = f
    
    future_ret = np.full(n, np.nan)
    future_ret[:-HORIZON] = c[HORIZON:]/c[:-HORIZON]-1
    y = (future_ret > 0).astype(float)
    
    fn = ["ret_10m","ret_20m","ret_30m","ret_60m","ret_120m","sigma","sigma_zscore",
          "up_ratio_10m","up_ratio_30m","up_ratio_60m","vol_imb","price_pos_2h","trend_strength"]
    X = np.column_stack([feats[k] for k in fn])
    return X, y, fn, feats, sig

def train_model(model_type, X_tr, y_tr):
    if model_type == "logreg":
        mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)+1e-8
        X_s = (X_tr - mu)/sd
        m = LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")
        m.fit(X_s, y_tr)
        return m, mu, sd, "logreg"
    elif model_type == "rf":
        m = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=20,
                                    random_state=42, n_jobs=-1)
        m.fit(X_tr, y_tr)
        return m, None, None, "rf"
    elif model_type == "xgb":
        m = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8, 
                              reg_alpha=1.0, reg_lambda=1.0,
                              objective="binary:logistic", random_state=42, verbosity=0)
        m.fit(X_tr, y_tr)
        return m, None, None, "xgb"

def predict_model(m, X, mu, sd, mtype):
    if mtype == "logreg":
        X_s = (X - mu)/sd
    else:
        X_s = X
    return m.predict_proba(X_s)[:, 1]

def run_one_config(X, y, feat_names, feats, sig, hour, n, 
                   train_days=10, top_h=8, sigma_q=0.7, model_type="logreg",
                   fixed_hours=True):
    n_days = n // BARS_PER_DAY
    idx_60 = feat_names.index("ret_60m")
    
    # 选固定小时池
    if fixed_hours:
        tr_s, tr_e = 0, min(train_days * BARS_PER_DAY, n)
        tr_hour = hour[tr_s+400:tr_e-HORIZON]
        tr_f60 = feats["ret_60m"][tr_s+400:tr_e-HORIZON]
        tr_y_part = y[tr_s+400:tr_e-HORIZON]
        valid = ~np.isnan(tr_f60) & ~np.isnan(tr_y_part)
        h_acc = {}
        for h in range(24):
            m = valid & (tr_hour == h)
            if m.sum() < 30: continue
            h_acc[h] = ((-tr_f60[m] > 0).astype(float) == tr_y_part[m]).mean()
        hour_pool = set(sorted(h_acc, key=h_acc.get, reverse=True)[:top_h])
    
    all_correct, all_probs = [], []
    
    for test_day in range(train_days, n_days):
        tr_s = max(0, (test_day - train_days) * BARS_PER_DAY)
        tr_e = test_day * BARS_PER_DAY
        te_s = test_day * BARS_PER_DAY
        te_e = min((test_day + 1) * BARS_PER_DAY, n)
        
        tr_mask = slice(tr_s + 400, tr_e - HORIZON)
        te_mask = slice(te_s + 400, te_e - HORIZON)
        
        X_tr = X[tr_mask]; y_tr = y[tr_mask]
        h_tr = hour[tr_mask]; s_tr = sig[tr_mask]
        
        keep = ~np.isnan(X_tr).any(axis=1) & ~np.isnan(y_tr)
        if fixed_hours:
            keep &= np.isin(h_tr, list(hour_pool))
        X_tr, y_tr = X_tr[keep], y_tr[keep]
        s_tr = s_tr[keep]; h_tr = h_tr[keep]
        
        if len(X_tr) < 300: continue
        sig_thr = np.quantile(s_tr, sigma_q)
        keep2 = s_tr >= sig_thr
        X_tr_f, y_tr_f = X_tr[keep2], y_tr[keep2]
        if len(X_tr_f) < 80: continue
        
        m, mu, sd, mtype = train_model(model_type, X_tr_f, y_tr_f)
        
        X_te = X[te_mask]; y_te = y[te_mask]
        h_te = hour[te_mask]; s_te = sig[te_mask]
        
        keep = ~np.isnan(X_te).any(axis=1) & ~np.isnan(y_te)
        if fixed_hours:
            keep &= np.isin(h_te, list(hour_pool))
        X_te, y_te = X_te[keep], y_te[keep]
        s_te = s_te[keep]
        
        keep2 = s_te >= sig_thr
        X_te_f, y_te_f = X_te[keep2], y_te[keep2]
        if len(X_te_f) < 5: continue
        
        prob = predict_model(m, X_te_f, mu, sd, mtype)
        pred = (prob > 0.5).astype(int)
        all_correct.extend((pred == y_te_f).tolist())
        all_probs.extend(prob.tolist())
    
    return np.array(all_correct).astype(float), np.array(all_probs)

def evaluate(arr, probs, label, W=30, THR=0.47):
    n = len(arr)
    acc = arr.mean()
    min100 = min(arr[i:i+100].mean() for i in range(n-99))*100 if n>=100 else 0
    
    # Regime Monitor
    recent, exec = [], []
    for cv in arr:
        should = (np.mean(recent[-W:]) >= THR) if len(recent) >= W else True
        if should: exec.append(cv)
        recent.append(float(cv))
    exec = np.array(exec)
    acc_e = exec.mean()
    min100_e = min(exec[i:i+100].mean() for i in range(len(exec)-99))*100 if len(exec)>=100 else 0
    exp = acc_e * 0.8 - (1 - acc_e)
    daily = len(exec) / 19.0  # 29天 - 10d train = 19天测试
    
    # shuffle
    np.random.seed(42); shuf = arr.copy(); np.random.shuffle(shuf)
    recent2, exec2 = [], []
    for cv in shuf:
        should = (np.mean(recent2[-W:]) >= THR) if len(recent2) >= W else True
        if should: exec2.append(cv)
        recent2.append(float(cv))
    exec2 = np.array(exec2)
    min100_s = min(exec2[i:i+100].mean() for i in range(len(exec2)-99))*100 if len(exec2)>=100 else 0
    
    return {
        "label": label, "n": n, "acc": acc*100, "min100": min100,
        "n_mon": len(exec), "acc_mon": acc_e*100, "min100_mon": min100_e,
        "exp_mon": exp*100, "daily": daily, "shuf_drop": min100_e - min100_s
    }

def main():
    c, vol, sell_vol, hour, n = load()
    X, y, feat_names, feats, sig = features(c, vol, sell_vol, n)
    print(f"{n} bars = {n/8640:.1f}d | {len(feat_names)} features")
    
    results = []
    
    configs = [
        # (train_days, top_h, sigma_q, model, fixed_hours, label)
        (10, 8, 0.5, "logreg", True, "T10 H8 sigTop50% LR fix"),
        (10, 8, 0.6, "logreg", True, "T10 H8 sigTop40% LR fix"),
        (10, 8, 0.7, "logreg", True, "T10 H8 sigTop30% LR fix"),
        (10, 6, 0.7, "logreg", True, "T10 H6 sigTop30% LR fix"),
        (10, 4, 0.7, "logreg", True, "T10 H4 sigTop30% LR fix"),
        (15, 8, 0.7, "logreg", True, "T15 H8 sigTop30% LR fix"),
        (10, 8, 0.7, "xgb",    True, "T10 H8 sigTop30% XGB fix"),
        (10, 8, 0.7, "rf",     True, "T10 H8 sigTop30% RF fix"),
        (10, 8, 0.7, "logreg", False, "T10 H8 sigTop30% LR dyn"),
        (10, 6, 0.6, "xgb",    True, "T10 H6 sigTop40% XGB fix"),
        (15, 8, 0.6, "xgb",    True, "T15 H8 sigTop40% XGB fix"),
        (10, 4, 0.6, "xgb",    True, "T10 H4 sigTop40% XGB fix"),
        (10, 8, 0.5, "xgb",    True, "T10 H8 sigTop50% XGB fix"),
        # 低波动试试
        (10, 8, 0.0, "logreg", True, "T10 H8 LOWsig LR fix"),
        (10, 8, 0.0, "xgb",    True, "T10 H8 LOWsig XGB fix"),
    ]
    
    print("\n" + "="*130)
    print(f"{'Config':<28s} {'n':>6s} {'acc':>6s} {'m100':>6s} {'n_m':>6s} {'a_m':>6s} {'m100m':>6s} {'exp':>6s} {'d/d':>5s} {'drop':>6s}")
    print("="*130)
    
    for td, th, sq, mt, fh, label in configs:
        if sq == 0.0:
            # 低波动区间: 取sigma bottom 30% instead
            arr, probs = run_one_config_LOW(X, y, feat_names, feats, sig, hour, n, td, th, mt, fh)
        else:
            arr, probs = run_one_config(X, y, feat_names, feats, sig, hour, n, td, th, sq, mt, fh)
        if len(arr) < 50: 
            print(f"{label:<28s} TOO FEW")
            continue
        # 找最优regime参数
        best = None
        for W in [20, 30, 50]:
            for THR in [0.45, 0.47, 0.49]:
                r = evaluate(arr, probs, label, W, THR)
                if best is None or r["min100_mon"] > best["min100_mon"]:
                    best = r
                    best["W"] = W; best["THR"] = THR
        
        results.append(best)
        r = best
        print(f"{label:<28s} {r['n']:6d} {r['acc']:5.1f}% {r['min100']:5.1f}% {r['n_mon']:6d} {r['acc_mon']:5.1f}% {r['min100_mon']:5.1f}% {r['exp_mon']:5.1f}c {r['daily']:5.1f} {r['shuf_drop']:5.1f}%  W={r['W']} thr={r['THR']}")
    
    # Top5 排名
    print("\n" + "="*130)
    print("TOP5 by (min100_mon, acc_mon)")
    print("="*130)
    results.sort(key=lambda x: (x["min100_mon"], x["acc_mon"]), reverse=True)
    for i, r in enumerate(results[:5]):
        print(f"#{i+1} {r['label']:<28s} n={r['n_mon']:5d} acc={r['acc_mon']:.1f}% min100={r['min100_mon']:.1f}% exp={r['exp_mon']:.1f}c W={r['W']} thr={r['THR']}")
    
    # 严格筛选: min100>=45%, acc>=60%, n>=1000, shuffle_drop>3
    print("\n满足 min100>=45% & acc>=60% & n>=1000 & shuf_drop>3:")
    good = [r for r in results if r["min100_mon"] >= 45 and r["acc_mon"] >= 60 
            and r["n_mon"] >= 1000 and r["shuf_drop"] > 3]
    if not good:
        print("  无满足条件的配置 😭")
        good = [r for r in results if r["min100_mon"] >= 40 and r["acc_mon"] >= 55 and r["shuf_drop"] > 0]
        print(f"  放宽到 min100>=40% & acc>=55% & shuf_drop>0: {len(good)}个")
    for r in good[:5]:
        print(f"  ✅ {r['label']} n={r['n_mon']} acc={r['acc_mon']:.1f}% min100={r['min100_mon']:.1f}% exp={r['exp_mon']:.1f}c")

def run_one_config_LOW(X, y, feat_names, feats, sig, hour, n, train_days, top_h, model_type, fixed_hours):
    """低波动版本: sigma bottom 30%"""
    n_days = n // BARS_PER_DAY
    idx_60 = feat_names.index("ret_60m")
    
    if fixed_hours:
        tr_s, tr_e = 0, min(train_days * BARS_PER_DAY, n)
        tr_hour = hour[tr_s+400:tr_e-HORIZON]
        tr_f60 = feats["ret_60m"][tr_s+400:tr_e-HORIZON]
        tr_y_part = y[tr_s+400:tr_e-HORIZON]
        valid = ~np.isnan(tr_f60) & ~np.isnan(tr_y_part)
        h_acc = {}
        for h in range(24):
            m = valid & (tr_hour == h)
            if m.sum() < 30: continue
            h_acc[h] = ((-tr_f60[m] > 0).astype(float) == tr_y_part[m]).mean()
        hour_pool = set(sorted(h_acc, key=h_acc.get, reverse=True)[:top_h])
    
    all_correct, all_probs = [], []
    
    for test_day in range(train_days, n_days):
        tr_s = max(0, (test_day - train_days) * BARS_PER_DAY)
        tr_e = test_day * BARS_PER_DAY
        te_s = test_day * BARS_PER_DAY
        te_e = min((test_day + 1) * BARS_PER_DAY, n)
        
        tr_mask = slice(tr_s + 400, tr_e - HORIZON)
        te_mask = slice(te_s + 400, te_e - HORIZON)
        
        X_tr = X[tr_mask]; y_tr = y[tr_mask]
        h_tr = hour[tr_mask]; s_tr = sig[tr_mask]
        
        keep = ~np.isnan(X_tr).any(axis=1) & ~np.isnan(y_tr)
        if fixed_hours: keep &= np.isin(h_tr, list(hour_pool))
        X_tr, y_tr = X_tr[keep], y_tr[keep]; s_tr = s_tr[keep]
        
        if len(X_tr) < 300: continue
        sig_thr = np.quantile(s_tr, 0.3)  # bottom 30%
        keep2 = s_tr <= sig_thr
        X_tr_f, y_tr_f = X_tr[keep2], y_tr[keep2]
        if len(X_tr_f) < 80: continue
        
        m, mu, sd, mtype = train_model(model_type, X_tr_f, y_tr_f)
        
        X_te = X[te_mask]; y_te = y[te_mask]
        h_te = hour[te_mask]; s_te = sig[te_mask]
        
        keep = ~np.isnan(X_te).any(axis=1) & ~np.isnan(y_te)
        if fixed_hours: keep &= np.isin(h_te, list(hour_pool))
        X_te, y_te = X_te[keep], y_te[keep]; s_te = s_te[keep]
        
        keep2 = s_te <= sig_thr
        X_te_f, y_te_f = X_te[keep2], y_te[keep2]
        if len(X_te_f) < 5: continue
        
        prob = predict_model(m, X_te_f, mu, sd, mtype)
        pred = (prob > 0.5).astype(int)
        all_correct.extend((pred == y_te_f).tolist())
        all_probs.extend(prob.tolist())
    
    return np.array(all_correct).astype(float), np.array(all_probs)

if __name__ == "__main__":
    main()
