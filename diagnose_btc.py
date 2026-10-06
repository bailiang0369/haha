#!/usr/bin/env python3
"""彻底诊断BTC aggTrades数据 - 找出哪些小时/波动率/条件下真有预测力"""
import polars as pl
import numpy as np
import glob
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

DATA_DIR = "/workspace/data/aggtrades"
HORIZON_BARS = 60  # 10min at 10s

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
            pl.col("price").mean().alias("vwap"),
            pl.col("qty").sum().alias("vol"),
            (pl.col("qty") * (~pl.col("is_buyer_maker")).cast(int)).sum().alias("sell_vol"),
        ]).sort("bucket")
        all_bars.append(bar)
    bars = pl.concat(all_bars).sort("bucket")
    c = bars["close"].to_numpy()
    vwap = bars["vwap"].to_numpy()
    vol = bars["vol"].to_numpy()
    sell_vol = bars["sell_vol"].to_numpy()
    bucket = bars["bucket"].to_numpy()
    hour = (bucket % 86400000000) // 3600000000
    return c, vwap, vol, sell_vol, hour, len(c)

def build_features(c, vwap, vol, sell_vol, n):
    ret = c[1:]/c[:-1]-1
    features = {}
    # 多周期动量
    for w, name in [(60,"ret_10m"), (180,"ret_30m"), (360,"ret_60m"), (720,"ret_120m")]:
        f = np.full(n, np.nan)
        f[w:] = c[w:]/c[:-w]-1
        features[name] = f
    # 波动率
    sigma = np.full(n, np.nan)
    for i in range(60, n): sigma[i] = np.std(ret[i-60:i])
    features["sigma_10m"] = sigma
    # 价格偏离VWAP
    features["vwap_dev"] = np.full(n, np.nan)
    features["vwap_dev"][1:] = c[1:]/vwap[1:]-1
    # Volume imbalance (buy/(buy+sell))
    features["vol_imb"] = np.full(n, np.nan)
    buy = vol - sell_vol
    total = vol
    features["vol_imb"] = np.where(total>0, buy/total, 0.5)
    # 未来收益
    future_ret = np.full(n, np.nan)
    future_ret[:-HORIZON_BARS] = c[HORIZON_BARS:]/c[:-HORIZON_BARS]-1
    y = (future_ret>0).astype(float)
    return features, y

def main():
    c, vwap, vol, sell_vol, hour, n = load()
    print(f"{n} bars = {n/8640:.1f}d")
    features, y = build_features(c, vwap, vol, sell_vol, n)
    
    # ===== 1. 单特征AUC =====
    print("\n" + "="*60)
    print("[1] 单特征 AUC (全量数据, 无过滤)")
    print("="*60)
    for name, f in features.items():
        mask = ~np.isnan(f) & ~np.isnan(y)
        if mask.sum() < 1000: continue
        auc = roc_auc_score(y[mask], f[mask])
        # 反转的AUC (动量反转信号)
        auc_rev = roc_auc_score(y[mask], -f[mask])
        best = max(auc, auc_rev)
        direction = "反转动量" if auc_rev > auc else "追涨"
        print(f"  {name:15s}: AUC={auc:.4f}  revAUC={auc_rev:.4f}  → 选{direction}={best:.4f}")
    
    # ===== 2. 分小时看AUC =====
    print("\n" + "="*60)
    print("[2] 分小时 AUC (ret_60m反转信号)")
    print("="*60)
    f60 = features["ret_60m"]
    for h in range(24):
        mask = (hour == h) & ~np.isnan(f60) & ~np.isnan(y)
        if mask.sum() < 50: continue
        auc = roc_auc_score(y[mask], -f60[mask])  # 反转
        acc = ((-f60[mask] > 0) == y[mask]).mean()
        print(f"  UTC{h:02d}: n={mask.sum():5d} AUC={auc:.4f} acc={acc*100:.1f}%")
    
    # ===== 3. 分波动率看AUC =====
    print("\n" + "="*60)
    print("[3] 分波动率分位 AUC (ret_60m反转)")
    print("="*60)
    sig = features["sigma_10m"]
    mask_all = ~np.isnan(f60) & ~np.isnan(y) & ~np.isnan(sig)
    for q_name, lo, hi in [("q10-30", 0.1, 0.3), ("q30-50", 0.3, 0.5), 
                            ("q50-70", 0.5, 0.7), ("q70-90", 0.7, 0.9), ("q90-100", 0.9, 1.0)]:
        l, h = np.quantile(sig[mask_all], [lo, hi])
        mask = mask_all & (sig >= l) & (sig < h)
        if mask.sum() < 100: continue
        auc = roc_auc_score(y[mask], -f60[mask])
        acc = ((-f60[mask] > 0) == y[mask]).mean()
        print(f"  {q_name}: n={mask.sum():5d} AUC={auc:.4f} acc={acc*100:.1f}%")
    
    # ===== 4. LogReg组合特征 =====
    print("\n" + "="*60)
    print("[4] LogReg多特征 AUC (walk-forward style)")
    print("="*60)
    feature_names = ["ret_10m","ret_30m","ret_60m","ret_120m","sigma_10m","vwap_dev","vol_imb"]
    X_full = np.column_stack([features[k] for k in feature_names])
    
    # 简单walk-forward: 10d train, 6d test
    BARS_PER_DAY = 8640
    n_days = n // BARS_PER_DAY
    aucs, accs, weights_list = [], [], []
    for test_day in range(10, n_days):
        tr_s, tr_e = (test_day-10)*BARS_PER_DAY, test_day*BARS_PER_DAY
        te_s, te_e = test_day*BARS_PER_DAY, min((test_day+1)*BARS_PER_DAY, n)
        
        tr_mask = slice(tr_s+400, tr_e-HORIZON_BARS)
        te_mask = slice(te_s+400, te_e-HORIZON_BARS)
        
        # 清NaN
        X_tr = X_full[tr_mask]; y_tr = y[tr_mask]
        keep = ~np.isnan(X_tr).any(axis=1)
        X_tr, y_tr = X_tr[keep], y_tr[keep]
        
        X_te = X_full[te_mask]; y_te = y[te_mask]
        keep = ~np.isnan(X_te).any(axis=1)
        X_te, y_te = X_te[keep], y_te[keep]
        
        if len(X_tr) < 500 or len(X_te) < 50: continue
        
        # 标准化
        mu = X_tr.mean(axis=0); sd = X_tr.std(axis=0)+1e-8
        X_trs = (X_tr - mu)/sd
        X_tes = (X_te - mu)/sd
        
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(X_trs, y_tr)
        prob = lr.predict_proba(X_tes)[:,1]
        auc = roc_auc_score(y_te, prob)
        pred = (prob > 0.5).astype(int)
        acc = (pred == y_te).mean()
        aucs.append(auc); accs.append(acc)
        weights_list.append(lr.coef_[0])
    
    print(f"  测试天数: {len(aucs)}")
    print(f"  平均 AUC={np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  平均 acc={np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%")
    
    avg_w = np.mean(weights_list, axis=0)
    print(f"\n  平均特征权重 (正=追涨, 负=反转):")
    for nm, w in zip(feature_names, avg_w):
        print(f"    {nm:15s}: {w:+.4f}")
    
    # ===== 5. 加Top小时 + 波动率过滤后的LogReg =====
    print("\n" + "="*60)
    print("[5] LogReg + Top8h + sigma top30% 过滤")
    print("="*60)
    all_preds, all_trues, all_probs = [], [], []
    BARS_PER_DAY = 8640
    n_days = n // BARS_PER_DAY
    for test_day in range(10, n_days):
        tr_s, tr_e = (test_day-10)*BARS_PER_DAY, test_day*BARS_PER_DAY
        te_s, te_e = test_day*BARS_PER_DAY, min((test_day+1)*BARS_PER_DAY, n)
        
        # 训练集选Top小时
        tr_hour = hour[tr_s+400:tr_e-HORIZON_BARS]
        tr_f60 = features["ret_60m"][tr_s+400:tr_e-HORIZON_BARS]
        tr_y_part = y[tr_s+400:tr_e-HORIZON_BARS]
        valid = ~np.isnan(tr_f60) & ~np.isnan(tr_y_part)
        h_acc = {h: ((-tr_f60[valid & (tr_hour==h)] > 0) == tr_y_part[valid & (tr_hour==h)]).mean() 
                 for h in range(24) if (valid & (tr_hour==h)).sum() >= 20}
        top_h = set(sorted(h_acc, key=h_acc.get, reverse=True)[:8])
        
        # sigma阈值
        tr_sig = sig[tr_s+400:tr_e-HORIZON_BARS]
        sig_thr = np.quantile(tr_sig[~np.isnan(tr_sig)], 0.7)
        
        # 训练LogReg
        tr_mask_slice = slice(tr_s+400, tr_e-HORIZON_BARS)
        X_tr = X_full[tr_mask_slice]; y_tr = y[tr_mask_slice]
        h_tr = hour[tr_mask_slice]; s_tr = sig[tr_mask_slice]
        keep = ~np.isnan(X_tr).any(axis=1) & ~np.isnan(y_tr) & (np.isin(h_tr, list(top_h))) & (s_tr >= sig_thr)
        X_tr, y_tr = X_tr[keep], y_tr[keep]
        
        te_mask_slice = slice(te_s+400, te_e-HORIZON_BARS)
        X_te = X_full[te_mask_slice]; y_te = y[te_mask_slice]
        h_te = hour[te_mask_slice]; s_te = sig[te_mask_slice]
        keep = ~np.isnan(X_te).any(axis=1) & ~np.isnan(y_te) & (np.isin(h_te, list(top_h))) & (s_te >= sig_thr)
        X_te, y_te = X_te[keep], y_te[keep]
        
        if len(X_tr) < 200 or len(X_te) < 30: continue
        
        mu = X_tr.mean(axis=0); sd = X_tr.std(axis=0)+1e-8
        X_trs = (X_tr - mu)/sd; X_tes = (X_te - mu)/sd
        
        lr = LogisticRegression(C=0.5, max_iter=1000)
        lr.fit(X_trs, y_tr)
        prob = lr.predict_proba(X_tes)[:,1]
        pred = (prob > 0.5).astype(int)
        
        all_preds.extend(pred.tolist())
        all_trues.extend(y_te.tolist())
        all_probs.extend(prob.tolist())
    
    all_preds = np.array(all_preds); all_trues = np.array(all_trues); all_probs = np.array(all_probs)
    acc = (all_preds == all_trues).mean()
    auc = roc_auc_score(all_trues, all_probs)
    arr = (all_preds == all_trues).astype(float)
    min100 = min(arr[i:i+100].mean() for i in range(len(arr)-99))*100 if len(arr)>=100 else 0
    print(f"  n={len(all_preds):,} | acc={acc*100:.1f}% | AUC={auc:.4f} | min100={min100:.1f}%")
    
    # 概率阈值过滤
    for pt in [0.5, 0.55, 0.6, 0.65]:
        m = (all_probs > pt) | (all_probs < 1-pt)
        if m.sum() < 50: continue
        acc2 = (all_preds[m] == all_trues[m]).mean()
        arr2 = (all_preds[m] == all_trues[m]).astype(float)
        min100_2 = min(arr2[i:i+100].mean() for i in range(len(arr2)-99))*100 if len(arr2)>=100 else 0
        print(f"  |prob-0.5|>{pt-0.5:.2f}: n={m.sum():,} acc={acc2*100:.1f}% min100={min100_2:.1f}%")
    
    print("\nDone.")

if __name__ == "__main__":
    main()
