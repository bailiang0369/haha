#!/usr/bin/env python3
"""
BTCUSDT Binary Options — PRODUCTION (实盘级验证版)
===================================================
核心问题: Regime Monitor 的"暂停后永不恢复"陷阱
解决: 对比多种Monitor设计, 找到实盘可工作的版本
"""
import polars as pl
import numpy as np
import glob
import json
import time
from sklearn.linear_model import LogisticRegression

DATA_DIR = "/workspace/data/aggtrades"
HORIZON = 60
BARS_PER_DAY = 8640
TRAIN_DAYS = 15
TOP_HOURS = 8
SIGMA_Q = 0.7

def load():
    files = sorted(glob.glob(f"{DATA_DIR}/*.csv"))
    print(f"加载 {len(files)} 天...")
    all_bars = []
    for f in files:
        df = pl.read_csv(f, has_header=False,
                         new_columns=["agg_id","price","qty","first_id","last_id",
                                      "ts_us","is_buyer_maker","is_best"])
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
    ret1 = c[1:] / c[:-1] - 1
    feats = {}
    for w, name in [(60, "ret_10m"), (120, "ret_20m"), (180, "ret_30m"), (360, "ret_60m"), (720, "ret_120m")]:
        f = np.full(n, np.nan); f[w:] = c[w:]/c[:-w]-1; feats[name] = f
    sig = np.full(n, np.nan)
    for i in range(60, n): sig[i] = np.std(ret1[i-60:i])
    feats["sigma"] = sig
    f = np.full(n, np.nan)
    for i in range(60*24, n):
        win = sig[i-60*24:i]
        f[i] = (sig[i] - np.nanmean(win)) / (np.nanstd(win)+1e-9)
    feats["sigma_zscore"] = f
    bar_sign = np.sign(ret1); bar_sign = np.concatenate([[0], bar_sign])
    for w, name in [(60, "up_ratio_10m"), (180, "up_ratio_30m"), (360, "up_ratio_60m")]:
        f = np.full(n, np.nan)
        for i in range(w, n): f[i] = (bar_sign[i-w:i] > 0).mean()
        feats[name] = f
    buy = vol - sell_vol
    feats["vol_imb"] = np.where(vol > 0, buy/(vol+1e-9), 0.5)
    f = np.full(n, np.nan)
    for i in range(720, n):
        win = c[i-720:i]; hi, lo = win.max(), win.min()
        f[i] = (c[i] - lo)/(hi-lo+1e-9)
    feats["price_pos_2h"] = f
    f = np.full(n, np.nan)
    for i in range(360, n):
        r60 = c[i]/c[i-360]-1; r10 = c[i]/c[i-60]-1
        f[i] = r60 * r10
    feats["trend_strength"] = f
    future_ret = np.full(n, np.nan)
    future_ret[:-HORIZON] = c[HORIZON:]/c[:-HORIZON]-1
    y = (future_ret > 0).astype(float)
    fn = ["ret_10m","ret_20m","ret_30m","ret_60m","ret_120m","sigma","sigma_zscore",
          "up_ratio_10m","up_ratio_30m","up_ratio_60m","vol_imb","price_pos_2h","trend_strength"]
    X = np.column_stack([feats[k] for k in fn])
    return X, y, fn, feats, sig

def get_signals_only():
    """只跑walk-forward, 输出所有候选交易的结果"""
    c, vol, sell_vol, hour, n = load()
    X, y, feat_names, feats, sig = features(c, vol, sell_vol, n)
    n_days = n // BARS_PER_DAY
    
    # 选固定小时池
    tr_s, tr_e = 0, TRAIN_DAYS * BARS_PER_DAY
    tr_hour = hour[tr_s+400:tr_e-HORIZON]
    tr_f60 = feats["ret_60m"][tr_s+400:tr_e-HORIZON]
    tr_y_part = y[tr_s+400:tr_e-HORIZON]
    valid = ~np.isnan(tr_f60) & ~np.isnan(tr_y_part)
    h_acc = {}
    for h in range(24):
        m = valid & (tr_hour == h)
        if m.sum() < 30: continue
        h_acc[h] = ((-tr_f60[m]>0).astype(float) == tr_y_part[m]).mean()
    hour_pool = set(sorted(h_acc, key=h_acc.get, reverse=True)[:TOP_HOURS])
    print(f"Top{TOP_HOURS}h: {sorted(hour_pool)}")
    
    all_rows = []
    for test_day in range(TRAIN_DAYS, n_days):
        tr_s = max(0, (test_day-TRAIN_DAYS)*BARS_PER_DAY)
        tr_e = test_day*BARS_PER_DAY
        te_s = test_day*BARS_PER_DAY
        te_e = min((test_day+1)*BARS_PER_DAY, n)
        
        tr_slice = slice(tr_s+400, tr_e-HORIZON)
        te_slice = slice(te_s+400, te_e-HORIZON)
        X_tr = X[tr_slice]; y_tr = y[tr_slice]; h_tr = hour[tr_slice]; s_tr = sig[tr_slice]
        keep = ~np.isnan(X_tr).any(axis=1) & ~np.isnan(y_tr) & np.isin(h_tr, list(hour_pool))
        X_tr, y_tr = X_tr[keep], y_tr[keep]; s_tr = s_tr[keep]
        if len(X_tr) < 300: continue
        sig_thr = np.quantile(s_tr, SIGMA_Q)
        keep2 = s_tr >= sig_thr
        X_tr_f, y_tr_f = X_tr[keep2], y_tr[keep2]
        if len(X_tr_f) < 80: continue
        mu = X_tr_f.mean(axis=0); sd = X_tr_f.std(axis=0)+1e-8
        X_tr_s = (X_tr_f-mu)/sd
        lr = LogisticRegression(C=0.5, max_iter=2000)
        lr.fit(X_tr_s, y_tr_f)
        
        X_te = X[te_slice]; y_te = y[te_slice]; h_te = hour[te_slice]; s_te = sig[te_slice]
        keep = ~np.isnan(X_te).any(axis=1) & ~np.isnan(y_te) & np.isin(h_te, list(hour_pool))
        X_te, y_te = X_te[keep], y_te[keep]; h_te, s_te = h_te[keep], s_te[keep]
        keep2 = s_te >= sig_thr
        X_te_f, y_te_f = X_te[keep2], y_te[keep2]
        if len(X_te_f) < 5: continue
        X_te_s = (X_te_f-mu)/sd
        prob = lr.predict_proba(X_te_s)[:,1]
        pred = (prob>0.5).astype(int)
        for p, t, pr, hh in zip(pred, y_te_f, prob, h_te[keep2]):
            all_rows.append({"correct": float(p==t), "hour": int(hh), "prob": float(pr)})
    
    arr = np.array([r["correct"] for r in all_rows])
    print(f"Total candidates: {len(arr):,}, base acc={arr.mean()*100:.1f}%")
    return arr, all_rows, n_days - TRAIN_DAYS

def simulate_monitor(arr, W, THR, max_pause=None, probe_every=None, mode="exec_only"):
    """
    通用Monitor模拟器
    
    Args:
        arr: 所有候选交易的正确与否序列 (float 0/1)
        W: 滑窗大小
        THR: 暂停阈值 (过去W笔胜率<THR时暂停)
        max_pause: 最大暂停笔数 (None=无限暂停)
        probe_every: 暂停期间每隔N笔允许1笔(None=不probe)
        mode: "full_update"=每次都update, "exec_only"=只update已执行
    """
    recent = []
    paused = False
    pause_count = 0
    probe_count = 0
    exec_arr = []
    skipped_arr = []
    
    for cv in arr:
        # 决策
        if paused:
            if max_pause and pause_count >= max_pause:
                should = True  # 强制恢复
                paused = False
                pause_count = 0
            elif probe_every and probe_count >= probe_every:
                should = True  # 探针交易
                probe_count = 0
            else:
                should = False
                pause_count += 1
                if probe_every: probe_count += 1
        else:
            should = (np.mean(recent[-W:]) >= THR) if len(recent) >= W else True
            if not should:
                paused = True
                pause_count = 1
                probe_count = 1
        
        if should:
            exec_arr.append(cv)
        else:
            skipped_arr.append(cv)
        
        # 更新
        if mode == "full_update":
            recent.append(float(cv))
        elif should:  # exec_only + 执行了
            recent.append(float(cv))
    
    return np.array(exec_arr)

def evaluate(arr, label):
    n = len(arr)
    acc = arr.mean()
    min100 = min(arr[i:i+100].mean() for i in range(n-99))*100 if n>=100 else 0
    min500 = min(arr[i:i+500].mean() for i in range(n-499))*100 if n>=500 else 0
    exp = acc*0.8 - (1-acc)
    return {"label": label, "n": n, "acc": acc*100, "min100": min100, 
            "min500": min500, "exp": exp*100}

def main():
    print("="*80)
    print("BTCUSDT — Regime Monitor 实盘级对比")
    print("="*80)
    
    arr, all_rows, test_days = get_signals_only()
    
    print(f"\n测试天数: {test_days}")
    print(f"\n{'='*80}")
    print("Monitor配置对比 (exec_only模式 — 实盘严格模拟)")
    print(f"{'='*80}")
    print(f"{'Config':<28s} {'n':>6s} {'acc':>6s} {'m100':>6s} {'m500':>6s} {'exp':>6s} {'d/d':>5s}")
    print("-"*80)
    
    configs = [
        # (W, THR, max_pause, probe_every, label)
        (None, None, None, None, "NO MONITOR (base)"),
        (10, 0.45, None, None, "W10 thr0.45 无限暂停"),
        (20, 0.47, None, None, "W20 thr0.47 无限暂停"),
        (30, 0.49, None, None, "W30 thr0.49 无限暂停"),
        (10, 0.50, None, None, "W10 thr0.50 无限暂停"),
        # 加max_pause
        (10, 0.40, 50, None, "W10 thr0.4 暂停≤50笔"),
        (10, 0.45, 30, None, "W10 thr0.45 暂停≤30笔"),
        (10, 0.45, 10, None, "W10 thr0.45 暂停≤10笔"),
        (20, 0.47, 30, None, "W20 thr0.47 暂停≤30笔"),
        # 加probe
        (10, 0.45, None, 10, "W10 thr0.45 probe每10笔"),
        (10, 0.45, None, 5, "W10 thr0.45 probe每5笔"),
        (20, 0.47, None, 10, "W20 thr0.47 probe每10笔"),
        # max_pause + probe
        (10, 0.45, 20, 5, "W10 thr0.45 p≤20 + probe5"),
        (10, 0.50, 20, 5, "W10 thr0.50 p≤20 + probe5"),
        (5, 0.45, None, None, "W5 thr0.45 无限暂停"),
        (5, 0.50, None, 10, "W5 thr0.50 probe10"),
        # full_update 模式 (对比)
        ("FU", 20, 0.47, None, None, "FU W20 thr0.47"),
        ("FU", 10, 0.45, None, None, "FU W10 thr0.45"),
    ]
    
    results = []
    for cfg in configs:
        if cfg[0] == "FU":
            _, W, THR, mp, pe, label = cfg
            exec_arr = simulate_monitor(arr, W, THR, mp, pe, mode="full_update")
        elif cfg[0] is None:
            label = cfg[-1]
            exec_arr = arr
        else:
            W, THR, mp, pe, label = cfg
            exec_arr = simulate_monitor(arr, W, THR, mp, pe, mode="exec_only")
        
        if len(exec_arr) < 10:
            print(f"{label:<28s} n={len(exec_arr):4d} (TOO FEW)")
            continue
        
        r = evaluate(exec_arr, label)
        r["daily"] = len(exec_arr) / test_days
        results.append(r)
        
        flag = "💀" if r["acc"] < 45 else "⚠️" if r["acc"] < 52 else "✅" if r["acc"] > 60 else "  "
        print(f"{label:<28s} {r['n']:6d} {r['acc']:5.1f}% {r['min100']:5.1f}% {r['min500']:5.1f}% {r['exp']:5.1f}c {r['daily']:5.0f} {flag}")
    
    # === Shuffle 对每个好配置验证 ===
    print(f"\n{'='*80}")
    print("Shuffle验证 (对TOP配置)")
    print(f"{'='*80}")
    np.random.seed(42)
    shuf = arr.copy(); np.random.shuffle(shuf)
    
    for r in results:
        if r["acc"] < 55 or r["n"] < 100: continue
        
        label = r["label"]
        # 从label解析参数
        if "FU" in label:
            W = 20 if "W20" in label else 10
            THR = 0.47 if "0.47" in label else 0.45
            mode = "full_update"
        else:
            W = int(label.split("W")[1].split()[0])
            THR = float(label.split("thr")[1].split()[0])
            mode = "exec_only"
        
        # 简化: 只对固定W/THR做shuffle
        if "暂停" in label or "probe" in label:
            continue
        
        real_arr = simulate_monitor(arr, W, THR, None, None, mode)
        shuf_arr = simulate_monitor(shuf, W, THR, None, None, mode)
        
        real_m100 = min(real_arr[i:i+100].mean() for i in range(len(real_arr)-99))*100 if len(real_arr)>=100 else 0
        shuf_m100 = min(shuf_arr[i:i+100].mean() for i in range(len(shuf_arr)-99))*100 if len(shuf_arr)>=100 else 0
        drop = real_m100 - shuf_m100
        passed = "✅" if drop >= 5 else "⚠️" if drop >= 2 else "❌"
        
        print(f"  {label:<28s}: real_m100={real_m100:.0f}% shuf_m100={shuf_m100:.0f}% drop={drop:+.0f}% {passed}")
    
    # === 满足所有条件的配置 ===
    print(f"\n{'='*80}")
    print("最终可接受配置 (exec_only, 实盘可运行)")
    print(f"{'='*80}")
    good = [r for r in results 
            if r["acc"] >= 55 and r["min100"] >= 40 
            and r["n"] >= 500 and "FU" not in r["label"]]
    if not good:
        print("  没有完全满足条件的 😭")
        # 放宽min100
        good = [r for r in results 
                if r["acc"] >= 52 and r["min100"] >= 30 
                and r["n"] >= 100 and "FU" not in r["label"]]
        print(f"  放宽到 acc≥52% min100≥30% n≥100: {len(good)}个")
    
    for r in good:
        print(f"  ✅ {r['label']:<28s} n={r['n']:5d} acc={r['acc']:.1f}% min100={r['min100']:.1f}% exp={r['exp']:.1f}c daily={r['daily']:.0f}")
    
    return results

if __name__ == "__main__":
    main()
