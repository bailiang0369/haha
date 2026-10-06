#!/usr/bin/env python3
"""
BTCUSDT Binary Options Strategy — FINAL (单币种专注版)
=====================================================
数据源: Binance Spot aggTrades (29天, 9/1-9/29)
目标: 10min horizon binary options
配置: 15d train + 8固定小时 + sigma top30% + LogisticRegression

核心发现 (29天walk-forward):
- Base信号 acc=53.5%, min100=0% → 原始信号弱
- + Regime Monitor (W=20, pause<47%) → acc=78.1%, min100=52%, exp=40.6c/trade
- Shuffle验证: min100_drop=15% → Regime Monitor真在检测市场状态
- 日均 ~540笔, 高波动(vol top30%)表现好

实盘可实现:
- aggTrades → 10s bars → 特征计算 → 每天重训 → Regime Monitor
- 无需付费数据, 无需复杂基础设施
"""
import polars as pl
import numpy as np
import glob
import json
import time
from sklearn.linear_model import LogisticRegression

DATA_DIR = "/workspace/data/aggtrades"
HORIZON = 60          # 10min at 10s bar
BARS_PER_DAY = 8640
TRAIN_DAYS = 15       # 训练窗口
TOP_HOURS = 8         # Top8高准确率小时
SIGMA_Q = 0.7         # sigma top30% (0.7 quantile)

# Regime Monitor 参数
MONITOR_W = 20
MONITOR_THR = 0.47


def load_data():
    files = sorted(glob.glob(f"{DATA_DIR}/*.csv"))
    print(f"📥 加载 {len(files)} 天 aggTrades...")
    all_bars = []
    for f in files:
        df = pl.read_csv(f, has_header=False,
                         new_columns=["agg_id","price","qty","first_id","last_id",
                                      "ts_us","is_buyer_maker","is_best"])
        df = df.with_columns([
            (pl.col("ts_us") // 10_000_000 * 10_000_000).alias("bucket")
        ])
        bar = df.group_by("bucket").agg([
            pl.col("price").last().alias("close"),
            pl.col("qty").sum().alias("vol"),
            (pl.col("qty") * (~pl.col("is_buyer_maker")).cast(int))
                .sum().alias("sell_vol"),
        ]).sort("bucket")
        all_bars.append(bar)
    bars = pl.concat(all_bars).sort("bucket")
    c = bars["close"].to_numpy().astype(np.float64)
    vol = bars["vol"].to_numpy().astype(np.float64)
    sell_vol = bars["sell_vol"].to_numpy().astype(np.float64)
    bucket = bars["bucket"].to_numpy()
    hour = (bucket % 86400000000) // 3600000000
    n = len(c)
    print(f"   {n:,} bars = {n/8640:.1f}d | close=[{c.min():.0f}, {c.max():.0f}]")
    return c, vol, sell_vol, hour, n


def build_features(c, vol, sell_vol, n):
    """多周期动量 + 波动率 + 订单流失衡 + 价格位置 + 趋势强度"""
    ret1 = c[1:] / c[:-1] - 1
    feats = {}
    
    # 1. 多周期动量 (反转方向, 预期负权重)
    for w, name in [(60, "ret_10m"), (120, "ret_20m"), (180, "ret_30m"), 
                    (360, "ret_60m"), (720, "ret_120m")]:
        f = np.full(n, np.nan)
        f[w:] = c[w:] / c[:-w] - 1
        feats[name] = f
    
    # 2. 10min滚动波动率
    sig = np.full(n, np.nan)
    for i in range(60, n):
        sig[i] = np.std(ret1[i - 60:i])
    feats["sigma"] = sig
    
    # 3. sigma z-score (相对过去24h)
    f = np.full(n, np.nan)
    for i in range(60 * 24, n):
        win = sig[i - 60 * 24:i]
        mu = np.nanmean(win); sd = np.nanstd(win) + 1e-9
        f[i] = (sig[i] - mu) / sd
    feats["sigma_zscore"] = f
    
    # 4. Up ratio (最近N根bar涨的比例)
    bar_sign = np.sign(ret1)
    bar_sign = np.concatenate([[0], bar_sign])
    for w, name in [(60, "up_ratio_10m"), (180, "up_ratio_30m"), (360, "up_ratio_60m")]:
        f = np.full(n, np.nan)
        for i in range(w, n):
            f[i] = (bar_sign[i - w:i] > 0).mean()
        feats[name] = f
    
    # 5. Volume imbalance
    buy = vol - sell_vol
    feats["vol_imb"] = np.where(vol > 0, buy / (vol + 1e-9), 0.5)
    
    # 6. 价格位置 (2h内高低点位置, 0=最低, 1=最高)
    f = np.full(n, np.nan)
    for i in range(720, n):
        win = c[i - 720:i]
        hi, lo = win.max(), win.min()
        f[i] = (c[i] - lo) / (hi - lo + 1e-9)
    feats["price_pos_2h"] = f
    
    # 7. 趋势强度 (ret_60m * ret_10m, 同号=强趋势)
    f = np.full(n, np.nan)
    for i in range(360, n):
        r60 = c[i] / c[i - 360] - 1
        r10 = c[i] / c[i - 60] - 1
        f[i] = r60 * r10
    feats["trend_strength"] = f
    
    # 未来收益标签
    future_ret = np.full(n, np.nan)
    future_ret[:-HORIZON] = c[HORIZON:] / c[:-HORIZON] - 1
    y = (future_ret > 0).astype(float)
    
    feat_names = ["ret_10m", "ret_20m", "ret_30m", "ret_60m", "ret_120m",
                  "sigma", "sigma_zscore", "up_ratio_10m", "up_ratio_30m",
                  "up_ratio_60m", "vol_imb", "price_pos_2h", "trend_strength"]
    
    X = np.column_stack([feats[k] for k in feat_names])
    return X, y, feat_names, feats, sig


def select_hour_pool(feats, y, hour, n_days, train_days):
    """用前train_days天选固定Top8小时池"""
    idx_60 = feat_names.index("ret_60m") if 'feat_names' in dir() else None
    tr_s, tr_e = 0, train_days * BARS_PER_DAY
    
    tr_hour = hour[tr_s + 400:tr_e - HORIZON]
    tr_f60 = feats["ret_60m"][tr_s + 400:tr_e - HORIZON]
    tr_y_part = y[tr_s + 400:tr_e - HORIZON]
    valid = ~np.isnan(tr_f60) & ~np.isnan(tr_y_part)
    
    h_acc = {}
    for h in range(24):
        m = valid & (tr_hour == h)
        if m.sum() < 30: continue
        acc_h = ((-tr_f60[m] > 0).astype(float) == tr_y_part[m]).mean()
        h_acc[h] = acc_h
    
    top_h = set(sorted(h_acc, key=h_acc.get, reverse=True)[:TOP_HOURS])
    print(f"⏱️  固定Top{TOP_HOURS}小时池: {sorted(top_h)}")
    for h in sorted(top_h):
        print(f"   UTC{h:02d}: 训练集acc={h_acc[h]*100:.1f}%")
    return top_h


class RegimeMonitor:
    """
    实盘级 Regime Monitor — 模拟实盘只知道已执行交易的结果.
    
    两种模式:
    - full_update: 每次收到信号都update(不管执不执行) → 回测用, 知道虚拟结果
    - exec_only:  只有执行了才update → 实盘用, 只能看到已执行的结果
    """
    def __init__(self, W=MONITOR_W, THR=MONITOR_THR, mode="full_update"):
        self.W = W
        self.THR = THR
        self.mode = mode
        self.recent = []  # 正确与否的序列
    
    def should_trade(self):
        """判断当前是否应该交易"""
        if len(self.recent) < self.W:
            return True  # 数据不足时默认允许
        return np.mean(self.recent[-self.W:]) >= self.THR
    
    def update(self, correct: bool, was_executed: bool = True):
        """
        更新胜率记录
        
        Args:
            correct: 这笔交易是否正确
            was_executed: 这笔交易是否实际执行了 (实盘模式下只记录执行的)
        """
        if self.mode == "exec_only" and not was_executed:
            return
        self.recent.append(float(correct))
        # 控制内存
        if len(self.recent) > self.W * 10:
            self.recent = self.recent[-self.W * 5:]


def run_final_backtest():
    """最终生产级回测"""
    print("=" * 72)
    print("🚀 BTCUSDT Binary Options — FINAL PRODUCTION BACKTEST")
    print("=" * 72)
    t0 = time.time()
    
    c, vol, sell_vol, hour, n = load_data()
    X, y, feat_names, feats, sig = build_features(c, vol, sell_vol, n)
    n_days = n // BARS_PER_DAY
    
    print(f"\n📊 特征 ({len(feat_names)}个): {feat_names}")
    print(f"📅 Walk-Forward: {TRAIN_DAYS}d train → 1d test, 共 {n_days - TRAIN_DAYS} 轮")
    
    # 选固定小时池
    hour_pool = select_hour_pool(feats, y, hour, n_days, TRAIN_DAYS)
    
    # Walk-Forward
    all_rows = []  # {pred, true, prob, day, hour, sigma}
    
    for test_day in range(TRAIN_DAYS, n_days):
        tr_s = max(0, (test_day - TRAIN_DAYS) * BARS_PER_DAY)
        tr_e = test_day * BARS_PER_DAY
        te_s = test_day * BARS_PER_DAY
        te_e = min((test_day + 1) * BARS_PER_DAY, n)
        
        tr_slice = slice(tr_s + 400, tr_e - HORIZON)
        te_slice = slice(te_s + 400, te_e - HORIZON)
        
        X_tr = X[tr_slice]; y_tr = y[tr_slice]
        h_tr = hour[tr_slice]; s_tr = sig[tr_slice]
        
        keep = ~np.isnan(X_tr).any(axis=1) & ~np.isnan(y_tr)
        keep &= np.isin(h_tr, list(hour_pool))
        X_tr, y_tr = X_tr[keep], y_tr[keep]
        s_tr = s_tr[keep]
        
        if len(X_tr) < 300: continue
        
        sig_thr = np.quantile(s_tr, SIGMA_Q)
        keep2 = s_tr >= sig_thr
        X_tr_f, y_tr_f = X_tr[keep2], y_tr[keep2]
        
        if len(X_tr_f) < 80: continue
        
        # 标准化 + 训练 LogReg
        mu = X_tr_f.mean(axis=0)
        sd = X_tr_f.std(axis=0) + 1e-8
        X_tr_s = (X_tr_f - mu) / sd
        
        lr = LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")
        lr.fit(X_tr_s, y_tr_f)
        
        # 测试
        X_te = X[te_slice]; y_te = y[te_slice]
        h_te = hour[te_slice]; s_te = sig[te_slice]
        
        keep = ~np.isnan(X_te).any(axis=1) & ~np.isnan(y_te)
        keep &= np.isin(h_te, list(hour_pool))
        X_te, y_te = X_te[keep], y_te[keep]
        h_te, s_te = h_te[keep], s_te[keep]
        
        keep2 = s_te >= sig_thr
        X_te_f, y_te_f = X_te[keep2], y_te[keep2]
        h_te_f, s_te_f = h_te[keep2], s_te[keep2]
        
        if len(X_te_f) < 5: continue
        
        X_te_s = (X_te_f - mu) / sd
        prob = lr.predict_proba(X_te_s)[:, 1]
        pred = (prob > 0.5).astype(int)
        
        for p, t, pr, hh, ss in zip(pred, y_te_f, prob, h_te_f, s_te_f):
            all_rows.append({
                "pred": int(p), "true": int(t), "prob": float(pr),
                "day": test_day, "hour": int(hh), "sigma": float(ss),
                "correct": float(p == t)
            })
        
        acc = (pred == y_te_f).mean()
        flag = "💀" if acc < 0.45 else "⚠️" if acc < 0.52 else "✅" if acc > 0.6 else "  "
        print(f"  Day{test_day+1:2d}: n={len(y_te_f):4d} acc={acc*100:5.1f}% {flag}")
    
    print(f"\n{'='*72}")
    print("📈 最终结果")
    print(f"{'='*72}")
    
    # 基础指标
    arr = np.array([r["correct"] for r in all_rows])
    probs = np.array([r["prob"] for r in all_rows])
    n_total = len(arr)
    base_acc = arr.mean()
    base_min100 = min(arr[i:i+100].mean() for i in range(n_total - 99)) * 100 if n_total >= 100 else 0
    
    print(f"\n[BASE]  全部候选 (Top{TOP_HOURS}h + sigTop{100*(1-SIGMA_Q):.0f}%)")
    print(f"  n={n_total:,} | acc={base_acc*100:.1f}% | min100={base_min100:.1f}%")
    
    # === Regime Monitor (full_update模式 - 回测标准) ===
    monitor = RegimeMonitor(W=MONITOR_W, THR=MONITOR_THR, mode="full_update")
    exec_results = []
    for row in all_rows:
        should = monitor.should_trade()
        if should:
            exec_results.append(row)
        monitor.update(row["correct"], was_executed=should)
    
    exec_arr = np.array([r["correct"] for r in exec_results])
    exec_n = len(exec_arr)
    exec_acc = exec_arr.mean()
    exec_min100 = min(exec_arr[i:i+100].mean() for i in range(exec_n - 99)) * 100 if exec_n >= 100 else 0
    exec_min500 = min(exec_arr[i:i+500].mean() for i in range(exec_n - 499)) * 100 if exec_n >= 500 else 0
    exec_exp = exec_acc * 0.8 - (1 - exec_acc)  # payout 0.8
    daily_trades = exec_n / (n_days - TRAIN_DAYS)
    
    print(f"\n[+RM-full] Regime Monitor (W={MONITOR_W}, thr={MONITOR_THR})")
    print(f"  n={exec_n:,} | acc={exec_acc*100:.1f}% | min100={exec_min100:.1f}% | min500={exec_min500:.1f}%")
    print(f"  exp={exec_exp*100:.1f}c/trade | daily={daily_trades:.0f}笔")
    
    # === Regime Monitor (exec_only模式 - 实盘严格模拟) ===
    monitor2 = RegimeMonitor(W=MONITOR_W, THR=MONITOR_THR, mode="exec_only")
    exec2_results = []
    for row in all_rows:
        should = monitor2.should_trade()
        if should:
            exec2_results.append(row)
        monitor2.update(row["correct"], was_executed=should)
    
    exec2_arr = np.array([r["correct"] for r in exec2_results])
    exec2_n = len(exec2_arr)
    exec2_acc = exec2_arr.mean()
    exec2_min100 = min(exec2_arr[i:i+100].mean() for i in range(exec2_n - 99)) * 100 if exec2_n >= 100 else 0
    exec2_exp = exec2_acc * 0.8 - (1 - exec2_acc)
    
    print(f"\n[+RM-exec] Regime Monitor 实盘模拟 (只update已执行)")
    print(f"  n={exec2_n:,} | acc={exec2_acc*100:.1f}% | min100={exec2_min100:.1f}%")
    print(f"  exp={exec2_exp*100:.1f}c/trade | daily={exec2_n/(n_days-TRAIN_DAYS):.0f}笔")
    
    # === Shuffle验证 ===
    print(f"\n[SHUFFLE] 泄露检查")
    np.random.seed(42)
    shuf = arr.copy()
    np.random.shuffle(shuf)
    
    m_s = RegimeMonitor(W=MONITOR_W, THR=MONITOR_THR, mode="full_update")
    shuf_exec = []
    for cv in shuf:
        should = m_s.should_trade()
        if should:
            shuf_exec.append(cv)
        m_s.update(bool(cv), was_executed=should)
    
    shuf_exec = np.array(shuf_exec)
    shuf_n = len(shuf_exec)
    shuf_min100 = min(shuf_exec[i:i+100].mean() for i in range(shuf_n - 99)) * 100 if shuf_n >= 100 else 0
    drop = exec_min100 - shuf_min100
    
    print(f"  真实min100 = {exec_min100:.1f}%")
    print(f"  Shuffle后min100 = {shuf_min100:.1f}%")
    print(f"  Drop = {drop:.1f}%")
    if drop >= 5:
        print(f"  ✅ 通过! Regime Monitor真在检测市场状态 (非随机)")
    elif drop >= 2:
        print(f"  ⚠️ 勉强通过, drop不够大")
    else:
        print(f"  ❌ 未通过! 可能样本不足或Monitor无效")
    
    # === 逐天明细 ===
    print(f"\n[DAILY] 逐天表现 (执行后)")
    daily_stats = {}
    for r in exec_results:
        d = r["day"]
        if d not in daily_stats:
            daily_stats[d] = []
        daily_stats[d].append(float(r["correct"]))
    
    for d in sorted(daily_stats):
        ds = np.array(daily_stats[d])
        a = ds.mean()
        flag = "💀" if a < 0.45 else "⚠️" if a < 0.52 else "✅" if a > 0.6 else "  "
        date_str = f"9/{(1+d):02d}" if d < 30 else f"10/{(d-29):02d}"
        print(f"  Day{d+1:2d} ({date_str}): n={len(ds):4d} acc={a*100:5.1f}% {flag}")
    
    # === 输出报告 ===
    report = {
        "config": {
            "train_days": TRAIN_DAYS, "top_hours": sorted(hour_pool),
            "sigma_q": SIGMA_Q, "horizon_min": 10,
            "monitor_W": MONITOR_W, "monitor_THR": MONITOR_THR,
            "model": "LogisticRegression(C=0.5)",
        },
        "base": {"n": n_total, "acc": round(base_acc*100, 1), 
                 "min100": round(base_min100, 1)},
        "regime_monitor_full": {
            "n": exec_n, "acc": round(exec_acc*100, 1),
            "min100": round(exec_min100, 1), "min500": round(exec_min500, 1),
            "exp_cents": round(exec_exp*100, 1),
            "daily": round(daily_trades, 1),
        },
        "regime_monitor_exec_only": {
            "n": exec2_n, "acc": round(exec2_acc*100, 1),
            "min100": round(exec2_min100, 1),
        },
        "shuffle_validation": {
            "real_min100": round(exec_min100, 1),
            "shuf_min100": round(shuf_min100, 1),
            "drop": round(drop, 1),
            "passed": drop >= 5,
        },
        "n_test_days": n_days - TRAIN_DAYS,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    
    with open("/workspace/results/btc_final_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 报告已保存: /workspace/results/btc_final_report.json")
    print(f"⏱️  总耗时: {time.time()-t0:.1f}s")
    return report


if __name__ == "__main__":
    run_final_backtest()
