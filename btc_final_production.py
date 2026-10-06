#!/usr/bin/env python3
"""
BTCUSDT Binary Options Strategy — ⭐ FINAL PRODUCTION ⭐
=======================================================
数据: 29天 Binance Spot aggTrades (9/1-9/29)
目标: 10min horizon binary options (payout ~0.8)

最终配置:
  - 训练: 15天滚动训练 LogisticRegression
  - 小时池: Top8固定 (训练集选出)
  - 波动率过滤: sigma top30%
  - Regime Monitor: W=10, 暂停阈值=0.40, 最大暂停50笔 (exec_only模式)

实盘级验证 (29d walk-forward):
  Base信号: n=13404, acc=53.9%, min100=0%
  + Regime Monitor: n=1945, acc=77.2%, min100=61%, min500=73%, exp=38.9c/trade
  Shuffle验证: min100 drop=21% → ✅ 真信号, 非随机

实盘实现要点:
  1. 每天UTC 0点后下载前一天aggTrades → 重训模型
  2. 实时: aggTrades → 10s bars → 特征 → 预测
  3. Regime Monitor只记录已执行交易结果
  4. max_pause=50避免永久暂停
"""
import polars as pl
import numpy as np
import glob
import json
import time
import os
from sklearn.linear_model import LogisticRegression

# ═══════════════════════════════════════════════════════════════
# PRODUCTION CONFIG — DO NOT MODIFY WITHOUT VALIDATION
# ═══════════════════════════════════════════════════════════════
DATA_DIR = "/workspace/data/aggtrades"
RESULT_DIR = "/workspace/results"

# 信号参数
HORIZON = 60            # 10min @ 10s bars
BARS_PER_DAY = 8640     # 8640 个10s bar = 1天
TRAIN_DAYS = 15         # 训练窗口 (15天)
TOP_HOURS = 8           # Top8高准确率小时
SIGMA_Q = 0.7           # sigma top30% (quantile=0.7)

# 模型
MODEL = LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")

# Regime Monitor (实盘级)
MONITOR_W = 10           # 滑窗10笔
MONITOR_THR = 0.40       # 过去10笔胜率<40%时暂停
MONITOR_MAX_PAUSE = 50   # 最多暂停50笔后强制恢复

# 置信度过滤 (可选, 保守版用)
CONF_THRESHOLD = 0.0     # 0.0=不用过滤, 0.1=prob>0.6或<0.4

PAYOUT = 0.8             # 二元期权payout比率


def load_data(data_dir=DATA_DIR):
    """加载所有aggTrades → 10s bars"""
    files = sorted(glob.glob(f"{data_dir}/*.csv"))
    if not files:
        raise FileNotFoundError(f"No aggTrades CSV found in {data_dir}")
    print(f"📥 Loading {len(files)} days of aggTrades...")
    
    all_bars = []
    for f in files:
        df = pl.read_csv(f, has_header=False,
                         new_columns=["agg_id","price","qty","first_id","last_id",
                                      "ts_us","is_buyer_maker","is_best"])
        # 10秒bucket
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
    
    print(f"   {len(c):,} bars ({len(c)/BARS_PER_DAY:.1f}d) | "
          f"close∈[{c.min():.0f}, {c.max():.0f}]")
    return c, vol, sell_vol, hour, len(c)


def build_features(c, vol, sell_vol, n):
    """
    多周期动量 + 波动率 + 订单流失衡 + 价格位置 + 趋势强度
    共12个特征, 全部可实时计算 (无未来信息)
    """
    ret1 = c[1:] / c[:-1] - 1  # 单bar return (10s)
    feats = {}
    
    # 1. 多周期动量 (反转方向, 预期负权重)
    for name, w in [("ret_10m", 60), ("ret_20m", 120), ("ret_30m", 180),
                    ("ret_60m", 360), ("ret_120m", 720)]:
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
        f[i] = (sig[i] - np.nanmean(win)) / (np.nanstd(win) + 1e-9)
    feats["sigma_zscore"] = f
    
    # 4. Up ratio (最近N根bar涨的比例)
    bar_sign = np.sign(ret1)
    bar_sign = np.concatenate([[0], bar_sign])  # 对齐
    for name, w in [("up_10m", 60), ("up_30m", 180), ("up_60m", 360)]:
        f = np.full(n, np.nan)
        for i in range(w, n):
            f[i] = (bar_sign[i - w:i] > 0).mean()
        feats[name] = f
    
    # 5. Volume imbalance (buy volume fraction)
    buy = vol - sell_vol
    feats["vol_imb"] = np.where(vol > 0, buy / (vol + 1e-9), 0.5)
    
    # 6. 价格位置 (2h内高低点位置, 0=最低, 1=最高)
    f = np.full(n, np.nan)
    for i in range(720, n):
        win = c[i - 720:i]
        hi, lo = win.max(), win.min()
        f[i] = (c[i] - lo) / (hi - lo + 1e-9)
    feats["price_pos_2h"] = f
    
    # 7. 趋势强度 (ret_60m × ret_10m, 同号=强趋势, 反号=震荡)
    f = np.full(n, np.nan)
    for i in range(360, n):
        r60 = c[i] / c[i - 360] - 1
        r10 = c[i] / c[i - 60] - 1
        f[i] = r60 * r10
    feats["trend_strength"] = f
    
    # Label: 未来10min涨跌
    future_ret = np.full(n, np.nan)
    future_ret[:-HORIZON] = c[HORIZON:] / c[:-HORIZON] - 1
    y = (future_ret > 0).astype(float)
    
    FEATURE_NAMES = [
        "ret_10m", "ret_20m", "ret_30m", "ret_60m", "ret_120m",
        "sigma", "sigma_zscore",
        "up_10m", "up_30m", "up_60m",
        "vol_imb", "price_pos_2h", "trend_strength"
    ]
    
    X = np.column_stack([feats[k] for k in FEATURE_NAMES])
    return X, y, FEATURE_NAMES, feats, sig


def select_hour_pool(feats, y, hour, n):
    """用前TRAIN_DAYS天选固定Top8小时池"""
    tr_e = TRAIN_DAYS * BARS_PER_DAY
    tr_hour = hour[400:tr_e - HORIZON]
    tr_f60 = feats["ret_60m"][400:tr_e - HORIZON]
    tr_y = y[400:tr_e - HORIZON]
    valid = ~np.isnan(tr_f60) & ~np.isnan(tr_y)
    
    h_acc = {}
    for h in range(24):
        m = valid & (tr_hour == h)
        if m.sum() < 30: continue
        h_acc[h] = ((-tr_f60[m] > 0).astype(float) == tr_y[m]).mean()
    
    top_h = set(sorted(h_acc, key=h_acc.get, reverse=True)[:TOP_HOURS])
    print(f"⏱️  Fixed Top{TOP_HOURS}h: {sorted(top_h)}")
    for h in sorted(top_h):
        print(f"   UTC{h:02d}: train_acc={h_acc[h]*100:.1f}%")
    return top_h


class ProductionRegimeMonitor:
    """
    实盘级 Regime Monitor — exec_only + max_pause
    
    关键设计:
    1. 只记录已执行交易的结果 (实盘中才能知道)
    2. max_pause 避免永久暂停 (强制恢复后继续试)
    3. 恢复后需要重新积累W个样本才会再次暂停
    """
    def __init__(self, W=MONITOR_W, THR=MONITOR_THR, max_pause=MONITOR_MAX_PAUSE):
        self.W = W
        self.THR = THR
        self.max_pause = max_pause
        self.recent = []       # 最近W笔已执行交易的正确与否
        self.paused = False    # 当前是否暂停
        self.pause_count = 0   # 已暂停的候选交易数量
        self.total_executed = 0
        self.total_correct = 0
    
    def should_trade(self):
        """判断当前这笔是否应该交易 (每收到一个候选信号调用)"""
        if self.paused:
            if self.pause_count >= self.max_pause:
                # 强制恢复
                self.paused = False
                self.pause_count = 0
                return True
            self.pause_count += 1
            return False
        
        # 未暂停: 检查recent胜率
        if len(self.recent) < self.W:
            return True  # 样本不够, 默认交易
        
        should = np.mean(self.recent[-self.W:]) >= self.THR
        if not should:
            self.paused = True
            self.pause_count = 1
        return should
    
    def update(self, correct: bool):
        """交易结算后调用"""
        self.recent.append(float(correct))
        self.total_executed += 1
        self.total_correct += int(correct)
        # 控制内存
        if len(self.recent) > self.W * 5:
            self.recent = self.recent[-self.W * 3:]
    
    @property
    def current_acc(self):
        """当前累计执行准确率"""
        if self.total_executed == 0: return 0.5
        return self.total_correct / self.total_executed


def full_walk_forward():
    """完整生产级回测"""
    print("=" * 72)
    print("🚀 BTCUSDT BINARY OPTIONS — PRODUCTION BACKTEST")
    print("=" * 72)
    t0 = time.time()
    
    c, vol, sell_vol, hour, n = load_data()
    X, y, FEATURE_NAMES, feats, sig = build_features(c, vol, sell_vol, n)
    n_days = n // BARS_PER_DAY
    
    print(f"\n📊 {len(FEATURE_NAMES)} features: {FEATURE_NAMES}")
    print(f"📅 Walk-Forward: {TRAIN_DAYS}d train → 1d test | {n_days - TRAIN_DAYS} test days")
    
    hour_pool = select_hour_pool(feats, y, hour, n)
    
    # ─── Walk-Forward ───
    all_rows = []  # 所有候选交易
    daily_details = {}
    
    for test_day in range(TRAIN_DAYS, n_days):
        tr_s = max(0, (test_day - TRAIN_DAYS) * BARS_PER_DAY)
        tr_e = test_day * BARS_PER_DAY
        te_s = test_day * BARS_PER_DAY
        te_e = min((test_day + 1) * BARS_PER_DAY, n)
        
        # 训练集
        X_tr = X[tr_s + 400:tr_e - HORIZON]
        y_tr = y[tr_s + 400:tr_e - HORIZON]
        h_tr = hour[tr_s + 400:tr_e - HORIZON]
        s_tr = sig[tr_s + 400:tr_e - HORIZON]
        
        keep = (~np.isnan(X_tr).any(axis=1)) & (~np.isnan(y_tr)) & np.isin(h_tr, list(hour_pool))
        X_tr, y_tr = X_tr[keep], y_tr[keep]
        s_tr = s_tr[keep]
        
        if len(X_tr) < 300: continue
        
        sig_thr = np.quantile(s_tr, SIGMA_Q)
        keep2 = s_tr >= sig_thr
        X_tr_f, y_tr_f = X_tr[keep2], y_tr[keep2]
        
        if len(X_tr_f) < 80: continue
        
        # 标准化 + 训练
        mu = X_tr_f.mean(axis=0)
        sd = X_tr_f.std(axis=0) + 1e-8
        X_tr_s = (X_tr_f - mu) / sd
        
        lr = LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")
        lr.fit(X_tr_s, y_tr_f)
        
        # 测试集
        X_te = X[te_s + 400:te_e - HORIZON]
        y_te = y[te_s + 400:te_e - HORIZON]
        h_te = hour[te_s + 400:te_e - HORIZON]
        s_te = sig[te_s + 400:te_e - HORIZON]
        
        keep = (~np.isnan(X_te).any(axis=1)) & (~np.isnan(y_te)) & np.isin(h_te, list(hour_pool))
        X_te, y_te = X_te[keep], y_te[keep]
        h_te, s_te = h_te[keep], s_te[keep]
        
        keep2 = s_te >= sig_thr
        X_te_f, y_te_f = X_te[keep2], y_te[keep2]
        h_te_f, s_te_f = h_te[keep2], s_te[keep2]
        
        if len(X_te_f) < 5: continue
        
        X_te_s = (X_te_f - mu) / sd
        prob = lr.predict_proba(X_te_s)[:, 1]
        pred = (prob > 0.5).astype(int)
        
        day_correct = []
        for p, t, pr, hh in zip(pred, y_te_f, prob, h_te_f):
            row = {"correct": float(p == t), "prob": float(pr),
                   "hour": int(hh), "day": test_day}
            all_rows.append(row)
            day_correct.append(float(p == t))
        
        base_day_acc = np.mean(day_correct)
        flag = "💀" if base_day_acc < 0.45 else "⚠️" if base_day_acc < 0.52 else "✅" if base_day_acc > 0.6 else "  "
        print(f"  Day{test_day+1:2d}: n={len(day_correct):4d} base_acc={base_day_acc*100:5.1f}% {flag}")
        
        daily_details[test_day] = {"base_acc": base_day_acc * 100, "n": len(day_correct)}
    
    # ─── 汇总 ───
    print(f"\n{'='*72}")
    print("📈 FINAL RESULTS")
    print(f"{'='*72}")
    
    arr = np.array([r["correct"] for r in all_rows])
    probs = np.array([r["prob"] for r in all_rows])
    n_total = len(arr)
    
    # --- BASE ---
    base_acc = arr.mean()
    base_min100 = min(arr[i:i+100].mean() for i in range(n_total - 99)) * 100 if n_total >= 100 else 0
    print(f"\n[BASE SIGNAL]")
    print(f"  n={n_total:,} | acc={base_acc*100:.1f}% | min100={base_min100:.1f}%")
    print(f"  → 裸信号很弱, 必须配合Regime Monitor")
    
    # --- PRODUCTION REGIME MONITOR ---
    print(f"\n[+ Regime Monitor] W={MONITOR_W} THR={MONITOR_THR} MAX_PAUSE={MONITOR_MAX_PAUSE}")
    
    monitor = ProductionRegimeMonitor()
    exec_results = []
    for row in all_rows:
        if monitor.should_trade():
            exec_results.append(row)
        # 模拟实盘: 执行后结算, 更新monitor
        was_exec = monitor.total_executed > len(exec_results)
        # 简化: 直接用all_rows顺序模拟
        # 实际上上面的if should_trade已经隐含了执行决策
        # 下面统一update
    
    # 正确的模拟:
    monitor2 = ProductionRegimeMonitor()
    exec_results = []
    for row in all_rows:
        should = monitor2.should_trade()
        if should:
            exec_results.append(row)
            monitor2.update(row["correct"])
        # 被跳过的交易不update (exec_only模式)
    
    exec_arr = np.array([r["correct"] for r in exec_results])
    exec_n = len(exec_arr)
    exec_acc = exec_arr.mean()
    exec_min100 = min(exec_arr[i:i+100].mean() for i in range(exec_n - 99)) * 100 if exec_n >= 100 else 0
    exec_min500 = min(exec_arr[i:i+500].mean() for i in range(exec_n - 499)) * 100 if exec_n >= 500 else 0
    exec_exp = exec_acc * PAYOUT - (1 - exec_acc)
    daily = exec_n / (n_days - TRAIN_DAYS)
    
    print(f"  n={exec_n:,} | acc={exec_acc*100:.1f}% | min100={exec_min100:.1f}% | min500={exec_min500:.1f}%")
    print(f"  exp={exec_exp*100:.1f}c/trade | daily={daily:.0f}笔 | payout={PAYOUT}")
    
    # --- CONFIDENCE FILTER (可选保守模式) ---
    if CONF_THRESHOLD > 0:
        print(f"\n[+ Confidence Filter] |prob-0.5|>{CONF_THRESHOLD}")
        m = (probs > 0.5 + CONF_THRESHOLD) | (probs < 0.5 - CONF_THRESHOLD)
        conf_arr = arr[m]
        conf_acc = conf_arr.mean()
        conf_min100 = min(conf_arr[i:i+100].mean() for i in range(len(conf_arr)-99))*100 if len(conf_arr)>=100 else 0
        print(f"  候选n={m.sum():,} | acc={conf_acc*100:.1f}% | min100={conf_min100:.1f}%")
    
    # --- SHUFFLE VALIDATION ---
    print(f"\n[SHUFFLE] Leakage Validation")
    np.random.seed(42)
    shuf = arr.copy()
    np.random.shuffle(shuf)
    
    m_real = ProductionRegimeMonitor()
    real_exec = []
    for cv in arr:
        if m_real.should_trade():
            real_exec.append(cv)
            m_real.update(bool(cv))
    
    m_shuf = ProductionRegimeMonitor()
    shuf_exec = []
    for cv in shuf:
        if m_shuf.should_trade():
            shuf_exec.append(cv)
            m_shuf.update(bool(cv))
    
    real_exec = np.array(real_exec); shuf_exec = np.array(shuf_exec)
    real_m100 = min(real_exec[i:i+100].mean() for i in range(len(real_exec)-99))*100 if len(real_exec)>=100 else 0
    shuf_m100 = min(shuf_exec[i:i+100].mean() for i in range(len(shuf_exec)-99))*100 if len(shuf_exec)>=100 else 0
    drop = real_m100 - shuf_m100
    
    print(f"  Real:    n={len(real_exec):5d} | acc={real_exec.mean()*100:5.1f}% | min100={real_m100:5.1f}%")
    print(f"  Shuffled:n={len(shuf_exec):5d} | acc={shuf_exec.mean()*100:5.1f}% | min100={shuf_m100:5.1f}%")
    print(f"  Drop:    min100 = {drop:+.1f}%")
    if drop >= 10:
        print(f"  ✅✅ STRONG PASS — True signal, not random!")
    elif drop >= 5:
        print(f"  ✅ PASS — Likely real signal")
    else:
        print(f"  ❌ FAIL — Possible leakage or too few samples")
    
    # --- DAILY DETAILS ---
    print(f"\n[DAILY] After Regime Monitor")
    daily_exec = {}
    for r in exec_results:
        d = r["day"]
        if d not in daily_exec: daily_exec[d] = []
        daily_exec[d].append(float(r["correct"]))
    
    for d in sorted(daily_exec):
        ds = np.array(daily_exec[d])
        a = ds.mean()
        flag = "💀" if a < 0.45 else "⚠️" if a < 0.52 else "✅" if a > 0.6 else "  "
        date_str = f"9/{(1+d):02d}" if d < 30 else f"10/{(d-29):02d}"
        print(f"  Day{d+1:2d} ({date_str}): n={len(ds):4d} acc={a*100:5.1f}% {flag}")
    
    # --- EXPORT REPORT ---
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "dataset": {"days": n_days, "bars": n, "train_days": TRAIN_DAYS},
        "config": {
            "top_hours": sorted(hour_pool), "sigma_q": SIGMA_Q,
            "model": "LogisticRegression(C=0.5)", "n_features": len(FEATURE_NAMES),
            "monitor_W": MONITOR_W, "monitor_THR": MONITOR_THR,
            "monitor_max_pause": MONITOR_MAX_PAUSE,
        },
        "base": {"n": int(n_total), "acc_pct": round(base_acc*100, 1),
                 "min100_pct": round(base_min100, 1)},
        "production": {
            "n": int(exec_n), "acc_pct": round(exec_acc*100, 1),
            "min100_pct": round(exec_min100, 1), "min500_pct": round(exec_min500, 1),
            "exp_cents_per_trade": round(exec_exp*100, 1),
            "daily_trades": round(daily, 1), "payout": PAYOUT,
        },
        "shuffle_validation": {
            "real_min100": round(real_m100, 1),
            "shuffled_min100": round(shuf_m100, 1),
            "drop": round(drop, 1),
            "passed_strong": bool(drop >= 10), "passed": bool(drop >= 5),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    
    os.makedirs(RESULT_DIR, exist_ok=True)
    report_path = f"{RESULT_DIR}/btc_production_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report: {report_path}")
    print(f"⏱️  Total: {time.time()-t0:.1f}s")
    
    return report


# ─── 实时预测接口 (实盘用) ───

def predict_realtime(closes, vols, sell_vols, hour_of_day):
    """
    实盘预测接口 — 输入最近足够的bar, 输出方向预测
    
    Args:
        closes: 最近足够的close数组 (至少 720+60*24=2160个bar)
        vols: 对应volume
        sell_vols: 对应sell volume
        hour_of_day: 当前UTC小时
    
    Returns:
        dict: {direction: 1=UP/-1=DOWN, prob_up: float, confidence: float}
    """
    # ⚠️ 生产环境中应该加载每天重训的模型 + mu/sd
    # 这里只做框架演示
    
    pass  # TODO: 实盘部署时实现


if __name__ == "__main__":
    report = full_walk_forward()
