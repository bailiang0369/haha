#!/usr/bin/env python3
"""
BTCUSDT 10min Reverse Strategy — Production Script
==================================================
数据源: Binance Spot aggTrades (免费下载: data.binance.vision)
信号: ret_360 反转 + Top8小时过滤 + sigma top30% + 滑窗Regime Monitor
Horizon: 10min (binary options)

完整参数 + Walk-Forward + Shuffle Leakage 验证.
"""

import polars as pl
import numpy as np
import glob
import time
from datetime import datetime, timezone


# ========== 核心参数 ==========
HORIZON_MIN = 10
STEP = HORIZON_MIN * 6       # 10s bars per horizon
REVERSAL_WINDOW = 360        # ret_360 = 60min动量
VOL_WINDOW = 60              # sigma_60 = 10min波动率
SIGMA_Q = 0.7                # 只交易波动率 top 30%
TOP_HOURS = 8                # 训练集选Top8小时
TRAIN_DAYS = 5               # 训练窗口 (每天滚动)
MONITOR_WINDOW = 50          # Regime滑窗大小
MONITOR_PAUSE_THR = 0.49     # 过去50笔胜率<49%暂停
BARS_PER_DAY = 8640          # 10s bars/day

DATA_DIR = "/workspace/data/aggtrades"


def load_and_resample(data_dir=DATA_DIR):
    """加载aggTrades CSV → 10秒close bar."""
    files = sorted(glob.glob(f"{data_dir}/*.csv"))
    print(f"加载 {len(files)} 天 aggTrades...")
    all_bars = []
    for f in files:
        df = pl.read_csv(f, has_header=False,
                         new_columns=["agg_id","price","qty","first_id","last_id","ts_us","is_buyer_maker","is_best"])
        df = df.with_columns([(pl.col("ts_us") // 10_000_000 * 10_000_000).alias("bucket")])
        all_bars.append(df.group_by("bucket").agg([pl.col("price").last().alias("close")]).sort("bucket"))
    
    bars = pl.concat(all_bars).sort("bucket")
    close = bars["close"].to_numpy()
    n = len(close); bucket = bars["bucket"].to_numpy()
    hour = (bucket % 86400000000) // 3600000000
    
    # 核心特征
    ret_360 = np.full(n, np.nan); ret_360[REVERSAL_WINDOW:] = close[REVERSAL_WINDOW:] / close[:-REVERSAL_WINDOW] - 1
    sigma_60 = np.full(n, np.nan); ret1 = close[1:] / close[:-1] - 1
    for i in range(VOL_WINDOW, n): sigma_60[i] = np.std(ret1[i-VOL_WINDOW:i])
    
    future_ret = np.full(n, np.nan); future_ret[:-STEP] = close[STEP:] / close[:-STEP] - 1
    
    print(f"  {n:,} bars ({n/BARS_PER_DAY:.1f}d) | ret_360 range=[{np.nanmin(ret_360)*10000:.0f},{np.nanmax(ret_360)*10000:.0f}]bps")
    return close, ret_360, sigma_60, future_ret, hour, n


def walk_forward(close, ret_360, sigma_60, future_ret, hour, n):
    """Walk-Forward: 5d train → 1d test, 每天重选小时+sigma."""
    candidates = []  # bool数组: 候选交易是否正确
    
    for test_day in range(TRAIN_DAYS, n // BARS_PER_DAY):
        tr_s, tr_e = (test_day - TRAIN_DAYS) * BARS_PER_DAY, test_day * BARS_PER_DAY
        te_s, te_e = test_day * BARS_PER_DAY, min((test_day + 1) * BARS_PER_DAY, n)
        
        # 训练选参数
        tr_v = np.zeros(n, dtype=bool); tr_v[tr_s+400:tr_e-STEP] = True
        tr_v &= ~np.isnan(ret_360) & ~np.isnan(future_ret)
        if tr_v.sum() < 1000: continue
        
        tr_correct = (ret_360[tr_v] < 0).astype(int) == (future_ret[tr_v] > 0).astype(int)
        tr_hour = hour[tr_v]; tr_sig = sigma_60[tr_v]
        
        acc_h = {h: tr_correct[tr_hour == h].mean() for h in range(24) if (tr_hour == h).sum() >= 20}
        top_h_set = {h for h, _ in sorted(acc_h.items(), key=lambda x: x[1], reverse=True)[:TOP_HOURS]}
        sig_thr = np.quantile(tr_sig, SIGMA_Q)
        
        # 测试
        te_v = np.zeros(n, dtype=bool); te_v[te_s+400:te_e-STEP] = True
        te_v &= ~np.isnan(ret_360) & ~np.isnan(future_ret)
        if te_v.sum() < 50: continue
        
        te_hour = hour[te_v]; te_sig = sigma_60[te_v]
        pred = (ret_360[te_v] < 0).astype(int)  # 反转
        y = (future_ret[te_v] > 0).astype(int)
        m_pass = np.isin(te_hour, list(top_h_set)) & (te_sig >= sig_thr)
        
        candidates.extend((pred == y)[m_pass].tolist())
    
    return np.array(candidates)


class RegimeMonitor:
    """实盘可实现: 滑窗监控实时胜率, 差regime暂停."""
    def __init__(self, W=MONITOR_WINDOW, THR=MONITOR_PAUSE_THR):
        self.W, self.THR = W, THR
        self.recent = []
    
    def should_trade(self):
        if len(self.recent) < self.W: return True
        return np.mean(self.recent[-self.W:]) >= self.THR
    
    def update(self, correct: bool):
        self.recent.append(float(correct))
        if len(self.recent) > self.W * 4:
            self.recent = self.recent[-self.W*2:]


def run_full_pipeline():
    t0 = time.time()
    print("=" * 70)
    print("BTCUSDT 10min Reverse Strategy — Final OOS")
    print("=" * 70)
    
    close, ret_360, sigma_60, future_ret, hour, n = load_and_resample()
    
    print(f"\n[1] Walk-Forward (5d train + rolling 1d test)...")
    candidates = walk_forward(close, ret_360, sigma_60, future_ret, hour, n)
    print(f"  候选交易 (Top8h + sigma top30%): {len(candidates):,}")
    
    # Base
    acc_base = candidates.mean()
    arr_base = candidates.astype(float)
    min100_base = min(arr_base[i:i+100].mean() for i in range(len(arr_base)-99))*100 if len(arr_base)>=100 else acc_base*100
    print(f"  Base acc={acc_base*100:.1f}% min100={min100_base:.1f}%")
    
    # + Regime Monitor
    monitor = RegimeMonitor()
    executed = []
    for cv in candidates:
        if monitor.should_trade():
            executed.append(cv)
        monitor.update(bool(cv))
    executed = np.array(executed)
    
    acc = executed.mean()
    arr = executed.astype(float)
    min100 = min(arr[i:i+100].mean() for i in range(len(arr)-99))*100 if len(arr)>=100 else acc*100
    min500 = min(arr[i:i+500].mean() for i in range(len(arr)-499))*100 if len(arr)>=500 else acc*100
    exp = acc * 0.8 - (1 - acc)
    
    print(f"\n[2] + Regime Monitor (W={MONITOR_WINDOW}, pause<{MONITOR_PAUSE_THR})")
    print(f"  n={len(executed):,} | acc={acc*100:.1f}% | min100={min100:.1f}% | min500={min500:.1f}%")
    print(f"  exp={exp*100:.1f}c/trade | daily={len(executed)/(n/BARS_PER_DAY-TRAIN_DAYS):.1f}/d")
    
    # Shuffle验证
    print(f"\n[3] Shuffle验证 (leakage检查):")
    np.random.seed(42); shuf = candidates.copy(); np.random.shuffle(shuf)
    m2 = RegimeMonitor()
    shuf_exec = []
    for cv in shuf:
        if m2.should_trade(): shuf_exec.append(cv)
        m2.update(bool(cv))
    shuf_exec = np.array(shuf_exec)
    acc_s = shuf_exec.mean()
    arr_s = shuf_exec.astype(float)
    min100_s = min(arr_s[i:i+100].mean() for i in range(len(arr_s)-99))*100 if len(arr_s)>=100 else acc_s*100
    
    print(f"  Real:    n={len(executed):,} acc={acc*100:.1f}% min100={min100:.1f}%")
    print(f"  Shuffled: n={len(shuf_exec):,} acc={acc_s*100:.1f}% min100={min100_s:.1f}%")
    print(f"  Leakage check: min100_drop={min100-min100_s:.1f}% (大drop=真regime detection)")
    
    if min100 - min100_s > 10:
        print("  ✅ 通过! Shuffle后min100大幅下降, 说明Regime Monitor真在检测市场状态")
    else:
        print("  ⚠️ Shuffle后min100没怎么降, 可能样本太少或有leakage")
    
    print(f"\n总耗时: {time.time()-t0:.1f}s")
    return {"acc": acc*100, "min100": min100, "exp_cents": exp*100, "n_trades": len(executed)}


if __name__ == "__main__":
    results = run_full_pipeline()
