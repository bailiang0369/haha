"""
risk_control.py — 风控过滤器

Crash zone 根因 (ARCHIVE_2026-10-05.md 第四节):
  1. 连续亏损熔断: 模型方向完全反了, conf 反而升高, 连续 25min 准确率 4%
  2. Regime detection: 市场从震荡 → 趋势 切换时, 模型反应慢 5-10min
  3. σ 突变: rolling volatility 飙升, 市场不稳定

本模块实现三种过滤器:
  A. ConsecutiveLossCircuitBreaker — 连续亏 N 笔 → 暂停 M 分钟
  B. VolatilityRegimeDetector      — rolling σ 突变 → 降权或暂停
  C. MomentumOverheatDetector      — 连续同方向 ret → 市场过热

用法:
  python3 risk_control.py                    # 用默认参数跑
  python3 risk_control.py --loss-n 8 --loss-pause 30 --vol-z 3
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime, timezone
import lightgbm as lgb
import sys


def to_ts(y, m, d, h=0, mi=0):
    return int(datetime(y, m, d, h, mi, tzinfo=timezone.utc).timestamp() * 1000)


def load_data(snap_dir: Path):
    """加载快照 + walk-forward 训练."""
    spot = pl.concat([pl.read_parquet(f).with_columns(
        pl.col("*").name.map(lambda c: f"spot_{c}" if c not in ("ts_ms","ts") else c)
    ) for f in sorted((snap_dir/"binance_spot").glob("*.parquet"))]).sort("ts_ms")
    fut = pl.concat([pl.read_parquet(f).with_columns(
        pl.col("*").name.map(lambda c: f"fut_{c}" if c not in ("ts_ms","ts") else c)
    ) for f in sorted((snap_dir/"binance_futures").glob("*.parquet"))]).sort("ts_ms")
    merged = fut.join_asof(spot, on="ts_ms", strategy="backward", tolerance=5000)
    drop = ["_prices", "_qtys", "ts", "label", "future_ret", "_right"]
    feat_cols = [c for c in merged.columns if not any(k in c for k in drop)]
    return merged, feat_cols


def walk_forward_predict(merged, feat_cols):
    """返回所有 walk-forward 预测结果 (逐笔)."""
    X_all = merged.select(feat_cols).to_numpy()
    ts    = merged["ts_ms"].to_numpy()
    mid   = merged["fut_mid"].to_numpy()

    step = 30
    future_ret = np.full(len(mid), np.nan)
    for i in range(len(mid)-step):
        future_ret[i] = (mid[i+step]-mid[i])/mid[i]

    day_starts = sorted(set(ts // 86400000))
    ret0 = future_ret[(ts >= to_ts(2026,9,4)) & (ts < to_ts(2026,9,11)) & ~np.isnan(future_ret)]
    thr_label = abs(np.quantile(ret0, 0.25))

    rows = []
    for day_i in range(7, len(day_starts)):
        train_lo = day_starts[day_i-7] * 86400000
        train_hi = day_starts[day_i]   * 86400000
        test_lo  = day_starts[day_i]   * 86400000
        test_hi  = (day_starts[day_i]+1) * 86400000
        train_mask = (ts >= train_lo) & (ts < train_hi)
        test_mask  = (ts >= test_lo)  & (ts < test_hi)
        train_y_ret = future_ret[train_mask]
        train_has = ~np.isnan(train_y_ret) & (np.abs(train_y_ret) > thr_label)
        if train_has.sum() < 1000: continue
        X_tr = X_all[train_mask][train_has]
        y_tr = (train_y_ret[train_has] > 0).astype(int)
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                                min_child_samples=200, verbose=-1, random_state=42)
        m.fit(X_tr, y_tr)
        test_idx = np.where(test_mask)[0]
        real_ret = future_ret[test_idx]
        test_has = ~np.isnan(real_ret) & (np.abs(real_ret) > thr_label)
        X_te = X_all[test_idx][test_has]
        proba = m.predict_proba(X_te)[:, 1]
        real_dir = (real_ret[test_has] > 0).astype(int)
        # 同时保留 sigma 和 mid_ret 用于 regime 检测
        sigma_vals = merged.select("fut_sigma").to_numpy().ravel()[test_idx[test_has]]
        mid_ret_vals = merged.select("fut_mid_ret_10s").to_numpy().ravel()[test_idx[test_has]]

        for j in range(len(X_te)):
            rows.append({
                "ts": int(ts[test_idx[test_has][j]]),
                "proba": float(proba[j]),
                "real_dir": int(real_dir[j]),
                "sigma": float(sigma_vals[j]),
                "mid_ret_10s": float(mid_ret_vals[j]),
            })
    return rows


# ============================================================
# 风控过滤器
# ============================================================

class RiskControlResult:
    """单笔预测是否允许交易."""
    __slots__ = ("allow", "reason", "adjusted_confidence")

    def __init__(self, allow: bool, reason: str = "", adj_conf: float = None):
        self.allow = allow
        self.reason = reason
        self.adjusted_confidence = adj_conf  # 降权后的 confidence


class ConsecutiveLossCircuitBreaker:
    """连续亏损熔断: 连续亏 N 笔 → 暂停 PAUSE_MIN 分钟."""

    def __init__(self, n_loss: int = 8, pause_min: int = 30):
        self.n_loss = n_loss
        self.pause_ms = pause_min * 60 * 1000
        self.loss_streak = 0
        self.pause_until = 0

    def check(self, ts: int, pred_correct: bool) -> bool:
        """返回是否允许交易."""
        if ts < self.pause_until:
            return False
        if pred_correct:
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            if self.loss_streak >= self.n_loss:
                self.pause_until = ts + self.pause_ms
                self.loss_streak = 0
        return True


class VolatilityRegimeDetector:
    """σ 突变检测: rolling σ > Z × 历史分位数 → 降权."""

    def __init__(self, z_threshold: float = 3.0, window: int = 1000):
        self.z_threshold = z_threshold
        self.window = window
        self.sigma_history = []

    def check(self, sigma: float) -> float:
        """返回 confidence 调节因子 (0~1). 1.0=正常, <1=降权."""
        if len(self.sigma_history) >= self.window:
            self.sigma_history.pop(0)
        self.sigma_history.append(sigma)
        if len(self.sigma_history) < 100:
            return 1.0  # 预热期不做
        hist = np.array(self.sigma_history)
        mu, sd = hist.mean(), hist.std() + 1e-6
        z = abs(sigma - mu) / sd
        if z > self.z_threshold:
            # 线性降权: z=3 → 0.5, z=5 → 0.0
            adj = max(0.0, 1.0 - (z - self.z_threshold) * 0.25)
            return adj
        return 1.0


class MomentumOverheatDetector:
    """动量过热检测: 连续 N 个 bar 同方向 ret 且累计 ret > 阈值 → 降权."""

    def __init__(self, n_consec: int = 5, cum_ret_threshold: float = 0.002):
        self.n_consec = n_consec
        self.cum_ret_threshold = cum_ret_threshold
        self.recent_rets = []

    def check(self, ret_10s: float) -> float:
        """返回 confidence 调节因子."""
        self.recent_rets.append(ret_10s)
        if len(self.recent_rets) > self.n_consec:
            self.recent_rets.pop(0)
        if len(self.recent_rets) < self.n_consec:
            return 1.0
        rets = np.array(self.recent_rets)
        same_dir = np.all(rets > 0) or np.all(rets < 0)
        cum = abs(rets.sum())
        if same_dir and cum > self.cum_ret_threshold:
            return 0.3  # 强降权
        return 1.0


def run_with_filters(rows, conf_threshold: float = 0.7,
                     loss_n: int = 8, loss_pause: int = 30,
                     vol_z: float = 3.0):
    """
    带风控过滤器的 walk-forward 回测.

    返回:
      trades: list of dict — 最终执行的交易 (含风控标记)
      skipped: dict — 各过滤器跳过的数量
    """
    # 先预测 (按 confidence 过滤)
    trade_candidates = []
    for r in rows:
        conf = abs(r["proba"] - 0.5) * 2
        if conf >= conf_threshold:
            pred_dir = 1 if r["proba"] > 0.5 else 0
            real_dir = r["real_dir"]
            correct = (pred_dir == real_dir)
            trade_candidates.append({
                **r, "conf": conf, "pred_dir": pred_dir,
                "correct": correct, "allow": True, "reason": "",
                "adj_conf": conf,
            })

    # 初始化风控
    breaker = ConsecutiveLossCircuitBreaker(n_loss=loss_n, pause_min=loss_pause)
    vol_det = VolatilityRegimeDetector(z_threshold=vol_z)
    mom_det = MomentumOverheatDetector(n_consec=5, cum_ret_threshold=0.002)

    skipped = {"breaker": 0, "vol": 0, "momentum": 0}
    trades = []

    for t in trade_candidates:
        # 先做降权 (影响 confidence, 不允许完全跳过)
        vol_adj = vol_det.check(t["sigma"])
        mom_adj = mom_det.check(t["mid_ret_10s"])
        adj_conf = t["conf"] * vol_adj * mom_adj
        t["adj_conf"] = adj_conf

        # 波动率/动量过热 → confidence 降到 0.3 以下 → 不交易
        if adj_conf < 0.3:
            if vol_adj < 1.0 and mom_adj < 1.0:
                skipped["vol"] += 1
            elif vol_adj < 1.0:
                skipped["vol"] += 1
            else:
                skipped["momentum"] += 1
            t["allow"] = False
            continue

        # 连续亏损熔断
        if not breaker.check(t["ts"], t["correct"]):
            skipped["breaker"] += 1
            t["allow"] = False
            continue

        # 允许交易, 但后续检查结果时要喂给 breaker
        breaker.check(t["ts"], t["correct"])  # 更新 streak
        trades.append(t)

    return trades, skipped


def analyze(trades, skipped, label=""):
    """分析风控后结果."""
    if not trades:
        print(f"  {label}: 0 trades")
        return {}
    arr = np.array([t["correct"] for t in trades])
    ts_arr = np.array([t["ts"] for t in trades])
    win_streaks = []
    cur = 0
    for c in arr:
        if c: cur += 1
        else:
            if cur > 0: win_streaks.append(cur)
            cur = 0
    win_streaks.append(cur)
    win_streaks = [s for s in win_streaks if s > 0]

    # 滚动 100 笔准确率
    if len(arr) >= 100:
        rolling = np.array([arr[i:i+100].mean() for i in range(len(arr)-99)])
        min_roll = rolling.min()
        p05 = np.percentile(rolling, 5)
    else:
        min_roll = arr.mean()
        p05 = arr.mean()

    out = {
        "n": len(trades),
        "acc": arr.mean(),
        "min_rolling_100": min_roll,
        "p05_rolling_100": p05,
    }
    print(f"  {label}: {len(trades):,} trades  acc={arr.mean()*100:.1f}%  "
          f"min-roll100={min_roll*100:.1f}%  P05-roll100={p05*100:.1f}%")
    print(f"    跳过: 熔断={skipped['breaker']:,}  vol={skipped['vol']:,}  momentum={skipped['momentum']:,}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snap-dir", default="/workspace/snapshots")
    parser.add_argument("--conf", type=float, default=0.7)
    parser.add_argument("--loss-n", type=int, default=8, help="连续亏 N 笔触发熔断")
    parser.add_argument("--loss-pause", type=int, default=30, help="熔断后暂停分钟数")
    parser.add_argument("--vol-z", type=float, default=3.0, help="σ 突变 z-score 阈值")
    parser.add_argument("--no-filter", action="store_true", help="不启用风控, 只跑 baseline 对照")
    args = parser.parse_args()

    snap_dir = Path(args.snap_dir)

    print(f"{'='*70}")
    print(f"  risk_control.py — 风控过滤器实验")
    print(f"  confidence threshold = {args.conf}")
    print(f"  熔断: 连续亏 {args.loss_n} 笔 → 暂停 {args.loss_pause} min")
    print(f"  vol σ-z 阈值: {args.vol_z}")
    print(f"{'='*70}\n")

    print("[1/3] 加载数据 ...")
    merged, feat_cols = load_data(snap_dir)
    print(f"  merged shape: {merged.shape}")

    print("\n[2/3] Walk-forward 训练 + 预测 ...")
    rows = walk_forward_predict(merged, feat_cols)
    print(f"  预测完成: {len(rows):,} samples")

    # ---- 对照: 无风控 ----
    print(f"\n[3/3] 风控对比实验 (conf≥{args.conf})")
    print(f"  {'-'*70}")

    baseline_trades, baseline_skip = run_with_filters(
        rows, conf_threshold=args.conf, loss_n=99999, loss_pause=0, vol_z=999
    )
    analyze(baseline_trades, baseline_skip, "BASELINE (无风控)")

    if not args.no_filter:
        for n_loss in [5, 8, 10]:
            for pause in [15, 30, 60]:
                trades, skipped = run_with_filters(
                    rows, conf_threshold=args.conf,
                    loss_n=n_loss, loss_pause=pause, vol_z=args.vol_z
                )
                analyze(trades, skipped, f"熔断 n={n_loss} pause={pause}min")

    print(f"\n{'='*70}")
    print("  核心问题验证: 风控后最差窗口会不会从 4% 变成 > 50%?")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
