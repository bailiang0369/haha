"""事件合约回测引擎.

事件合约假设:
  - 到期时间 = PREDICT_HORIZON_MIN (3 min)
  - 买方在时刻 t 决定买涨 (UP) 或买跌 (DOWN), 投入本金 S
  - 若方向判断正确, 到期获利 S * (EVENT_ODDS - 1); 否则亏掉 S
  - 手续费 EVENT_FEE_PCT 开仓时扣除
  - 模型置信度低于 MIN_CONFIDENCE 不下注
  - 每次下注占总资金 POSITION_SIZE_PCT, 单笔固定比例, 永不加杠杆
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

import config

log = logging.getLogger("backtest")

Side = Literal["UP", "DOWN"]


@dataclass
class TradeLog:
    t: pd.Timestamp
    side: Side
    size: float           # 投入本金 (USDT)
    entry_price: float
    exit_price: float
    exit_time: pd.Timestamp
    pnl: float
    ret: float            # 单笔收益率 = pnl / size
    confidence: float
    correct: bool


@dataclass
class BacktestResult:
    trades: list[TradeLog]
    equity: pd.Series
    stats: dict


def run(
    predictions: pd.DataFrame,
    close_series: pd.Series,
    initial: float = config.INITIAL_CAPITAL,
) -> BacktestResult:
    """跑回测.

    Parameters
    ----------
    predictions : DataFrame
        index 为 close_time, 列需包含:
          pred      int (0 DOWN / 1 NEUTRAL / 2 UP)
          conf_up   float [0,1]
          conf_down float [0,1]
    close_series : Series
        原始 K 线的 close, index = open_time. 会内部对齐到 close_time.
    """
    close = close_series.copy()
    close.index = close.index + pd.to_timedelta(1, unit="m")
    close = close.reindex(predictions.index)

    h = config.PREDICT_HORIZON_MIN
    pnl_list: list[TradeLog] = []
    capital = initial
    equity_curve = {predictions.index[0]: initial}
    last_capital = capital

    for t, row in predictions.iterrows():
        # 记录时间戳上的资金变化
        equity_curve[t] = last_capital  # 先记下未动资金
        if row["pred"] == 1:            # NEUTRAL 不下注
            continue

        side: Side = "UP" if row["pred"] == 2 else "DOWN"
        conf = row["conf_up"] if side == "UP" else row["conf_down"]
        if conf < config.MIN_CONFIDENCE:
            continue

        entry_price = close.loc[t]
        exit_time = t + pd.to_timedelta(h, unit="m")
        if exit_time not in close.index or pd.isna(close.loc[exit_time]):
            continue
        exit_price = close.loc[exit_time]

        size = capital * config.POSITION_SIZE_PCT
        fee = size * config.EVENT_FEE_PCT
        size_after_fee = size - fee

        # 结算
        if (side == "UP" and exit_price > entry_price) or (side == "DOWN" and exit_price < entry_price):
            pnl = size_after_fee * (config.EVENT_ODDS - 1)   # 赢
            correct = True
        else:
            pnl = -size_after_fee                            # 亏
            correct = False
        capital += pnl
        last_capital = capital

        pnl_list.append(TradeLog(
            t=t, side=side, size=size_after_fee,
            entry_price=entry_price, exit_price=exit_price,
            exit_time=exit_time, pnl=pnl,
            ret=pnl / (size_after_fee + 1e-9),
            confidence=conf, correct=correct,
        ))
        equity_curve[t] = capital

    equity = pd.Series(equity_curve).sort_index()
    stats = _compute_stats(pnl_list, equity, initial)
    log.info(f"backtest done. trades={len(pnl_list)} ROI={stats['roi_pct']:.2f}%")
    return BacktestResult(trades=pnl_list, equity=equity, stats=stats)


def _compute_stats(trades: list[TradeLog], equity: pd.Series, initial: float) -> dict:
    final = equity.iloc[-1]
    roi = final / initial - 1
    if not trades:
        return {"trades": 0, "roi_pct": roi * 100, "sharpe": np.nan, "max_dd": np.nan}

    correct = sum(1 for t in trades if t.correct)
    win_rate = correct / len(trades)
    pnls = np.array([t.pnl for t in trades])
    rets = np.array([t.ret for t in trades])
    avg_win = float(np.mean(rets[rets > 0])) if (rets > 0).any() else 0.0
    avg_loss = float(np.mean(rets[rets < 0])) if (rets < 0).any() else 0.0
    profit_factor = (pnls[pnls > 0].sum() / abs(pnls[pnls < 0].sum())) \
        if (pnls < 0).any() else float("inf")

    # 日度夏普 (假设按分钟采样 equity, 年化)
    eq_ret = equity.pct_change().dropna()
    sharpe = float(eq_ret.mean() / (eq_ret.std() + 1e-9) * np.sqrt(24 * 60))

    # 最大回撤
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax
    max_dd = float(dd.min())

    return {
        "trades": len(trades),
        "win_rate": win_rate,
        "roi_pct": roi * 100,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "sharpe_ann": sharpe,
        "max_drawdown": max_dd,
        "final_capital": final,
    }
