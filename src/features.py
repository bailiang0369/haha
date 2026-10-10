"""特征工程: 从 K 线 / 逐笔 / depth 构造 1 分钟对齐的特征矩阵.

以 K 线 close_time 为特征时间戳，表示"这一分钟结束时对未来的观察视角".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# K 线特征
# ---------------------------------------------------------------------------
def build_kline_features(df: pd.DataFrame) -> pd.DataFrame:
    """df 必须是以 open_time 为 index 的 1m K 线 DataFrame.

    输出以 close_time 为 index，每个 close_time 对应这一分钟收盘时的特征快照.
    """
    df = df.copy()
    # 以 close_time 对齐，特征表示 "截至 t 时刻" 的观测
    df["t"] = df.index + pd.to_timedelta(1, unit="m")  # open_time + 1m = close_time
    df.set_index("t", inplace=True)
    df.sort_index(inplace=True)

    out = pd.DataFrame(index=df.index)
    out["ret_1m"] = df["close"].pct_change(1)
    out["ret_3m"] = df["close"].pct_change(3)
    out["ret_5m"] = df["close"].pct_change(5)
    out["ret_15m"] = df["close"].pct_change(15)
    out["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # 振幅类
    out["body"] = (df["close"] - df["open"]).abs() / df["open"]
    out["range_hl"] = (df["high"] - df["low"]) / df["open"]
    out["upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"]
    out["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"]

    # 成交量 + 滚动
    out["volume"] = df["volume"]
    out["quote_volume"] = df["quote_volume"]
    out["taker_buy_vol"] = df["taker_buy_volume"]
    out["taker_sell_vol"] = df["volume"] - df["taker_buy_volume"]
    out["taker_buy_ratio"] = df["taker_buy_volume"] / (df["volume"] + 1e-9)
    for w in [5, 15, 30, 60]:
        out[f"volume_mean_{w}"] = df["volume"].rolling(w).mean()
        out[f"volume_ratio_{w}"] = df["volume"] / out[f"volume_mean_{w}"]
        out[f"taker_buy_ratio_mean_{w}"] = out["taker_buy_ratio"].rolling(w).mean()

    # MA / 位置类 (close 相对 MA 的偏离)
    close = df["close"]
    for w in [3, 5, 10, 20, 50]:
        ma = close.rolling(w).mean()
        out[f"ma_pos_{w}"] = (close - ma) / ma

    # RSI
    out["rsi_14"] = _rsi(close, 14)
    out["rsi_6"] = _rsi(close, 6)

    # 波动率 (已实现波动)
    out["rv_5"] = out["log_return"].rolling(5).std()
    out["rv_15"] = out["log_return"].rolling(15).std()

    # 新高/新低距离
    out["hh_dist_10"] = close / close.rolling(10).max() - 1
    out["ll_dist_10"] = close / close.rolling(10).min() - 1

    return out


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - 100 / (1 + rs)


# ---------------------------------------------------------------------------
# 逐笔成交特征 (可选; 没数据时留空, 由调用方再合并)
# ---------------------------------------------------------------------------
def build_trades_features(trades: pd.DataFrame, kline_index: pd.DatetimeIndex) -> pd.DataFrame:
    """trades: 原始逐笔 DataFrame, 含 time/price/qty/is_maker_sell.

    先按 1 分钟 resample, 然后把每个分钟的聚合统计挂到 kline_index 上.
    """
    if trades is None or trades.empty:
        return pd.DataFrame(index=kline_index)

    t = trades.copy()
    t["is_buy"] = ~t["is_maker_sell"]  # 主动买入
    t.set_index("time", inplace=True)
    # 把逐笔对齐到 "所属分钟" = 下一个 1m K 线的 open_time
    # 这样 resample 后刚好对应我们以 close_time 为索引的 kline 特征
    by_min = t.resample("1min", label="right", closed="right")

    agg = pd.DataFrame(index=kline_index)
    agg["trades_count"] = by_min.size().reindex(agg.index).fillna(0)
    agg["buy_vol"] = by_min.apply(lambda g: g.loc[g["is_buy"], "qty"].sum()).reindex(agg.index).fillna(0)
    agg["sell_vol"] = by_min.apply(lambda g: g.loc[~g["is_buy"], "qty"].sum()).reindex(agg.index).fillna(0)
    agg["big_trade_count"] = by_min.apply(lambda g: (g["qty"] >= 5).sum()).reindex(agg.index).fillna(0)
    agg["buy_sell_imbalance"] = (agg["buy_vol"] - agg["sell_vol"]) / (agg["buy_vol"] + agg["sell_vol"] + 1e-9)

    for w in [3, 10, 30]:
        agg[f"trades_count_ma_{w}"] = agg["trades_count"].rolling(w).mean()
        agg[f"buy_sell_imbalance_mean_{w}"] = agg["buy_sell_imbalance"].rolling(w).mean()
    return agg


# ---------------------------------------------------------------------------
# Depth (L2) 特征 (需要预先按分钟抓快照; 这里提供对单个快照的特征)
# ---------------------------------------------------------------------------
def depth_features_from_snapshot(snapshot: dict) -> dict:
    """币安 /fapi/v1/depth 返回 bids/asks 列表. 直接对一个快照算特征."""
    bids = np.array(snapshot["bids"], dtype=float)[: config.DEPTH_LEVELS]  # [price, qty]
    asks = np.array(snapshot["asks"], dtype=float)[: config.DEPTH_LEVELS]

    mid = (bids[0, 0] + asks[0, 0]) / 2
    spread = asks[0, 0] - bids[0, 0]
    bid_vol = bids[:, 1].sum()
    ask_vol = asks[:, 1].sum()
    imb = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)

    # 加权中间价 (size-weighted mid)
    wmid = (bids[:, 0] * asks[:, 1]).sum() + (asks[:, 0] * bids[:, 1]).sum()
    wmid /= (bids[:, 1].sum() + asks[:, 1].sum() + 1e-9)

    return {
        "mid": mid,
        "spread_bps": spread / mid * 1e4,
        "bid_vol_top": bid_vol,
        "ask_vol_top": ask_vol,
        "depth_imbalance": imb,
        "weighted_mid": wmid,
        "wmid_dev": (wmid - mid) / (mid + 1e-9),
    }


def build_depth_features(snapshots: list[tuple[pd.Timestamp, dict]], kline_index: pd.DatetimeIndex) -> pd.DataFrame:
    """把一串 (timestamp, snapshot) 转成按 1m 对齐的特征 DataFrame."""
    if not snapshots:
        return pd.DataFrame(index=kline_index)
    rows = []
    for ts, snap in snapshots:
        feat = depth_features_from_snapshot(snap)
        feat["t"] = ts.floor("min") + pd.Timedelta(1, unit="m")  # 对齐到下一个 close_time
        rows.append(feat)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(index=kline_index)
    df.set_index("t", inplace=True)
    # 同一分钟内可能多次快照, 取最后一个 (最接近该分钟收盘)
    df = df[~df.index.duplicated(keep="last")]
    df = df.reindex(kline_index).sort_index()
    # 前后各 1 个缺失可接受, 其余线性填充
    df = df.ffill(limit=2).bfill(limit=2)
    # 衍生滚动
    out = df.copy()
    out["depth_imbalance_ma_3"] = out["depth_imbalance"].rolling(3).mean()
    out["spread_bps_ma_5"] = out["spread_bps"].rolling(5).mean()
    out["wmid_dev_abs"] = out["wmid_dev"].abs()
    return out


# ---------------------------------------------------------------------------
# 顶层: 把所有数据源合成一个矩阵
# ---------------------------------------------------------------------------
def build_feature_matrix(
    klines: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    depth_snapshots: list[tuple[pd.Timestamp, dict]] | None = None,
) -> pd.DataFrame:
    kf = build_kline_features(klines)
    out = kf.copy()
    index = out.index

    if trades is not None and not trades.empty:
        tf = build_trades_features(trades, index)
        out = out.join(tf, how="left")

    if depth_snapshots:
        df = build_depth_features(depth_snapshots, index)
        out = out.join(df, how="left")

    out.dropna(how="all", axis=1, inplace=True)
    # 每行至少需要 K 线部分的前 LOOKBACK_KLINES 根过后才完整
    out = out.iloc[config.LOOKBACK_KLINES:]
    out.ffill(inplace=True)
    return out
