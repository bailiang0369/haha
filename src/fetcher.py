"""从币安拉取 U 本位合约数据: K 线 / 逐笔成交 / L2 depth.

所有下载均基于币安公开 REST API，K 线和 depth 无需 API key；
historicalTrades 需 key，若缺失则 trades 会退化为 depth 快照合成。

运行时会从 .env 加载 BINANCE_API_KEY / BINANCE_API_SECRET (可选)。
"""
from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

import config

load_dotenv()
if os.getenv("BINANCE_API_KEY"):
    config.BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")

log = logging.getLogger("fetcher")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# 低级别 HTTP
# ---------------------------------------------------------------------------
def _get(path: str, params: dict, use_key: bool = False) -> list | dict:
    url = config.BINANCE_BASE + path
    headers = {}
    if use_key and config.BINANCE_API_KEY:
        headers["X-MBX-APIKEY"] = config.BINANCE_API_KEY
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=config.REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.json()
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                wait = 2 ** attempt + 1
                log.warning(f"429 rate limited, sleep {wait}s ...")
                time.sleep(wait)
                continue
            if e.response is not None and e.response.status_code == 418:
                log.error("IP banned, abort")
                raise
            log.warning(f"HTTP {e} retry {attempt}/5")
            time.sleep(1.5 * (attempt + 1))
        except requests.RequestException as e:
            log.warning(f"network error {e} retry {attempt}/5")
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed GET {path} params={params}")


def _sleep() -> None:
    time.sleep(config.REQUEST_INTERVAL)


# ---------------------------------------------------------------------------
# K 线
# ---------------------------------------------------------------------------
def fetch_klines(
    symbol: str = config.SYMBOL,
    interval: str = config.TIMEFRAME,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    limit: int = 1000,
    auto_paginate: bool = True,
) -> pd.DataFrame:
    """拉取 1m K 线 (币安每次返回最多 1000 条，自动翻页)."""
    rows: list[list] = []
    cur_start = start_ms
    with tqdm(desc="klines", unit="batch") as pbar:
        while True:
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            if cur_start is not None:
                params["startTime"] = cur_start
            if end_ms is not None:
                params["endTime"] = end_ms
            batch = _get("/fapi/v1/klines", params)
            if not batch:
                break
            rows.extend(batch)
            pbar.update(1)
            if not auto_paginate or len(batch) < limit:
                break
            cur_start = batch[-1][6] + 1  # close_time + 1ms 避免重叠
            if end_ms is not None and cur_start >= end_ms:
                break
            _sleep()

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open", "high", "low", "close", "volume",
              "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    df.sort_index(inplace=True)
    log.info(f"klines rows={len(df)} from {df.index.min()} -> {df.index.max()}")
    return df


# ---------------------------------------------------------------------------
# 逐笔成交
# ---------------------------------------------------------------------------
def fetch_trades_snapshot(limit: int = 500) -> pd.DataFrame:
    """最近 500 条逐笔(免 key)."""
    data = _get("/fapi/v1/trades", {"symbol": config.SYMBOL, "limit": limit})
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    for c in ["price", "qty"]:
        df[c] = df[c].astype(float)
    df.rename(columns={"isBuyerMaker": "is_maker_sell"}, inplace=True)
    return df


def fetch_historical_trades(
    symbol: str = config.SYMBOL,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    from_id: Optional[int] = None,
    limit: int = 1000,
    max_batches: int = 500,
) -> pd.DataFrame:
    """拉历史逐笔(需 API key). 没 key 会抛错."""
    if not config.BINANCE_API_KEY:
        raise RuntimeError("BINANCE_API_KEY not set; historicalTrades requires it")
    all_rows = []
    cur_from = from_id
    with tqdm(desc="historicalTrades", unit="batch") as pbar:
        for _ in range(max_batches):
            params = {"symbol": symbol, "limit": limit}
            if cur_from is not None:
                params["fromId"] = cur_from
            if start_ms is not None:
                params["startTime"] = start_ms
            if end_ms is not None:
                params["endTime"] = end_ms
            batch = _get("/fapi/v1/historicalTrades", params, use_key=True)
            if not batch:
                break
            all_rows.extend(batch)
            pbar.update(1)
            if len(batch) < limit:
                break
            # by trade id is monotonic increasing in response
            cur_from = batch[-1]["id"] + 1
            if end_ms is not None and batch[-1]["time"] >= end_ms:
                break
            _sleep()
    df = pd.DataFrame(all_rows).drop_duplicates(subset=["id"]) if all_rows else pd.DataFrame()
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
        for c in ["price", "qty"]:
            df[c] = df[c].astype(float)
        df.rename(columns={"isBuyerMaker": "is_maker_sell"}, inplace=True)
        df.sort_values("time", inplace=True)
    return df


# ---------------------------------------------------------------------------
# L2 depth (top 1000 levels, 最新快照)
# ---------------------------------------------------------------------------
def fetch_depth_snapshot(limit: int = 1000) -> dict:
    """一次性深度快照. 返回 {'bids': [[px,qty],...], 'asks': [...], 'time': ms}."""
    data = _get("/fapi/v1/depth", {"symbol": config.SYMBOL, "limit": limit})
    data["time"] = int(time.time() * 1000)
    return data


# ---------------------------------------------------------------------------
# 高层: 一键下载 & 缓存
# ---------------------------------------------------------------------------
def download_klines(
    days: int = 60,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    df = fetch_klines(start_ms=start_ms, end_ms=end_ms)
    out_path = out_path or (config.DATA_DIR / f"{config.SYMBOL}_1m_klines_{days}d.parquet")
    df.to_parquet(out_path)
    log.info(f"saved -> {out_path}")
    return df


def download_historical_trades(
    days: int = 3,
    out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """逐笔历史(需 API key). 默认只拉 3 天，量太大。"""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 24 * 60 * 60 * 1000
    df = fetch_historical_trades(start_ms=start_ms, end_ms=end_ms)
    out_path = out_path or (config.DATA_DIR / f"{config.SYMBOL}_trades_{days}d.parquet")
    df.to_parquet(out_path)
    log.info(f"saved -> {out_path}")
    return df


if __name__ == "__main__":
    # 快速 smoke test: 拉最近 3 天 1m K 线
    df = download_klines(days=3)
    print(df.head(3))
    print(f"shape={df.shape}")
