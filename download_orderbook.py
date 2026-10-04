#!/usr/bin/env python3
"""BTC L2 订单簿下载器 - CryptoHFT Data bulk CLI wrapper.

内部通过 cryptohftdata Python SDK 遍历 market × exchange 组合，
调用 cryptohftdata bulk 下载小时级 parquet 文件。支持中断续跑。

用法示例:
  python3 download_orderbook.py \
    --start 2026-09-04 --end 2026-10-04 \
    --assets BTC --market both --exchanges binance \
    --output /workspace/data \
    --api-key $CRYPTOHFTDATA_API_KEY
"""
import argparse
import json
import os
import sys
import subprocess
import time
from datetime import datetime, date, timedelta
from pathlib import Path

# SDK exchange 名称映射: 用户输入 -> cryptohftdata 实际 exchange 值
EXCHANGE_MAP = {
    "binance": ["binance_spot", "binance_futures"],
    "okx":     ["okx_spot", "okx_futures"],
    "bybit":   ["bybit_spot", "bybit_futures"],
    "bitget":  ["bitget_spot", "bitget_futures"],
    "kraken":  ["kraken_spot", "kraken_futures"],
    "hyperliquid": ["hyperliquid_spot", "hyperliquid_futures"],
}

EXCHANGE_ONLY = {
    "binance_spot":   ["binance_spot"],
    "binance_futures": ["binance_futures"],
    "okx_spot":       ["okx_spot"],
    "okx_futures":    ["okx_futures"],
}


def expand_exchanges(exchange_arg: str, market_arg: str):
    """把用户传入的 exchanges + market 参数展开成 SDK exchange 列表."""
    raw = [e.strip().lower() for e in exchange_arg.split(",")]
    markets = set(market_arg.strip().lower().split(","))

    result = []
    for r in raw:
        if r in EXCHANGE_ONLY:
            result.extend(EXCHANGE_ONLY[r])
        elif r in EXCHANGE_MAP:
            pair = EXCHANGE_MAP[r]
            if markets == {"spot"}:
                result.append(pair[0])
            elif markets == {"futures"}:
                result.append(pair[1])
            else:  # both
                result.extend(pair)
        else:
            raise ValueError(f"Unknown exchange: {r}")
    # 去重保序
    seen = set()
    out = []
    for e in result:
        if e not in seen:
            seen.add(e)
            out.append(e)
    return out


def count_hourly_files(output_dir: str) -> int:
    """统计 hive layout 下已有的 .parquet 文件数量."""
    p = Path(output_dir)
    if not p.exists():
        return 0
    return sum(1 for _ in p.rglob("*.parquet"))


def count_expected_files(start: str, end: str, exchanges, symbols):
    total_days = (date.fromisoformat(end) - date.fromisoformat(start)).days + 1
    per_exchange = total_days * 24 * len(symbols)
    return per_exchange * len(exchanges)


def run_bulk(exchange: str, symbols, start, end, dest, api_key):
    """调用 cryptohftdata bulk 下载一个 exchange 的 orderbook 数据."""
    cmd = [
        "cryptohftdata", "bulk",
        "--exchange", exchange,
        "--data-type", "orderbook",
        "--start", start,
        "--end", end,
        "--symbols", *symbols,
        "--dest", dest,
        "--layout", "hive",
        "--workers", "16",
        "--yes",
        "--quiet",
    ]
    env = os.environ.copy()
    if api_key:
        env["CRYPTOHFTDATA_API_KEY"] = api_key

    print(f"[{datetime.now().isoformat()}] RUN: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=False)
    dt = time.time() - t0
    print(f"[{datetime.now().isoformat()}] DONE {exchange} rc={proc.returncode} elapsed={dt:.1f}s", flush=True)
    return proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--assets", default="BTC")
    ap.add_argument("--market", default="both",
                    help="spot / futures / both (也可逗号分隔)")
    ap.add_argument("--exchanges", default="binance")
    ap.add_argument("--output", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--data-type", default="orderbook")  # 预留
    args = ap.parse_args()

    # 解析 symbols (资产就是 symbols, 用 USDT 对)
    symbols = []
    for a in args.assets.split(","):
        a = a.strip().upper()
        symbols.append(f"{a}USDT")

    api_key = args.api_key or os.environ.get("CRYPTOHFTDATA_API_KEY", "")
    exchanges = expand_exchanges(args.exchanges, args.market)

    expected = count_expected_files(args.start, args.end, exchanges, symbols)

    os.makedirs(args.output, exist_ok=True)

    print("=" * 72, flush=True)
    print("BTC L2 Orderbook Downloader", flush=True)
    print("=" * 72, flush=True)
    print(f"  Start:     {args.start}", flush=True)
    print(f"  End:       {args.end}", flush=True)
    print(f"  Symbols:   {symbols}", flush=True)
    print(f"  Exchanges: {exchanges}", flush=True)
    print(f"  Output:    {args.output}", flush=True)
    print(f"  API key:   {'***SET***' if api_key else '***MISSING***'}", flush=True)
    print(f"  Expected:  {expected} parquet files", flush=True)
    print("=" * 72, flush=True)

    overall_rc = 0
    for ex in exchanges:
        rc = run_bulk(ex, symbols, args.start, args.end, args.output, api_key)
        if rc != 0:
            overall_rc = rc
            print(f"WARN: {ex} bulk returned rc={rc}, continuing with remaining", flush=True)

    # 最终统计
    total = count_hourly_files(args.output)
    pct = (total / expected * 100) if expected > 0 else 0

    print("=" * 72, flush=True)
    print(f"SUMMARY: total={total}/{expected}  ({pct:.1f}%)", flush=True)
    print("=" * 72, flush=True)

    sys.exit(overall_rc)


if __name__ == "__main__":
    main()
