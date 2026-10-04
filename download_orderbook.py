#!/usr/bin/env python3
"""
Wrapper around cryptohftdata bulk CLI for Binance BTCUSDT orderbook downloads.

Covers both spot (binance_spot) and USDⓈ-M futures (binance_futures) markets
in a date range. Interrupted runs resume from where they stopped (the cryptohftdata
bulk tool skips already-present files).

Usage:
    python3 download_orderbook.py \
        --start 2026-09-04 --end 2026-10-04 \
        --assets BTC --market both --exchanges binance \
        --output /workspace/data --api-key "$(cat /workspace/.cryptohft_key)"
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone


EXCHANGE_MAP = {
    ("binance", "spot"): "binance_spot",
    ("binance", "futures"): "binance_futures",
    ("bybit", "spot"): "bybit_spot",
    ("bybit", "futures"): "bybit",           # bybit futures is "bybit"
    ("okx", "spot"): "okx_spot",
    ("okx", "futures"): "okx_futures",
    ("kraken", "spot"): "kraken_spot",
    ("kraken", "futures"): "kraken_derivatives",
}


def build_plan(exchanges: list[str], markets: list[str], asset: str) -> list[tuple[str, str]]:
    plan = []
    for ex in exchanges:
        for m in markets:
            key = (ex, m)
            if key not in EXCHANGE_MAP:
                print(f"[WARN] unsupported exchange/market: {ex}/{m} — skipping", file=sys.stderr)
                continue
            plan.append((EXCHANGE_MAP[key], f"{asset}USDT"))
    return plan


def run_bulk(
    exchange: str,
    symbol: str,
    start: str,
    end: str,
    dest: str,
    api_key: str | None,
    workers: int,
) -> int:
    cmd = [
        "cryptohftdata", "bulk",
        "--exchange", exchange,
        "--data-type", "orderbook",
        "--start", start,
        "--end", end,
        "--symbols", symbol,
        "--dest", dest,
        "--layout", "mirror",
        "--yes",
        "--workers", str(workers),
    ]
    env = os.environ.copy()
    if api_key:
        env["CRYPTOHFTDATA_API_KEY"] = api_key
        cmd += ["--api-key", api_key]

    print(f"[{ts()}] >>> {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0
    print(f"[{ts()}] <<< {exchange} {symbol} exit={proc.returncode} elapsed={elapsed:.0f}s", flush=True)
    return proc.returncode


def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")
    ap.add_argument("--assets", default="BTC", help="comma-separated base assets, e.g. BTC,ETH")
    ap.add_argument("--market", default="both", choices=["spot", "futures", "both"])
    ap.add_argument("--exchanges", default="binance", help="comma-separated, e.g. binance,bybit")
    ap.add_argument("--output", default="/workspace/data")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--workers", type=int, default=4, help="concurrent downloads (default 4 for api endpoint free tier)")
    args = ap.parse_args()

    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    exchanges = [e.strip().lower() for e in args.exchanges.split(",") if e.strip()]
    if args.market == "both":
        markets = ["spot", "futures"]
    else:
        markets = [args.market]

    os.makedirs(args.output, exist_ok=True)

    plan: list[tuple[str, str]] = []
    for asset in assets:
        plan.extend(build_plan(exchanges, markets, asset))

    print(f"[{ts()}] Plan: {len(plan)} bulk download(s)", flush=True)
    for ex, sym in plan:
        print(f"    - {ex}  {sym}", flush=True)

    failures = 0
    for exchange, symbol in plan:
        rc = run_bulk(
            exchange=exchange,
            symbol=symbol,
            start=args.start,
            end=args.end,
            dest=args.output,
            api_key=args.api_key,
            workers=args.workers,
        )
        if rc != 0:
            print(f"[{ts()}] !!! bulk download FAILED for {exchange}/{symbol} (rc={rc})", file=sys.stderr, flush=True)
            failures += 1

    if failures:
        print(f"[{ts()}] Done with {failures} failure(s).", file=sys.stderr, flush=True)
        return 1
    print(f"[{ts()}] All downloads complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
