#!/usr/bin/env python3
"""CryptoHFTData L2 orderbook bulk downloader.

Wrapper around cryptohftdata SDK that accepts user-facing CLI flags
(--start / --end / --assets / --market / --exchanges / --output / --api-key)
and translates them into one or more download_bulk calls.

Markets: spot | futures | both
Exchanges: binance (expands to binance_spot + binance_futures when market=both)
Assets: comma-separated symbols like BTC or BTC,ETH  (treated as BTCUSDT, ETHUSDT)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

import cryptohftdata as chd

MARKET_TO_EXCHANGE = {
    "spot": ["binance_spot"],
    "futures": ["binance_futures"],
    "both": ["binance_spot", "binance_futures"],
}


def asset_to_symbol(asset: str) -> str:
    """BTC -> BTCUSDT, BTCUSDT -> BTCUSDT (idempotent)."""
    a = asset.strip().upper()
    if a.endswith("USDT") or a.endswith("USDC") or a.endswith("FDUSD"):
        return a
    return f"{a}USDT"


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def iter_days(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d = date.fromordinal(d.toordinal() + 1)


def run_download(
    api_key: str,
    symbols: list[str],
    start: str,
    end: str,
    exchanges: list[str],
    output: Path,
) -> int:
    output.mkdir(parents=True, exist_ok=True)

    client = chd.CryptoHFTDataClient(api_key=api_key)
    failures = 0

    for ex in exchanges:
        for sym in symbols:
            print(
                f"[download_orderbook] START exchange={ex} symbol={sym} "
                f"start={start} end={end} dest={output}",
                flush=True,
            )
            try:
                result = client.download_bulk(
                    exchange=ex,
                    data_type="orderbook",
                    start=start,
                    end=end,
                    symbols=[sym],
                    dest=str(output),
                    confirm=False,
                    transport="auto",
                    max_workers=16,
                )
                print(
                    f"[download_orderbook] DONE  exchange={ex} symbol={sym} "
                    f"path={result.path} downloaded={result.downloaded} "
                    f"skipped={result.skipped} errors={result.errors}",
                    flush=True,
                )
            except Exception as exc:
                failures += 1
                print(
                    f"[download_orderbook] ERROR exchange={ex} symbol={sym}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    return failures


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download L2 orderbook via CryptoHFTData SDK")
    p.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")
    p.add_argument("--assets", required=True, help="comma-separated e.g. BTC or BTC,ETH")
    p.add_argument("--market", required=True, choices=["spot", "futures", "both"])
    p.add_argument("--exchanges", required=True, help="binance (for now)")
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument("--api-key", required=False, default=None, help="or read from CRYPTOHFTDATA_API_KEY")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args(argv)

    api_key = args.api_key or os.environ.get("CRYPTOHFTDATA_API_KEY")
    if not api_key:
        print("ERROR: --api-key or CRYPTOHFTDATA_API_KEY required", file=sys.stderr)
        return 2

    symbols = [asset_to_symbol(a) for a in args.assets.split(",") if a.strip()]
    exchanges = MARKET_TO_EXCHANGE[args.market]

    failures = run_download(
        api_key=api_key,
        symbols=symbols,
        start=args.start,
        end=args.end,
        exchanges=exchanges,
        output=Path(args.output),
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
