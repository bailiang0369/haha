#!/usr/bin/env python3
"""Download BTC L2 orderbook from CryptoHFTData.

Thin wrapper around `cryptohftdata bulk` CLI. Supports spot+futures markets,
resumes interrupted runs automatically. Writes a .watch_state.json summary
so the watchdog can report progress.

Usage:
    python3 download_orderbook.py --start 2026-09-04 --end 2026-10-04 \\
        --assets BTC --market both --exchanges binance \\
        --output /workspace/data --api-key <key>
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SPOT_EXCHANGES = {
    "binance": "binance_spot",
    "okx": "okx_spot",
    "bybit": "bybit_spot",
    "bitget": "bitget_spot",
    "kraken": "kraken_spot",
}
FUTURES_EXCHANGES = {
    "binance": "binance_futures",
    "okx": "okx_futures",
    "bybit": "bybit_futures",
    "bitget": "bitget_futures",
    "hyperliquid": "hyperliquid",
    "kraken": "kraken_futures",
}


def run_bulk(exchange: str, symbols: list, start: str, end: str,
             dest: str, api_key: str, workers: int) -> int:
    """Invoke `cryptohftdata bulk` for one exchange/market."""
    cmd = [
        "cryptohftdata", "bulk",
        "--exchange", exchange,
        "--data-type", "orderbook",
        "--start", start,
        "--end", end,
        "--symbols", *symbols,
        "--dest", dest,
        "--workers", str(workers),
        "--yes",
        "--api-key", api_key,
    ]
    # Filter out empty strings from symbols unpack
    cmd = [c for c in cmd if c]
    print(f"[download_orderbook] running: {' '.join(cmd[:8])} ... (symbols={symbols})", flush=True)
    result = subprocess.run(cmd)
    return result.returncode


def count_parquets(output_dir: str) -> dict:
    """Count downloaded .parquet files grouped by exchange prefix."""
    counts = {"spot": 0, "futures": 0, "total": 0}
    for path in glob.glob(f"{output_dir}/**/*.parquet", recursive=True):
        counts["total"] += 1
        p = path.lower()
        if "futures" in p or "hyperliquid" in p or "perpetual" in p:
            counts["futures"] += 1
        else:
            counts["spot"] += 1
    return counts


def disk_avail_gb(path: str) -> float:
    s = os.statvfs(path)
    return (s.f_bavail * s.f_frsize) / (1024**3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--assets", default="BTC", help="Comma-separated, e.g. BTC")
    ap.add_argument("--market", default="both", choices=["spot", "futures", "both"])
    ap.add_argument("--exchanges", default="binance", help="Comma-separated, e.g. binance,okx")
    ap.add_argument("--output", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--state-file", default=None,
                    help="Where to write .watch_state.json (default: <output>/.watch_state.json)")
    args = ap.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_file = args.state_file or str(output_dir / ".watch_state.json")

    assets = [a.strip().upper() for a in args.assets.split(",")]
    # Resolve symbols — for BTC we download BTCUSDT (primary perpetual+spot)
    # plus any common pairs. Default to BTCUSDT.
    symbols_map = {
        "BTC": "BTCUSDT",
        "ETH": "ETHUSDT",
        "SOL": "SOLUSDT",
    }
    symbols = [symbols_map.get(a, f"{a}USDT") for a in assets]

    exchanges = [e.strip().lower() for e in args.exchanges.split(",")]
    markets = ["spot", "futures"] if args.market == "both" else [args.market]

    plan = []
    for ex in exchanges:
        if "spot" in markets and ex in SPOT_EXCHANGES:
            plan.append(SPOT_EXCHANGES[ex])
        if "futures" in markets and ex in FUTURES_EXCHANGES:
            plan.append(FUTURES_EXCHANGES[ex])

    # Expected: days * 24 hours * exchanges * symbols
    from datetime import date
    d1 = date.fromisoformat(args.start)
    d2 = date.fromisoformat(args.end)
    days = (d2 - d1).days + 1
    expected = days * 24 * len(plan) * len(symbols)

    print(f"[download_orderbook] plan={plan}, symbols={symbols}, days={days}, expected={expected}", flush=True)

    start_ts = time.time()
    all_ok = True
    for ex in plan:
        rc = run_bulk(ex, symbols, args.start, args.end,
                      str(output_dir), args.api_key, args.workers)
        if rc != 0:
            print(f"[download_orderbook] FAIL exchange={ex} rc={rc}", flush=True)
            all_ok = False
        else:
            print(f"[download_orderbook] DONE exchange={ex}", flush=True)

    elapsed = int(time.time() - start_ts)
    counts = count_parquets(str(output_dir))
    state = {
        "total": counts["total"],
        "spot": counts["spot"],
        "futures": counts["futures"],
        "expected": expected,
        "progress": round(counts["total"] / expected * 100, 2) if expected else 0,
        "status": "completed" if all_ok else "partial",
        "elapsed_sec": elapsed,
        "disk_avail_gb": round(disk_avail_gb(str(output_dir)), 1),
        "started_at": int(start_ts),
        "finished_at": int(time.time()),
        "plan": plan,
        "symbols": symbols,
        "start": args.start,
        "end": args.end,
    }
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)
    print(f"[download_orderbook] STATE written to {state_file}", flush=True)
    print(json.dumps(state, indent=2), flush=True)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
