#!/usr/bin/env python3
"""Binance BTCUSDT spot + futures L2 orderbook bulk download wrapper."""
from __future__ import annotations
import argparse, os, subprocess, sys, time
from datetime import datetime, timezone

EXCHANGE_MAP = {
    ("binance", "spot"):    "binance_spot",
    ("binance", "futures"): "binance_futures",
    ("bybit",   "spot"):    "bybit_spot",
    ("bybit",   "futures"): "bybit",
    ("okx",     "spot"):    "okx_spot",
    ("okx",     "futures"): "okx_futures",
}

def ts(): return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def bulk(exchange, symbol, start, end, dest, api_key, workers):
    cmd = ["cryptohftdata", "bulk",
           "--exchange", exchange, "--data-type", "orderbook",
           "--start", start, "--end", end,
           "--symbols", symbol, "--dest", dest,
           "--layout", "mirror", "--yes", "--workers", str(workers)]
    env = os.environ.copy()
    if api_key:
        env["CRYPTOHFTDATA_API_KEY"] = api_key
        cmd += ["--api-key", api_key]
    print(f"[{ts()}] >>> {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, env=env).returncode
    print(f"[{ts()}] <<< {exchange} {symbol} rc={rc} elapsed={time.time()-t0:.0f}s", flush=True)
    return rc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end",   required=True)
    ap.add_argument("--assets",  default="BTC")
    ap.add_argument("--market",  default="both", choices=["spot","futures","both"])
    ap.add_argument("--exchanges", default="binance")
    ap.add_argument("--output",  default="/workspace/data")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    assets    = [x.strip() for x in a.assets.split(",")    if x.strip()]
    exchanges = [x.strip().lower() for x in a.exchanges.split(",") if x.strip()]
    markets   = ["spot","futures"] if a.market=="both" else [a.market]
    os.makedirs(a.output, exist_ok=True)

    plan = []
    for asset in assets:
        for ex in exchanges:
            for m in markets:
                k = (ex, m)
                if k in EXCHANGE_MAP:
                    plan.append((EXCHANGE_MAP[k], f"{asset}USDT"))
                else:
                    print(f"[WARN] unsupported {ex}/{m}", file=sys.stderr)

    print(f"[{ts()}] Plan: {len(plan)} bulk download(s)", flush=True)
    for ex, sym in plan: print(f"    - {ex}  {sym}", flush=True)

    fails = 0
    for ex, sym in plan:
        if bulk(ex, sym, a.start, a.end, a.output, a.api_key, a.workers) != 0:
            fails += 1
    print(f"[{ts()}] Done. failures={fails}", flush=True)
    return 1 if fails else 0

if __name__ == "__main__": sys.exit(main())
