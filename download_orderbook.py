#!/usr/bin/env python3
"""BTC L2 orderbook bulk downloader for CryptoHFTData REST API.

Features:
- Resumable: skips existing valid files
- Concurrent download via ThreadPoolExecutor (default 4 workers)
- Graceful shutdown: state written after every file
- Rate-limit aware: auto-retry on 429
"""
import argparse
import datetime as dt
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

API_BASE = "https://api.cryptohftdata.com/download"
STATE_PATH = Path("/workspace/.watch_state.json")
MIN_FILE_BYTES = 50_000

SYMBOL_MAP = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
}
EXCHANGE_MAP = {
    ("binance", "spot"): "binance_spot",
    ("binance", "futures"): "binance_futures",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--assets", default="BTC")
    p.add_argument("--market", default="both", choices=["spot", "futures", "both"])
    p.add_argument("--exchanges", default="binance")
    p.add_argument("--output", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--request-gap", type=float, default=0.3,
                   help="min seconds between starting new requests (60/min = 1s)")
    p.add_argument("--retries", type=int, default=5)
    return p.parse_args()


def daterange(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def build_file_list(args):
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    assets = [a.strip().upper() for a in args.assets.split(",")]
    exchanges = [e.strip().lower() for e in args.exchanges.split(",")]
    markets = []
    if args.market in ("spot", "both"):
        markets.append("spot")
    if args.market in ("futures", "both"):
        markets.append("futures")
    files = []
    for asset in assets:
        symbol = SYMBOL_MAP.get(asset, f"{asset}USDT")
        for exch in exchanges:
            for mkt in markets:
                ex_dir = EXCHANGE_MAP.get((exch, mkt))
                if not ex_dir:
                    continue
                for day in daterange(start, end):
                    for h in range(24):
                        fname = f"{symbol}_orderbook.parquet"
                        remote = f"{ex_dir}/{day.isoformat()}/{h:02d}/{fname}"
                        local = Path(args.output) / ex_dir / day.isoformat() / f"{h:02d}" / fname
                        files.append({
                            "remote": remote,
                            "local": str(local),
                            "symbol": symbol,
                            "market": mkt,
                        })
    return files


def file_ok(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        return False
    return p.stat().st_size >= MIN_FILE_BYTES


_last_request_ts = 0.0
_rate_lock = threading.Lock()


def _rate_wait(gap: float):
    global _last_request_ts
    with _rate_lock:
        now = time.monotonic()
        wait = gap - (now - _last_request_ts)
        if wait > 0:
            time.sleep(wait)
        _last_request_ts = time.monotonic()


def download_one(remote: str, local: str, api_key: str, gap: float, retries: int) -> bool:
    _rate_wait(gap)
    url = f"{API_BASE}?file={remote}&api_key={api_key}"
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            s = requests.Session()
            r = s.get(url, timeout=120, stream=True,
                      headers={"User-Agent": "ob-dl/2.0"})
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 120))
                print(f"  429 rate-limited, sleeping {wait}s", file=sys.stderr)
                time.sleep(wait)
                _rate_wait(gap)
                continue
            if r.status_code != 200:
                print(f"  HTTP {r.status_code} for {remote}", file=sys.stderr)
                if attempt < retries:
                    time.sleep(2 * attempt)
                continue
            tmp = local + ".part"
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
            size = os.path.getsize(tmp)
            if size < MIN_FILE_BYTES:
                os.remove(tmp)
                return False
            os.replace(tmp, local)
            return True
        except Exception as e:
            print(f"  err ({attempt}/{retries}) {remote}: {e}", file=sys.stderr)
            if attempt < retries:
                time.sleep(3 * attempt)
    return False


class Counter:
    def __init__(self):
        self.lock = threading.Lock()
        self.done = 0
        self.spot = 0
        self.futures = 0
        self.failed = 0


def main():
    args = parse_args()
    files = build_file_list(args)
    expected = len(files)

    # Split into skip vs download
    to_download = []
    skipped = 0
    spot_done = 0
    futures_done = 0
    for f in files:
        if file_ok(f["local"]):
            skipped += 1
            if f["market"] == "spot":
                spot_done += 1
            else:
                futures_done += 1
        else:
            to_download.append(f)

    total_done = skipped
    print(f"[init] expected={expected}, already_ok={skipped}, to_download={len(to_download)}",
          file=sys.stderr)

    counter = Counter()
    counter.done = skipped
    counter.spot = spot_done
    counter.futures = futures_done
    counter.failed = 0
    started = time.time()
    output_root = Path(args.output)

    def total_bytes():
        if not output_root.exists():
            return 0
        return sum(p.stat().st_size for p in output_root.rglob("*.parquet")
                   if p.is_file())

    def disk_avail_mb():
        try:
            usage = shutil.disk_usage(str(output_root.parent))
            return usage.free / (1024 * 1024)
        except Exception:
            return 0.0

    def write_state():
        d, sp, fu, fa = counter.done, counter.spot, counter.futures, counter.failed
        pct = 100.0 * d / expected if expected else 0.0
        state = {
            "total": d,
            "spot": sp,
            "futures": fu,
            "expected": expected,
            "progress": f"{pct:.1f}%",
            "size_mb": round(total_bytes() / (1024 * 1024), 1),
            "status": "complete" if d >= expected else "running",
            "failed": fa,
            "disk_avail_mb": round(disk_avail_mb(), 0),
            "start": args.start,
            "end": args.end,
            "assets": args.assets,
            "market": args.market,
            "exchanges": args.exchanges,
            "workers": args.concurrency,
            "updated_at": dt.datetime.utcnow().isoformat() + "Z",
            "elapsed_s": round(time.time() - started, 1),
        }
        try:
            tmp = STATE_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=2))
            os.replace(tmp, STATE_PATH)
        except Exception as e:
            print(f"[state] write failed: {e}", file=sys.stderr)

    write_state()  # initial

    def worker(fn):
        ok = download_one(fn["remote"], fn["local"], args.api_key,
                          args.request_gap, args.retries)
        with counter.lock:
            counter.done += 1
            if fn["market"] == "spot":
                counter.spot += 1
            else:
                counter.futures += 1
            if not ok:
                counter.failed += 1
        return ok, fn

    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = {ex.submit(worker, f): f for f in to_download}
        for fut in as_completed(futures):
            try:
                ok, fn = fut.result()
                mkt = fn["market"]
                if ok:
                    print(f"  OK {mkt[:3]} {fn['remote']}", flush=True)
                else:
                    print(f"  FAIL {mkt[:3]} {fn['remote']}", file=sys.stderr, flush=True)
            except Exception as e:
                print(f"  worker exception: {e}", file=sys.stderr)
                with counter.lock:
                    counter.done += 1
                    counter.failed += 1

            with counter.lock:
                cur_done = counter.done
                cur_spot = counter.spot
                cur_fut = counter.futures
                cur_failed = counter.failed

            write_state()

            if cur_done % 20 == 0 or cur_done >= expected:
                elapsed = time.time() - started
                rate = cur_done / elapsed if elapsed > 0 else 0
                pct = 100.0 * cur_done / expected
                size_mb = total_bytes() / (1024 * 1024)
                print(f"[{cur_done}/{expected}] spot={cur_spot} fut={cur_fut} "
                      f"fail={cur_failed} {pct:.1f}% {size_mb:.0f}MB "
                      f"rate={rate:.2f}/s eta={(expected-cur_done)/rate:.0f}s",
                      flush=True)

    elapsed = time.time() - started
    write_state()
    d, sp, fu, fa = counter.done, counter.spot, counter.futures, counter.failed
    print(f"\nDONE: {d}/{expected} (spot={sp} futures={fu} failed={fa}) "
          f"in {elapsed:.0f}s", flush=True)
    sys.exit(0 if d >= expected else 1)


if __name__ == "__main__":
    main()
