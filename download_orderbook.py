#!/usr/bin/env python3
"""CryptoHFTData hourly orderbook bulk downloader.

Downloads hourly Parquet/Zstd orderbook files from cryptohftdata REST API
into a mirror of the upstream R2 layout under --output. Progress is
continuously persisted to <output>/.download_state.json so a restart can
resume skipping already-present hourly files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import random
from dataclasses import dataclass, asdict

# Flush every print so the log written by the watchdog stays live.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.parse import urlencode

import requests


API_BASE = "https://api.cryptohftdata.com/v1"
REQUIRED_MARKETS = {
    "spot": ["binance_spot"],
    "futures": ["binance_futures"],
}
DATA_TYPE = "orderbook"
SYMBOL = "BTCUSDT"


def daterange(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def build_file_list(
    start: date,
    end: date,
    exchanges: List[str],
    markets: List[str],
) -> List[str]:
    """Return upstream R2 object keys, e.g. binance_futures/2026-10-03/12/BTCUSDT_orderbook.parquet."""
    paths: List[str] = []
    for exchange in exchanges:
        for d in daterange(start, end):
            for h in range(24):
                paths.append(
                    f"{exchange}/{d.isoformat()}/{h:02d}/{SYMBOL}_{DATA_TYPE}.parquet"
                )
    return paths


def market_to_exchanges(market: str, exchanges: List[str]) -> List[str]:
    """Translate `--market both|spot|futures` into concrete REST exchange ids."""
    if market == "both":
        out = []
        for m in ("spot", "futures"):
            out.extend(REQUIRED_MARKETS[m])
        return out
    if market in REQUIRED_MARKETS:
        return REQUIRED_MARKETS[market]
    # caller passed raw exchange ids
    return exchanges


def download_one(
    session: requests.Session,
    file_path: str,
    api_key: str,
    dest: Path,
    timeout: int = 120,
    max_retries: int = 4,
) -> tuple[bool, Optional[str]]:
    """Download one hourly file. Returns (success, error_reason).

    302 is expected (worker signs a presigned URL) -- we follow.
    404 = upstream has no object yet (legitimate for future hours / gaps).
    Anything else is a real failure we retry with backoff.
    """
    url = f"{API_BASE}/download"
    params = {"file": file_path, "api_key": api_key}

    for attempt in range(max_retries + 1):
        try:
            # stream=True so we can write directly to disk without holding
            # multi-MB parquet in memory; proxied via env HTTPS_PROXY.
            r = session.get(
                url,
                params=params,
                timeout=timeout,
                allow_redirects=True,
                stream=True,
            )
            if r.status_code == 404:
                return False, "not_found"
            if r.status_code == 429:
                sleep_s = float(r.headers.get("Retry-After") or 2 ** attempt)
                time.sleep(sleep_s + random.random())
                continue
            if r.status_code != 200:
                if attempt < max_retries:
                    time.sleep(2 ** attempt + random.random())
                    continue
                return False, f"http_{r.status_code}"

            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp, dest)
            return True, None
        except requests.exceptions.RequestException as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt + random.random())
                continue
            return False, f"network:{type(e).__name__}"
    return False, "exhausted"


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n //= 1024
    return f"{n}TB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")
    ap.add_argument("--assets", required=True, help="comma-sep, BTC for now")
    ap.add_argument("--market", required=True, choices=["spot", "futures", "both"])
    ap.add_argument("--exchanges", required=True, help="comma-sep exchange ids, e.g. binance")
    ap.add_argument("--output", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--sleep-between", type=float, default=0.25,
                    help="Seconds between each download to stay well under 60/min free tier.")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    # Restrict to BTC -- we don't auto-discover per-asset symbol yet.
    assets = [a.strip().upper() for a in args.assets.split(",")]
    if assets != ["BTC"]:
        print(f"[warn] asset list {assets} requested but symbol map only covers BTC -> BTCUSDT; "
              "others will be skipped", file=sys.stderr)
    exchanges_arg = [e.strip() for e in args.exchanges.split(",")]
    exchanges = market_to_exchanges(args.market, exchanges_arg)

    state_path = output / ".download_state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
        except json.JSONDecodeError:
            state = {}
    else:
        state = {}

    file_list = build_file_list(start, end, exchanges, args.market)
    expected = len(file_list)
    print(f"[init] {expected} hourly objects across {exchanges} from {start} to {end}")

    # A file is already done if it exists on disk OR was recorded as ok in state.
    def already_done(fp: str) -> bool:
        local = output / fp
        if local.exists() and local.stat().st_size > 0:
            return True
        return state.get(fp) == "ok"

    todo = [fp for fp in file_list if not already_done(fp)]
    print(f"[init] {len(file_list) - len(todo)} already on disk, {len(todo)} pending")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "btc-l2-orderbook-watchdog/1.0",
    })

    started = time.time()
    ok = missing = failed = 0
    bytes_total = 0

    for i, fp in enumerate(todo, 1):
        dest = output / fp
        if dest.exists() and dest.stat().st_size > 0:
            ok += 1
            continue

        success, reason = download_one(
            session, fp, args.api_key, dest,
            timeout=args.timeout, max_retries=args.max_retries,
        )
        if success:
            size = dest.stat().st_size
            bytes_total += size
            ok += 1
            state[fp] = "ok"
        elif reason == "not_found":
            missing += 1
            # leave state missing so next run retries (upstream may lag by a day)
        else:
            failed += 1
            state[fp] = reason

        # checkpoint every 20 files so we don't lose progress on crash
        if i % 20 == 0:
            state_path.write_text(json.dumps(state))
            elapsed = time.time() - started
            rate = ok / max(elapsed, 1e-6)
            print(f"[progress] {ok+missing+failed}/{expected} "
                  f"(ok={ok} missing={missing} failed={failed}) "
                  f"{human(bytes_total)} done @ {rate:.2f} files/s")

        if args.sleep_between > 0:
            time.sleep(args.sleep_between)

    state_path.write_text(json.dumps(state))
    elapsed = time.time() - started
    rate = ok / max(elapsed, 1e-6)
    print(f"[done] ok={ok} missing={missing} failed={failed} size={human(bytes_total)} "
          f"elapsed={elapsed/60:.1f}min rate={rate:.2f}/s")
    if failed:
        print(f"[warn] {failed} failed files; check {state_path} for reasons", file=sys.stderr)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
