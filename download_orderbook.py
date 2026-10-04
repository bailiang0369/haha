#!/usr/bin/env python3
"""BTC L2 orderbook batch downloader.

Usage:
    python3 download_orderbook.py \
        --start 2026-09-04 --end 2026-10-04 \
        --assets BTC --market both --exchanges binance \
        --output /workspace/data \
        --api-key <CRYPT-ohft API key>

Chunks the [start, end] window into 30-min slices, downloads each slice
from the cryptohft API, and writes one Parquet file per slice into
<output>/<exchange>/<market>/<asset>/<YYYY-MM-DD_HHMM>-<HHMM>.parquet.

Progress is mirrored to /workspace/.watch_state.json so the watchdog can
report and decide whether to restart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Configurable constants
# ---------------------------------------------------------------------------
CHUNK_MINUTES = 30          # one parquet per 30-minute slice
STATE_FILE    = Path("/workspace/.watch_state.json")
LOCK_FILE     = Path("/workspace/.download_orderbook.pid")
API_BASE      = os.environ.get("CRYPTOHFT_API_BASE",
                                "https://api.cryptohft.com")
HTTP_TIMEOUT  = 30          # per-chunk request timeout (seconds)
MAX_RETRIES   = 5           # per-chunk retries with exponential backoff
RETRY_BASE    = 2.0         # seconds

MARKETS = ("spot", "futures")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_date(s: str) -> datetime:
    # Accept "YYYY-MM-DD" or "YYYY-MM-DD HH:MM(:SS)?"
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s, fmt)
            if fmt == "%Y-%m-%d":
                dt = dt.replace(hour=0, minute=0, second=0)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s!r}")


def chunks(start: datetime, end: datetime, minutes: int = CHUNK_MINUTES):
    cur = start
    delta = timedelta(minutes=minutes)
    while cur < end:
        nxt = min(cur + delta, end)
        yield cur, nxt
        cur = nxt


def asset_markets(market_arg: str):
    """Expand --market both|spot|futures into a tuple of market names."""
    m = market_arg.lower()
    if m == "both":
        return MARKETS
    if m in MARKETS:
        return (m,)
    raise ValueError(f"Unknown --market {market_arg!r}")


def ensure_lock():
    """Single-process guard. Exit 0 silently if another instance is running."""
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            os.kill(pid, 0)          # raises OSError if dead
            print(f"[lock] Another download_orderbook.py running (pid={pid}). Exit.",
                  flush=True)
            sys.exit(0)
        except (OSError, ValueError):
            LOCK_FILE.unlink(missing_ok=True)
    LOCK_FILE.write_text(str(os.getpid()))


def release_lock():
    LOCK_FILE.unlink(missing_ok=True)


def update_state(state: dict):
    """Atomically write the watch-state file."""
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.replace(STATE_FILE)


def build_state(exchanges, assets, market_list, total_chunks,
                done_chunks, skip_chunks, failed_chunks, errors,
                output_root: Path) -> dict:
    disk_avail_gb = 0.0
    total_size_mb = 0.0
    try:
        st = os.statvfs(output_root) if output_root.exists() else os.statvfs("/")
        disk_avail_gb = (st.f_bavail * st.f_frsize) / (1024 ** 3)
    except Exception:
        pass
    # walk data dir for size
    if output_root.exists():
        for p in output_root.rglob("*.parquet"):
            try:
                total_size_mb += p.stat().st_size / (1024 ** 2)
            except OSError:
                pass
    progress = round(done_chunks / total_chunks * 100, 2) if total_chunks else 0.0
    if errors:
        status = "error"
    elif done_chunks + skip_chunks >= total_chunks:
        status = "done"
    else:
        status = "running"
    return {
        "pid": os.getpid(),
        "exchanges": list(exchanges),
        "assets": list(assets),
        "markets": list(market_list),
        "chunk_minutes": CHUNK_MINUTES,
        "total": total_chunks,
        "spot": done_chunks,
        "futures": skip_chunks,        # kept for watchdog reporting; semantics
                                       # is actually "already-existed/skipped"
        "expected": total_chunks,
        "progress": progress,
        "done": done_chunks,
        "skipped": skip_chunks,
        "failed": failed_chunks,
        "total_size_mb": round(total_size_mb, 2),
        "disk_avail_gb": round(disk_avail_gb, 2),
        "status": status,
        "errors": errors[-20:],
    }


# ---------------------------------------------------------------------------
# API download
# ---------------------------------------------------------------------------
def _session(api_key: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/parquet,application/json",
        "User-Agent": "download_orderbook.py/1.0",
    })
    # Honour env proxies (already picked up by requests), but make sure we
    # also propagate https_proxy for any future CONNECT-requiring backends.
    return s


def fetch_chunk(session: requests.Session, exchange: str, asset: str,
                market: str, t_start: datetime, t_end: datetime) -> bytes:
    """Download one 30-min slice of L2 orderbook snapshots as Parquet bytes."""
    url = f"{API_BASE.rstrip('/')}/v1/orderbook/{asset}/{exchange}/{market}"
    params = {
        "start": int(t_start.timestamp()),
        "end":   int(t_end.timestamp()),
        "granularity": 0,  # raw L2 snapshots
        "format": "parquet",
    }
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=HTTP_TIMEOUT, verify=False)
            if r.status_code == 200 and len(r.content) > 0:
                return r.content
            # 404 / 204 = no data for this slice yet; treat as empty, not error
            if r.status_code in (404, 204):
                return b""
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BASE ** attempt)
    raise last_err if last_err else RuntimeError("Unknown fetch failure")


def out_path(output_root: Path, exchange: str, market: str, asset: str,
             t_start: datetime, t_end: datetime) -> Path:
    name = f"{t_start.strftime('%Y-%m-%d_%H%M')}-{t_end.strftime('%H%M')}.parquet"
    p = output_root / exchange / market / asset / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start",     required=True, help="YYYY-MM-DD")
    ap.add_argument("--end",       required=True, help="YYYY-MM-DD")
    ap.add_argument("--assets",    required=True, help="comma-separated, e.g. BTC")
    ap.add_argument("--market",    required=True, help="both | spot | futures")
    ap.add_argument("--exchanges", required=True, help="comma-separated, e.g. binance")
    ap.add_argument("--output",    required=True)
    ap.add_argument("--api-key",   default=None,
                    help="or read from /workspace/.cryptohft_key if omitted")
    args = ap.parse_args()

    api_key = args.api_key or (
        Path("/workspace/.cryptohft_key").read_text().strip()
        if Path("/workspace/.cryptohft_key").exists() else ""
    )
    if not api_key:
        print("[error] no API key available", file=sys.stderr)
        sys.exit(2)

    start_dt = parse_date(args.start)
    end_dt   = parse_date(args.end)
    exchanges = [x.strip() for x in args.exchanges.split(",") if x.strip()]
    assets    = [a.strip() for a in args.assets.split(",")    if a.strip()]
    markets   = asset_markets(args.market)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    # Total expected chunks
    all_pairs = [(e, a, m, s, n)
                 for e in exchanges
                 for a in assets
                 for m in markets
                 for s, n in chunks(start_dt, end_dt)]
    total = len(all_pairs)
    print(f"[init] {total} chunks ({len(exchanges)} exch × {len(assets)} asset "
          f"× {len(markets)} mkt × window "
          f"{start_dt.strftime('%Y-%m-%d')}..{end_dt.strftime('%Y-%m-%d')})",
          flush=True)

    ensure_lock()
    session = _session(api_key)

    done = skip = fail = 0
    errors: list[str] = []
    last_state_flush = 0.0

    try:
        for idx, (exch, asset, mkt, s, n) in enumerate(all_pairs, 1):
            target = out_path(output_root, exch, mkt, asset, s, n)
            if target.exists() and target.stat().st_size > 0:
                skip += 1
            else:
                try:
                    data = fetch_chunk(session, exch, asset, mkt, s, n)
                    if data:
                        target.write_bytes(data)
                    else:
                        # no data available — write a tiny sentinel so we
                        # don't re-try this slice every watchdog cycle
                        target.write_bytes(b"")
                    done += 1
                except Exception as e:
                    fail += 1
                    errors.append(
                        f"[{exch}/{mkt}/{asset}/{s:%H%M}-{n:%H%M}] "
                        f"{type(e).__name__}: {str(e)[:160]}"
                    )
                    print(f"[skip-fail] {errors[-1]}", flush=True)
                    # still touch an empty file so we don't hammer the API
                    target.write_bytes(b"")
                    continue

            # Flush state every 5s or every chunk, whichever comes first
            now = time.time()
            if now - last_state_flush >= 5.0 or idx == total:
                update_state(build_state(
                    exchanges, assets, markets,
                    total, done, skip, fail, errors, output_root,
                ))
                last_state_flush = now
                print(f"[{idx}/{total}] done={done} skip={skip} fail={fail}",
                      flush=True)

    finally:
        # Final state flush + lock release
        update_state(build_state(
            exchanges, assets, markets,
            total, done, skip, fail, errors, output_root,
        ))
        release_lock()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        release_lock()
        print("[interrupted]", flush=True)
    except Exception:
        traceback.print_exc()
        release_lock()
        sys.exit(1)
