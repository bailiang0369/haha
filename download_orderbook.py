#!/usr/bin/env python3
"""CryptoHFTData BTC L2 orderbook bulk downloader.

Downloads hourly Parquet orderbook snapshots from CryptoHFTData for the given
asset(s), exchange(s), market(s), and date range.  Resumes from existing files
on re-invocation and writes a .watch_state.json progress snapshot that a
watchdog (watch_download.sh) can read.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
import fcntl
from pathlib import Path
from typing import List

STATE_PATH = Path("/workspace/.watch_state.json")
LOCK_PATH = Path("/workspace/.download_orderbook.pid")
SYMBOL_SUFFIX = "USDT"


# ---------------------------------------------------------------------------
# state helpers
# ---------------------------------------------------------------------------
def disk_avail_bytes(path: Path) -> int:
    st = os.statvfs(str(path))
    return st.f_bavail * st.f_frsize


def scan_output(root: Path) -> dict:
    """Scan output directory and return per-market file counts and total size."""
    counts = {"spot": 0, "futures": 0}
    total_bytes = 0
    if not root.exists():
        return counts, total_bytes
    for p in root.rglob("*.parquet"):
        name = p.name.lower()
        size = p.stat().st_size
        total_bytes += size
        if "spot" in name or "spot" in str(p).lower():
            # exchange directory names like binance_spot, binance_futures
            parts = p.parts
            exchange_dir = next((x for x in parts if "binance" in x.lower() or "okx" in x.lower()), "")
            if "spot" in exchange_dir:
                counts["spot"] += 1
            elif "futures" in exchange_dir:
                counts["futures"] += 1
            else:
                # fallback: just count futures first, then spot
                counts["futures"] += 1
        else:
            counts["futures"] += 1
    return counts, total_bytes


def classify_exchange_dir(dir_name: str) -> str:
    low = dir_name.lower()
    if "futures" in low or "swap" in low or "perp" in low:
        return "futures"
    return "spot"


def scan_output_v2(root: Path) -> tuple[dict, int]:
    """More reliable scan: use parent directory name (binance_spot, ...)."""
    counts = {"spot": 0, "futures": 0}
    total_bytes = 0
    if not root.exists():
        return counts, total_bytes
    for p in root.rglob("*.parquet"):
        size = p.stat().st_size
        total_bytes += size
        # walk up until we find an exchange dir (contains underscore)
        parent = p
        market = None
        for _ in range(6):
            parent = parent.parent
            if "_" in parent.name:
                market = classify_exchange_dir(parent.name)
                break
        if market == "futures":
            counts["futures"] += 1
        elif market == "spot":
            counts["spot"] += 1
        else:
            counts["futures"] += 1  # default fallback
    return counts, total_bytes


def write_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str))
    tmp.rename(STATE_PATH)


def read_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


# ---------------------------------------------------------------------------
# generation of expected file list
# ---------------------------------------------------------------------------
def iter_hourly(start: dt.date, end: dt.date) -> List[dt.datetime]:
    """Hourly boundaries from start 00:00 to end 23:00 inclusive."""
    hours: List[dt.datetime] = []
    cur = dt.datetime.combine(start, dt.time.min)
    end_limit = dt.datetime.combine(end, dt.time(23, 0))
    while cur <= end_limit:
        hours.append(cur)
        cur += dt.timedelta(hours=1)
    return hours


def build_file_paths(assets: List[str], exchanges: List[str], markets: List[str],
                     hours: List[dt.datetime]) -> List[tuple[str, str, str]]:
    """Return list of (exchange_dir, local_subdir, filename).

    exchange_dir   -> e.g. "binance_spot" (used for both API --file and local path)
    local_subdir   -> relative dir under --output
    filename       -> e.g. "BTCUSDT_orderbook.parquet"
    """
    out: List[tuple[str, str, str]] = []
    for ex in exchanges:
        for mkt in markets:
            ex_dir = f"{ex}_{mkt}"           # e.g. binance_spot, binance_futures
            for asset in assets:
                symbol = f"{asset}{SYMBOL_SUFFIX}"
                fn = f"{symbol}_orderbook.parquet"
                for h in hours:
                    rel = f"{ex_dir}/{h.strftime('%Y-%m-%d')}/{h.strftime('%H')}/{fn}"
                    out.append((ex_dir, rel, fn))
    return out


# ---------------------------------------------------------------------------
# lock
# ---------------------------------------------------------------------------
class FileLock:
    def __init__(self, path: Path):
        self.path = path
        self.fh = None

    def __enter__(self):
        self.fh = open(self.path, "w")
        try:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pid = self.path.read_text().strip() or "?"
            raise SystemExit(f"another download_orderbook instance running (pid {pid}); abort.")
        self.fh.write(str(os.getpid()))
        self.fh.flush()
        return self

    def __exit__(self, *a):
        try:
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        self.fh.close()
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# downloader
# ---------------------------------------------------------------------------
def download_one(api_path: str, local_path: Path, api_key: str) -> bool:
    """Invoke `cryptohftdata download` for a single object.

    Returns True on success, False on failure.
    """
    local_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CRYPTOHFTDATA_API_KEY"] = api_key
    # --output must be the exact destination file path
    cmd = [
        sys.executable, "-m", "cryptohftdata.cli", "download",
        "--file", api_path,
        "--output", str(local_path),
    ]
    try:
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {api_path}", file=sys.stderr)
        return False
    if res.returncode != 0:
        print(f"  [FAIL rc={res.returncode}] {api_path}", file=sys.stderr)
        if res.stderr:
            print(res.stderr.strip()[:500], file=sys.stderr)
        return False
    # sanity: file must exist and be non-empty
    if not local_path.exists() or local_path.stat().st_size < 1000:
        print(f"  [EMPTY] {api_path}", file=sys.stderr)
        return False
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CryptoHFTData orderbook bulk downloader")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--assets", default="BTC", help="comma-separated, e.g. BTC,ETH")
    p.add_argument("--market", default="both", choices=["spot", "futures", "both"])
    p.add_argument("--exchanges", default="binance", help="comma-separated, e.g. binance,okx")
    p.add_argument("--output", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-retries", type=int, default=3)
    return p.parse_args()


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def main() -> int:
    args = parse_args()

    # dates
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    if end < start:
        print("end before start", file=sys.stderr)
        return 2

    assets = [a.strip() for a in args.assets.split(",") if a.strip()]
    exchanges = [e.strip() for e in args.exchanges.split(",") if e.strip()]
    if args.market == "both":
        markets = ["spot", "futures"]
    else:
        markets = [args.market]

    hours = iter_hourly(start, end)
    plan = build_file_paths(assets, exchanges, markets, hours)
    expected = len(plan)

    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # initial state
    spot_now, futures_now, total_now = (0, 0, 0)
    counts, size_total = scan_output_v2(out_root)
    spot_now = counts["spot"]
    futures_now = counts["futures"]
    total_now = spot_now + futures_now
    disk = disk_avail_bytes(out_root)

    state = {
        "config": {
            "start": args.start,
            "end": args.end,
            "assets": assets,
            "market": markets,
            "exchanges": exchanges,
            "output": str(out_root),
        },
        "expected": expected,
        "total": total_now,
        "spot": spot_now,
        "futures": futures_now,
        "progress": f"{total_now}/{expected} ({100*total_now/max(expected,1):.1f}%)",
        "size": human_bytes(size_total),
        "size_bytes": size_total,
        "disk_avail": human_bytes(disk),
        "disk_avail_bytes": disk,
        "status": "running",
        "last_update": dt.datetime.utcnow().isoformat() + "Z",
    }
    write_state(state)

    print(f"[download_orderbook] plan: {expected} files "
          f"({len(hours)}h x {len(markets)}mkt x {len(exchanges)}ex x {len(assets)}asset)")
    print(f"[download_orderbook] existing: spot={spot_now} futures={futures_now} "
          f"size={human_bytes(size_total)} disk_avail={human_bytes(disk)}")

    if args.dry_run:
        for api, local, _fn in plan[:5]:
            print(f"  DRY {api} -> {local}")
        print(f"  ... ({expected-5} more)")
        return 0

    # ---- single-instance lock ----
    with FileLock(LOCK_PATH):
        done = total_now
        ok = 0
        skip = 0
        fail = 0

        for idx, (_ex_dir, api_path, _fn) in enumerate(plan, 1):
            local_path = out_root / api_path
            if local_path.exists() and local_path.stat().st_size >= 1000:
                skip += 1
                done += 1
                continue

            # retry loop
            success = False
            for attempt in range(1, args.max_retries + 1):
                if download_one(api_path, local_path, args.api_key):
                    success = True
                    break
                time.sleep(2 * attempt)
            if success:
                ok += 1
            else:
                fail += 1

            done += 1

            # refresh counts every N downloads (or every 30s on skips)
            if idx % 10 == 0 or idx == expected:
                counts, size_total = scan_output_v2(out_root)
                spot_c = counts["spot"]
                futures_c = counts["futures"]
                disk = disk_avail_bytes(out_root)
                state.update({
                    "total": spot_c + futures_c,
                    "spot": spot_c,
                    "futures": futures_c,
                    "progress": f"{spot_c+futures_c}/{expected} ({100*(spot_c+futures_c)/max(expected,1):.1f}%)",
                    "size": human_bytes(size_total),
                    "size_bytes": size_total,
                    "disk_avail": human_bytes(disk),
                    "disk_avail_bytes": disk,
                    "last_update": dt.datetime.utcnow().isoformat() + "Z",
                })
                write_state(state)
                print(f"  [{idx}/{expected}] ok={ok} skip={skip} fail={fail} "
                      f"spot={spot_c} futures={futures_c} size={human_bytes(size_total)}")

        # final scan
        counts, size_total = scan_output_v2(out_root)
        disk = disk_avail_bytes(out_root)
        state.update({
            "total": counts["spot"] + counts["futures"],
            "spot": counts["spot"],
            "futures": counts["futures"],
            "progress": f"{counts['spot']+counts['futures']}/{expected} "
                       f"({100*(counts['spot']+counts['futures'])/max(expected,1):.1f}%)",
            "size": human_bytes(size_total),
            "size_bytes": size_total,
            "disk_avail": human_bytes(disk),
            "disk_avail_bytes": disk,
            "status": "completed" if fail == 0 else f"completed_with_failures({fail})",
            "last_update": dt.datetime.utcnow().isoformat() + "Z",
        })
        write_state(state)
        print(f"[download_orderbook] DONE ok={ok} skip={skip} fail={fail} "
              f"total={state['total']}/{expected}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
