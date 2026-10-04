#!/usr/bin/env python3
"""CryptoHFT orderbook parquet downloader — hourly, parallel, resumable."""
import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timedelta, date
from pathlib import Path

import requests

API = "https://api.cryptohftdata.com/v1/download"
STATE_PATH = "/workspace/.watch_state.json"
LOG_PATH = "/workspace/download_orderbook.log"
RATE_LIMIT = 0.25  # seconds between requests to stay under 60/min

# ----- helpers -----

def log(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def load_state() -> dict:
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: dict) -> None:
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_PATH)


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def disk_avail(path: str) -> int:
    try:
        s = os.statvfs(path)
        return s.f_bavail * s.f_frsize
    except Exception:
        return -1


# ----- core download -----

def build_jobs(args):
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()

    markets = {"spot": ["spot"], "futures": ["futures"], "both": ["spot", "futures"]}[args.market]
    exchanges = [e.strip() for e in args.exchanges.split(",")]
    assets = [a.strip().upper() for a in args.assets.split(",")]

    jobs = []
    for d in daterange(start, end):
        for h in range(24):
            for ex in exchanges:
                for m in markets:
                    for a in assets:
                        fname = f"{ex}_{m}/{d.isoformat()}/{h:02d}/{a}USDT_orderbook.parquet"
                        local = Path(args.output) / fname
                        jobs.append({
                            "api_file": fname,
                            "local": local,
                            "market": m,
                            "exchange": ex,
                            "asset": a,
                            "date": d.isoformat(),
                            "hour": h,
                        })
    return jobs


def download_one(job: dict, api_key: str, retries: int = 3) -> tuple[str, int]:
    url = f"{API}?file={job['api_file']}"
    if api_key:
        url += f"&api_key={api_key}"

    job["local"].parent.mkdir(parents=True, exist_ok=True)
    tmp_path = job["local"].with_suffix(".parquet.tmp")

    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60, allow_redirects=True) as r:
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}"
                    if r.status_code in (404, 451):
                        # non-retriable
                        return ("missing", 0)
                    time.sleep(min(2 ** attempt, 10))
                    continue
                total = 0
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        total += len(chunk)
                if total < 1024:
                    last_err = "file too small"
                    continue
                os.replace(tmp_path, job["local"])
                return ("ok", total)
        except Exception as e:
            last_err = str(e)
            time.sleep(min(2 ** attempt, 10))
    log(f"  FAIL {job['api_file']}: {last_err}")
    return ("fail", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--assets", default="BTC")
    ap.add_argument("--market", choices=["spot", "futures", "both"], default="both")
    ap.add_argument("--exchanges", default="binance")
    ap.add_argument("--output", default="/workspace/data")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    # API key fallback from file
    if not args.api_key:
        kp = Path("/workspace/.cryptohft_key")
        if kp.exists():
            args.api_key = kp.read_text().strip()

    jobs = build_jobs(args)
    total = len(jobs)

    # count existing
    spot_expected = sum(1 for j in jobs if j["market"] == "spot")
    fut_expected = sum(1 for j in jobs if j["market"] == "futures")

    existing = sum(1 for j in jobs if j["local"].exists() and j["local"].stat().st_size > 1024)
    downloaded = 0
    spot_ok = 0
    fut_ok = 0
    failed = 0
    missing = 0
    bytes_total = 0

    state = {
        "start": args.start,
        "end": args.end,
        "assets": args.assets,
        "market": args.market,
        "exchanges": args.exchanges,
        "expected_total": total,
        "expected_spot": spot_expected,
        "expected_futures": fut_expected,
        "spot": 0,
        "futures": 0,
        "downloaded": 0,
        "failed": 0,
        "missing": 0,
        "bytes": 0,
        "progress": 0.0,
        "status": "running",
        "disk_avail_bytes": disk_avail(args.output),
        "disk_avail": human_bytes(disk_avail(args.output)),
        "last_update": datetime.utcnow().isoformat() + "Z",
    }
    save_state(state)
    log(f"START total={total} spot={spot_expected} fut={fut_expected} existing={existing}")

    try:
        for i, job in enumerate(jobs, 1):
            if job["local"].exists() and job["local"].stat().st_size > 1024:
                downloaded += 1
                if job["market"] == "spot":
                    spot_ok += 1
                else:
                    fut_ok += 1
                bytes_total += job["local"].stat().st_size
                continue

            result, size = download_one(job, args.api_key)
            if result == "ok":
                downloaded += 1
                bytes_total += size
                if job["market"] == "spot":
                    spot_ok += 1
                else:
                    fut_ok += 1
            elif result == "missing":
                missing += 1
            else:
                failed += 1

            if i % 5 == 0 or i == total:
                state.update({
                    "spot": spot_ok,
                    "futures": fut_ok,
                    "downloaded": downloaded,
                    "failed": failed,
                    "missing": missing,
                    "bytes": bytes_total,
                    "progress": round((downloaded + missing) / total * 100, 2),
                    "disk_avail_bytes": disk_avail(args.output),
                    "disk_avail": human_bytes(disk_avail(args.output)),
                    "last_update": datetime.utcnow().isoformat() + "Z",
                })
                save_state(state)
                log(f"PROGRESS {i}/{total} downloaded={downloaded} spot={spot_ok} fut={fut_ok} failed={failed} size={human_bytes(bytes_total)}")

            time.sleep(RATE_LIMIT)

    except KeyboardInterrupt:
        log("INTERRUPTED by user")
        state["status"] = "interrupted"
    except Exception as e:
        log(f"ERROR: {e}")
        state["status"] = f"error: {e}"
    else:
        state["status"] = "done" if failed == 0 else "done_with_failures"
        log(f"DONE downloaded={downloaded} failed={failed} missing={missing} total={total} size={human_bytes(bytes_total)}")

    state.update({
        "spot": spot_ok,
        "futures": fut_ok,
        "downloaded": downloaded,
        "failed": failed,
        "missing": missing,
        "bytes": bytes_total,
        "progress": round((downloaded + missing) / total * 100, 2),
        "disk_avail_bytes": disk_avail(args.output),
        "disk_avail": human_bytes(disk_avail(args.output)),
        "last_update": datetime.utcnow().isoformat() + "Z",
    })
    save_state(state)


if __name__ == "__main__":
    main()
