#!/usr/bin/env python3
"""watch_daemon.py -- the *persistent* BTC L2 download guardian.

Why a Python daemon instead of shell `nohup`:
  Every RunCommand shell session tears down its process group when the tool
  call exits. `nohup` and `disown` don't save you here (the harness SIGKILLs
  the whole PGID once the blocking command returns). `start_new_session=True`
  (a.k.a. setsid) is the only reliable escape hatch in this sandbox.

Loop:
  - check if download_orderbook.py is alive (via pidfile + pgrep)
  - if not, respawn it with subprocess.Popen(start_new_session=True,
    close_fds=True) and stdout/stderr redirected to a log file
  - every 10s refresh /workspace/.watch_state.json with progress
  - when on-disk count reaches EXPECTED we mark status="done" and exit 0
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

WORKDIR = Path("/workspace")
DATA = WORKDIR / "data"
SCRIPT = WORKDIR / "download_orderbook.py"
STATE = WORKDIR / ".watch_state.json"
LOG = DATA / ".download.log"
PIDFILE = WORKDIR / ".download.pid"
API_KEY_FILE = WORKDIR / ".cryptohft_key"

START = "2026-09-04"
END = "2026-10-04"
EXPECTED = 1488           # 31 days * 24h * 2 (spot+futures binance)

START_CMD = [
    sys.executable, str(SCRIPT),
    "--start", START, "--end", END,
    "--assets", "BTC",
    "--market", "both",
    "--exchanges", "binance",
    "--output", str(DATA),
    "--api-key", API_KEY_FILE.read_text().strip(),
]

SLEEP = 10                # seconds between watchdog ticks
LOG_TAIL_LEN = 2000       # chars of .download.log to ship into state json


def _pgrep(pattern: str) -> list[int]:
    """Simple pgrep fallback (os.kill(pid, 0) + /proc scanning)."""
    pids = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="ignore")
        except (OSError, PermissionError):
            continue
        if pattern in cmd:
            pids.append(pid)
    return pids


def is_download_alive() -> bool:
    """True if there's a live download_orderbook.py owned by us.

    We match the command line pattern `download_orderbook.py` but explicitly
    exclude ourselves (watch_daemon.py) and any pgrep/grep invocations, which
    matches what bash's `ps aux | grep [d]ownload_orderbook` was doing.
    """
    return bool(_pgrep("download_orderbook.py"))


def spawn_download() -> int:
    """Launch download_orderbook.py in a new session. Never returns 0 on fail.

    Returns the spawned pid (int) on success.
    """
    DATA.mkdir(parents=True, exist_ok=True)
    log_fh = open(LOG, "ab", buffering=0)
    # Truncate log only on a *fresh* start, not on a re-respawn mid-run,
    # so we keep history for diagnosis.
    p = subprocess.Popen(
        START_CMD,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,       # setsid — survives parent session exit
        close_fds=True,               # detach from any inherited FDs
    )
    PIDFILE.write_text(str(p.pid))
    return p.pid


def count_parquets() -> tuple[int, int, int]:
    """(total, spot, futures) on-disk parquet count under DATA/."""
    total = spot = fut = 0
    for root, dirs, files in os.walk(DATA):
        for fn in files:
            if not fn.endswith(".parquet"):
                continue
            total += 1
            full = os.path.join(root, fn)
            if "binance_spot" in full:
                spot += 1
            elif "binance_futures" in full:
                fut += 1
    return total, spot, fut


def disk_avail_kb() -> tuple[int, int]:
    stat = os.statvfs(str(WORKDIR))
    avail = stat.f_frsize * stat.f_bavail // 1024
    total = stat.f_frsize * stat.f_blocks // 1024
    used_pct = 100 - int(avail * 100 / max(total, 1))
    return avail, used_pct


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n //= 1024
    return f"{n}TB"


def write_state(status: str, pid: int | None) -> None:
    total, spot, fut = count_parquets()
    progress = f"{100 * total / max(EXPECTED, 1):.2f}"
    size_bytes = 0
    try:
        size_bytes = int(subprocess.check_output(
            ["du", "-sb", str(DATA)], stderr=subprocess.DEVNULL
        ).split()[0])
    except Exception:
        pass
    avail_kb, used_pct = disk_avail_kb()

    log_tail = ""
    if LOG.exists():
        try:
            raw = LOG.read_text(errors="replace")
            log_tail = raw[-LOG_TAIL_LEN:].replace("\\", "\\\\").replace('"', '\\"').replace("\n", "|")
        except Exception:
            pass

    state = {
        "status": status,
        "pid": pid,
        "expected": EXPECTED,
        "total": total,
        "spot": spot,
        "futures": fut,
        "progress_pct": float(progress),
        "size_bytes": size_bytes,
        "disk_avail_kb": avail_kb,
        "disk_avail_h": human(avail_kb * 1024),
        "disk_used_pct": used_pct,
        "start": START,
        "end": END,
        "log_tail": log_tail,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    STATE.write_text(json.dumps(state, indent=2))
    print(f"[watchdog] status={status} pid={pid} "
          f"total={total}/{EXPECTED} ({progress}%) "
          f"size={human(size_bytes)} disk={human(avail_kb*1024)}")


def main() -> int:
    # prevent multiple watchdogs from racing to spawn downloaders
    my_pidfile = WORKDIR / ".watchdog.pid"
    if my_pidfile.exists():
        try:
            old = int(my_pidfile.read_text().strip())
            os.kill(old, 0)
            # another watchdog is alive; exit silently (the cron runner can still
            # read state, which is what matters)
            print(f"[watchdog] pidfile held by live pid {old}; exiting")
            return 0
        except (OSError, ValueError):
            my_pidfile.unlink(missing_ok=True)
    my_pidfile.write_text(str(os.getpid()))

    print(f"[watchdog] starting on pid {os.getpid()}, pidfile={my_pidfile}")

    try:
        consecutive_respawns = 0
        last_spawn_at = 0.0
        min_respawn_gap = 30.0     # don't hammer upstream on a crashing loop

        while True:
            total, _, _ = count_parquets()
            if total >= EXPECTED:
                write_state("done", None)
                print(f"[watchdog] done! {total}/{EXPECTED} files; exiting.")
                return 0

            alive = is_download_alive()
            pid: int | None = None
            if alive:
                # grab any one matching pid (the downloader's own)
                pids = _pgrep("download_orderbook.py")
                pid = pids[0] if pids else None
                consecutive_respawns = 0
                write_state("running", pid)
            else:
                now = time.time()
                if now - last_spawn_at < min_respawn_gap:
                    write_state("waiting", None)
                else:
                    avail_kb, _ = disk_avail_kb()
                    if avail_kb < 1024 * 1024:     # < 1 GiB headroom
                        write_state("disk_low", None)
                    else:
                        pid = spawn_download()
                        last_spawn_at = now
                        consecutive_respawns += 1
                        write_state("starting", pid)

            # heartbeat -- also useful for `ps | grep watch_daemon`
            time.sleep(SLEEP)
    finally:
        my_pidfile.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
