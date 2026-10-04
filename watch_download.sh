#!/usr/bin/env bash
# watch_download.sh -- single-entry cron guard for the BTC L2 download pipeline.
#
# The persistent supervisor is watch_daemon.py, which is spawned via
# Python subprocess.Popen(start_new_session=True, close_fds=True). That is the
# only way to escape TRAE sandbox shell-session teardown; plain `nohup`/`disown`
# get SIGKILLed when the blocking RunCommand returns.
#
# Responsibilities:
#   1. File / API-key preflight; abort with status=error JSON if broken.
#   2. Detect whether a *python* watch_daemon.py is already alive (must match
#      both the `python` argv[0] AND contain `watch_daemon.py` in cmdline so
#      we don't self-match the bash script that runs this check).
#   3. If none alive, spawn watch_daemon.py detached via a tiny python helper
#      one-liner that Popen(start_new_session=True).
#   4. Read the .watch_state.json written by the watchdog so the cron caller
#      gets a compact snapshot.
#
# Exit:
#   0   healthy (daemon is, or was just started; state exists)
#   2   misconfiguration (missing file / api key / disk)
set -euo pipefail

WORKDIR=/workspace
DAEMON="$WORKDIR/watch_daemon.py"
STATE="$WORKDIR/.watch_state.json"
DATA="$WORKDIR/data"
API_KEY_FILE="$WORKDIR/.cryptohft_key"

mkdir -p "$DATA"

# ---- preflight -------------------------------------------------------------
for f in "$DAEMON" "$WORKDIR/download_orderbook.py" "$API_KEY_FILE"; do
  if [[ ! -f "$f" ]]; then
    echo "{\"status\":\"error\",\"msg\":\"missing $f\"}" > "$STATE" 2>/dev/null || true
    exit 2
  fi
done

DISK_AVAIL_KB=$(df -k "$WORKDIR" | awk 'NR==2 {print $4}')
if [[ "$DISK_AVAIL_KB" -lt $((1024*1024)) ]]; then
  echo "{\"status\":\"disk_low\",\"avail_kb\":$DISK_AVAIL_KB}" > "$STATE" 2>/dev/null || true
  exit 2
fi

# ---- is a python watch_daemon.py already alive? ----------------------------
# Loop /proc numerically; require cmdline starts with a python interpreter
# AND contains `watch_daemon.py` so we don't match ourselves or the zsh
# parent that hosts this script (whose cmdline includes the bash body).
WATCHDOG_PID=""
for pid_dir in /proc/[0-9]*; do
  pid="${pid_dir##*/}"
  argv0=$(readlink "$pid_dir/exe" 2>/dev/null || true)
  case "$argv0" in
    *python*) ;;
    *) continue ;;
  esac
  cmdline=$(tr '\0' ' ' < "$pid_dir/cmdline" 2>/dev/null || true)
  if echo "$cmdline" | grep -q "watch_daemon.py"; then
    WATCHDOG_PID="$pid"
    break
  fi
done

if [[ -n "$WATCHDOG_PID" ]]; then
  echo "[watch_download.sh] watch_daemon.py already alive (pid=$WATCHDOG_PID); reading state"
  exit 0
fi

# ---- spawn daemon detached via Python Popen -------------------------------
echo "[watch_download.sh] spawning watch_daemon.py detached"
DAEMON_PID=$(python3 -c "
import subprocess, sys
p = subprocess.Popen(
    [sys.executable, '$DAEMON'],
    stdout=open('$DATA/.daemon.log', 'ab', buffering=0),
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    start_new_session=True,
    close_fds=True,
)
print(p.pid, flush=True)
")
echo "[watch_download.sh] spawned watch_daemon.py pid=$DAEMON_PID"

# Wait a tick so the daemon writes its first state before the caller reads.
sleep 2
exit 0
