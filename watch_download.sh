#!/usr/bin/env bash
# ----------------------------------------------------------------------
# watch_download.sh  -- watchdog for BTC L2 orderbook bulk download
#
# Each invocation:
#   1. Ensures the sandbox environment is restored (pip deps, API key).
#   2. Ensures download_orderbook.py exists (git-tracked recovery).
#   3. Detects a running download_orderbook.py; never starts a second one.
#   4. If nothing is running, launches download_orderbook.py in the
#      background with nohup and writes the PID to a marker file.
#   5. Reads /workspace/.watch_state.json and prints a human summary.
#
# Idempotent.  Safe to call from cron every N minutes.
# ----------------------------------------------------------------------
set -euo pipefail

WORKDIR="/workspace"
LOGDIR="$WORKDIR/logs"
LOGFILE="$LOGDIR/watch_download.log"
STATE="$WORKDIR/.watch_state.json"
PIDFILE="$WORKDIR/.download_orderbook.pid"
WATCH_PIDFILE="$WORKDIR/.watch_download.pid"

# ---- config mirrors the scheduled task's download command -------------
START="2026-09-04"
END="2026-10-04"
ASSETS="BTC"
MARKET="both"
EXCHANGES="binance"
OUTPUT="$WORKSPACE/data"
API_KEY_FILE="$WORKSPACE/.cryptohft_key"

echo "========================================="
echo "watch_download.sh @ $(date -u '+%Y-%m-%d %H:%M:%SZ')"
echo "========================================="

# ---- 0. disk -----------------------------------------------------------
DISK_AVAIL=$(df -h "$WORKDIR" | awk 'NR==2{print $4}')
echo "[disk] avail=$DISK_AVAIL"

# ---- 1. env restore ---------------------------------------------------
ensure_pip() {
    python3 -c "import requests, pandas, pyarrow, cryptohftdata" 2>/dev/null && return 0
    echo "[pip] installing requests pandas pyarrow cryptohftdata ..."
    pip install requests pandas pyarrow cryptohftdata -q 2>&1 | tail -3
}

ensure_api_key() {
    if [ -s "$API_KEY_FILE" ]; then return 0; fi
    echo "[key] restoring API key ..."
    echo '7b01d0d6c45dbf65f025ad40ec8aafb1f0540d64c9892c1f386cf3ebca47c9f8' > "$API_KEY_FILE"
    chmod 600 "$API_KEY_FILE"
}

ensure_scripts() {
    # download_orderbook.py and this script must be git tracked for the
    # sandbox-reset recovery to work.  If they were never committed, we
    # still try git checkout and only proceed if they exist.
    if [ ! -f "$WORKDIR/download_orderbook.py" ]; then
        echo "[git] checkout download_orderbook.py ..."
        git -C "$WORKDIR" checkout download_orderbook.py 2>&1 || true
    fi
    if [ ! -f "$WORKDIR/watch_download.sh" ]; then
        echo "[git] checkout watch_download.sh ..."
        git -C "$WORKDIR" checkout watch_download.sh 2>&1 || true
    fi
    chmod +x "$WORKDIR/watch_download.sh" 2>/dev/null || true
}

ensure_pip
ensure_api_key
ensure_scripts

if [ ! -f "$WORKDIR/download_orderbook.py" ]; then
    echo "[FATAL] download_orderbook.py missing and cannot be recovered from git."
    echo "        This means the script was never committed."
    exit 1
fi

mkdir -p "$LOGDIR"
mkdir -p "$OUTPUT"

# ---- 2. single-downloader check ---------------------------------------
is_downloader_running() {
    pgrep -f "python3 $WORKDIR/download_orderbook.py" >/dev/null 2>&1 || \
    pgrep -f "python.*download_orderbook\.py" >/dev/null 2>&1
}

if is_downloader_running; then
    DL_PID=$(pgrep -f "python.*download_orderbook\.py" | head -1)
    echo "[lock] download_orderbook.py already running (pid=$DL_PID)."
else
    echo "[lock] no download_orderbook.py running -> launching."
    API_KEY=$(cat "$API_KEY_FILE")
    nohup python3 "$WORKDIR/download_orderbook.py" \
        --start "$START" \
        --end "$END" \
        --assets "$ASSETS" \
        --market "$MARKET" \
        --exchanges "$EXCHANGES" \
        --output "$OUTPUT" \
        --api-key "$API_KEY" \
        >> "$LOGDIR/download_orderbook.log" 2>&1 &
    DL_PID=$!
    echo "$DL_PID" > "$PIDFILE"
    echo "[launch] pid=$DL_PID  log=$LOGDIR/download_orderbook.log"
    sleep 3
    if ! kill -0 "$DL_PID" 2>/dev/null; then
        echo "[WARN] download_orderbook.py died immediately; check $LOGDIR/download_orderbook.log"
    fi
fi

# ---- 3. single-watchdog lock -----------------------------------------
exec 9>"$WATCH_PIDFILE"
if ! flock -n 9; then
    echo "[lock] another watch_download.sh running; exit."
    exit 0
fi
echo "$$" > "$WATCH_PIDFILE"

# ---- 4. state report --------------------------------------------------
if [ -f "$STATE" ]; then
    echo "--- .watch_state.json ---"
    cat "$STATE"
    echo "-------------------------"
else
    echo "[state] .watch_state.json not present yet; waiting for first write."
fi

echo "[done] $(date -u '+%Y-%m-%d %H:%M:%SZ')"
