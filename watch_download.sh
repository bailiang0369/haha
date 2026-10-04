#!/usr/bin/env bash
# watch_download.sh - watchdog for BTC L2 orderbook bulk downloader.
#
# Responsibilities:
#   1. Ensure at most ONE download_orderbook.py process is running.
#   2. If no downloader is running AND expected > total, launch it via setsid.
#   3. If downloader crashed mid-flight, restart it (resumable).
#   4. Dump a human-readable status summary to stdout on every run.
#
# Designed to be called by a scheduler / cron repeatedly.
# Uses setsid so child survives sandbox shell exit (PPID becomes 1).

set -u

WORKSPACE="/workspace"
STATE_FILE="${WORKSPACE}/.watch_state.json"
LOG_FILE="${WORKSPACE}/.watch_log"
PID_FILE="${WORKSPACE}/.watch_download.pid"

API_KEY="7b01d0d6c45dbf65f025ad40ec8aafb1f0540d64c9892c1f386cf3ebca47c9f8"
START_DATE="2026-09-04"
END_DATE="2026-10-04"
ASSETS="BTC"
MARKET="both"
EXCHANGES="binance"
OUTPUT="${WORKSPACE}/data"

# --- PID lock to prevent two watchdogs from running at once ---
exec 9>"$PID_FILE"
if ! flock -n 9; then
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] another watch_download.sh is running, exiting"
    exit 0
fi

log() { echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$LOG_FILE"; }

# --- Fresh env check (sandbox reset recovery) ---
if ! python3 -c "import requests, pandas, pyarrow" 2>/dev/null; then
    log "pip packages missing, installing..."
    pip install requests pandas pyarrow -q 2>&1 | tail -3 | tee -a "$LOG_FILE"
fi

if [ ! -f "${WORKSPACE}/.cryptohft_key" ]; then
    log "API key missing, writing..."
    echo "$API_KEY" > "${WORKSPACE}/.cryptohft_key"
    chmod 600 "${WORKSPACE}/.cryptohft_key"
fi

# --- Script recovery ---
if [ ! -f "${WORKSPACE}/download_orderbook.py" ]; then
    log "ERROR: download_orderbook.py missing - sandbox reset without git restore!"
    exit 1
fi

# --- Process detection (count python3 downloader processes) ---
DL_COUNT=$(pgrep -fc 'python3 download_orderbook.py' 2>/dev/null || echo 0)

# Read state
TOTAL=0; EXPECTED=0; SPOT=0; FUTURES=0; PROGRESS="0%"
SIZE=0; STATUS="no state yet"; DISK=0
if [ -f "$STATE_FILE" ]; then
    TOTAL=$(python3 -c "import json;print(json.load(open('$STATE_FILE')).get('total',0))" 2>/dev/null || echo 0)
    EXPECTED=$(python3 -c "import json;print(json.load(open('$STATE_FILE')).get('expected',0))" 2>/dev/null || echo 0)
    SPOT=$(python3 -c "import json;print(json.load(open('$STATE_FILE')).get('spot',0))" 2>/dev/null || echo 0)
    FUTURES=$(python3 -c "import json;print(json.load(open('$STATE_FILE')).get('futures',0))" 2>/dev/null || echo 0)
    PROGRESS=$(python3 -c "import json;print(json.load(open('$STATE_FILE')).get('progress','0%'))" 2>/dev/null || echo "0%")
    SIZE=$(python3 -c "import json;print(json.load(open('$STATE_FILE')).get('size_mb',0))" 2>/dev/null || echo 0)
    STATUS=$(python3 -c "import json;print(json.load(open('$STATE_FILE')).get('status','unknown'))" 2>/dev/null || echo "unknown")
    DISK=$(python3 -c "import json;print(json.load(open('$STATE_FILE')).get('disk_avail_mb',0))" 2>/dev/null || echo 0)
fi

REAL_DISK=$(df -m "$WORKSPACE" 2>/dev/null | awk 'NR==2 {print $4}')
[ -z "$REAL_DISK" ] && REAL_DISK="?"

log "=== STATUS ==="
log "  downloader procs:   $DL_COUNT"
log "  total/expected:     ${TOTAL}/${EXPECTED}"
log "  spot/futures:       ${SPOT}/${FUTURES}"
log "  progress:           ${PROGRESS}"
log "  size_mb:            ${SIZE}"
log "  status:             ${STATUS}"
log "  disk_avail_mb(state): ${DISK}"
log "  disk_avail_mb(df):  ${REAL_DISK}"

# --- Decision ---
if [ "$DL_COUNT" -ge 1 ]; then
    log "download_orderbook.py already running ($DL_COUNT proc), nothing to do"
    exit 0
fi

if [ "$STATUS" = "complete" ] && [ "$TOTAL" -ge "$EXPECTED" ]; then
    log "ALL DONE: $TOTAL/$EXPECTED files, $SIZE MB"
    exit 0
fi

# Launch via setsid so PPID becomes 1 (survives sandbox shell exit)
log "starting download_orderbook.py with setsid (concurrency=4)..."
setsid python3 "${WORKSPACE}/download_orderbook.py" \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --assets "$ASSETS" \
    --market "$MARKET" \
    --exchanges "$EXCHANGES" \
    --output "$OUTPUT" \
    --api-key "$API_KEY" \
    --concurrency 4 \
    --request-gap 0.2 \
    > "${WORKSPACE}/.dl_stdout.log" \
    2> "${WORKSPACE}/.dl_stderr.log" </dev/null &
DL_PID=$!
log "launched pid=$DL_PID"
disown "$DL_PID" 2>/dev/null || true
sleep 4
if kill -0 "$DL_PID" 2>/dev/null; then
    PPID=$(ps -o ppid= -p "$DL_PID" 2>/dev/null | tr -d ' ')
    log "pid $DL_PID alive (ppid=$PPID)"
else
    log "WARNING: pid $DL_PID died!"
    tail -20 "${WORKSPACE}/.dl_stderr.log" | tee -a "$LOG_FILE"
fi
