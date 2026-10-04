#!/usr/bin/env bash
# BTC orderbook download watchdog
# - ensures a single instance of download_orderbook.py runs
# - emits /workspace/.watch_state.json snapshots
# - safe to invoke repeatedly (no concurrency)

set -u

PIDFILE=/workspace/.download_orderbook.pid
STATE=/workspace/.watch_state.json
LOG=/workspace/watch_download.log
API_KEY_FILE=/workspace/.cryptohft_key
DL_SCRIPT=/workspace/download_orderbook.py

START_DATE="2026-09-04"
END_DATE="2026-10-04"
ASSETS="BTC"
MARKET="both"
EXCHANGES="binance"
OUTPUT="/workspace/data"

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"
}

# ---- env recover ----
ensure_env() {
  # pip deps
  python3 -c "import requests, pandas, pyarrow" 2>/dev/null || {
    log "pip deps missing, installing..."
    pip install requests pandas pyarrow -q 2>&1 | tail -1
  }
  # api key
  if [ ! -s "$API_KEY_FILE" ]; then
    log "API key file missing, writing default..."
    echo '7b01d0d6c45dbf65f025ad40ec8aafb1f0540d64c9892c1f386cf3ebca47c9f8' > "$API_KEY_FILE"
    chmod 600 "$API_KEY_FILE"
  fi
  # download script
  if [ ! -f "$DL_SCRIPT" ]; then
    log "download_orderbook.py missing — cannot recover (was untracked), aborting."
    exit 2
  fi
}

# ---- lock (single instance) ----
already_running() {
  if [ -f "$PIDFILE" ]; then
    local pid
    pid=$(cat "$PIDFILE" 2>/dev/null)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
    # stale pidfile
    rm -f "$PIDFILE"
  fi
  # also check by ps (defense in depth)
  if ps aux | grep '[d]ownload_orderbook.py' | grep -v grep >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

start_downloader() {
  local api_key pid
  api_key=$(cat "$API_KEY_FILE")
  log "START download_orderbook.py (start=$START_DATE end=$END_DATE assets=$ASSETS market=$MARKET)"
  setsid python3 "$DL_SCRIPT" \
    --start "$START_DATE" --end "$END_DATE" \
    --assets "$ASSETS" --market "$MARKET" --exchanges "$EXCHANGES" \
    --output "$OUTPUT" --api-key "$api_key" \
    </dev/null >>/workspace/download_orderbook.log 2>&1 &
  pid=$!
  disown "$pid" 2>/dev/null || true
  echo "$pid" > "$PIDFILE"
  sleep 3
  if ! kill -0 "$pid" 2>/dev/null; then
    log "ERROR downloader died immediately (setsid)"
    rm -f "$PIDFILE"
    return 1
  fi
  log "downloader PID=$pid (setsid+disown)"
}

# ---- main ----
ensure_env

if already_running; then
  pid=$(cat "$PIDFILE" 2>/dev/null || ps aux | grep '[d]ownload_orderbook.py' | awk '{print $2}' | head -1)
  log "download_orderbook.py already running PID=$pid — skip start"
else
  start_downloader
fi

# emit state summary one-shot (the python process keeps updating STATE itself)
if [ -f "$STATE" ]; then
  echo "--- .watch_state.json ---"
  cat "$STATE"
  echo ""
  total=$(python3 -c "import json; print(json.load(open('$STATE')).get('expected_total', '?'))" 2>/dev/null)
  downloaded=$(python3 -c "import json; print(json.load(open('$STATE')).get('downloaded', '?'))" 2>/dev/null)
  progress=$(python3 -c "import json; print(json.load(open('$STATE')).get('progress', '?'))" 2>/dev/null)
  log "STATE total=$total downloaded=$downloaded progress=$progress%"
else
  log "STATE not yet written — waiting for first PROGRESS tick..."
fi
