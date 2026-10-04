#!/usr/bin/env bash
# BTC L2 orderbook watchdog.
#
# Responsibilities (single instance per workspace):
#   1. Report the current download state.
#   2. If download_orderbook.py is not running but the task is not yet done,
#      start it (only one instance).
#   3. Keep a per-run log in /workspace/watch_download.log.
#
# Trigger this from cron or manually: bash /workspace/watch_download.sh

set -euo pipefail

WS=/workspace
LOG="$WS/watch_download.log"
STATE="$WS/.watch_state.json"
WATCH_PID="$WS/.watch_download.pid"
DOWN_PID="$WS/.download_orderbook.pid"
KEY_FILE="$WS/.cryptohft_key"

# --- command used to launch the downloader. Keep in sync with the task spec. ---
DOWN_CMD=(python3 "$WS/download_orderbook.py"
          --start     2026-09-04
          --end       2026-10-04
          --assets    BTC
          --market    both
          --exchanges binance
          --output    "$WS/data"
          --api-key   "7b01d0d6c45dbf65f025ad40ec8aafb1f0540d64c9892c1f386cf3ebca47c9f8")

log() {
    local msg="[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*"
    echo "$msg" | tee -a "$LOG"
}

# ---------------------------------------------------------------------------
# 1) Single-instance guard for the watchdog itself
# ---------------------------------------------------------------------------
if [ -f "$WATCH_PID" ]; then
    old_pid=$(cat "$WATCH_PID" 2>/dev/null || true)
    if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
        log "[watch] Another watch_download.sh is running (pid=$old_pid). Exit."
        exit 0
    fi
fi
echo $$ > "$WATCH_PID"
trap 'rm -f "$WATCH_PID"' EXIT

# ---------------------------------------------------------------------------
# 2) Sanity: pip deps + API key + downloader script
# ---------------------------------------------------------------------------
if ! python3 -c "import requests, pandas, pyarrow" 2>/dev/null; then
    log "[watch] python deps missing, installing ..."
    pip install -q requests pandas pyarrow >> "$LOG" 2>&1 || true
fi
if [ ! -f "$KEY_FILE" ]; then
    echo '7b01d0d6c45dbf65f025ad40ec8aafb1f0540d64c9892c1f386cf3ebca47c9f8' \
        > "$KEY_FILE"
    chmod 600 "$KEY_FILE"
    log "[watch] Recreated API key file"
fi
if [ ! -f "$WS/download_orderbook.py" ]; then
    log "[watch] download_orderbook.py missing -- git restore attempt"
    git -C "$WS" checkout download_orderbook.py >> "$LOG" 2>&1 || true
fi

# ---------------------------------------------------------------------------
# 3) Detect any live download_orderbook.py process (ps + our PID file)
# ---------------------------------------------------------------------------
ps_count=$(ps -eo comm,args | awk '$1=="python3" && /download_orderbook.py/ {print}' | wc -l || true)

running_pid=""
if [ -f "$DOWN_PID" ]; then
    candidate=$(cat "$DOWN_PID" 2>/dev/null || true)
    if [ -n "$candidate" ] && kill -0 "$candidate" 2>/dev/null; then
        running_pid="$candidate"
    fi
fi

log "[watch] downloader_processes=$ps_count registered_pid=$running_pid"

# ---------------------------------------------------------------------------
# 4) Start downloader if needed
# ---------------------------------------------------------------------------
need_start=0
if [ "$ps_count" -eq 0 ] && [ -z "$running_pid" ]; then
    # Only skip starting if the state file says we are fully done.
    if [ -f "$STATE" ]; then
        total=$(python3 -c "import json;print(json.load(open('$STATE')).get('total',0))" 2>/dev/null || echo 0)
        done=$(python3  -c "import json;print(json.load(open('$STATE')).get('done',0))"   2>/dev/null || echo 0)
        status=$(python3 -c "import json;print(json.load(open('$STATE')).get('status',''))" 2>/dev/null || echo "")
        if [ "$status" = "done" ] || { [ "$total" -gt 0 ] && [ "$done" -ge "$total" ]; }; then
            log "[watch] Task already complete (total=$total done=$done). Nothing to do."
        else
            need_start=1
        fi
    else
        need_start=1
    fi
fi

if [ "$need_start" -eq 1 ]; then
    log "[watch] Starting downloader (python3 $WS/download_orderbook.py ...)"
    nohup "${DOWN_CMD[@]}" >> "$LOG" 2>&1 &
    new_pid=$!
    echo "$new_pid" > "$DOWN_PID"
    log "[watch] Downloader pid=$new_pid launched"
    sleep 3   # give it a moment to start writing state
fi

# ---------------------------------------------------------------------------
# 5) Read & report state (one-line JSON for easy automation, plus human table)
# ---------------------------------------------------------------------------
report_state() {
    local title="$1"
    if [ ! -f "$STATE" ]; then
        log "[watch][$title] no .watch_state.json yet"
        return
    fi
    python3 - "$STATE" "$title" << 'PY' | tee -a "$LOG"
import json, sys
s = json.load(open(sys.argv[1]))
t = sys.argv[2]
print(f"\n=== watch_state [{t}] ===")
for k in ("pid","total","spot","futures","expected","progress",
          "done","skipped","failed","total_size_mb","disk_avail_gb","status"):
    print(f"  {k:14s} = {s.get(k,'-')}")
print(f"  updated_at     = {s.get('updated_at','-')}")
if s.get("errors"):
    print(f"  errors({len(s['errors'])}) :")
    for e in s["errors"][-5:]:
        print(f"    - {e}")
print(json.dumps({
    "title": t,
    "total":    s.get("total",0),
    "spot":     s.get("spot",0),
    "futures":  s.get("futures",0),
    "expected": s.get("expected",0),
    "progress": s.get("progress",0),
    "size_mb":  s.get("total_size_mb",0),
    "disk_gb":  s.get("disk_avail_gb",0),
    "status":   s.get("status","?"),
    "done":     s.get("done",0),
    "failed":   s.get("failed",0),
}, ensure_ascii=False))
PY
}

report_state "watch_run"

log "[watch] Exit cleanly."
