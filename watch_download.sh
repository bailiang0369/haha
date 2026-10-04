#!/usr/bin/env bash
# watch_download.sh — watchdog for BTC L2 orderbook bulk download
#
# Responsibilities:
#   1. Single-instance lock (flock) so two watchdogs never race.
#   2. Detect whether the downloader is alive; if dead and it was our turn twice, restart it.
#   3. Emit a concise status snapshot into .watch_state.json every run.
#   4. Print a human-readable one-line summary to stdout.
#
# Usage:
#   bash /workspace/watch_download.sh            # run once (intended as cron entry or manual call)
#   bash /workspace/watch_download.sh --loop      # keep running, sleep SLEEP_SEC between ticks
set -u

ROOT="/workspace"
DATA_DIR="$ROOT/data"
STATE_FILE="$ROOT/.watch_state.json"
PID_FILE="$ROOT/.download.pid"
LOG_FILE="$ROOT/.download.log"
LOCK_FILE="$ROOT/.watch.lock"
DOWNLOAD_PY="$ROOT/download_orderbook.py"
KEY_FILE="$ROOT/.cryptohft_key"

# Config (BTCUSDT, spot + futures, 2026-09-04 .. 2026-10-04)
START_DATE="2026-09-04"
END_DATE="2026-10-04"
ASSETS="BTC"
EXCHANGES="binance"
MARKET="both"

# Two consecutive "dead" runs before we restart — avoids transient false negatives.
MAX_MISS_BEFORE_RESTART=2
SLEEP_SEC=120

# --- helpers -----------------------------------------------------------------

now_utc() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

human_size() {
    # bytes → human
    local b=$1
    if   (( b >= 1024**4 )); then awk -v b="$b" 'BEGIN{printf "%.1fT", b/1024**4}'
    elif (( b >= 1024**3 )); then awk -v b="$b" 'BEGIN{printf "%.1fG", b/1024**3}'
    elif (( b >= 1024**2 )); then awk -v b="$b" 'BEGIN{printf "%.1fM", b/1024**2}'
    elif (( b >= 1024    )); then awk -v b="$b" 'BEGIN{printf "%.1fK", b/1024}'
    else echo "${b}B"
    fi
}

count_files() {
    local dir="$1"
    find "$dir" -name "*.parquet" -type f 2>/dev/null | wc -l
}

# --- status collection -------------------------------------------------------

collect_status() {
    local status="running"
    local note=""
    local spot_files=0 futures_files=0 total_files=0

    # Process alive?
    local alive=0
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE" 2>/dev/null)
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            alive=1
        fi
    fi
    # Cross-check: any download_orderbook.py still running?
    if (( alive == 0 )); then
        if pgrep -f "download_orderbook.py.*BTCUSDT" >/dev/null 2>&1; then
            alive=1
        fi
    fi

    # Count spot / futures
    if [[ -d "$DATA_DIR" ]]; then
        spot_files=$(find "$DATA_DIR/binance_spot" -name "*BTCUSDT_orderbook.parquet" -type f 2>/dev/null | wc -l)
        futures_files=$(find "$DATA_DIR/binance_futures" -name "*BTCUSDT_orderbook.parquet" -type f 2>/dev/null | wc -l)
    fi
    total_files=$(( spot_files + futures_files ))

    # Expected: 2 markets × 24h × 31 days (inclusive Sep 4..Oct 4) ≈ 1488
    local expected=1488

    # Disk
    local disk_total disk_used disk_avail disk_pct
    disk_avail=$(df -B1 "$ROOT" | awk 'NR==2 {print $4}')
    disk_total=$(df -B1 "$ROOT" | awk 'NR==2 {print $2}')
    disk_used=$(df -B1 "$ROOT"  | awk 'NR==2 {print $3}')
    disk_pct=$(awk -v u="$disk_used" -v t="$disk_total" 'BEGIN{printf "%.1f", (u*100.0)/t}')

    local size_bytes=0
    if [[ -d "$DATA_DIR" ]]; then
        size_bytes=$(du -sb "$DATA_DIR" 2>/dev/null | awk '{print $1}')
        size_bytes=${size_bytes:-0}
    fi

    local progress_pct=0
    if (( expected > 0 )); then
        progress_pct=$(awk -v c="$total_files" -v e="$expected" 'BEGIN{printf "%.1f", (c*100.0)/e}')
    fi

    # Determine status
    if (( alive == 0 )); then
        # Read miss counter from last state
        local prev_miss=0
        if [[ -f "$STATE_FILE" ]]; then
            prev_miss=$(python3 -c "import json,sys;print(json.load(open('$STATE_FILE')).get('miss_streak',0))" 2>/dev/null || echo 0)
        fi
        local miss=$(( prev_miss + 1 ))

        if (( miss >= MAX_MISS_BEFORE_RESTART )); then
            status="restarting"
            note="downloader was missing $miss ticks — spawning new process"
            restart_downloader  # defined below
            miss=0              # reset streak after spawn
        else
            status="idle"
            note="downloader missing ($miss/$MAX_MISS_BEFORE_RESTART before restart)"
        fi
    else
        status="running"
        note="downloader alive (PID $(cat "$PID_FILE" 2>/dev/null || echo "unknown"))"
        miss=0
    fi

    # Write state JSON
    python3 - "$STATE_FILE" <<PYEOF
import json, sys
state = {
    "timestamp": "$(now_utc)",
    "status": "$status",
    "note": "$note",
    "miss_streak": $miss,
    "pid": $(cat "$PID_FILE" 2>/dev/null || echo 0),
    "spot_files": $spot_files,
    "futures_files": $futures_files,
    "total_files": $total_files,
    "expected_files": $expected,
    "progress_pct": float("$progress_pct"),
    "size_bytes": $size_bytes,
    "size": "$(human_size $size_bytes)",
    "disk_avail_bytes": $disk_avail,
    "disk_avail": "$(human_size $disk_avail)",
    "disk_used_bytes": $disk_used,
    "disk_used_pct": float("$disk_pct"),
    "start_date": "$START_DATE",
    "end_date": "$END_DATE",
    "assets": "$ASSETS",
    "exchanges": "$EXCHANGES",
    "market": "$MARKET",
}
with open(sys.argv[1], "w") as f:
    json.dump(state, f, indent=2)
PYEOF

    # Human summary
    local spot_h futures_h total_h expected_h size_h avail_h
    spot_h="$spot_files"
    futures_h="$futures_files"
    total_h="$total_files"
    expected_h="$expected"
    size_h="$(human_size $size_bytes)"
    avail_h="$(human_size $disk_avail)"

    echo "[$(now_utc)]  文件: ${total_h} / ${expected_h} (${progress_pct}%)  Spot: ${spot_h}  Futures: ${futures_h}  Size: ${size_h}"
    echo "                磁盘可用: ${avail_h}  占用: ${disk_pct}%"
    echo "                状态: ${status}  (${note})"
}

# --- restart -----------------------------------------------------------------

restart_downloader() {
    # Single-spawn guard — never start a 2nd downloader.
    if [[ -f "$PID_FILE" ]]; then
        local old_pid
        old_pid=$(cat "$PID_FILE" 2>/dev/null)
        if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
            echo "[$(now_utc)]  已有下载进程 PID=$old_pid 在跑，跳过重启。"
            return 0
        fi
    fi

    if pgrep -f "download_orderbook.py.*BTCUSDT" >/dev/null 2>&1; then
        echo "[$(now_utc)]  pgrep 发现下载进程存活，跳过重启。"
        return 0
    fi

    # Ensure deps
    pip install requests pandas pyarrow cryptohftdata -q 2>/dev/null || true

    # Build key arg
    local key_arg=""
    if [[ -f "$KEY_FILE" ]]; then
        local k
        k=$(cat "$KEY_FILE" 2>/dev/null | tr -d '\n ')
        if [[ -n "$k" ]]; then
            key_arg="--api-key $k"
        fi
    fi

    mkdir -p "$DATA_DIR"

    # setsid detaches the process group from the controlling terminal / sandbox shell,
    # so it survives the watcher's own process exit. Plain nohup + & + disown is NOT
    # sufficient in this environment — the child gets reaped within seconds.
    echo "[$(now_utc)]  启动下载进程 (setsid → $LOG_FILE) ..."
    setsid bash -c "python3 '$DOWNLOAD_PY' \
        --start '$START_DATE' --end '$END_DATE' \
        --assets '$ASSETS' --market '$MARKET' --exchanges '$EXCHANGES' \
        --output '$DATA_DIR' \
        $key_arg \
        >> '$LOG_FILE' 2>&1" </dev/null >/dev/null 2>&1 &

    local new_pid=$!
    echo "$new_pid" > "$PID_FILE"
    sleep 1
    echo "[$(now_utc)]  新 PID=$new_pid"
}

# --- main --------------------------------------------------------------------

tick() {
    collect_status
}

loop_mode=0
for arg in "$@"; do [[ "$arg" == "--loop" ]] && loop_mode=1; done

(
    # Single-instance lock — shared across manual and cron invocations.
    if ! flock -n 9; then
        echo "[$(now_utc)]  另一个 watch_download.sh 正在运行 (flock $LOCK_FILE)，退出。"
        exit 0
    fi

    # If downloader has never been started yet, start it once before reporting.
    if [[ ! -f "$PID_FILE" ]] || ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        if ! pgrep -f "download_orderbook.py.*BTCUSDT" >/dev/null 2>&1; then
            echo "[$(now_utc)]  首次运行：初始化下载进程。"
            restart_downloader
        fi
    fi

    tick
    if (( loop_mode )); then
        while sleep "$SLEEP_SEC"; do tick; done
    fi
) 9>"$LOCK_FILE"
