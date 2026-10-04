#!/usr/bin/env bash
# watch_download.sh — watchdog for CryptoHFTData bulk download
set -u
ROOT="/workspace"
DATA_DIR="$ROOT/data"
STATE="$ROOT/.watch_state.json"
PIDF="$ROOT/.download.pid"
LOGF="$ROOT/.download.log"
LOCKF="$ROOT/.watch.lock"
DOWNLOAD_PY="$ROOT/download_orderbook.py"
KEYF="$ROOT/.cryptohft_key"

START="2026-09-04"
END="2026-10-04"
ASSETS="BTC"
EXCHANGES="binance"
MARKET="both"
MAX_MISS=2

now() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
hsize() {
    local b=$1
    if   (( b >= 1024**4 )); then awk -v b="$b" 'BEGIN{printf "%.1fT", b/1024**4}'
    elif (( b >= 1024**3 )); then awk -v b="$b" 'BEGIN{printf "%.1fG", b/1024**3}'
    elif (( b >= 1024**2 )); then awk -v b="$b" 'BEGIN{printf "%.1fM", b/1024**2}'
    elif (( b >= 1024    )); then awk -v b="$b" 'BEGIN{printf "%.1fK", b/1024}'
    else echo "${b}B"
    fi
}

# --- restart ---
restart() {
    # Never spawn if one is already alive.
    [[ -f "$PIDF" ]] && { local p; p=$(cat "$PIDF" 2>/dev/null); [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null && { echo "[$(now)] 已有下载 PID=$p，跳过。"; return 0; }; }
    pgrep -f "download_orderbook.py.*${ASSETS}USDT" >/dev/null 2>&1 && { echo "[$(now)] pgrep 发现进程，跳过。"; return 0; }

    pip install requests pandas pyarrow cryptohftdata -q 2>/dev/null || true

    local karg=""
    [[ -f "$KEYF" ]] && { local k; k=$(cat "$KEYF" 2>/dev/null | tr -d '\n '); [[ -n "$k" ]] && karg="--api-key $k"; }

    mkdir -p "$DATA_DIR"
    # setsid = detach from sandbox terminal (nohup & disown gets reaped here)
    setsid bash -c "python3 '$DOWNLOAD_PY' --start '$START' --end '$END' \
        --assets '$ASSETS' --market '$MARKET' --exchanges '$EXCHANGES' \
        --output '$DATA_DIR' $karg >> '$LOGF' 2>&1" </dev/null >/dev/null 2>&1 &
    echo $! > "$PIDF"
    echo "[$(now)] 启动下载 (setsid), PID=$(cat "$PIDF")"
}

# --- collect status & output JSON ---
collect() {
    local alive=0 pid=""
    [[ -f "$PIDF" ]] && { pid=$(cat "$PIDF" 2>/dev/null); [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && alive=1; }
    (( alive == 0 )) && pgrep -f "download_orderbook.py.*${ASSETS}USDT" >/dev/null 2>&1 && alive=1

    local spot=0 fut=0
    [[ -d "$DATA_DIR/binance_spot" ]] && spot=$(find "$DATA_DIR/binance_spot"  -name "*BTCUSDT_orderbook.parquet" -type f 2>/dev/null | wc -l)
    [[ -d "$DATA_DIR/binance_futures" ]] && fut=$(find "$DATA_DIR/binance_futures" -name "*BTCUSDT_orderbook.parquet" -type f 2>/dev/null | wc -l)
    local total=$(( spot + fut ))
    local expected=1488

    local d_avail d_total d_used d_pct
    d_avail=$(df -B1 "$ROOT" | awk 'NR==2{print $4}')
    d_total=$(df -B1 "$ROOT" | awk 'NR==2{print $2}')
    d_used=$(df -B1 "$ROOT"  | awk 'NR==2{print $3}')
    d_pct=$(awk -v u="$d_used" -v t="$d_total" 'BEGIN{printf "%.1f", (u*100.0)/t}')
    local sz=0
    [[ -d "$DATA_DIR" ]] && sz=$(du -sb "$DATA_DIR" 2>/dev/null | awk '{print $1}')
    sz=${sz:-0}
    local pct=0
    (( expected > 0 )) && pct=$(awk -v c="$total" -v e="$expected" 'BEGIN{printf "%.1f", (c*100.0)/e}')

    # Miss streak
    local prev_miss=0
    [[ -f "$STATE" ]] && prev_miss=$(python3 -c "import json;print(json.load(open('$STATE')).get('miss_streak',0))" 2>/dev/null || echo 0)
    local miss=$(( prev_miss + 1 ))

    local status note
    if (( alive == 0 )); then
        if (( miss >= MAX_MISS )); then
            status="restarting"; note="missing ${miss} ticks → spawning"; restart; miss=0
        else
            status="idle";       note="missing (${miss}/${MAX_MISS}) before restart"
        fi
    else
        status="running"; note="alive PID=${pid:-unknown}"; miss=0
    fi

    python3 - "$STATE" <<PYEOF
import json,sys
json.dump({
  "timestamp":"$(now)","status":"$status","note":"$note","miss_streak":$miss,
  "pid":${pid:-0},
  "spot_files":$spot,"futures_files":$fut,"total_files":$total,
  "expected_files":$expected,"progress_pct":float("$pct"),
  "size_bytes":$sz,"size":"$(hsize $sz)",
  "disk_avail_bytes":$d_avail,"disk_avail":"$(hsize $d_avail)",
  "disk_used_bytes":$d_used,"disk_used_pct":float("$d_pct"),
  "start_date":"$START","end_date":"$END","assets":"$ASSETS",
  "exchanges":"$EXCHANGES","market":"$MARKET"
}, open(sys.argv[1],"w"), indent=2)
PYEOF

    echo "[$(now)]  文件: ${total} / ${expected} (${pct}%)  Spot: ${spot}  Futures: ${fut}  Size: $(hsize $sz)"
    echo "              磁盘可用: $(hsize $d_avail)  占用: ${d_pct}%"
    echo "              状态: ${status}  (${note})"
}

# --- main ---
loop=0
for a in "$@"; do [[ "$a" == "--loop" ]] && loop=1; done

(
    flock -n 9 || { echo "[$(now)] 另一个 watcher 在跑 (flock $LOCKF)"; exit 0; }
    [[ -f "$PIDF" ]] || ! kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null && \
        ! pgrep -f "download_orderbook.py.*${ASSETS}USDT" >/dev/null 2>&1 && { echo "[$(now)] 首次运行 → 初始化"; restart; }
    collect
    (( loop )) && while sleep 120; do collect; done
) 9>"$LOCKF"
