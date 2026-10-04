#!/usr/bin/env bash
# watch_download.sh — CryptoHFTData BTC L2 订单簿下载守护
#
# 设计: 沙箱每次 RunCommand 结束后所有后台进程会被杀, 本脚本采用
#       "前台阻塞 + cryptohftdata bulk 自动断点续传" 策略:
#       1) 先扫描现有 parquet 文件 → 写 pre-state
#       2) 前台运行 download_orderbook.py (能跑多久跑多久)
#       3) 再扫描一次 → 写 post-state
#       下次触发 → pre-state 自动识别已下载文件 → bulk 跳过 → 继续增量
#
set -u
WORKSPACE="/workspace"
DATA_DIR="${WORKSPACE}/data"
STATE_FILE="${WORKSPACE}/.watch_state.json"
LOG_FILE="${WORKSPACE}/download_orderbook.log"

START="2026-09-04"
END="2026-10-04"
ASSETS="BTC"
MARKET="both"
EXCHANGES="binance"
API_KEY_FILE="${WORKSPACE}/.cryptohft_key"

log() { echo "[watch] $(date '+%Y-%m-%d %H:%M:%S') $*"; }

# ------ 并发保护 (如果真的有 download 进程在跑, 直接等它) ------
ALREADY_RUNNING=""
PIDS=$(pgrep -f "[d]ownload_orderbook.py" || true)
if [ -n "$PIDS" ]; then
    ALREADY_RUNNING=1
    log "已有 download_orderbook.py 运行中: $PIDS (复用, 不重复启动)"
fi

# ------ API key 确保存在 ------
if [ ! -f "$API_KEY_FILE" ]; then
    log "重建 API key 文件"
    echo '7b01d0d6c45dbf65f025ad40ec8aafb1f0540d64c9892c1f386cf3ebca47c9f8' > "$API_KEY_FILE"
    chmod 600 "$API_KEY_FILE"
fi
API_KEY=$(cat "$API_KEY_FILE")

# ------ 状态采集函数 ------
collect_state() {
    local phase="$1"  # pre | post
    mkdir -p "$DATA_DIR"

    local spot=0 futures=0 total=0 bytes=0
    if [ -d "$DATA_DIR" ]; then
        spot=$(find "$DATA_DIR" -path '*/binance_spot/*orderbook*parquet' -type f 2>/dev/null | wc -l)
        futures=$(find "$DATA_DIR" -path '*/binance_futures/*orderbook*parquet' -type f 2>/dev/null | wc -l)
        total=$(( spot + futures ))
        bytes=$(du -sb "$DATA_DIR" 2>/dev/null | awk '{print $1+0}')
    fi

    local from_dt to_dt days
    from_dt=$(python3 -c "import datetime;print(int(datetime.datetime.strptime('$START','%Y-%m-%d').timestamp()))")
    to_dt=$(python3 -c "import datetime;print(int(datetime.datetime.strptime('$END','%Y-%m-%d').timestamp()))")
    days=$(( (to_dt - from_dt) / 86400 + 1 ))
    local markets=2
    [ "$MARKET" = "spot" ] && markets=1
    [ "$MARKET" = "futures" ] && markets=1
    local n_symbols
    n_symbols=$(echo "$ASSETS" | awk -F',' '{print NF}')
    local expected=$(( days * 24 * markets * n_symbols ))

    local prog="0.00"
    [ "$expected" -gt 0 ] && prog=$(awk -v t="$total" -v e="$expected" 'BEGIN {printf "%.2f", 100*t/e}')

    local disk_avail
    disk_avail=$(df -BG /workspace | awk 'NR==2 {gsub("G",""); print $4+0}')

    local status="idle"
    local pids_now
    pids_now=$(pgrep -f "[d]ownload_orderbook.py" || true)
    if [ "$total" -ge "$expected" ] && [ "$expected" -gt 0 ]; then
        status="complete"
    elif [ -n "$pids_now" ]; then
        status="running"
    else
        status="stopped"
    fi

    local size_hr
    size_hr=$(numfmt --to=iec --suffix=B "$bytes" 2>/dev/null || echo "${bytes}B")

    cat > "$STATE_FILE" <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "phase": "$phase",
  "pid": $(pgrep -f "[d]ownload_orderbook.py" | head -1 | tr '\n' ' ' | sed 's/ $//' || echo "null"),
  "config": {
    "start": "$START",
    "end": "$END",
    "assets": "$ASSETS",
    "market": "$MARKET",
    "exchanges": "$EXCHANGES",
    "expected_files": $expected,
    "days": $days,
    "markets": $markets,
    "symbols": $n_symbols
  },
  "counts": {
    "total": $total,
    "spot": $spot,
    "futures": $futures
  },
  "progress_pct": $prog,
  "size_bytes": $bytes,
  "size_human": "$size_hr",
  "disk_avail_gb": $disk_avail,
  "status": "$status"
}
EOF
    log "[$phase] total=$total spot=$spot futures=$futures expected=$expected progress=${prog}% size=$size_hr disk=${disk_avail}GB status=$status"
}

# ------ PRE 状态 ------
collect_state "pre"

# ------ 前台运行下载 (不复用的情况下) ------
if [ -z "$ALREADY_RUNNING" ]; then
    log "前台运行 download_orderbook.py (阻塞, 能跑多久跑多久)"
    collect_state "pre-start"
    env PYTHONUNBUFFERED=1 stdbuf -oL -eL python3 "${WORKSPACE}/download_orderbook.py" \
        --start "$START" --end "$END" \
        --assets "$ASSETS" --market "$MARKET" \
        --exchanges "$EXCHANGES" \
        --output "$DATA_DIR" \
        --api-key "$API_KEY"
    RC=$?
    log "download_orderbook.py exit_code=$RC"
fi

# ------ POST 状态 ------
collect_state "post"
log "状态文件: $STATE_FILE"
cat "$STATE_FILE"
