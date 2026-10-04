#!/usr/bin/env bash
# watch_download.sh — 守护 CryptoHFTData BTC L2 订单簿下载
#
# 职责:
#   1) 检查并发: 已有 download_orderbook.py 在跑 → 只做状态汇报, 不重复启动
#   2) 无下载进程 → 启动 download_orderbook.py (后台, 写 PID)
#   3) 扫描 /workspace/data, 统计 spot / futures 已下载 parquet 文件数
#   4) 根据 START/END/MARKET 计算 expected 总量
#   5) 输出 /workspace/.watch_state.json

set -u
WORKSPACE="/workspace"
DATA_DIR="${WORKSPACE}/data"
STATE_FILE="${WORKSPACE}/.watch_state.json"
PID_FILE="${WORKSPACE}/.download_orderbook.pid"
LOG_FILE="${WORKSPACE}/download_orderbook.log"

# ---------- 固定配置 (与下载命令对齐) ----------
START="2026-09-04"
END="2026-10-04"
ASSETS="BTC"
MARKET="both"
EXCHANGES="binance"
API_KEY_FILE="${WORKSPACE}/.cryptohft_key"
# ---------------------------------------------

log() { echo "[watch] $(date '+%Y-%m-%d %H:%M:%S') $*"; }

# ------ 1) 并发检查 ------
DOWNLOAD_PID=""
if [ -f "$PID_FILE" ]; then
    DOWNLOAD_PID=$(cat "$PID_FILE" 2>/dev/null || true)
    if [ -n "$DOWNLOAD_PID" ] && kill -0 "$DOWNLOAD_PID" 2>/dev/null; then
        log "已有 download_orderbook.py 运行中 PID=${DOWNLOAD_PID}"
    else
        log "PID 文件存在但进程已死, 清理"
        rm -f "$PID_FILE"
        DOWNLOAD_PID=""
    fi
fi
# 二次保险: 用 ps 再 grep 一次
if [ -z "$DOWNLOAD_PID" ]; then
    PIDS=$(pgrep -f "[d]ownload_orderbook.py" || true)
    if [ -n "$PIDS" ]; then
        DOWNLOAD_PID=$(echo "$PIDS" | head -n1)
        log "发现 download_orderbook.py PID=${DOWNLOAD_PID} (pgrep)"
        echo "$DOWNLOAD_PID" > "$PID_FILE"
    fi
fi

# ------ 2) 没有下载进程 → 启动 ------
if [ -z "$DOWNLOAD_PID" ]; then
    if [ ! -f "$API_KEY_FILE" ]; then
        log "API key 文件缺失, 重建"
        echo '7b01d0d6c45dbf65f025ad40ec8aafb1f0540d64c9892c1f386cf3ebca47c9f8' > "$API_KEY_FILE"
        chmod 600 "$API_KEY_FILE"
    fi
    API_KEY=$(cat "$API_KEY_FILE")

    log "启动 download_orderbook.py 后台进程"
    nohup env PYTHONUNBUFFERED=1 stdbuf -oL -eL python3 "${WORKSPACE}/download_orderbook.py" \
        --start "$START" --end "$END" \
        --assets "$ASSETS" --market "$MARKET" \
        --exchanges "$EXCHANGES" \
        --output "$DATA_DIR" \
        --api-key "$API_KEY" \
        >> "$LOG_FILE" 2>&1 &
    NEW_PID=$!
    echo "$NEW_PID" > "$PID_FILE"
    DOWNLOAD_PID="$NEW_PID"
    log "已启动 PID=${NEW_PID}, 写入 ${PID_FILE}"
    sleep 2
fi

# ------ 3) 扫描已下载文件 ------
mkdir -p "$DATA_DIR"

SPOT_COUNT=0
FUTURES_COUNT=0
TOTAL_COUNT=0
TOTAL_BYTES=0

if [ -d "$DATA_DIR" ]; then
    SPOT_COUNT=$(find "$DATA_DIR" -path '*/binance_spot/*orderbook*parquet' -type f 2>/dev/null | wc -l)
    FUTURES_COUNT=$(find "$DATA_DIR" -path '*/binance_futures/*orderbook*parquet' -type f 2>/dev/null | wc -l)
    TOTAL_COUNT=$(( SPOT_COUNT + FUTURES_COUNT ))
    TOTAL_BYTES=$(du -sb "$DATA_DIR" 2>/dev/null | awk '{print $1+0}')
fi

# ------ 4) 计算 expected ------
# 天数 (含首尾), 市场数, symbol 数
from_dt=$(date -d "$START" +%s 2>/dev/null || python3 -c "import datetime;print(int(datetime.datetime.strptime('$START','%Y-%m-%d').timestamp()))")
to_dt=$(date -d "$END" +%s 2>/dev/null || python3 -c "import datetime;print(int(datetime.datetime.strptime('$END','%Y-%m-%d').timestamp()))")
DAYS=$(( (to_dt - from_dt) / 86400 + 1 ))
HOURS_PER_DAY=24

case "$MARKET" in
    spot)   MARKETS=1 ;;
    futures) MARKETS=1 ;;
    both)   MARKETS=2 ;;
esac
# assets 数量 (BTC -> 1 symbol)
N_SYMBOLS=$(echo "$ASSETS" | awk -F',' '{print NF}')
EXPECTED=$(( DAYS * HOURS_PER_DAY * MARKETS * N_SYMBOLS ))

# ------ 5) 磁盘空间 ------
DISK_AVAIL=$(df -BG /workspace | awk 'NR==2 {print $4}' | tr -d 'G')
DISK_USED=$(df -h /workspace | awk 'NR==2 {print $3}' | tr -d '%')

# ------ 6) 进度 & 状态 ------
if [ "$EXPECTED" -gt 0 ]; then
    PROGRESS=$(awk -v t="$TOTAL_COUNT" -v e="$EXPECTED" 'BEGIN {printf "%.2f", 100*t/e}')
else
    PROGRESS="0.00"
fi

# status: 完成 / 运行中 / 等待启动 / 已停止
STATUS="idle"
if [ "$DOWNLOAD_PID" != "" ] && kill -0 "$DOWNLOAD_PID" 2>/dev/null; then
    if [ "$TOTAL_COUNT" -ge "$EXPECTED" ] && [ "$EXPECTED" -gt 0 ]; then
        STATUS="complete"
    else
        STATUS="running"
    fi
else
    if [ "$TOTAL_COUNT" -ge "$EXPECTED" ] && [ "$EXPECTED" -gt 0 ]; then
        STATUS="complete"
    else
        STATUS="stopped"
    fi
fi

SIZE_HR=$(numfmt --to=iec --suffix=B "$TOTAL_BYTES" 2>/dev/null || echo "${TOTAL_BYTES}B")

# ------ 7) 写 JSON ------
cat > "$STATE_FILE" <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "pid": ${DOWNLOAD_PID:-null},
  "config": {
    "start": "$START",
    "end": "$END",
    "assets": "$ASSETS",
    "market": "$MARKET",
    "exchanges": "$EXCHANGES",
    "expected_files": $EXPECTED,
    "days": $DAYS,
    "markets": $MARKETS,
    "symbols": $N_SYMBOLS
  },
  "counts": {
    "total": $TOTAL_COUNT,
    "spot": $SPOT_COUNT,
    "futures": $FUTURES_COUNT
  },
  "progress_pct": $PROGRESS,
  "size_bytes": $TOTAL_BYTES,
  "size_human": "$SIZE_HR",
  "disk_avail_gb": $DISK_AVAIL,
  "status": "$STATUS"
}
EOF

log "状态: total=${TOTAL_COUNT} spot=${SPOT_COUNT} futures=${FUTURES_COUNT} expected=${EXPECTED} progress=${PROGRESS}% size=${SIZE_HR} disk=${DISK_AVAIL}GB status=${STATUS}"
log "状态文件已写入 ${STATE_FILE}"
cat "$STATE_FILE"
