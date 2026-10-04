#!/usr/bin/env bash
# watch_download.sh — BTC L2 订单簿下载守护脚本
# 职责:
#   1) 单实例锁 (禁止并发 download_orderbook.py)
#   2) 扫描已下载分片 / 读取 .dl_state.json
#   3) 汇总 spot / futures 数量、进度、磁盘空间, 写入 .watch_state.json
#   4) 进程不存在且未完成时, 自动拉起下载 (nohup 后台跑)
#
# 用法:  bash /workspace/watch_download.sh
# cron / 定时任务每次触发直接跑就行, 脚本幂等。

set -u

WORKSPACE="${WORKSPACE:-/workspace}"
DATA_DIR="${DATA_DIR:-/workspace/data}"
DL_PY="${DL_PY:-$WORKSPACE/download_orderbook.py}"
STATE_JSON="${WORKSPACE}/.watch_state.json"
DL_STATE="${DATA_DIR}/.dl_state.json"
WATCH_LOG="${WORKSPACE}/.watch.log"
WATCH_PID="${WORKSPACE}/.watch.pid"

START="2026-09-04"
END="2026-10-04"
ASSETS="BTC"
MARKET="both"
EXCHANGES="binance"
API_KEY="$(cat "$WORKSPACE/.cryptohft_key" 2>/dev/null | tr -d '[:space:]')"

exec >> "$WATCH_LOG" 2>&1
echo "===== $(date -Is) watch_download.sh run ====="

# ---------- 单实例锁 ----------
if [[ -f "$WATCH_PID" ]]; then
    OLD_PID=$(cat "$WATCH_PID" 2>/dev/null)
    if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[watch] another watch running (pid=$OLD_PID), skip."
        exit 0
    fi
fi
echo $$ > "$WATCH_PID"
trap 'rm -f "$WATCH_PID"' EXIT

# ---------- 检查下载进程 ----------
DL_PROCS=$(ps aux | grep '[d]ownload_orderbook.py' | wc -l)
echo "[watch] download_orderbook.py instances = $DL_PROCS"

# ---------- 磁盘空间 ----------
DISK_KB=$(df -k "$WORKSPACE" | awk 'NR==2 {print $4}')
DISK_AVAIL_H=$(df -h "$WORKSPACE" | awk 'NR==2 {print $4}')
echo "[watch] disk avail = ${DISK_AVAIL_H} (${DISK_KB} KB)"

# ---------- 扫描实际分片 ----------
mkdir -p "$DATA_DIR"
# spot parquet 数 (bids + asks = 2 文件/小时)
SPOT_N=$(find "$DATA_DIR" -path "*/spot/*.parquet" 2>/dev/null | wc -l)
FUT_N=$(find "$DATA_DIR" -path "*/futures/*.parquet" 2>/dev/null | wc -l)
TOTAL_FILES=$((SPOT_N + FUT_N))

# 总字节数
SIZE_BYTES=$(du -sb "$DATA_DIR" 2>/dev/null | awk '{print $1}')
[[ -z "$SIZE_BYTES" ]] && SIZE_BYTES=0
SIZE_H=$(du -sh "$DATA_DIR" 2>/dev/null | awk '{print $1}')

# ---------- 读取 / 回退计算 expected ----------
DAYS=$(( $(date -d "$END" +%s) - $(date -d "$START" +%s) ))
DAYS=$((DAYS / 86400))           # 不含 END 当日, 让 30 天 = 2880
EXPECTED=$(( DAYS * 24 * 2 * 2 ))   # days * hours * 2(bids+asks) * 2(spot+futures)

# 如果 .dl_state.json 存在, 优先用里面的 total
if [[ -f "$DL_STATE" ]]; then
    JS_TOTAL=$(python3 -c "import json,sys;print(json.load(open('$DL_STATE')).get('total',''))" 2>/dev/null)
    if [[ -n "$JS_TOTAL" ]]; then EXPECTED="$JS_TOTAL"; fi
fi

# ---------- 状态字段 ----------
if (( EXPECTED > 0 )); then
    PROGRESS=$(awk -v d="$TOTAL_FILES" -v e="$EXPECTED" 'BEGIN{printf "%.2f", 100*d/e}')
else
    PROGRESS="0.00"
fi

if (( DL_PROCS > 0 )); then
    STATUS="running"
elif (( TOTAL_FILES >= EXPECTED )); then
    STATUS="done"
else
    STATUS="idle"
fi

# ---------- 写 .watch_state.json ----------
python3 - "$STATE_JSON" <<PYEOF
import json, sys, os
s = {
    "ts":         __import__("datetime").datetime.utcnow().isoformat() + "Z",
    "total":      $TOTAL_FILES,
    "spot":       $SPOT_N,
    "futures":    $FUT_N,
    "expected":   $EXPECTED,
    "progress":   float("$PROGRESS"),
    "size":       "$SIZE_H",
    "size_bytes": $SIZE_BYTES,
    "status":     "$STATUS",
    "disk_avail": "$DISK_AVAIL_H",
    "disk_kb":    $DISK_KB,
    "dl_procs":   $DL_PROCS,
    "start":      "$START",
    "end":        "$END",
    "assets":     "$ASSETS",
    "market":     "$MARKET",
    "exchanges":  "$EXCHANGES",
}
open(sys.argv[1], "w").write(json.dumps(s, indent=2))
print(s)
PYEOF
echo "[watch] wrote $STATE_JSON  status=$STATUS  progress=${PROGRESS}%  total=$TOTAL_FILES/$EXPECTED"

# ---------- 自动拉起 ----------
if (( DL_PROCS == 0 )) && [[ "$STATUS" != "done" ]]; then
    if [[ -z "$API_KEY" ]]; then
        echo "[watch] API key missing, abort start."
        exit 1
    fi
    echo "[watch] no download process & not done, run batch-size=192 in foreground..."
    # 前台跑 batch, 避免 sandbox 回收后台进程
    python3 "$DL_PY" \
        --start "$START" --end "$END" \
        --assets "$ASSETS" --market "$MARKET" --exchanges "$EXCHANGES" \
        --output "$DATA_DIR" --api-key "$API_KEY" \
        --no-api --batch-size 192 \
        >> "$WORKSPACE/.download.log" 2>&1
    echo "[watch] batch exit=$?"
fi

# 成功退出
exit 0
