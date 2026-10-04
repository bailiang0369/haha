#!/bin/bash
# BTC L2 订单簿下载 守护脚本
# 功能: 环境恢复 + 进程守护 + 进度监控 + 状态上报

WORKSPACE=/workspace
DATA=$WORKSPACE/data
PY=/root/.pyenv/versions/3.14.7/bin/python3
KEY=$WORKSPACE/.cryptohft_key
STATE=$WORKSPACE/.watch_state.json
TOTAL_EXPECTED=2880

echo "==== $(date '+%Y-%m-%d %H:%M:%S') ===="

# ---- 1. 依赖恢复 ----
if ! $PY -c "import requests, pandas, pyarrow" 2>/dev/null; then
  echo "[恢复] pip install..."
  pip install requests pandas pyarrow -q 2>/dev/null
fi

# ---- 2. API key 恢复 ----
if [ ! -f "$KEY" ] || [ ! -s "$KEY" ]; then
  echo "[恢复] API key 丢失, 从 git 无法恢复, 手动重建..."
  # 如果 git 里存了就 git show 出来
  git -C $WORKSPACE show HEAD:.cryptohft_key 2>/dev/null > "$KEY"
  chmod 600 "$KEY"
fi

# ---- 3. 进程守护 ----
if ps aux | grep -q "[d]ownload_orderbook.py"; then
  STATUS="running"
else
  STATUS="restarting"
  echo "[重启] 下载进程不存在"
  nohup $PY $WORKSPACE/download_orderbook.py \
    --start 2026-09-04 --end 2026-10-04 \
    --assets BTC --market both --exchanges binance \
    --output $DATA \
    --api-key "$(cat $KEY)" &
fi

# ---- 4. 进度统计 ----
WS=$(find $DATA -name '*.parquet' 2>/dev/null | wc -l)
SPOT=$(find $DATA/binance_spot -name '*.parquet' 2>/dev/null | wc -l)
FUTS=$(find $DATA/binance_futures -name '*.parquet' 2>/dev/null | wc -l)
SIZE=$(du -sh $DATA 2>/dev/null | cut -f1)
PCT=$(echo "scale=1; $WS * 100 / $TOTAL_EXPECTED" | bc 2>/dev/null || echo "0")
DISK=$(df -h $WORKSPACE | tail -1 | awk '{print $4}')

# ---- 5. 状态 ----
echo "  文件: $WS / $TOTAL_EXPECTED (${PCT}%)  Spot: $SPOT  Futures: $FUTS  Size: $SIZE"
echo "  磁盘可用: $DISK  状态: $STATUS"

cat > $STATE << EOF
{"timestamp":"$(date -Iseconds)","status":"$STATUS","total":$WS,"spot":$SPOT,"futures":$FUTS,"expected":$TOTAL_EXPECTED,"progress":$PCT,"size":"$SIZE","disk_avail":"$DISK"}
EOF

# ---- 6. 完成检测 ----
if [ "$WS" -ge "$TOTAL_EXPECTED" ]; then
  echo "🎉 下载完成! 所有 $TOTAL_EXPECTED 个文件已就绪"
fi
