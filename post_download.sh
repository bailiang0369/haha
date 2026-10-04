#!/bin/bash
# 下载完成后自动跑快照处理
# 用法: bash post_download.sh
set -e
export PATH="/root/.pyenv/bin:$PATH"

echo "=== post_download.sh 启动 ==="
echo "文件: $(find /workspace/data -name '*.parquet' | wc -l) / 1440"

# 等下载完成
for i in $(seq 1 200); do
  alive=$(ps aux | grep '[d]ownload_orderbook' | wc -l)
  n=$(find /workspace/data -name '*.parquet' 2>/dev/null | wc -l)
  echo "[$i] $n/1440  进程=$([ "$alive" -gt 0 ] && echo RUNNING || echo DONE)"
  if [ "$alive" -eq 0 ] && [ "$n" -ge 1400 ]; then
    echo "✅ 下载完成!"
    break
  fi
  sleep 30
done

# 装依赖
pip install polars numpy -q 2>&1 | tail -1

# 跑快照
echo "=== 开始快照处理 ==="
cd /workspace && python3 process_snapshots.py \
  --start 2026-09-04 --end 2026-10-04 \
  --markets binance_spot binance_futures \
  --interval 10 --top_n 50

echo "=== 完成 ==="
echo "快照文件: $(find /workspace/snapshots -name '*.parquet' | wc -l)"
du -sh /workspace/snapshots
