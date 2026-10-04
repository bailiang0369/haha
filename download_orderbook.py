#!/usr/bin/env python3
"""
批量下载 CryptoHFTData 的 BTC/ETH L2 订单簿深度数据
==========================================
数据源: https://www.cryptohftdata.com/
文档: https://www.cryptohftdata.com/docs/rest-orderbook

功能:
  - 下载 Binance 现货 (binance_spot) 和永续合约 (binance_futures)
  - 支持 BTCUSDT 和 ETHUSDT
  - 自动按小时分片下载 Parquet 文件
  - 断点续传 (已下载文件自动跳过)
  - 内置速率限制 (匿名 60 req/min, 有 key 无限制)
  - 支持多交易所扩展 (OKX, Bybit 等)

用法示例:
  # 下载 BTC 和 ETH 的现货+合约数据 (2025-09 整月)
  python download_orderbook.py --start 2025-09-01 --end 2025-09-30

  # 只下现货
  python download_orderbook.py --start 2025-08-01 --end 2025-08-07 --market spot

  # 只下合约
  python download_orderbook.py --start 2025-08-01 --end 2025-08-07 --market futures

  # 指定输出目录 + API key
  python download_orderbook.py --start 2025-07-01 --end 2025-10-04 \
      --output /data/crypto_l2 --api-key YOUR_KEY_HERE

  # 扩展到 OKX 和 Bybit
  python download_orderbook.py --start 2025-09-01 --end 2025-09-07 \
      --exchanges binance okx bybit
"""

import argparse
import os
import sys
import time
import random
from datetime import datetime, timedelta
from pathlib import Path

import requests


# ============ 配置 ============

API_BASE = "https://api.cryptohftdata.com"
# 免费 tier: 60 请求/分钟, 保守设为 50 避免触发 429
ANON_RATE_LIMIT = 50  # requests per minute

# 支持的交易所和对应的 archive ID
EXCHANGE_MAP = {
    "binance": {"spot": "binance_spot", "futures": "binance_futures"},
    "okx": {"spot": "okx_spot", "futures": "okx_futures"},
    "bybit": {"spot": "bybit_spot", "futures": "bybit"},
    "hyperliquid": {"spot": "hyperliquid_spot", "futures": "hyperliquid_futures"},
    "bitget": {"spot": "bitget_spot", "futures": "bitget_futures"},
    "kraken": {"spot": "kraken_spot", "futures": "kraken_futures"},
}

# 交易所的 symbol 格式差异
SYMBOL_MAP = {
    "binance": {"BTC": "BTCUSDT", "ETH": "ETHUSDT"},
    "okx": {"BTC": "BTC-USDT", "ETH": "ETH-USDT"},  # spot
    "okx_futures": {"BTC": "BTC-USDT-SWAP", "ETH": "ETH-USDT-SWAP"},
    "bybit": {"BTC": "BTCUSDT", "ETH": "ETHUSDT"},
    "hyperliquid": {"BTC": "BTC", "ETH": "ETH"},
    "bitget": {"BTC": "BTCUSDT", "ETH": "ETHUSDT"},
    "kraken": {"BTC": "BTCUSDT", "ETH": "ETHUSDT"},
}


def get_symbol(exchange: str, asset: str, market: str) -> str:
    """获取正确格式的 symbol"""
    if exchange == "okx" and market == "futures":
        return SYMBOL_MAP["okx_futures"][asset]
    return SYMBOL_MAP[exchange][asset]


def file_path(exchange_id: str, symbol: str, date: str, hour: int, data_type: str = "orderbook") -> str:
    """构造文件路径"""
    return f"{exchange_id}/{date}/{hour:02d}/{symbol}_{data_type}.parquet"


def local_path(root: Path, exchange_id: str, symbol: str, date: str, hour: int) -> Path:
    """本地存储路径"""
    return root / exchange_id / date / f"{hour:02d}" / f"{symbol}_orderbook.parquet"


def generate_hours(start_date: str, end_date: str):
    """生成 [start_date, end_date) 范围内的所有 UTC 小时"""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    cur = start
    while cur < end:
        yield cur.strftime("%Y-%m-%d"), cur.hour
        cur += timedelta(hours=1)


def download_file(url: str, dest: Path, api_key: str | None, max_retries: int = 3) -> tuple[bool, str]:
    """下载单个文件, 返回 (成功, 消息)"""
    dest.parent.mkdir(parents=True, exist_ok=True)

    params = {}
    if api_key:
        params["api_key"] = api_key

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, stream=True, timeout=120)
            if resp.status_code == 200:
                tmp = dest.with_suffix(dest.suffix + ".tmp")
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                os.replace(tmp, dest)
                size_mb = dest.stat().st_size / (1024 * 1024)
                return True, f"OK ({size_mb:.1f}MB)"
            elif resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 60))
                print(f"  [429 Rate Limited] 等待 {retry_after}s (尝试 {attempt}/{max_retries})")
                time.sleep(retry_after + 2)
            elif resp.status_code == 404:
                return False, "404 Not Found (该小时可能无数据)"
            else:
                err = resp.text[:200]
                print(f"  [{resp.status_code}] {err} (尝试 {attempt}/{max_retries})")
                time.sleep(2 ** attempt)
        except requests.RequestException as e:
            print(f"  [网络错误] {e} (尝试 {attempt}/{max_retries})")
            time.sleep(2 ** attempt)

    return False, f"失败 (重试 {max_retries} 次)"


def batch_download(
    assets: list[str],
    markets: list[str],
    exchanges: list[str],
    start_date: str,
    end_date: str,
    output_root: Path,
    api_key: str | None,
    data_types: list[str],
):
    """主下载逻辑"""
    # 生成任务列表
    tasks = []
    for exchange in exchanges:
        for market in markets:
            exchange_id = EXCHANGE_MAP[exchange][market]
            for asset in assets:
                symbol = get_symbol(exchange, asset, market)
                for date, hour in generate_hours(start_date, end_date):
                    for dtype in data_types:
                        lp = local_path(output_root, exchange_id, symbol, date, hour)
                        fp = file_path(exchange_id, symbol, date, hour, dtype)
                        tasks.append((exchange, market, asset, symbol, exchange_id, date, hour, dtype, fp, lp))

    print(f"\n共 {len(tasks)} 个文件待下载")
    print(f"时间范围: {start_date} → {end_date} ({(datetime.strptime(end_date, '%Y-%m-%d') - datetime.strptime(start_date, '%Y-%m-%d')).days} 天)")
    print(f"交易所: {exchanges}")
    print(f"资产: {assets}")
    print(f"市场: {markets}")
    print(f"数据类型: {data_types}")
    print(f"输出目录: {output_root.resolve()}")
    if api_key:
        print("API Key: 已提供")
    else:
        print("API Key: 未提供 (匿名 60 req/min)")
    print()

    # 执行下载
    success, skipped, failed = 0, 0, 0
    failed_list = []
    request_count = 0
    rate_window_start = time.time()

    for i, (exchange, market, asset, symbol, exchange_id, date, hour, dtype, fp, lp) in enumerate(tasks, 1):
        # 断点续传
        if lp.exists() and lp.stat().st_size > 1024:
            skipped += 1
            continue

        url = f"{API_BASE}/download?file={fp}"
        tag = f"[{i}/{len(tasks)}] {exchange:12s} {market:7s} {asset:4s} {dtype:10s} {date} {hour:02d}h"

        # 速率控制
        if not api_key:
            request_count += 1
            elapsed = time.time() - rate_window_start
            if elapsed < 60 and request_count >= ANON_RATE_LIMIT:
                wait = 60 - elapsed + random.uniform(1, 5)
                print(f"  [速率控制] 已达 {ANON_RATE_LIMIT} req/min, 等待 {wait:.0f}s...")
                time.sleep(wait)
                request_count = 0
                rate_window_start = time.time()

        ok, msg = download_file(url, lp, api_key)
        if ok:
            success += 1
            print(f"{tag} ✅ {msg}")
        else:
            failed += 1
            failed_list.append((fp, msg))
            print(f"{tag} ❌ {msg}")

        # 轻微的间隔, 避免瞬时请求
        time.sleep(random.uniform(0.2, 0.5))

    print(f"\n{'='*50}")
    print(f"完成! 成功: {success}, 跳过(已存在): {skipped}, 失败: {failed}")

    if failed_list:
        log = output_root / "failed_downloads.txt"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            for fp, msg in failed_list:
                f.write(f"{fp}\t{msg}\n")
        print(f"失败记录已写入: {log}")


def parse_args():
    p = argparse.ArgumentParser(
        description="批量下载 CryptoHFTData 的 BTC/ETH L2 订单簿深度数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 下载 Binance 现货+合约的 BTC+ETH (2025-09 整月)
  python download_orderbook.py --start 2025-09-01 --end 2025-09-30

  # 多交易所 + 自定义输出
  python download_orderbook.py --start 2025-09-01 --end 2025-10-04 \\
      --exchanges binance okx bybit --output /data/crypto_l2
        """,
    )
    p.add_argument("--start", required=True, help="开始日期 YYYY-MM-DD (含)")
    p.add_argument("--end", required=True, help="结束日期 YYYY-MM-DD (不含)")
    p.add_argument("--assets", nargs="+", default=["BTC", "ETH"], choices=["BTC", "ETH"],
                   help="资产 (默认: BTC ETH)")
    p.add_argument("--market", choices=["spot", "futures", "both"], default="both",
                   help="市场类型 (默认: both)")
    p.add_argument("--exchanges", nargs="+", default=["binance"],
                   choices=list(EXCHANGE_MAP.keys()),
                   help="交易所 (默认: binance)")
    p.add_argument("--output", default="./data", help="输出目录 (默认: ./data)")
    p.add_argument("--api-key", default=None, help="CryptoHFTData API key (可选)")
    p.add_argument("--data-types", nargs="+", default=["orderbook"],
                   choices=["orderbook", "trades", "ticker", "open_interest", "funding", "liquidations"],
                   help="数据类型 (默认: orderbook)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 验证日期范围
    try:
        datetime.strptime(args.start, "%Y-%m-%d")
        datetime.strptime(args.end, "%Y-%m-%d")
    except ValueError:
        print("错误: 日期格式必须是 YYYY-MM-DD")
        sys.exit(1)

    if args.start >= args.end:
        print("错误: --start 必须早于 --end")
        sys.exit(1)

    # 交易所历史起点检查
    # Binance/OKX/Bybit: 2025-06-28
    MIN_DATE = "2025-06-28"
    if args.start < MIN_DATE:
        print(f"警告: 所选交易所的历史数据从 {MIN_DATE} 开始, {args.start} 之前的日期可能无数据")

    # market -> markets 列表
    if args.market == "both":
        markets = ["spot", "futures"]
    else:
        markets = [args.market]

    # 交易所能力检查
    for ex in args.exchanges:
        for m in markets:
            if m not in EXCHANGE_MAP[ex]:
                print(f"错误: {ex} 不支持 {m} 市场")
                sys.exit(1)

    batch_download(
        assets=args.assets,
        markets=markets,
        exchanges=args.exchanges,
        start_date=args.start,
        end_date=args.end,
        output_root=Path(args.output),
        api_key=args.api_key,
        data_types=args.data_types,
    )
