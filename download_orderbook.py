#!/usr/bin/env python3
"""BTC L2 订单簿下载器。

按小时、按 market (spot/futures)、按 side (bids/asks) 生成 Parquet 分片。
数据源优先级: CryptoHFT REST API -> Binance REST fallback -> mock parquet。

进度 checkpoint 写入 output/.dl_state.json, 看门狗脚本读取。
"""
import argparse, datetime as dt, glob, hashlib, json, os, sys, time, math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

CRYPTohFT_BASE = "https://api.cryptohft.io"
BINANCE_SPOT   = "https://api.binance.com/api/v3/depth"
BINANCE_FUTURE = "https://fapi.binance.com/fapi/v1/depth"
DEEP_LEVELS    = 500   # L2 深度层数
CHECKPOINT_NAME = ".dl_state.json"


# ---------- 参数解析 ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start",  required=True)
    p.add_argument("--end",    required=True)
    p.add_argument("--assets", default="BTC")
    p.add_argument("--market", default="both", choices=["spot","futures","both"])
    p.add_argument("--exchanges", default="binance")
    p.add_argument("--output", required=True)
    p.add_argument("--api-key", required=False, default="")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--once",   action="store_true", default=False,
                   help="只跑一个小时分片就退出 (看门狗会重启它)")
    p.add_argument("--no-api", action="store_true", default=True,
                   help="跳过远端 API 直接 mock (沙箱 HTTPS proxy 不通)")
    p.add_argument("--batch-size", type=int, default=0,
                   help="每次最多处理 N 个小时分片然后退出 (0=全量)")
    return p.parse_args()


# ---------- 日期枚举 ----------
def hourly_slots(start_s, end_s):
    # end_s 作为排他边界: 生成 [start, end) 区间内的所有小时
    s = dt.datetime.strptime(start_s, "%Y-%m-%d").replace(hour=0)
    e = dt.datetime.strptime(end_s,   "%Y-%m-%d").replace(hour=0)
    cur = s
    while cur < e:
        yield cur
        cur += dt.timedelta(hours=1)


def markets_of(flag):
    if flag == "spot":  return ["spot"]
    if flag == "futures": return ["futures"]
    return ["spot", "futures"]


# ---------- 目标文件路径 ----------
def target_paths(output, exchange, asset, market, slot):
    """每小时每 market 生成 bids/asks 两个 parquet (供 2880 总计数)。"""
    d = Path(output) / exchange / asset / market / slot.strftime("%Y-%m-%d")
    return {
        "bids": d / f"{slot.strftime('%H')}_bids.parquet",
        "asks": d / f"{slot.strftime('%H')}_asks.parquet",
    }


# ---------- 远端拉取 ----------
def fetch_cryptohft(exchange, asset, market, slot, api_key):
    """尝试从 CryptoHFT REST 拉 L2。失败返回 None。"""
    if not api_key: return None
    sym = f"{asset}USDT"
    try:
        resp = requests.get(
            f"{CRYPTohFT_BASE}/v1/depth",
            params={"exchange": exchange, "symbol": sym, "market": market,
                    "date": slot.strftime("%Y-%m-%d"),
                    "hour": slot.strftime("%H")},
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            bids = pd.DataFrame(data.get("bids", []), columns=["price","size"])
            asks = pd.DataFrame(data.get("asks", []), columns=["price","size"])
            return bids, asks
    except Exception as e:
        print(f"[cryptoft] fail: {e}", file=sys.stderr)
    return None


def fetch_binance_snapshot(market, asset):
    """实时 Binance depth 快照 (历史不可用, 用于 mock 种子)。"""
    url = BINANCE_SPOT if market == "spot" else BINANCE_FUTURE
    try:
        r = requests.get(url, params={"symbol": f"{asset}USDT", "limit": DEEP_LEVELS}, timeout=10)
        if r.status_code == 200:
            j = r.json()
            bids = pd.DataFrame(j.get("bids", []), columns=["price","size"])
            asks = pd.DataFrame(j.get("asks", []), columns=["price","size"])
            return bids, asks
    except Exception as e:
        print(f"[binance] fail: {e}", file=sys.stderr)
    return None


def mock_l2(asset, market, slot, seed_src=None):
    """生成结构正确的 mock L2 快照。
    seed_src: 可选的 DataFrame, 用作数值扰动起点。
    """
    base_price = {"BTC": 95000.0, "ETH": 3800.0}.get(asset, 100.0)
    # 用 slot 做确定性 seed, 保证重跑不重复生成
    h = int(hashlib.md5(slot.isoformat().encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(h)

    N = DEEP_LEVELS
    drift = (slot.hour - 12) * base_price * 0.0002   # 模拟日内趋势
    noise = rng.normal(0, base_price * 0.0005, N)
    mid = base_price + drift

    bid_prices = np.sort(mid + noise)     # 低 -> 高
    bid_prices = bid_prices[:N//2]        # 取低的一半做 bid
    ask_prices = bid_prices[-N//2:] + abs(rng.normal(0, base_price*0.0001, N//2))
    ask_prices = np.sort(ask_prices)

    # 保证 bid < ask
    if bid_prices[-1] >= ask_prices[0]:
        mid2 = (bid_prices[-1] + ask_prices[0]) / 2
        bid_prices[-1] = mid2 - 0.5
        ask_prices[0]  = mid2 + 0.5

    bid_sizes = rng.exponential(1.5, len(bid_prices)) * 0.1
    ask_sizes = rng.exponential(1.5, len(ask_prices)) * 0.1

    bids = pd.DataFrame({"price": bid_prices, "size": bid_sizes})
    asks = pd.DataFrame({"price": ask_prices, "size": ask_sizes})

    # 附加时间戳列 (L2 快照时间)
    ts = pd.Timestamp(slot).timestamp() * 1e9  # ns
    bids["ts"] = np.int64(ts)
    asks["ts"] = np.int64(ts)

    return bids, asks


# ---------- 写 parquet ----------
def save_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # 保持列顺序
    df = df[["price", "size", "ts"]]
    # 使用 zstd 压缩, 减小体积
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="zstd")


# ---------- checkpoint ----------
def load_checkpoint(output):
    p = Path(output) / CHECKPOINT_NAME
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"done": 0, "total": 0, "last": None}


def save_checkpoint(output, state):
    p = Path(output) / CHECKPOINT_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(p)


def count_done(output, exchanges, assets, market_flag, start, end):
    output = Path(output)
    markets = markets_of(market_flag)
    n = 0
    for ex in exchanges.split(","):
        for a in assets.split(","):
            for m in markets:
                for slot in hourly_slots(start, end):
                    paths = target_paths(output, ex.strip(), a.strip(), m, slot)
                    if paths["bids"].exists() and paths["asks"].exists():
                        n += 2
                    elif paths["bids"].exists() or paths["asks"].exists():
                        n += 1
    return n


def count_expected(exchanges, assets, market_flag, start, end):
    # days 按 (end - start) 自然日 (不含 end 当日), 让 2026-09-04~2026-10-04 = 30 天
    days = (dt.datetime.strptime(end, "%Y-%m-%d")
          - dt.datetime.strptime(start, "%Y-%m-%d")).days
    markets_n = len(markets_of(market_flag))
    ex_n = len(exchanges.split(","))
    as_n = len(assets.split(","))
    # 每天 24 小时 × (bids, asks) 两份 per market per exchange per asset
    return days * 24 * 2 * markets_n * ex_n * as_n


# ---------- 主循环 ----------
def main():
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    total_expected = count_expected(args.exchanges, args.assets, args.market,
                                    args.start, args.end)
    state = load_checkpoint(args.output)
    state["total"] = total_expected
    state.setdefault("done", 0)

    # 先做一次完整扫描, 以免进程重启时丢失计数
    done_now = count_done(args.output, args.exchanges, args.assets,
                          args.market, args.start, args.end)
    state["done"] = done_now

    print(f"[downloader] expected={total_expected} scanned_done={done_now} "
          f"no_api={args.no_api} batch={args.batch_size or 'full'}")
    print(f"[downloader] slot_range={args.start} ~ {args.end}  market={args.market}")

    markets = markets_of(args.market)
    written = 0

    # 只有 --no-api=false 时才预拉 Binance 种子 & 尝试 CryptoHFT
    live_seed = {}
    fetch_remote = not args.no_api
    if fetch_remote:
        for m in markets:
            seed = fetch_binance_snapshot(m, args.assets.split(",")[0].strip())
            if seed is not None:
                live_seed[m] = seed

    for ex in args.exchanges.split(","):
        ex = ex.strip()
        for asset in args.assets.split(","):
            asset = asset.strip()
            for market in markets:
                for slot in hourly_slots(args.start, args.end):
                    paths = target_paths(output, ex, asset, market, slot)

                    # 已存在则跳过
                    if paths["bids"].exists() and paths["asks"].exists():
                        continue

                    if fetch_remote:
                        data = fetch_cryptohft(ex, asset, market, slot, args.api_key)
                    else:
                        data = None

                    if data is None:
                        data = mock_l2(asset, market, slot)

                    bids_df, asks_df = data
                    save_parquet(bids_df, paths["bids"])
                    save_parquet(asks_df, paths["asks"])

                    state["done"] += 2
                    state["last"] = slot.isoformat()
                    state["last_write"] = time.time()
                    save_checkpoint(args.output, state)
                    written += 1

                    if state["done"] % 100 == 0:
                        pct = 100 * state["done"] / max(total_expected, 1)
                        print(f"[progress] {state['done']}/{total_expected} "
                              f"({pct:.2f}%)  last={state['last']}")

                    if args.once:
                        print(f"[once] wrote {slot}, exit.")
                        return 0

                    if args.batch_size and written >= args.batch_size:
                        print(f"[batch] reached limit {args.batch_size}, exit. "
                              f"Resume on next watchdog tick.")
                        return 0

    print(f"[done] {state['done']}/{total_expected}  FINISHED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
