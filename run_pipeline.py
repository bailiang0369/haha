#!/usr/bin/env python3
"""
BTCUSDT 订单簿 + aggTrades → 监督学习特征 Pipeline

数据源：
  - CryptoHFTData (Binance Futures + Spot)：L2 depth update parquets
  - Binance data.binance.vision：aggTrades CSV

输出：
  - /workspace/output/features_BTCUSDT_{spot,futures}_{YYYY-MM-DD}.parquet
  - 每行 = 每根 1m K 线 close_time 的完整特征 + 标签

架构：
  - 逐小时处理 orderbook（单小时 ~2GB 内存，7 小时一起装不下）
  - Python dict 维护 L2Book（逐事件状态机，polars 不适合做这个）
  - polars 用于 aggTrades 处理 / K 线生成 / 特征表拼接
  - 多窗口特征并行累计（10s / 30s / 60s / 5min）
  - close_time 严格对齐，避免 look-ahead bias
"""

from __future__ import annotations

import gc
import glob
import os
import time
from collections import defaultdict
from typing import Iterable

import polars as pl
from bisect import bisect_left, insort

# ============================================================
# 配置
# ============================================================
DATA_DIR = "/workspace/data"
OUT_DIR = "/workspace/output"
SYMBOL = "BTCUSDT"
DATE = "2026-09-02"

BOOK_WINDOW = 0.0025          # mid ± 0.25%
WINDOWS = [10, 30, 60, 300]   # 多窗口特征（秒）
LARGE_QTY = 0.5               # 大额阈值（BTC）

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# L2Book Replay（纯 Python，单线程）
# ============================================================
class L2Book:
    """维护 mid ± BOOK_WINDOW 范围内的 L2 订单簿。"""

    __slots__ = ("bids", "asks", "_bid_sorted", "_ask_sorted",
                 "price_window", "mid", "last_event_time", "last_update_id")

    def __init__(self, price_window: float = BOOK_WINDOW):
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self._bid_sorted: list[float] = []   # asc, best bid = [-1]
        self._ask_sorted: list[float] = []   # asc, best ask = [0]
        self.price_window = price_window
        self.mid = 0.0
        self.last_event_time = 0
        self.last_update_id = None

    def _maintain_window(self):
        if self.mid <= 0:
            return
        lo = self.mid * (1 - self.price_window)
        hi = self.mid * (1 + self.price_window)
        while self._bid_sorted and self._bid_sorted[0] < lo:
            self.bids.pop(self._bid_sorted.pop(0), None)
        while self._ask_sorted and self._ask_sorted[-1] > hi:
            self.asks.pop(self._ask_sorted.pop(), None)

    def update_mid(self):
        if self._bid_sorted and self._ask_sorted:
            self.mid = (self._bid_sorted[-1] + self._ask_sorted[0]) / 2

    def apply(self, price: float, qty: float, side: str):
        if side == "bid":
            if qty > 0:
                if price not in self.bids:
                    insort(self._bid_sorted, price)
                self.bids[price] = qty
            else:
                if price in self.bids:
                    self.bids.pop(price)
                    self._bid_sorted.pop(bisect_left(self._bid_sorted, price))
        else:
            if qty > 0:
                if price not in self.asks:
                    insort(self._ask_sorted, price)
                self.asks[price] = qty
            else:
                if price in self.asks:
                    self.asks.pop(price)
                    self._ask_sorted.pop(bisect_left(self._ask_sorted, price))
        self.update_mid()
        self._maintain_window()

    # ---- 读取 ----
    def best_bid(self):
        if self._bid_sorted:
            p = self._bid_sorted[-1]
            return p, self.bids[p]
        return None, None

    def best_ask(self):
        if self._ask_sorted:
            p = self._ask_sorted[0]
            return p, self.asks[p]
        return None, None

    def snapshot(self) -> dict:
        best_bid, bid_vol = self.best_bid()
        best_ask, ask_vol = self.best_ask()

        out = {
            "mid": self.mid,
            "spread": (best_ask - best_bid) if (best_bid and best_ask) else 0.0,
            "spread_bps": 0.0,
            "imbalance": 0.0,
            "top_bid_vol": bid_vol or 0.0,
            "top_ask_vol": ask_vol or 0.0,
            "depth_total": 0.0,
            "depth_bid": 0.0,
            "depth_ask": 0.0,
            "bid_count": len(self.bids),
            "ask_count": len(self.asks),
            "depth_01pct": 0.0,
            "depth_05pct": 0.0,
        }

        if self.mid > 0:
            out["spread_bps"] = out["spread"] / self.mid * 10000

        bid_sum = sum(self.bids.values())
        ask_sum = sum(self.asks.values())
        total = bid_sum + ask_sum
        out["depth_bid"] = bid_sum
        out["depth_ask"] = ask_sum
        out["depth_total"] = total
        out["imbalance"] = (bid_sum - ask_sum) / total if total > 0 else 0.0

        if self.mid > 0:
            d01 = self.mid * 0.001
            d05 = self.mid * 0.005
            out["depth_01pct"] = (
                sum(q for p, q in self.bids.items() if p >= self.mid - d01)
                + sum(q for p, q in self.asks.items() if p <= self.mid + d01)
            )
            out["depth_05pct"] = (
                sum(q for p, q in self.bids.items() if p >= self.mid - d05)
                + sum(q for p, q in self.asks.items() if p <= self.mid + d05)
            )

        return out


# ============================================================
# 多窗口事件统计
# ============================================================
class WindowStats:
    __slots__ = ("window", "events", "total")

    def __init__(self, window_sec: int):
        self.window = window_sec
        self.events: list[tuple[int, dict]] = []
        self.total: dict[str, float] = defaultdict(float)

    def expire(self, cutoff_ms: int):
        idx = 0
        while idx < len(self.events) and self.events[idx][0] < cutoff_ms:
            _, vals = self.events[idx]
            for k, v in vals.items():
                self.total[k] -= v
            idx += 1
        if idx > 0:
            del self.events[:idx]

    def add(self, etime_ms: int, vals: dict[str, float]):
        self.events.append((etime_ms, vals))
        for k, v in vals.items():
            self.total[k] += v


class EventAggregator:
    def __init__(self, windows: list[int] = WINDOWS):
        self.windows = windows
        self.bid_last_qty: dict[float, float] = {}
        self.ask_last_qty: dict[float, float] = {}
        self.ws = {w: WindowStats(w) for w in windows}

    def on_depth_update(self, etime_ms: int,
                         bid_changes: list[tuple[float, float]],
                         ask_changes: list[tuple[float, float]]):
        for w_sec, w in self.ws.items():
            w.expire(etime_ms - w_sec * 1000)

        ofi = {w: 0.0 for w in self.windows}
        la = {w: 0 for w in self.windows}
        laq = {w: 0.0 for w in self.windows}
        lc = {w: 0 for w in self.windows}
        lcq = {w: 0.0 for w in self.windows}

        for price, new_qty in bid_changes:
            prev = self.bid_last_qty.get(price, 0.0)
            for w in self.windows:
                if new_qty > prev:
                    ofi[w] += (new_qty - prev)
                elif new_qty < prev:
                    ofi[w] -= (prev - new_qty)
                if new_qty >= LARGE_QTY and prev < LARGE_QTY:
                    la[w] += 1
                    laq[w] += new_qty
                if new_qty == 0 and prev >= LARGE_QTY:
                    lc[w] += 1
                    lcq[w] += prev
            self.bid_last_qty[price] = new_qty

        for price, new_qty in ask_changes:
            prev = self.ask_last_qty.get(price, 0.0)
            for w in self.windows:
                if new_qty > prev:
                    ofi[w] -= (new_qty - prev)
                elif new_qty < prev:
                    ofi[w] += (prev - new_qty)
                if new_qty >= LARGE_QTY and prev < LARGE_QTY:
                    la[w] += 1
                    laq[w] += new_qty
                if new_qty == 0 and prev >= LARGE_QTY:
                    lc[w] += 1
                    lcq[w] += prev
            self.ask_last_qty[price] = new_qty

        for w in self.windows:
            self.ws[w].add(etime_ms, {
                "ofi": ofi[w], "large_add": la[w], "large_add_qty": laq[w],
                "large_cancel": lc[w], "large_cancel_qty": lcq[w],
            })

    def snapshot(self, etime_ms: int,
                 trade_stats: dict[int, dict]) -> dict[str, float]:
        out = {}
        for w in self.windows:
            self.ws[w].expire(etime_ms - w * 1000)
            st = self.ws[w].total
            tr = trade_stats.get(w, {})
            buy_vol = tr.get("buy_vol", 0.0)
            sell_vol = tr.get("sell_vol", 0.0)
            total_vol = buy_vol + sell_vol

            out[f"ofi_{w}s"] = st["ofi"]
            out[f"ofi_norm_{w}s"] = st["ofi"] / (total_vol + 1e-9)
            out[f"large_add_{w}s"] = st["large_add"]
            out[f"large_cancel_{w}s"] = st["large_cancel"]
            out[f"large_cancel_rate_{w}s"] = (
                st["large_cancel"] / st["large_add"] if st["large_add"] > 0 else 0.0)
            out[f"trade_count_{w}s"] = tr.get("trade_count", 0)
            out[f"buy_ratio_{w}s"] = buy_vol / total_vol if total_vol > 0 else 0.5
            out[f"large_trade_{w}s"] = tr.get("large_trade", 0)
            out[f"volume_{w}s"] = total_vol
            out[f"trade_rate_{w}s"] = tr.get("trade_count", 0) / w
        return out


# ============================================================
# 成交流 + K 线生成（polars）
# ============================================================
def load_aggtrades(exchange: str) -> pl.DataFrame:
    glob_pat = (f"{DATA_DIR}/aggtrades/futures/{SYMBOL}-aggTrades-{DATE}.csv"
                if exchange == "futures"
                else f"{DATA_DIR}/aggtrades/spot/{SYMBOL}-aggTrades-{DATE}.csv")
    dfs = []
    for f in sorted(glob.glob(glob_pat)):
        if exchange == "spot":
            # spot CSV 没 header，8 列（多 is_best_match）
            df = pl.read_csv(f, has_header=False, ignore_errors=True,
                             new_columns=["agg_trade_id", "price", "quantity",
                                          "first_trade_id", "last_trade_id",
                                          "transact_time", "is_buyer_maker", "is_best_match"])
            dfs.append(df.select([
                pl.col("price").cast(pl.Float64),
                pl.col("quantity").cast(pl.Float64),
                (pl.col("transact_time").cast(pl.Int64) / 1000).cast(pl.Int64).alias("ts_ms"),
                pl.col("is_buyer_maker").cast(pl.Boolean).alias("is_maker"),
            ]))
        else:
            df = pl.read_csv(f, ignore_errors=True)
            dfs.append(df.select([
                pl.col("price").cast(pl.Float64),
                pl.col("quantity").cast(pl.Float64),
                pl.col("transact_time").cast(pl.Int64).alias("ts_ms"),
                pl.col("is_buyer_maker").cast(pl.Boolean).alias("is_maker"),
            ]))
    df = pl.concat(dfs).sort("ts_ms")
    return df


def build_trade_window_stats(trades: pl.DataFrame, target_ts: int) -> dict[int, dict]:
    out = {}
    for w in WINDOWS:
        lo = target_ts - w * 1000
        win = trades.filter((pl.col("ts_ms") >= lo) & (pl.col("ts_ms") <= target_ts))
        if win.height == 0:
            out[w] = {"trade_count": 0, "buy_vol": 0.0, "sell_vol": 0.0,
                      "large_trade": 0, "large_trade_qty": 0.0}
            continue
        buy = win.filter(pl.col("is_maker"))["quantity"].sum() or 0.0
        sell = win.filter(~pl.col("is_maker"))["quantity"].sum() or 0.0
        large = win.filter(pl.col("quantity") >= LARGE_QTY)
        out[w] = {
            "trade_count": win.height,
            "buy_vol": buy,
            "sell_vol": sell,
            "large_trade": large.height,
            "large_trade_qty": large["quantity"].sum() or 0.0,
        }
    return out


def generate_klines(trades: pl.DataFrame) -> pl.DataFrame:
    trades = trades.with_columns(pl.from_epoch("ts_ms", time_unit="ms").alias("dt"))

    klines = trades.group_by_dynamic(
        "dt", every="1m", closed="left", label="right", offset="0s",
    ).agg([
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.col("price").last().alias("close"),
        pl.col("quantity").sum().alias("volume"),
        (pl.col("price") * pl.col("quantity")).sum().alias("quote_volume"),
        pl.col("ts_ms").last().alias("close_ts_ms"),
    ])

    klines = klines.with_columns([
        ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("kline_return"),
        ((pl.col("high") - pl.col("low")) / pl.col("open")).alias("kline_range"),
        ((pl.col("close") - pl.col("open")).abs() / pl.col("open")).alias("kline_body"),
        ((pl.col("high") - pl.max_horizontal(pl.col("close"), pl.col("open")))
         / pl.col("open")).alias("kline_upper_shadow"),
        ((pl.min_horizontal(pl.col("close"), pl.col("open")) - pl.col("low"))
         / pl.col("open")).alias("kline_lower_shadow"),
        (pl.col("quote_volume") / pl.col("volume")).alias("vwap"),
    ])
    klines = klines.with_columns([
        pl.col("close").shift(1).alias("prev_close"),
        pl.col("volume").shift(1).alias("prev_volume"),
    ]).with_columns([
        ((pl.col("close") / pl.col("prev_close") - 1)).alias("return_vs_prev"),
        ((pl.col("volume") / pl.col("prev_volume") - 1)).alias("volume_change"),
    ])
    return klines


# ============================================================
# 小时级处理（避免 OOM）
# ============================================================
def bootstrap_from_first_update(ob_grouped: pl.DataFrame,
                                 book: L2Book,
                                 aggregator: EventAggregator) -> int:
    """从第一条 update 初始化 book。返回第一条的 final_update_id。"""
    first = ob_grouped.row(0, named=True)
    for side, p, q in zip(first["sides"], first["prices"], first["quantities"]):
        if q <= 0:
            continue
        if side == "bid":
            book.bids[p] = q
            aggregator.bid_last_qty[p] = q
        else:
            book.asks[p] = q
            aggregator.ask_last_qty[p] = q
    book._bid_sorted = sorted(book.bids.keys())
    book._ask_sorted = sorted(book.asks.keys())
    book.update_mid()
    book._maintain_window()
    return int(first["final_update_id"])


def run_for_exchange(exchange: str):
    print(f"\n{'='*60}")
    print(f"[exchange={exchange}] 开始处理 {DATE}")
    print(f"{'='*60}")
    t0 = time.time()

    # ---- 1. 加载 trades + 生成 klines（小数据，一次加载） ----
    print("📦 加载 aggTrades ...")
    trades = load_aggtrades(exchange)
    print(f"   trades: {trades.height:,} 行, "
          f"ts_ms ∈ [{trades['ts_ms'][0]}, {trades['ts_ms'][-1]}]")

    print("📊 生成 1m K 线 ...")
    klines = generate_klines(trades)
    close_times = klines["close_ts_ms"].to_list()
    n_klines = len(close_times)
    print(f"   klines: {n_klines} 根")

    # ---- 2. 初始化全局状态（跨小时保留） ----
    book = L2Book(BOOK_WINDOW)
    aggregator = EventAggregator(WINDOWS)
    prev_final = None  # 首文件 bootstrap 后赋值
    feat_rows = []

    # ---- 3. 逐小时处理 orderbook ----
    ob_files = sorted(glob.glob(
        f"{DATA_DIR}/orderbook/{exchange}_{SYMBOL}_{DATE}_*.parquet"))
    if not ob_files:
        print(f"❌ 没找到 orderbook 文件")
        return

    print(f"📂 逐小时处理 {len(ob_files)} 个 orderbook parquet ...")
    bar_ptr = 0  # 当前推进到第几根 K 线
    total_updates = 0
    t_replay = time.time()

    for fidx, ob_file in enumerate(ob_files):
        print(f"\n   [{fidx+1}/{len(ob_files)}] {os.path.basename(ob_file)}")
        t_hour = time.time()

        # 加载单小时 orderbook + group_by
        ob = pl.read_parquet(ob_file)
        ob = ob.with_columns([
            pl.col("price").cast(pl.Float64),
            pl.col("quantity").cast(pl.Float64),
            pl.col("first_update_id").cast(pl.Int64),
            pl.col("prev_final_update_id").cast(pl.Int64),
            pl.col("final_update_id").cast(pl.Int64),
        ])
        ob = ob.sort(["event_time", "first_update_id", "side", "price"])
        ob_grouped = ob.group_by(["first_update_id"], maintain_order=True).agg([
            pl.col("event_time").first().alias("event_time"),
            pl.col("prev_final_update_id").first().alias("prev_final_update_id"),
            pl.col("final_update_id").first().alias("final_update_id"),
            pl.col("side").alias("sides"),
            pl.col("price").alias("prices"),
            pl.col("quantity").alias("quantities"),
        ])

        n_updates = ob_grouped.height
        update_times = ob_grouped["event_time"].to_list()
        total_updates += n_updates
        print(f"     {ob.height:,} 行 → {n_updates:,} 条 update, "
              f"[{update_times[0]} ~ {update_times[-1]}]")

        # bootstrap（首文件）或检查连续性（后续文件）
        if prev_final is None:
            prev_final = bootstrap_from_first_update(ob_grouped, book, aggregator)
            u_idx = 1  # 跳过 bootstrap 那条
            print(f"     bootstrap 完成: mid={book.mid:.2f}")
        else:
            first_pfid_raw = ob_grouped.row(0, named=True)["prev_final_update_id"]
            first_pfid = int(first_pfid_raw) if first_pfid_raw is not None else -1
            if first_pfid != prev_final:
                print(f"     ⚠️  跨小时 gap! 期望 pfid={prev_final}, 实际={first_pfid} → 重新 bootstrap")
                prev_final = bootstrap_from_first_update(ob_grouped, book, aggregator)
                u_idx = 1
            else:
                print(f"     update 链连续，继续")
                u_idx = 0  # 首条也需要 apply（它的 pfid 匹配）

        # 推进 close_times 并生成特征
        while bar_ptr < n_klines and update_times[-1] >= close_times[bar_ptr]:
            target = close_times[bar_ptr]
            # apply 所有 update_etimes < target
            while u_idx < n_updates and update_times[u_idx] < target:
                row = ob_grouped.row(u_idx, named=True)
                pfid_raw = row["prev_final_update_id"]
                final = int(row["final_update_id"])
                # prev_final_update_id 可能为 null → 当 gap 处理
                pfid = int(pfid_raw) if pfid_raw is not None else -1

                if pfid != prev_final:
                    # gap → 用当前 update bootstrap
                    book.bids.clear(); book.asks.clear()
                    aggregator = EventAggregator(WINDOWS)
                    for side, p, q in zip(row["sides"], row["prices"], row["quantities"]):
                        if q > 0:
                            (book.bids if side == "bid" else book.asks)[p] = q
                            if side == "bid": aggregator.bid_last_qty[p] = q
                            else: aggregator.ask_last_qty[p] = q
                    book._bid_sorted = sorted(book.bids.keys())
                    book._ask_sorted = sorted(book.asks.keys())
                    book.update_mid()
                    book._maintain_window()
                prev_final = final

                et = int(row["event_time"])
                bid_changes, ask_changes = [], []
                for side, p, q in zip(row["sides"], row["prices"], row["quantities"]):
                    book.apply(p, q, side)
                    if side == "bid": bid_changes.append((p, q))
                    else: ask_changes.append((p, q))
                aggregator.on_depth_update(et, bid_changes, ask_changes)
                u_idx += 1

            if book.mid > 0:
                snap = book.snapshot()
                trade_stats = build_trade_window_stats(trades, target)
                event_feat = aggregator.snapshot(target, trade_stats)
                kline_row = klines.row(bar_ptr, named=True)

                row_out = {"close_ts_ms": target}
                row_out.update({f"book_{k}": v for k, v in snap.items()})
                row_out.update(event_feat)
                row_out.update({
                    "kline_open": kline_row.get("open"),
                    "kline_high": kline_row.get("high"),
                    "kline_low": kline_row.get("low"),
                    "kline_close": kline_row.get("close"),
                    "kline_volume": kline_row.get("volume"),
                    "kline_quote_volume": kline_row.get("quote_volume"),
                    "kline_return": kline_row.get("kline_return"),
                    "kline_range": kline_row.get("kline_range"),
                    "kline_body": kline_row.get("kline_body"),
                    "kline_upper_shadow": kline_row.get("kline_upper_shadow"),
                    "kline_lower_shadow": kline_row.get("kline_lower_shadow"),
                    "kline_vwap": kline_row.get("vwap"),
                    "return_vs_prev": kline_row.get("return_vs_prev"),
                    "volume_change": kline_row.get("volume_change"),
                })
                feat_rows.append(row_out)

            bar_ptr += 1
            if bar_ptr % 100 == 0:
                elapsed = time.time() - t_replay
                print(f"     ★ bar {bar_ptr}/{n_klines}  "
                      f"[{bar_ptr/elapsed:.0f} bar/s]  "
                      f"mid={book.mid:.2f}")

        print(f"     小时耗时: {time.time() - t_hour:.1f}s  "
              f"已处理 bar {bar_ptr}/{n_klines}")

        # 释放单小时内存
        del ob, ob_grouped, update_times
        gc.collect()

    # 处理未覆盖的 close_times（book 不推进，但特征里 book 保持最后状态）
    while bar_ptr < n_klines:
        target = close_times[bar_ptr]
        if book.mid > 0:
            snap = book.snapshot()
            trade_stats = build_trade_window_stats(trades, target)
            event_feat = aggregator.snapshot(target, trade_stats)
            kline_row = klines.row(bar_ptr, named=True)
            row_out = {"close_ts_ms": target}
            row_out.update({f"book_{k}": v for k, v in snap.items()})
            row_out.update(event_feat)
            row_out.update({
                "kline_open": kline_row.get("open"),
                "kline_high": kline_row.get("high"),
                "kline_low": kline_row.get("low"),
                "kline_close": kline_row.get("close"),
                "kline_volume": kline_row.get("volume"),
                "kline_quote_volume": kline_row.get("quote_volume"),
                "kline_return": kline_row.get("kline_return"),
                "kline_range": kline_row.get("kline_range"),
                "kline_body": kline_row.get("kline_body"),
                "kline_upper_shadow": kline_row.get("kline_upper_shadow"),
                "kline_lower_shadow": kline_row.get("kline_lower_shadow"),
                "kline_vwap": kline_row.get("vwap"),
                "return_vs_prev": kline_row.get("return_vs_prev"),
                "volume_change": kline_row.get("volume_change"),
            })
            feat_rows.append(row_out)
        bar_ptr += 1

    # ---- 4. 过滤 dead rows（book 覆盖区域外 book_mid 不变的行） ----
    print(f"\n🏷️  生成标签 ...")
    feat_df = pl.DataFrame(feat_rows).sort("close_ts_ms")

    # 识别 book 真正推进过的行（book_mid 与前一行不同）
    feat_df = feat_df.with_columns([
        pl.col("book_mid").shift(1).alias("prev_book_mid"),
    ])
    feat_df = feat_df.with_columns([
        (pl.col("book_mid") != pl.col("prev_book_mid")).alias("book_changed"),
    ])
    n_live = feat_df.filter(pl.col("book_changed")).height
    n_dead = feat_df.height - n_live
    print(f"   book 覆盖区域内（book_changed=True）: {n_live} 行")
    print(f"   book 覆盖区域外（book 停滞）: {n_dead} 行 → 过滤掉")
    feat_df = feat_df.filter(pl.col("book_changed"))

    # 标签
    feat_df = feat_df.with_columns([
        pl.col("kline_close").shift(-1).alias("next_close"),
        pl.col("kline_close").shift(-2).alias("next2_close"),
    ])
    feat_df = feat_df.with_columns([
        ((pl.col("next_close") - pl.col("kline_close")) / pl.col("kline_close"))
        .alias("label_return_1m"),
        ((pl.col("next2_close") - pl.col("kline_close")) / pl.col("kline_close"))
        .alias("label_return_2m"),
    ])
    feat_df = feat_df.with_columns([
        (pl.col("label_return_1m") > 0).cast(pl.Int8).alias("label_dir_1m"),
    ])
    feat_df = feat_df.filter(pl.col("label_return_1m").is_not_null())

    # 清理中间列
    feat_df = feat_df.drop(["prev_book_mid", "book_changed", "next_close", "next2_close"])

    # ---- 5. 保存 ----
    out_path = f"{OUT_DIR}/features_{exchange}_{SYMBOL}_{DATE}.parquet"
    feat_df.write_parquet(out_path, compression="zstd")
    elapsed = time.time() - t0
    print(f"✅ [{exchange}] 完成！")
    print(f"   输出: {out_path}  ({feat_df.height} 行 × {feat_df.width} 列)")
    print(f"   耗时: {elapsed:.1f}s  (总计 {total_updates:,} 条 update)")
    print(f"   标签分布:")
    print(feat_df["label_dir_1m"].value_counts().sort("label_dir_1m"))
    print(f"   label_return_1m: mean={feat_df['label_return_1m'].mean():.6f}, "
          f"std={feat_df['label_return_1m'].std():.6f}")
    print(f"   mid 范围: {feat_df['book_mid'].min():.2f} ~ {feat_df['book_mid'].max():.2f} "
          f"({feat_df['book_mid'].n_unique()} 个不同值)")
    return feat_df


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    run_for_exchange("futures")
    run_for_exchange("spot")
