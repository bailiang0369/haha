#!/usr/bin/env python3
"""
L2 Orderbook → 10s snapshots 特征矩阵
  - 10s 均匀采样
  - σ 分桶 (rolling 10min std)
  - 7 桶: [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]σ
  - polars 处理, numpy 加速
"""
import argparse, sys, os, time
from pathlib import Path
import polars as pl
import numpy as np

pl.Config.set_fast_parse_dates(True)

BUCKETS = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2026-09-04")
    p.add_argument("--end",   default="2026-10-04")
    p.add_argument("--markets", nargs="+", default=["binance_spot","binance_futures"])
    p.add_argument("--interval", type=int, default=10)
    p.add_argument("--top_n", type=int, default=50)
    return p.parse_args()

def iter_hour_files(root: Path, date_str: str, market: str):
    d = root / market / date_str
    if not d.exists():
        return
    for h in sorted(d.iterdir()):
        if not h.is_dir(): continue
        for f in sorted(h.glob("*.parquet")):
            yield f

def pivot_snapshot(df_hour: pl.LazyFrame, top_n: int) -> pl.LazyFrame:
    """把逐行 depth 表变成 per-update 宽表"""
    # 取每档的 max qty (orderbook 累积快照语义: 同价位大的覆盖小的)
    agg = df_hour.group_by(["update_id","side","price"]).agg(pl.col("quantity").sum())

    bids = agg.filter(pl.col("side")=="bid").sort("update_id").group_by("update_id").agg([
        pl.col("price").sort(descending=True).head(top_n).alias("bid_prices"),
        pl.col("quantity").sort_by("price", descending=True).head(top_n).alias("bid_qtys"),
    ])
    asks = agg.filter(pl.col("side")=="ask").sort("update_id").group_by("update_id").agg([
        pl.col("price").sort(descending=False).head(top_n).alias("ask_prices"),
        pl.col("quantity").sort_by("price", descending=False).head(top_n).alias("ask_qtys"),
    ])

    return bids.join(asks, on="update_id", how="outer_coalesce")

def build_day_snapshots(day_files: list[Path], interval: int, top_n: int) -> pl.DataFrame:
    """处理一天的所有小时文件 → 10s 快照 + 特征"""
    if not day_files:
        return pl.DataFrame()

    # 读所有小时
    dfs = []
    for f in day_files:
        df = pl.read_parquet(f, columns=["update_id","price","quantity","side","final_update_id"]).with_columns(
            pl.col("final_update_id").alias("ts_ms")
        )
        if df.height == 0: continue
        dfs.append(df)

    raw = pl.concat(dfs).lazy()

    # Pivot
    piv = pivot_snapshot(raw, top_n).collect()
    if piv.height == 0:
        return pl.DataFrame()

    # 计算每快照的基础指标
    def list_sum(arr):
        return float(np.sum(arr)) if len(arr) > 0 else 0.0

    def list_first(arr):
        return float(arr[0]) if len(arr) > 0 else float("nan")

    bid_prices = piv["bid_prices"].to_list()
    bid_qtys   = piv["bid_qtys"].to_list()
    ask_prices = piv["ask_prices"].to_list()
    ask_qtys   = piv["ask_qtys"].to_list()
    ts_ms      = piv["update_id"].to_numpy().astype(np.int64)  # update_id 在 CryptoHFTData 就是 ms timestamp

    n = len(ts_ms)
    mid_arr   = np.full(n, np.nan)
    bb_arr    = np.full(n, np.nan)
    ba_arr    = np.full(n, np.nan)
    spread_arr = np.full(n, np.nan)
    spread_bps = np.full(n, np.nan)
    n_bids_arr = np.zeros(n, dtype=np.int32)
    n_asks_arr = np.zeros(n, dtype=np.int32)
    tbq_arr   = np.zeros(n, dtype=np.float64)
    taq_arr   = np.zeros(n, dtype=np.float64)

    for i in range(n):
        bp, bq = bid_prices[i], bid_qtys[i]
        ap, aq = ask_prices[i], ask_qtys[i]
        if len(bp) > 0 and len(ap) > 0:
            bb_arr[i] = bp[0]
            ba_arr[i] = ap[0]
            mid_arr[i] = (bp[0] + ap[0]) / 2
            spread_arr[i] = ap[0] - bp[0]
            spread_bps[i] = spread_arr[i] / mid_arr[i] * 10000
            n_bids_arr[i] = len(bp)
            n_asks_arr[i] = len(ap)
            tbq_arr[i] = np.sum(bq) if len(bq) > 0 else 0
            taq_arr[i] = np.sum(aq) if len(aq) > 0 else 0

    # 构造 DataFrame
    out = pl.DataFrame({
        "ts_ms": ts_ms,
        "mid": mid_arr,
        "best_bid": bb_arr,
        "best_ask": ba_arr,
        "spread": spread_arr,
        "spread_bps": spread_bps,
        "n_bids": n_bids_arr,
        "n_asks": n_asks_arr,
        "total_bid_qty": tbq_arr,
        "total_ask_qty": taq_arr,
        "bid_prices": bid_prices,
        "bid_qtys": bid_qtys,
        "ask_prices": ask_prices,
        "ask_qtys": ask_qtys,
    })

    # 排序 + 去 nan mid
    out = out.sort("ts_ms").filter(pl.col("mid").is_not_nan())
    if out.height == 0:
        return out

    # 按 interval 采样: 对齐到 interval 边界
    out = out.with_columns(
        (pl.col("ts_ms") // (interval * 1000) * (interval * 1000)).alias("bucket")
    ).group_by("bucket").agg(pl.col("*").first()).rename({"bucket":"ts_ms"}).sort("ts_ms")

    return out

def add_bucket_features(df: pl.DataFrame, top_n: int) -> pl.DataFrame:
    """在采样后的快照上加 σ 分桶 + 衍生特征"""
    if df.height < 60:
        return df

    mid = df["mid"].to_numpy()
    bid_prices = df["bid_prices"].to_list()
    bid_qtys   = df["bid_qtys"].to_list()
    ask_prices = df["ask_prices"].to_list()
    ask_qtys   = df["ask_qtys"].to_list()

    # Rolling σ (10min 窗口, 采样间隔 interval s)
    window = max(600 // 10, 30)  # 600s / 10s = 60
    sigma = np.full(len(mid), np.nan)
    for i in range(window, len(mid)):
        sigma[i] = np.std(mid[i-window:i])
    # 前 window 个用全局 std
    sigma[:window] = np.nanstd(mid[:window]) if window < len(mid) else np.nanstd(mid)

    bucket_cols = {}
    for b in BUCKETS:
        bucket_cols[f"bid_vol_-{b}σ"] = np.zeros(len(mid))
        bucket_cols[f"ask_vol_+{b}σ"] = np.zeros(len(mid))

    total_depth = np.zeros(len(mid))

    for i in range(len(mid)):
        m = mid[i]
        s = sigma[i]
        if np.isnan(s) or s < 0.01:
            s = max(m * 0.001, 1.0)

        for j in range(len(BUCKETS)):
            lo = -BUCKETS[j] * s
            hi = +BUCKETS[j] * s
            # bid 侧: price in [m + lo, m]
            for p, q in zip(bid_prices[i], bid_qtys[i]):
                d = p - m
                if lo <= d <= 0:
                    bucket_cols[f"bid_vol_-{BUCKETS[j]}σ"][i] += q
            # ask 侧: price in [m, m + hi]
            for p, q in zip(ask_prices[i], ask_qtys[i]):
                d = p - m
                if 0 <= d <= hi:
                    bucket_cols[f"ask_vol_+{BUCKETS[j]}σ"][i] += q

        total_depth[i] = bucket_cols[f"bid_vol_-5.0σ"][i] + bucket_cols[f"ask_vol_+5.0σ"][i]

    df = df.with_columns([pl.Series(k, v) for k, v in bucket_cols.items()])

    # 失衡特征
    imb_exprs = []
    for b in BUCKETS:
        imb_exprs.append(
            (pl.col(f"bid_vol_-{b}σ") - pl.col(f"ask_vol_+{b}σ"))
            / (pl.col(f"bid_vol_-{b}σ") + pl.col(f"ask_vol_+{b}σ") + 1e-9)
            .alias(f"imb_-{b}σ")
        )
    df = df.with_columns(imb_exprs)

    # top5 depth imbalance
    df = df.with_columns([
        (pl.col("total_bid_qty") - pl.col("total_ask_qty"))
        / (pl.col("total_bid_qty") + pl.col("total_ask_qty") + 1e-9).alias("depth_imbalance_top5"),
        pl.Series("sigma", sigma),
        pl.Series("total_depth", total_depth),
    ])

    # 时间列 + 动量
    df = df.with_columns([
        pl.from_epoch("ts_ms", time_unit="ms").alias("ts"),
        pl.col("mid").shift(1).alias("mid_prev"),
        pl.col("mid").pct_change(1).alias("mid_ret_10s"),
    ])
    df = df.with_columns([
        pl.col("mid").shift(6).alias("mid_prev_60s"),
        pl.col("mid").pct_change(6).alias("mid_ret_1m"),
    ])

    return df

def main():
    args = parse_args()
    root = Path("/workspace/data")
    out  = Path("/workspace/snapshots")
    out.mkdir(parents=True, exist_ok=True)

    dates = pl.date_range(pl.date(args.start), pl.date(args.end), eager=True).to_list()
    dates = [str(d) for d in dates]

    print(f"📅 {args.start} → {args.end}  |  采样 {args.interval}s  |  分桶 {BUCKETS}σ")
    print(f"  输出: {out}\n")

    t0 = time.time()
    total_snap = 0

    for market in args.markets:
        out_mkt = out / market
        out_mkt.mkdir(exist_ok=True)

        for d in dates:
            files = list(iter_hour_files(root, d, market))
            if not files:
                continue
            t1 = time.time()
            snap = build_day_snapshots(files, args.interval, args.top_n)
            if snap.height == 0: continue
            snap = add_bucket_features(snap, args.top_n)
            out_file = out_mkt / f"BTCUSDT_{d}.parquet"
            snap.write_parquet(out_file, compression="zstd")
            elapsed = time.time() - t1
            total_snap += snap.height
            size_kb = out_file.stat().st_size / 1024
            print(f"  {d[:10]} {market:18s} {snap.height:>7,} 快照, {size_kb:.1f}KB, {elapsed:.1f}s")

    total_elapsed = time.time() - t0
    print(f"\n✅ 总耗时 {total_elapsed:.1f}s  |  {total_snap:,} 快照")

if __name__ == "__main__":
    main()
