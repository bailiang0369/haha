#!/usr/bin/env python3
"""
CryptoHFTData L2 → 10s 快照特征矩阵
  - polars lazy + streaming engine (OOM 安全)
  - 10s 均匀采样
  - σ 分桶 (rolling 10min std)
"""
import argparse, os, sys, time
from pathlib import Path
from datetime import datetime, timedelta
import polars as pl
import numpy as np

BUCKETS = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]

def iter_hour_files(root: Path, date_str: str, market: str):
    d = root / market / date_str
    if not d.exists():
        return
    for h in sorted(d.iterdir()):
        if not h.is_dir(): continue
        for f in sorted(h.glob("*.parquet")):
            yield f

def pivot_hour(f: Path, top_n: int) -> pl.DataFrame:
    """用 lazy + streaming pivot 一个 hourly 文件"""
    df = pl.scan_parquet(f).with_columns([
        pl.col("price").cast(pl.Float64),
        pl.col("quantity").cast(pl.Float64),
        (pl.col("received_time") / 1_000_000).cast(pl.Int64).alias("ts_ms"),
    ])

    bids = df.filter(pl.col("side") == "bid").group_by(["final_update_id", "ts_ms"]).agg([
        pl.col("price").sort(descending=True).head(top_n).alias("bid_prices"),
        pl.col("quantity").sort_by("price", descending=True).head(top_n).alias("bid_qtys"),
    ]).select(["final_update_id", "ts_ms", "bid_prices", "bid_qtys"])

    asks = df.filter(pl.col("side") == "ask").group_by(["final_update_id"]).agg([
        pl.col("price").sort(descending=False).head(top_n).alias("ask_prices"),
        pl.col("quantity").sort_by("price", descending=False).head(top_n).alias("ask_qtys"),
    ]).select(["final_update_id", "ask_prices", "ask_qtys"])

    return bids.join(asks, on="final_update_id", how="left").collect(engine="streaming")

def build_day(day_files: list[Path], interval: int, top_n: int) -> pl.DataFrame:
    """每小时 pivot → 立即采样 → 再 concat 小表"""
    sampled_hours = []

    for f in day_files:
        piv = pivot_hour(f, top_n)
        if piv.height == 0:
            continue

        bid_prices = piv["bid_prices"].to_list()
        bid_qtys   = piv["bid_qtys"].to_list()
        ask_prices = piv["ask_prices"].to_list()
        ask_qtys   = piv["ask_qtys"].to_list()
        ts_ms      = piv["ts_ms"].to_numpy().astype(np.int64)

        n = len(ts_ms)
        mid_arr    = np.full(n, np.nan)
        bb_arr     = np.full(n, np.nan)
        ba_arr     = np.full(n, np.nan)
        spread_arr = np.full(n, np.nan)
        spread_bps = np.full(n, np.nan)
        n_bids_arr = np.zeros(n, dtype=np.int32)
        n_asks_arr = np.zeros(n, dtype=np.int32)
        tbq_arr    = np.zeros(n, dtype=np.float64)
        taq_arr    = np.zeros(n, dtype=np.float64)

        for i in range(n):
            bp = bid_prices[i] if bid_prices[i] is not None else []
            bq = bid_qtys[i] if bid_qtys[i] is not None else []
            ap = ask_prices[i] if ask_prices[i] is not None else []
            aq = ask_qtys[i] if ask_qtys[i] is not None else []
            if len(bp) > 0 and len(ap) > 0:
                bb_arr[i] = bp[0]; ba_arr[i] = ap[0]
                mid_arr[i] = (bp[0] + ap[0]) / 2
                spread_arr[i] = ap[0] - bp[0]
                spread_bps[i] = spread_arr[i] / mid_arr[i] * 10000
                n_bids_arr[i] = len(bp); n_asks_arr[i] = len(ap)
                tbq_arr[i] = np.sum(bq) if len(bq) > 0 else 0
                taq_arr[i] = np.sum(aq) if len(aq) > 0 else 0

        hourly = pl.DataFrame({
            "ts_ms": ts_ms, "mid": mid_arr, "best_bid": bb_arr, "best_ask": ba_arr,
            "spread": spread_arr, "spread_bps": spread_bps,
            "n_bids": n_bids_arr, "n_asks": n_asks_arr,
            "total_bid_qty": tbq_arr, "total_ask_qty": taq_arr,
            "bid_prices": bid_prices, "bid_qtys": bid_qtys,
            "ask_prices": ask_prices, "ask_qtys": ask_qtys,
        }).filter(pl.col("mid").is_not_nan())

        if hourly.height == 0:
            continue

        # 对这一小时内采样
        sampled = hourly.with_columns(
            (pl.col("ts_ms") // (interval * 1000) * (interval * 1000)).alias("bucket")
        ).drop("ts_ms").group_by("bucket").agg(pl.col("*").first()).rename({"bucket": "ts_ms"})

        if sampled.height > 0:
            sampled_hours.append(sampled)

    if not sampled_hours:
        return pl.DataFrame()

    out = pl.concat(sampled_hours).sort("ts_ms")
    return out

def add_bucket_features(df: pl.DataFrame) -> pl.DataFrame:
    """σ 分桶 + 衍生特征"""
    if df.height < 60:
        return df

    mid = df["mid"].to_numpy()
    bid_prices = df["bid_prices"].to_list()
    bid_qtys   = df["bid_qtys"].to_list()
    ask_prices = df["ask_prices"].to_list()
    ask_qtys   = df["ask_qtys"].to_list()

    window = 60
    sigma = np.full(len(mid), np.nan)
    for i in range(window, len(mid)):
        sigma[i] = np.std(mid[i-window:i])
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
            for p, q in zip(bid_prices[i] or [], bid_qtys[i] or []):
                d = p - m
                if lo <= d <= 0:
                    bucket_cols[f"bid_vol_-{BUCKETS[j]}σ"][i] += q
            for p, q in zip(ask_prices[i] or [], ask_qtys[i] or []):
                d = p - m
                if 0 <= d <= hi:
                    bucket_cols[f"ask_vol_+{BUCKETS[j]}σ"][i] += q
        total_depth[i] = bucket_cols[f"bid_vol_-5.0σ"][i] + bucket_cols[f"ask_vol_+5.0σ"][i]

    df = df.with_columns([pl.Series(k, v) for k, v in bucket_cols.items()])

    imb_exprs = []
    for b in BUCKETS:
        imb_exprs.append(
            (pl.col(f"bid_vol_-{b}σ") - pl.col(f"ask_vol_+{b}σ"))
            / (pl.col(f"bid_vol_-{b}σ") + pl.col(f"ask_vol_+{b}σ") + 1e-9)
            .alias(f"imb_-{b}σ")
        )
    df = df.with_columns(imb_exprs)

    df = df.with_columns([
        (pl.col("total_bid_qty") - pl.col("total_ask_qty"))
        / (pl.col("total_bid_qty") + pl.col("total_ask_qty") + 1e-9).alias("depth_imbalance_top5"),
        pl.Series("sigma", sigma),
        pl.Series("total_depth", total_depth),
        pl.from_epoch("ts_ms", time_unit="ms").alias("ts"),
        pl.col("mid").pct_change(1).alias("mid_ret_10s"),
        pl.col("mid").pct_change(6).alias("mid_ret_1m"),
    ])

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-09-04")
    parser.add_argument("--end",   default="2026-10-04")
    parser.add_argument("--markets", nargs="+", default=["binance_spot", "binance_futures"])
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--top_n", type=int, default=50)
    args = parser.parse_args()

    root = Path("/workspace/data")
    out  = Path("/workspace/snapshots")
    out.mkdir(parents=True, exist_ok=True)

    s = datetime.strptime(args.start, "%Y-%m-%d").date()
    e = datetime.strptime(args.end, "%Y-%m-%d").date()
    dates = []
    d = s
    while d <= e:
        dates.append(str(d))
        d += timedelta(days=1)

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
            snap = build_day(files, args.interval, args.top_n)
            if snap.height == 0:
                continue
            snap = add_bucket_features(snap)
            out_file = out_mkt / f"BTCUSDT_{d}.parquet"
            snap.write_parquet(out_file, compression="zstd")
            elapsed = time.time() - t1
            total_snap += snap.height
            size_kb = out_file.stat().st_size / 1024
            print(f"  {d} {market:18s} {snap.height:>7,} 快照, {size_kb:.1f}KB, {elapsed:.1f}s")

    print(f"\n✅ 总耗时 {time.time()-t0:.1f}s  |  {total_snap:,} 快照")

if __name__ == "__main__":
    main()
