"""
正确重建 CryptoHFTData L2 orderbook:
  snapshot 初始化 → 逐 update 维护 state → 每秒 last → 10s bar
输出: 每 10s 一个完整 orderbook summary (best_bid, best_ask, mid, spread, top5 imbalance)
"""
import polars as pl
import numpy as np
import glob, os, json, time
from collections import defaultdict
import warnings; warnings.filterwarnings('ignore')

ROOT = "data/binance_spot"
OUT = "data/correct_l2"
os.makedirs(OUT, exist_ok=True)

def rebuild_day(day_dir):
    """重建一天的数据, 返回每秒一个 state"""
    date_str = os.path.basename(day_dir)
    out_f = f"{OUT}/{date_str}.npz"
    if os.path.exists(out_f):
        print(f"  {date_str}: 已存在")
        return np.load(out_f, allow_pickle=True)
    
    files = sorted(glob.glob(f"{day_dir}/**/BTCUSDT_orderbook.parquet", recursive=True))
    all_et, all_bb, all_ba, all_vol_imbalance, all_top5_imbalance = [], [], [], [], []
    
    for f in files:
        try:
            df = pl.read_parquet(f).with_columns([
                pl.col("price").cast(pl.Float64),
                pl.col("quantity").cast(pl.Float64),
            ])
        except Exception as e:
            print(f"  skip {f}: {e}")
            continue
        
        snap = df.filter(pl.col("event_type") == "snapshot")
        updates = df.filter(pl.col("event_type") == "update").sort(["final_update_id","side","price"])
        
        if snap.height == 0: continue
        
        # snapshot 初始化
        bids_state = defaultdict(float)
        asks_state = defaultdict(float)
        for row in snap.iter_rows(named=True):
            if row['quantity'] > 0:
                if row['side'] == 'bid': bids_state[row['price']] = row['quantity']
                else: asks_state[row['price']] = row['quantity']
        
        # 初始快照的指标
        et0 = int(snap['event_time'][0])
        if bids_state and asks_state:
            bb0 = max(bids_state); ba0 = min(asks_state)
            top5_bid_vol = sum(sorted(bids_state.values(), reverse=True)[:5])
            top5_ask_vol = sum(sorted(asks_state.values(), reverse=True)[:5])
            all_et.append(et0); all_bb.append(bb0); all_ba.append(ba0)
            all_vol_imbalance.append((top5_bid_vol - top5_ask_vol) / (top5_bid_vol + top5_ask_vol + 1e-9))
            all_top5_imbalance.append((top5_bid_vol - top5_ask_vol) / (top5_bid_vol + top5_ask_vol + 1e-9))
        
        # updates 逐 fid 批量应用
        fid_groups = updates.group_by("final_update_id").agg([
            pl.col("event_time").first().alias("et_ms"),
            pl.col("side").alias("sides"),
            pl.col("price").alias("prices"),
            pl.col("quantity").alias("qtys"),
        ])
        
        for row in fid_groups.iter_rows(named=True):
            et_ms = int(row['et_ms'])
            for side, price, qty in zip(row['sides'], row['prices'], row['qtys']):
                if qty == 0:
                    if side == 'bid' and price in bids_state: del bids_state[price]
                    elif side == 'ask' and price in asks_state: del asks_state[price]
                else:
                    if side == 'bid': bids_state[price] = qty
                    else: asks_state[price] = qty
            
            if not bids_state or not asks_state: continue
            
            bb = max(bids_state); ba = min(asks_state)
            top5_bv = sum(sorted(bids_state.values(), reverse=True)[:5])
            top5_av = sum(sorted(asks_state.values(), reverse=True)[:5])
            total_bv = sum(bids_state.values())
            total_av = sum(asks_state.values())
            
            all_et.append(et_ms)
            all_bb.append(bb)
            all_ba.append(ba)
            all_vol_imbalance.append((total_bv - total_av) / (total_bv + total_av + 1e-9))
            all_top5_imbalance.append((top5_bv - top5_av) / (top5_bv + top5_av + 1e-9))
    
    if not all_et:
        return None
    
    # 排序 + 去重
    idx = np.argsort(all_et)
    arr = dict(
        et=np.array(all_et)[idx],
        bb=np.array(all_bb)[idx],
        ba=np.array(all_ba)[idx],
        vol_imb=np.array(all_vol_imbalance)[idx],
        top5_imb=np.array(all_top5_imbalance)[idx],
    )
    np.savez(out_f, **arr)
    print(f"  {date_str}: {len(all_et)} states → {out_f}")
    return arr

# 重建所有天
day_dirs = sorted(glob.glob(f"{ROOT}/2026-*"))
print(f"📦 需要重建 {len(day_dirs)} 天")
for d in day_dirs:
    rebuild_day(d)

# 加载所有天 → 每秒 last → 10s bar
print(f"\n📊 加载所有天 → 10s bar...")
all_10s = []
for f in sorted(glob.glob(f"{OUT}/*.npz")):
    data = np.load(f, allow_pickle=True)
    et, bb, ba, vi, ti = data['et'], data['bb'], data['ba'], data['vol_imb'], data['top5_imb']
    mid = (bb + ba) / 2
    
    # 每秒 last
    ts_s = et // 1000
    uniq_s, idx_s = np.unique(ts_s, return_index=True)
    s_min, s_max = int(uniq_s[0]), int(uniq_s[-1])
    all_s = np.arange(s_min, s_max + 1)
    
    mid_1s = np.full(len(all_s), np.nan)
    bb_1s = np.full(len(all_s), np.nan)
    ba_1s = np.full(len(all_s), np.nan)
    vi_1s = np.full(len(all_s), np.nan)
    ti_1s = np.full(len(all_s), np.nan)
    s_to_idx = {int(s): i for i, s in enumerate(uniq_s)}
    for j, s in enumerate(all_s):
        if int(s) in s_to_idx:
            orig_i = idx_s[s_to_idx[int(s)]]
            mid_1s[j] = mid[orig_i]; bb_1s[j] = bb[orig_i]; ba_1s[j] = ba[orig_i]
            vi_1s[j] = vi[orig_i]; ti_1s[j] = ti[orig_i]
    
    # forward fill
    valid = ~np.isnan(mid_1s)
    if not valid.any(): continue
    fi = np.where(valid)[0][0]
    for arr_1s in [mid_1s, bb_1s, ba_1s, vi_1s, ti_1s]:
        arr_1s[:fi] = arr_1s[fi]
    for j in range(fi+1, len(mid_1s)):
        if np.isnan(mid_1s[j]):
            mid_1s[j] = mid_1s[j-1]; bb_1s[j] = bb_1s[j-1]; ba_1s[j] = ba_1s[j-1]
            vi_1s[j] = vi_1s[j-1]; ti_1s[j] = ti_1s[j-1]
    
    # 10s: 每 10s 取最后一秒
    bucket = all_s // 10 * 10
    uniq_bk, last_idx = np.unique(bucket, return_index=True)
    all_10s.append((
        uniq_bk * 1000,
        mid_1s[last_idx], bb_1s[last_idx], ba_1s[last_idx],
        vi_1s[last_idx], ti_1s[last_idx]
    ))

# 拼接
ts_ms = np.concatenate([x[0] for x in all_10s])
mid = np.concatenate([x[1] for x in all_10s])
bb = np.concatenate([x[2] for x in all_10s])
ba = np.concatenate([x[3] for x in all_10s])
vi = np.concatenate([x[4] for x in all_10s])
ti = np.concatenate([x[5] for x in all_10s])

# 去重
uniq_ts, last_idx = np.unique(ts_ms, return_index=True)
ts_ms, mid, bb, ba, vi, ti = uniq_ts, mid[last_idx], bb[last_idx], ba[last_idx], vi[last_idx], ti[last_idx]

np.savez(f"{OUT}/all_10s.npz", ts_ms=ts_ms, mid=mid, bb=bb, ba=ba, vol_imb=vi, top5_imb=ti)

print(f"\n✅ 全部 10s bar: {len(mid)}")
print(f"  bb>ba: {(bb>ba).sum()}")
print(f"  spread unique: {np.unique(ba-bb)[:10]}")
diffs = np.abs(np.diff(mid))
print(f"  |d|>0.01: {(diffs>0.01).sum()} ({(diffs>0.01).mean()*100:.1f}%)")
print(f"  |d|>1: {(diffs>1).sum()} ({(diffs>1).mean()*100:.3f}%)")
print(f"  ret std: {np.nanstd(diffs/mid[:-1])*10000:.4f} bps")
print(f"  vol_imb range: [{vi.min():.3f}, {vi.max():.3f}], std={vi.std():.3f}")
print(f"  top5_imb range: [{ti.min():.3f}, {ti.max():.3f}], std={ti.std():.3f}")

# 保存完整 summary
summary = pl.DataFrame({
    "ts_ms": ts_ms, "mid": mid, "bb": bb, "ba": ba,
    "spread_bps": (ba - bb) / mid * 10000,
    "vol_imb": vi, "top5_imb": ti,
    "date": pl.from_epoch(ts_ms // 1000, unit="s").dt.date().cast(pl.Utf8),
})
summary.write_parquet(f"{OUT}/l2_summary.parquet")
print(f"\n📁 已保存: {OUT}/l2_summary.parquet")

