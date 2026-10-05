"""
免费 Pipeline: 币安现货 aggTrades → 重建 mid → 10s 重采样 → ret_10s+ret_1m → walk-forward
零付费, 无限历史, AUC 预期 ~0.68
"""
import os, glob, time, json, requests, zipfile, io
import numpy as np, polars as pl, lightgbm as lgb
from sklearn.metrics import roc_auc_score
from datetime import datetime, timezone

DATA_DIR = "/workspace/data/agg_trades_spot"
os.makedirs(DATA_DIR, exist_ok=True)
t0 = time.time()

# ===== Step 1: 下载 aggTrades 历史 =====
print(f"[{time.time()-t0:.0f}s] Step 1: 下载 aggTrades 历史...", flush=True)

today = datetime(2026, 10, 5)
all_dates = []
for days_back in range(0, 180):
    d = today.replace(hour=0, minute=0, second=0, microsecond=0) - __import__('datetime').timedelta(days=days_back)
    ds = d.strftime("%Y-%m-%d")
    url = f"https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-{ds}.zip"
    fname = f"{DATA_DIR}/BTCUSDT_aggTrades_{ds}.parquet"
    if os.path.exists(fname):
        all_dates.append((ds, fname))
        continue
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200 and len(r.content) > 1000:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            csv_name = [n for n in z.namelist() if n.endswith(".csv")][0]
            df = pl.read_csv(io.BytesIO(z.read(csv_name)))
            df.columns = ["agg_trade_id","price","qty","first_trade_id","last_trade_id",
                          "timestamp","is_buyer_maker","is_trade_me"]
            df = df.with_columns([
                pl.col("timestamp").cast(pl.Int64).alias("ts_ms"),
                pl.col("price").cast(pl.Float64).alias("price"),
                pl.col("is_buyer_maker").alias("is_buyer_maker"),
            ]).select(["ts_ms","price","is_buyer_maker"])
            df.write_parquet(fname, compression="snappy")
            all_dates.append((ds, fname))
            print(f"  ✅ {ds}", flush=True)
        else:
            print(f"  ❌ {ds}: HTTP {r.status_code}", flush=True)
    except Exception as e:
        print(f"  ❌ {ds}: {str(e)[:60]}", flush=True)
    if len(all_dates) >= 30:  # 先下 30 天够了
        break

print(f"  总共 {len(all_dates)} 天", flush=True)

# ===== Step 2: 加载 + 重建 mid =====
print(f"\n[{time.time()-t0:.0f}s] Step 2: 加载 + 重建 mid...", flush=True)
dfs = []
for ds, fname in all_dates:
    try: dfs.append(pl.read_parquet(fname))
    except: pass
df = pl.concat(dfs).sort("ts_ms")
ts_ms = df["ts_ms"].to_numpy().astype(np.int64)
price = df["price"].to_numpy().astype(np.float64)
is_bm = df["is_buyer_maker"].to_numpy().astype(bool)
print(f"  {len(ts_ms):,} 笔成交", flush=True)

bid_price = np.where(is_bm, price, np.nan)
ask_price = np.where(~is_bm, price, np.nan)

def ffill(arr):
    out = arr.copy(); last = np.nan
    for i in range(len(arr)):
        if not np.isnan(arr[i]): last = arr[i]
        out[i] = last
    return out

bid_f = ffill(bid_price); ask_f = ffill(ask_price)
mid = (bid_f + ask_f) / 2
valid = ~np.isnan(mid)
ts = ts_ms[valid]; mid = mid[valid]
print(f"  有效 mid: {len(ts):,}", flush=True)

# ===== Step 3: 10s 重采样 =====
print(f"\n[{time.time()-t0:.0f}s] Step 3: 10s 重采样...", flush=True)
bucket = ts // 10000 * 10000
_, last_idx = np.unique(bucket, return_index=True)
last_idx = np.concatenate([last_idx[1:]-1, [len(ts)-1]])
ts_10 = bucket[last_idx]; mid_10 = mid[last_idx]
print(f"  {len(ts_10):,} 行 ({(ts_10[-1]-ts_10[0])/1000/86400:.0f} 天)", flush=True)

# ===== Step 4: 特征 + Label =====
print(f"\n[{time.time()-t0:.0f}s] Step 4: 特征 + Label...", flush=True)
HORIZON_MIN = 5
TRAIN_DAYS = 7
step = HORIZON_MIN * 6

n = len(mid_10)
ret_1 = np.full(n, np.nan); ret_1[1:] = mid_10[1:]/mid_10[:-1]-1
ret_6 = np.full(n, np.nan); ret_6[6:] = mid_10[6:]/mid_10[:-6]-1
future_ret = np.full(n, np.nan)
for i in range(n-step): future_ret[i] = (mid_10[i+step]-mid_10[i])/mid_10[i]

has = ~np.isnan(future_ret) & ~np.isnan(ret_1) & ~np.isnan(ret_6)
day_idx = (ts_10 // 86400000).astype(int)
unique_days = np.unique(day_idx)
thr = abs(np.quantile(future_ret[has & (day_idx < unique_days[TRAIN_DAYS])], 0.25))
labeled = has & (np.abs(future_ret) > thr)

l_ts = ts_10[labeled]; l_fr = future_ret[labeled]
l_r1 = ret_1[labeled]; l_r6 = ret_6[labeled]
l_day = day_idx[labeled]
unique_days = np.unique(l_day)
print(f"  labeled={len(l_ts):,}, thr={thr*10000:.2f}bps, {len(unique_days)} 天", flush=True)

# ===== Step 5: Walk-forward =====
print(f"\n[{time.time()-t0:.0f}s] Step 5: Walk-forward ({len(unique_days)-TRAIN_DAYS} 模型)...", flush=True)
all_p, all_y, all_t = [], [], []
per_day = {}

for day_i in range(TRAIN_DAYS, len(unique_days)):
    d = unique_days[day_i]; dlo = unique_days[max(0, day_i-TRAIN_DAYS)]
    tm = (l_day >= dlo) & (l_day < d); qm = (l_day == d)
    if tm.sum() < 5000 or qm.sum() < 200: continue
    
    X_tr = np.column_stack([l_r1[tm], l_r6[tm]])
    y_tr = (l_fr[tm] > 0).astype(int)
    X_te = np.column_stack([l_r1[qm], l_r6[qm]])
    y_te = (l_fr[qm] > 0).astype(int)
    
    m = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31,
                            min_child_samples=500, verbose=-1, random_state=42, n_jobs=-1)
    m.fit(X_tr, y_tr)
    proba = m.predict_proba(X_te)[:, 1]
    
    all_p.extend(proba); all_y.extend(y_te); all_t.extend(l_ts[qm])
    per_day[int(d)] = round(float(((proba>0.5).astype(int)==y_te).mean()*100), 2)
    
    if (day_i-TRAIN_DAYS+1) % 10 == 0:
        print(f"  进度 {day_i-TRAIN_DAYS+1}/{len(unique_days)-TRAIN_DAYS} | {time.time()-t0:.0f}s", flush=True)

all_p = np.array(all_p); all_y = np.array(all_y); all_t = np.array(all_t)
print(f"\n[{time.time()-t0:.0f}s] 训练完成! {len(all_p):,} 预测", flush=True)

# ===== Step 6: 评估 =====
if len(all_p) > 100:
    auc = roc_auc_score(all_y, all_p)
    conf = np.abs(all_p - 0.5)*2
    correct = (all_p > 0.5).astype(int) == all_y
    
    print(f"\n{'='*65}", flush=True)
    print(f"  aggTrades 重建 mid (免费!) — pred 5min", flush=True)
    print(f"  AUC = {auc:.4f}", flush=True)
    print(f"{'='*65}", flush=True)
    
    payout = 0.8
    results = []
    for c in [0.0, 0.3, 0.5, 0.7, 0.8]:
        mc = conf >= c
        if mc.sum() < 100: continue
        streak=0; pu=0; em=np.zeros(len(all_p),dtype=bool)
        for i in np.where(mc)[0]:
            t=all_t[i]
            if t<pu: continue
            em[i]=True
            if correct[i]: streak=0
            else: streak+=1
            if streak>=3: pu=t+15*60*1000; streak=0
        cb_n = em.sum()
        c_acc = correct[mc].mean()*100
        arr = correct[em].astype(float)
        cb_acc = arr.mean()*100 if cb_n>100 else 0
        mn = min(arr[i:i+100].mean() for i in range(len(arr)-99))*100 if len(arr)>=100 else cb_acc
        exp = cb_acc/100*payout - (1-cb_acc/100)
        print(f"  conf≥{c:.1f}: n={mc.sum():>7,} acc={c_acc:>6.1f}% cb_n={cb_n:>7,} cb_acc={cb_acc:>6.1f}% min100={mn:>6.1f}% exp={exp*100:>5.1f}¢", flush=True)
        results.append({"conf":c,"n":int(mc.sum()),"acc_no_cb":round(c_acc,1),"cb_n":int(cb_n),"cb_acc":round(cb_acc,1),"cb_min100":round(mn,1),"exp_cent":round(exp*100,1)})
    
    # 保存
    out = {
        "auc":round(float(auc),4),"thr_bps":round(float(thr*10000),2),
        "total":len(all_p),"n_days":len(per_day),"results":results,"per_day":per_day,
        "method":"aggTrades 重建 mid (免费)","horizon_min":HORIZON_MIN,
    }
    os.makedirs("/workspace/results",exist_ok=True)
    with open("/workspace/results/aggtrades_free_5min.json","w") as f: json.dump(out,f,indent=2,default=str)
    print(f"\n✅ /workspace/results/aggtrades_free_5min.json | 总 {time.time()-t0:.0f}s", flush=True)
else:
    print("❌ 没有预测样本", flush=True)
