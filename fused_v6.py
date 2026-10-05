"""L2 v6 + aggTrades 融合回测 (干净版)"""
import os, glob, time, numpy as np, polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

t0=time.time(); BUCKET_MS=10_000; STEP_3MIN=18; TRAIN_DAYS=7

# 1. L2 v6 cache
print(f"[{time.time()-t0:.0f}s] 加载 L2 v6...")
l2 = pl.read_parquet('/workspace/data/l2_spot_mid_10s_v6.parquet').filter(pl.col('spread_bps') > -10)
l2_b = l2['bucket'].to_numpy(); l2_m = l2['mid'].to_numpy(); l2_sp = l2['spread_bps'].to_numpy()
del l2

# 2. aggTrades
print(f"[{time.time()-t0:.0f}s] 加载 aggTrades...")
agg_files = sorted(glob.glob('/workspace/data/agg_trades_full/*.parquet'))
dfs_a = [pl.read_parquet(f) for f in agg_files]
col_ts = 'ts_ms' if 'ts_ms' in dfs_a[0].columns else 'ts_us'
df_a = pl.concat(dfs_a).sort(col_ts)
ts_us = df_a[col_ts].to_numpy(); price_a = df_a['price'].to_numpy()
qty_a = df_a['qty'].to_numpy(); is_bm = df_a['is_bm'].to_numpy()
del df_a, dfs_a

bucket_a = (ts_us // 1000) // BUCKET_MS * BUCKET_MS
del ts_us
df_a2 = pl.DataFrame({'b':bucket_a,'p':price_a,'q':qty_a,'bm':is_bm.astype(int)})
del bucket_a, price_a, qty_a, is_bm

bars_a = df_a2.group_by('b').agg([
    pl.col('p').max().alias('h'), pl.col('p').min().alias('l'), pl.col('p').last().alias('c'),
    pl.col('q').sum().alias('vol'),
    (pl.col('q')*pl.col('bm')).sum().alias('vbm'),
    (pl.col('q')*(1-pl.col('bm'))).sum().alias('vtk'),
    ((pl.col('q')>1)*(1-pl.col('bm'))).sum().alias('nlg_tk'),
    (pl.col('q')>1).sum().alias('nlg'),
]).sort('b')
del df_a2

ab_b = bars_a['b'].to_numpy(); ab_c = bars_a['c'].to_numpy()
ab_h = bars_a['h'].to_numpy(); ab_l = bars_a['l'].to_numpy(); ab_vol = bars_a['vol'].to_numpy()
ab_vbm = bars_a['vbm'].to_numpy(); ab_vtk = bars_a['vtk'].to_numpy()
ab_nlg_tk = bars_a['nlg_tk'].to_numpy(); ab_nlg = bars_a['nlg'].to_numpy()
del bars_a

# 3. Align
common = np.intersect1d(l2_b, ab_b); ts = common; n = len(ts)
print(f"[{time.time()-t0:.0f}s] 对齐: {len(ts):,}")

def align(sb, sv, tb):
    idx = np.searchsorted(sb, tb); v = (idx < len(sb)) & (sb[idx]==tb)
    out = np.full(len(tb), np.nan); out[v] = sv[idx[v]]; return out

m = align(l2_b, l2_m, ts); sp = align(l2_b, l2_sp, ts)
del l2_b, l2_m, l2_sp

imb_b = align(ab_b, (ab_vtk-ab_vbm)/(ab_vol+1e-12), ts)
sv = align(ab_b, ab_vtk-ab_vbm, ts)
tk_f = align(ab_b, ab_vtk/(ab_vol+1e-12), ts)
lg_tk = align(ab_b, ab_nlg_tk-(ab_nlg-ab_nlg_tk), ts)
park = align(ab_b, (ab_h-ab_l)/(ab_c+1e-12), ts)
lv = align(ab_b, np.log(ab_vol+1), ts)
del ab_b, ab_c, ab_h, ab_l, ab_vol, ab_vbm, ab_vtk, ab_nlg_tk, ab_nlg

# 4. Rolling
def rs(x, w):
    x0 = np.where(np.isnan(x), 0.0, np.array(x, dtype=np.float64))
    cs = np.concatenate([[0.0], np.cumsum(x0)])
    out = np.full(len(x), np.nan); out[w-1:] = cs[w:] - cs[:-w]
    cnt = np.concatenate([[0.0], np.cumsum(np.where(np.isnan(x), 0.0, 1.0))])
    c2 = np.full(len(x), 0.0); c2[w-1:] = cnt[w:] - cnt[:-w]
    out[c2 < w*0.5] = np.nan; return out
def rm(x, w):
    s = rs(x, w); c = rs(np.where(np.isnan(x), 0.0, 1.0), w)
    mask = c >= w*0.5; r = np.full(len(x), np.nan); r[mask] = s[mask]/c[mask]; return r

l2_r1 = np.full(n, np.nan); l2_r1[1:] = m[1:]/m[:-1]-1
l2_r6 = np.full(n, np.nan); l2_r6[6:] = m[6:]/m[:-6]-1
l2_r18 = np.full(n, np.nan); l2_r18[18:] = m[18:]/m[:-18]-1
l2_sp_r6 = rm(sp, 6)

sv6 = rs(sv, 6); sv18 = rs(sv, 18)
tk6 = rm(tk_f, 6); tk18 = rm(tk_f, 18)
lg6 = rs(lg_tk, 6); lg18 = rs(lg_tk, 18)
park6 = rm(park, 6); lv6 = rm(lv, 6)

# future
future = np.full(n, np.nan)
for i in range(n - STEP_3MIN):
    a, bv = m[i], m[i+STEP_3MIN]
    if not np.isnan(a) and not np.isnan(bv) and a > 0: future[i] = (bv - a) / a

# 5. Labels + WF
print(f"[{time.time()-t0:.0f}s] Walk-Forward...")
NAMES = ['l2_r1','l2_r6','l2_r18','l2_sp','l2_sp_r6',
         'sv6','sv18','tk6','tk18','lg6','lg18','park6','lv6']
MATS = [l2_r1,l2_r6,l2_r18,sp,l2_sp_r6,
        sv6,sv18,tk6,tk18,lg6,lg18,park6,lv6]
X = np.column_stack(MATS)

day_idx = (ts // 86400000).astype(int); u_days = np.unique(day_idx)
has = ~np.isnan(future)
for ci in range(X.shape[1]): has &= ~np.isnan(X[:, ci])
thr = abs(np.quantile(future[has & (day_idx < u_days[TRAIN_DAYS])], 0.25))
labeled = has & (np.abs(future) > thr)
l_day = day_idx[labeled]; l_fr = future[labeled]
print(f"  labeled: {labeled.sum():,}, thr={thr*10000:.2f}bps, {len(u_days)} days")

def run_wf(idx_list, desc):
    Xs = X[labeled][:, idx_list]
    all_p, all_y, imp = [], [], np.zeros(Xs.shape[1])
    nm = 0
    for di in range(TRAIN_DAYS, len(u_days)):
        d, dlo = u_days[di], u_days[max(0, di-TRAIN_DAYS)]
        tm = (l_day >= dlo) & (l_day < d); qm = l_day == d
        if tm.sum() < 2000 or qm.sum() < 100: continue
        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                    min_child_samples=200, verbose=-1, random_state=42, n_jobs=-1)
        model.fit(Xs[tm], (l_fr[tm] > 0).astype(int))
        all_p.extend(model.predict_proba(Xs[qm])[:, 1])
        all_y.extend((l_fr[qm] > 0).astype(int))
        imp += model.feature_importances_; nm += 1
    if len(all_p) < 100: print(f"  {desc}: ❌"); return None, None
    ap = np.array(all_p); ay = np.array(all_y)
    auc = roc_auc_score(ay, ap)
    conf = np.abs(ap - 0.5) * 2; acc = (ap > 0.5).astype(int) == ay
    print(f"\n  [{desc}] {nm}m, {len(ap):,}p, AUC={auc:.4f}")
    for c in [0.0, 0.3, 0.5, 0.6, 0.7]:
        mc = conf >= c
        if mc.sum() < 50: continue
        print(f"    conf≥{c:.1f}: n={mc.sum():>7,}, acc={acc[mc].mean()*100:>5.1f}%")
    return auc, imp / max(nm, 1)

print(f"\n{'='*60}\n3min WF (v6 L2 + aggTrades)\n{'='*60}")

auc1, _ = run_wf([NAMES.index('l2_r1'), NAMES.index('l2_r6')], "① ret_10s+ret_1m (L2)")
auc2, _ = run_wf([NAMES.index(n) for n in ['sv6','sv18','tk6','tk18','lg6','lg18','park6','lv6']], "② aggTrades-only")
auc3, _ = run_wf([NAMES.index(n) for n in ['l2_r1','l2_r6','l2_r18','l2_sp','l2_sp_r6']], "③ L2-only")
auc4, imp = run_wf(list(range(len(NAMES))), "④ FUSED (全部 13 feat)")

if imp is not None:
    ord_ = np.argsort(-imp)
    print(f"\nTop 10:")
    for r, i in enumerate(ord_[:10]):
        print(f"  #{r+1:2d} {NAMES[i]:<15} gain={imp[i]:.1f}")

print(f"\n总耗时: {time.time()-t0:.0f}s")
print(f"\n{'='*60}")
for desc, a in [("① ret baseline",auc1),("② agg-only",auc2),("③ L2-only",auc3),("④ FUSED",auc4)]:
    print(f"  {desc:<20} AUC={a:.4f}")
PYEOF