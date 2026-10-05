"""
L2 Spot Orderbook + aggTrades Spot 融合回测
目标: 3min 涨跌预测, Walk-Forward 7d→1d
对比: Ret-only | aggTrades-only | L2-only | FUSED
"""
import os, glob, time, sys
import numpy as np, polars as pl
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

t0 = time.time()
BUCKET_MS = 10_000
STEP_3MIN = 18
TRAIN_DAYS = 7

def log(msg): print(msg, flush=True)

# ========== 1. 加载 L2 cache ==========
log(f"[{time.time()-t0:.0f}s] 1. 加载 L2 cache...")
l2 = pl.read_parquet('/workspace/data/l2_spot_mid_10s.parquet')
l2 = l2.filter(pl.col('spread_bps') > -10)  # 过滤坏 bucket
l2_b = l2['bucket'].to_numpy().astype(np.int64)
l2_m = l2['mid'].to_numpy().astype(np.float64)
l2_sp = l2['spread_bps'].to_numpy().astype(np.float64)
l2_bk = l2['book_imb'].to_numpy().astype(np.float64)
l2_nu = l2['nu'].to_numpy().astype(np.float64)
del l2

# ========== 2. 加载 aggTrades ==========
log(f"[{time.time()-t0:.0f}s] 2. 加载 aggTrades...")
agg_files = sorted(glob.glob('/workspace/data/agg_trades_full/*.parquet'))
dfs_a = [pl.read_parquet(f) for f in agg_files]
df_a = pl.concat(dfs_a).sort('ts_ms' if 'ts_ms' in dfs_a[0].columns else 'ts_us')
col_ts = 'ts_ms' if 'ts_ms' in dfs_a[0].columns else 'ts_us'
ts_us = df_a[col_ts].to_numpy().astype(np.int64)
price_a = df_a['price'].to_numpy().astype(np.float64)
qty_a = df_a['qty'].to_numpy().astype(np.float64)
is_bm = df_a['is_bm'].to_numpy().astype(bool)
del df_a, dfs_a
log(f"   aggTrades: {len(ts_us):,} 笔")

bucket_a = (ts_us // 1000) // BUCKET_MS * BUCKET_MS
del ts_us

df_a2 = pl.DataFrame({'b': bucket_a, 'p': price_a, 'q': qty_a, 'bm': is_bm.astype(int)})
del bucket_a, price_a, qty_a, is_bm

bars_a = df_a2.group_by('b').agg([
    pl.col('p').max().alias('h'),
    pl.col('p').min().alias('l'),
    pl.col('p').last().alias('c'),
    pl.col('q').sum().alias('vol'),
    (pl.col('q')*pl.col('bm')).sum().alias('vbm'),
    (pl.col('q')*(1-pl.col('bm'))).sum().alias('vtk'),
    ((pl.col('q')>1)*(1-pl.col('bm'))).sum().alias('nlg_tk'),
    (pl.col('q')>1).sum().alias('nlg'),
]).sort('b')
del df_a2

ab_b = bars_a['b'].to_numpy().astype(np.int64)
ab_c = bars_a['c'].to_numpy().astype(np.float64)
ab_h = bars_a['h'].to_numpy().astype(np.float64)
ab_l = bars_a['l'].to_numpy().astype(np.float64)
ab_vol = bars_a['vol'].to_numpy().astype(np.float64)
ab_vbm = bars_a['vbm'].to_numpy().astype(np.float64)
ab_vtk = bars_a['vtk'].to_numpy().astype(np.float64)
ab_nlg_tk = bars_a['nlg_tk'].to_numpy().astype(np.float64)
ab_nlg = bars_a['nlg'].to_numpy().astype(np.float64)
del bars_a
log(f"   aggTrades bars: {len(ab_c):,}")

# ========== 3. 对齐 ==========
log(f"[{time.time()-t0:.0f}s] 3. 对齐...")
common = np.intersect1d(l2_b, ab_b)
log(f"   交集: {len(common):,}")

def align(sb, sv, tb):
    idx = np.searchsorted(sb, tb)
    v = (idx < len(sb)) & (sb[idx] == tb)
    out = np.full(len(tb), np.nan)
    out[v] = sv[idx[v]]
    return out

ts = common
n = len(ts)

m = align(l2_b, l2_m, ts)
sp = align(l2_b, l2_sp, ts)
bk = align(l2_b, l2_bk, ts)
nu = align(l2_b, l2_nu, ts)
del l2_b, l2_m, l2_sp, l2_bk, l2_nu

imb_b = align(ab_b, (ab_vtk-ab_vbm)/(ab_vol+1e-12), ts)
sv = align(ab_b, ab_vtk-ab_vbm, ts)
tk_f = align(ab_b, ab_vtk/(ab_vol+1e-12), ts)
lg_tk = align(ab_b, ab_nlg_tk-(ab_nlg-ab_nlg_tk), ts)
park = align(ab_b, (ab_h-ab_l)/(ab_c+1e-12), ts)
lv = align(ab_b, np.log(ab_vol+1), ts)
del ab_b, ab_c, ab_h, ab_l, ab_vol, ab_vbm, ab_vtk, ab_nlg_tk, ab_nlg

# ========== Rolling ==========
def rs(x, w):
    x0 = np.where(np.isnan(x), 0.0, np.array(x, dtype=np.float64))
    cs = np.concatenate([[0.0], np.cumsum(x0)])
    out = np.full(len(x), np.nan)
    out[w-1:] = cs[w:] - cs[:-w]
    cnt = np.concatenate([[0.0], np.cumsum(np.where(np.isnan(x), 0.0, 1.0))])
    c2 = np.full(len(x), 0.0)
    c2[w-1:] = cnt[w:] - cnt[:-w]
    out[c2 < w*0.5] = np.nan
    return out

def rm(x, w):
    s = rs(x, w)
    c = rs(np.where(np.isnan(x), 0.0, 1.0), w)
    mask = c >= w*0.5
    r = np.full(len(x), np.nan)
    r[mask] = s[mask] / c[mask]
    return r

log(f"[{time.time()-t0:.0f}s] 4. 特征工程...")

# L2 ret (多窗口)
l2_r1 = np.full(n, np.nan); l2_r1[1:] = m[1:]/m[:-1]-1
l2_r6 = np.full(n, np.nan); l2_r6[6:] = m[6:]/m[:-6]-1
l2_r18 = np.full(n, np.nan); l2_r18[18:] = m[18:]/m[:-18]-1
l2_sp_r6 = rm(sp, 6); l2_bk_r6 = rm(bk, 6); l2_nu_r6 = rm(nu, 6)

# aggTrades rolling
agg_sv6 = rs(sv, 6); agg_sv18 = rs(sv, 18)
agg_tk6 = rm(tk_f, 6); agg_tk18 = rm(tk_f, 18)
agg_lg6 = rs(lg_tk, 6); agg_lg18 = rs(lg_tk, 18)
agg_park6 = rm(park, 6); agg_lv6 = rm(lv, 6)

# future_ret (用 L2 mid, 连续)
future = np.full(n, np.nan)
for i in range(n - STEP_3MIN):
    a, bv = m[i], m[i+STEP_3MIN]
    if not np.isnan(a) and not np.isnan(bv) and a > 0:
        future[i] = (bv - a) / a

del sv, tk_f, lg_tk

# ========== Walk-Forward ==========
log(f"[{time.time()-t0:.0f}s] 5. Walk-Forward 3min...")
day_idx = (ts // 86400000).astype(int)
u_days = np.unique(day_idx)

NAMES = [
    'l2_r1', 'l2_r6', 'l2_r18', 'l2_sp_r6', 'l2_bk_r6', 'l2_nu_r6',
    'sp', 'bk', 'nu',
    'imb_b', 'sv6', 'sv18', 'tk6', 'tk18', 'lg6', 'lg18', 'park6', 'lv6',
]
MATS = [l2_r1, l2_r6, l2_r18, l2_sp_r6, l2_bk_r6, l2_nu_r6,
        sp, bk, nu, imb_b, agg_sv6, agg_sv18, agg_tk6, agg_tk18, agg_lg6, agg_lg18, agg_park6, agg_lv6]

X_ALL = np.column_stack(MATS)

# Label mask
has = ~np.isnan(future)
for ci in range(X_ALL.shape[1]):
    has &= ~np.isnan(X_ALL[:, ci])

thr = abs(np.quantile(future[has & (day_idx < u_days[TRAIN_DAYS])], 0.25))
labeled = has & (np.abs(future) > thr)
l_day = day_idx[labeled]
l_fr = future[labeled]
log(f"   labeled: {labeled.sum():,}, thr={thr*10000:.2f}bps, {len(u_days)} days")

def run_wf(idx_list, desc):
    X_set = X_ALL[labeled][:, idx_list]
    all_p, all_y = [], []
    imp = np.zeros(X_set.shape[1])
    nm = 0
    for di in range(TRAIN_DAYS, len(u_days)):
        d = u_days[di]; dlo = u_days[max(0, di-TRAIN_DAYS)]
        tm = (l_day >= dlo) & (l_day < d)
        qm = l_day == d
        if tm.sum() < 2000 or qm.sum() < 100:
            continue
        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                    min_child_samples=200, verbose=-1, random_state=42, n_jobs=-1)
        model.fit(X_set[tm], (l_fr[tm] > 0).astype(int))
        all_p.extend(model.predict_proba(X_set[qm])[:, 1])
        all_y.extend((l_fr[qm] > 0).astype(int))
        imp += model.feature_importances_
        nm += 1
    if len(all_p) < 100:
        log(f"   {desc}: ❌ n={len(all_p)}")
        return None, None
    ap = np.array(all_p); ay = np.array(all_y)
    auc = roc_auc_score(ay, ap)
    conf = np.abs(ap - 0.5) * 2
    acc = (ap > 0.5).astype(int) == ay
    log(f"\n   [{desc}] {nm}m, {len(ap):,}p, AUC={auc:.4f}")
    for c in [0.0, 0.3, 0.5, 0.6, 0.7]:
        mc = conf >= c
        if mc.sum() < 50:
            continue
        c_acc = acc[mc].mean() * 100
        exp = c_acc/100 * 0.8 - (1 - c_acc/100)
        log(f"     conf≥{c:.1f}: n={mc.sum():>7,}, acc={c_acc:>5.1f}%, exp={exp*100:+.1f}¢")
    return auc, imp / max(nm, 1)

# ===== 4 组对比 =====
log(f"\n{'='*70}")
log("3min Horizon Walk-Forward (BTCUSDT Spot)")
log(f"{'='*70}")

ret_idx = [NAMES.index('l2_r1'), NAMES.index('l2_r6')]
auc1, _ = run_wf(ret_idx, "① Baseline: ret_10s(L2) + ret_1m(L2)")

agg_idx = [NAMES.index(n) for n in ['imb_b', 'sv6', 'sv18', 'tk6', 'tk18', 'lg6', 'lg18', 'park6', 'lv6']]
auc2, _ = run_wf(agg_idx, "② aggTrades-only (9 feat)")

l2_idx = [NAMES.index(n) for n in ['l2_r1', 'l2_r6', 'l2_r18', 'l2_sp_r6', 'l2_bk_r6', 'l2_nu_r6', 'sp', 'bk', 'nu']]
auc3, _ = run_wf(l2_idx, "③ L2-only (9 feat)")

all_idx = list(range(len(NAMES)))
auc4, imp = run_wf(all_idx, "④ FUSED L2+aggTrades (18 feat)")

# Top 重要性
if imp is not None:
    ord_ = np.argsort(-imp)
    log(f"\nTop 10 FUSED 特征重要性 (gain):")
    for r, i in enumerate(ord_[:10]):
        log(f"   #{r+1:2d} {NAMES[i]:<15} gain={imp[i]:.1f}")

log(f"\n{'='*70}")
log(f"总耗时: {time.time()-t0:.0f}s")
log("\n=== 最终 AUC 对比 (3min, walk-forward) ===")
log(f"   ① ret baseline:    {auc1:.4f}")
log(f"   ② aggTrades-only:  {auc2:.4f}")
log(f"   ③ L2-only:         {auc3:.4f}")
log(f"   ④ FUSED:           {auc4:.4f}")
log(f"\n   FUSED 比 L2-only 提升: {auc4-auc3:+.4f}")
log(f"   FUSED 比 agg-only 提升: {auc4-auc2:+.4f}")

# 保存结果
os.makedirs('/workspace/results', exist_ok=True)
results = {
    'horizon_min': 3,
    'train_days': TRAIN_DAYS,
    'date_range_ms': [int(common[0]), int(common[-1])],
    'auc': {
        'ret_baseline': float(auc1) if auc1 else None,
        'agg_only': float(auc2) if auc2 else None,
        'l2_only': float(auc3) if auc3 else None,
        'fused': float(auc4) if auc4 else None,
    },
    'feat_names': NAMES,
    'fused_imp_gain': imp.tolist() if imp is not None else None,
}
with open('/workspace/results/fused_3min.json', 'w') as f:
    import json; json.dump(results, f, indent=2)
log(f"\n✅ 结果保存: /workspace/results/fused_3min.json")
