#!/usr/bin/env python3
"""
特征工程模块 — 零过拟合版
==========================
设计原则:
  1. 只用 past-only 数据, 零未来泄露
  2. 分位数/中位数 > 均值 (抗极端值)
  3. 不用纯总成交量 (无流动性上下文没意义)
  4. 用主动买卖量、OFI、流动性特征
  5. 特征要有金融直觉, 不是瞎编

数据来源:
  - aggTrades: 主动买/卖量 (is_buyer_maker)
  - klines 1m: OHLC
  - L2 orderbook: bid/ask depth, spread

Feature 列表 (约 50 个):

A. 价格动量 (ret 多周期):
   - ret_1m, ret_5m, ret_15m, ret_30m, ret_60m, ret_120m
   - 每周期 ret 的滚动中位数 (不是均值!)

B. 波动率 (多周期):
   - sigma_5m_median, sigma_15m_median, sigma_60m_median
   - sigma ratio (短期/长期)

C. 主动买卖力量 (从 aggTrades):
   - agg_buy_vol, agg_sell_vol (每个 bar 内)
   - agg_buy_ratio = buy_vol / (buy_vol + sell_vol)
   - agg_buy_ratio 的滚动中位数
   - 大单比例 (qty > 95th percentile)

D. 订单流不平衡 (OFI):
   - OFI = (bid_size_change_if price_up) - (ask_size_change_if price_down)
   - 简化版: (主动买量 - 主动卖量) 的滚动分位数

E. 流动性特征 (从 L2 或 klines):
   - Amihud illiquidity = |ret| / turnover (用 median)
   - 收盘 spread proxy = (high-low)/close
   - depth imbalance (如果有 L2)

F. 时间特征:
   - hour_of_day (周期编码)
   - day_of_week

G. 趋势特征:
   - close vs MA_median_5, MA_median_10, MA_median_30
   - 不要用 MA (均值), 用 rolling median

Label:
  future_ret = close[t+H] / close[t] - 1
  二分类: future_ret > 0 → 1, 否则 0
"""
import polars as pl
import numpy as np
import glob
import gc


# ═══════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════

def load_aggtrades(data_dir: str) -> pl.DataFrame:
    """aggTrades → 分钟级主动买卖量"""
    files = sorted(glob.glob(f"{data_dir}/*.csv"))
    if not files:
        raise FileNotFoundError(f"No aggTrades in {data_dir}")
    dfs = [pl.read_csv(f, has_header=False, new_columns=[
        "agg_id", "price", "qty", "first_trade_id", "last_trade_id",
        "ts_us", "is_buyer_maker", "is_best_match",
    ]) for f in files]
    agg = pl.concat(dfs).sort("ts_us")
    del dfs; gc.collect()

    # 1m bucket
    agg = agg.with_columns(
        ((pl.col("ts_us") // 60_000_000) * 60).alias("bucket_ts")
    )
    by_min = agg.group_by("bucket_ts").agg([
        pl.col("qty").sum().alias("total_qty"),
        pl.col("qty").filter(pl.col("is_buyer_maker").not_()).sum().alias("active_buy_qty"),
        pl.col("qty").filter(pl.col("is_buyer_maker")).sum().alias("active_sell_qty"),
        pl.col("price").last().alias("close"),
        pl.col("price").first().alias("open"),
        pl.col("price").max().alias("high"),
        pl.col("price").min().alias("low"),
        pl.len().alias("trade_count"),
    ]).sort("bucket_ts")
    del agg; gc.collect()

    by_min = by_min.with_columns(
        pl.from_epoch("bucket_ts", time_unit="s").alias("ts")
    )
    return by_min.with_columns([
        pl.col("ts").dt.hour().alias("hour"),
        pl.col("ts").dt.weekday().alias("dow"),
        pl.col("ts").dt.date().alias("date"),
    ])


def load_klines(data_dir: str) -> pl.DataFrame:
    """Binance 1m klines → OHLCV
    Binance kline CSV 列 (12列, 无header):
      open_time, open, high, low, close, volume, close_time, quote_volume,
      trade_count, taker_buy_base, taker_buy_quote, ignore
    """
    files = sorted(glob.glob(f"{data_dir}/*.csv"))
    if not files:
        raise FileNotFoundError(f"No klines in {data_dir}")
    dfs = [pl.read_csv(f, has_header=False, new_columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trade_count",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]) for f in files]
    klines = pl.concat(dfs).with_columns([
        pl.col("open_time") // 1_000_000,  # us → s (Binance klines 给的是微秒)
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
        pl.col("quote_volume").cast(pl.Float64),
        pl.col("taker_buy_base").cast(pl.Float64),
        pl.col("taker_buy_quote").cast(pl.Float64),
    ])
    del dfs; gc.collect()

    klines = klines.with_columns(
        pl.from_epoch("open_time", time_unit="s").alias("ts")
    )
    klines = klines.with_columns([
        pl.col("ts").dt.hour().alias("hour"),
        pl.col("ts").dt.weekday().alias("dow"),
        pl.col("ts").dt.date().alias("date"),
    ])
    return klines.sort("open_time")


# ═══════════════════════════════════════════════════════════════
# 特征计算 (全部 past-only)
# ═══════════════════════════════════════════════════════════════

def compute_features(df: pl.DataFrame, horizon_min: int = 15) -> pl.DataFrame:
    """从 1m OHLCV + 主动买卖量 → past-only 特征

    输入 df 必须按 open_time 排序, 包含列:
      open_time, open, high, low, close, volume, quote_volume,
      taker_buy_base, taker_buy_quote, taker_sell_base,
      hour, dow, date, ts
    """
    out = df

    # ── A. ret 多周期 ──
    for w in [1, 5, 15, 30, 60, 120]:
        out = out.with_columns(
            (pl.col("close") / pl.col("close").shift(w) - 1).alias(f"ret_{w}m")
        )

    # ── B. 波动率 (rolling median of |ret|, 不用 std!) ──
    # 用户说用分位数中位数, 不用均值
    for w in [5, 15, 60]:
        out = out.with_columns(
            pl.col("ret_1m").abs().rolling_median(w).alias(f"absret_median_{w}m")
        )

    # sigma ratio: 短/长
    out = out.with_columns(
        (pl.col("absret_median_5m") / (pl.col("absret_median_60m") + 1e-10))
        .alias("sigma_ratio_5_60")
    )

    # ── C. 主动买卖力量 (从 klines taker_buy) ──
    # klines 自带 taker_buy_base, taker_sell_base (volume - taker_buy_base)
    out = out.with_columns(
        (pl.col("volume") - pl.col("taker_buy_base")).alias("taker_sell_base")
    )
    out = out.with_columns(
        (pl.col("taker_buy_base") / (pl.col("volume") + 1e-10)).alias("buy_ratio")
    )
    # 滚动中位数 of buy_ratio (抗极端)
    for w in [5, 15, 60]:
        out = out.with_columns(
            pl.col("buy_ratio").rolling_median(w).alias(f"buy_ratio_median_{w}m")
        )

    # 主动买卖量变化率
    out = out.with_columns([
        pl.col("taker_buy_base").rolling_median(15).alias("buy_vol_median_15m"),
        pl.col("taker_sell_base").rolling_median(15).alias("sell_vol_median_15m"),
    ])

    # OFI proxy: (buy - sell) / (buy + sell) rolling median
    out = out.with_columns(
        ((pl.col("taker_buy_base") - pl.col("taker_sell_base"))
         / (pl.col("taker_buy_base") + pl.col("taker_sell_base") + 1e-10))
        .rolling_median(15).alias("ofi_proxy_median_15m")
    )

    # ── D. 流动性特征 ──
    # Amihud illiquidity = median(|ret| / quote_volume)
    # 不是均值! 用户说用分位数中位数
    out = out.with_columns(
        (pl.col("ret_1m").abs() / (pl.col("quote_volume") + 1e-10)).alias("illiq_raw")
    )
    for w in [15, 60]:
        out = out.with_columns(
            pl.col("illiq_raw").rolling_median(w).alias(f"amihud_{w}m")
        )

    # 振幅 proxy
    out = out.with_columns(
        ((pl.col("high") - pl.col("low")) / (pl.col("close") + 1e-10)).alias("amplitude")
    )
    out = out.with_columns(
        pl.col("amplitude").rolling_median(15).alias("amplitude_median_15m")
    )

    # ── E. 趋势 (rolling median close, 不是 MA) ──
    for w in [5, 10, 30, 60]:
        out = out.with_columns(
            pl.col("close").rolling_median(w).alias(f"close_median_{w}m")
        )
    # close vs 各周期 median
    for w in [10, 30, 60]:
        out = out.with_columns(
            (pl.col("close") / pl.col(f"close_median_{w}m") - 1).alias(f"dev_median_{w}m")
        )

    # ── F. 时间编码 ──
    # hour 用 sin/cos 周期编码, 避免 23 点和 0 点差很远
    out = out.with_columns([
        (pl.col("hour") * 2 * np.pi / 24).sin().alias("hour_sin"),
        (pl.col("hour") * 2 * np.pi / 24).cos().alias("hour_cos"),
        # dow 也编码
        (pl.col("dow") * 2 * np.pi / 7).sin().alias("dow_sin"),
        (pl.col("dow") * 2 * np.pi / 7).cos().alias("dow_cos"),
    ])

    # ── G. Label (future_ret, 二分类) ──
    out = out.with_columns(
        (pl.col("close").shift(-horizon_min) / pl.col("close") - 1).alias("future_ret")
    )

    # drop_nulls (rolling windows + future_ret 两端)
    # 但保留 ret_1m 原始值用于 bootstrap
    out = out.drop_nulls()

    return out


# ═══════════════════════════════════════════════════════════════
# Bootstrap 信号存在性检验
# ═══════════════════════════════════════════════════════════════

def variance_ratio_test(returns: np.ndarray, lag: int = 5) -> dict:
    """Lo-MacKinlay Variance Ratio Test
    H0: returns 是随机游走 (VR = 1)
    VR < 1 → 均值回归 (负自相关)
    VR > 1 → 趋势 (正自相关)
    """
    n = len(returns)
    if n < lag * 2:
        return {"vr": np.nan, "z": np.nan, "p": np.nan}

    mu = np.mean(returns)
    sq = np.sum((returns - mu) ** 2) / (n - 1)

    # k-period return variance
    k_rets = np.cumsum(returns)
    k_rets = k_rets[lag::lag] - np.concatenate([[0], k_rets[:-lag][::lag]])[:len(k_rets[lag::lag])]
    k_sq = np.sum((k_rets - lag * mu) ** 2) / (n / lag - 1)

    vr = k_sq / (sq * lag)

    # 标准正态 z-stat
    se = np.sqrt((2 * (2 * lag - 1) * (lag - 1)) / (6 * n * lag))
    z = (vr - 1) / se
    p_val = 2 * (1 - _norm_cdf(abs(z)))

    return {"vr": vr, "z": z, "p": p_val}


def _norm_cdf(x):
    """标准正态 CDF 近似 (Abramowitz & Stegun)"""
    from scipy.stats import norm
    return norm.cdf(x)


def bootstrap_feature_ic(df: pl.DataFrame, feature_cols: list,
                         target_col: str = "future_ret", n_boot: int = 1000) -> pl.DataFrame:
    """Bootstrap 每个特征的 IC (Information Coefficient) 95% CI

    IC = Spearman 相关系数 (比 Pearson 更稳健, 不假设线性)
    如果 CI 跨 0 → 特征在 out-of-sample 下可能没用
    """
    from scipy.stats import spearmanr

    n = len(df)
    targets = df[target_col].to_numpy()
    results = []

    for feat in feature_cols:
        vals = df[feat].to_numpy()
        valid = ~np.isnan(vals) & ~np.isnan(targets)
        if valid.sum() < 100:
            results.append({"feature": feat, "ic_mean": np.nan, "ic_lo": np.nan,
                           "ic_hi": np.nan, "p_value": np.nan})
            continue

        x = vals[valid]
        y = targets[valid]

        # 全样本 IC
        ic_full, p_full = spearmanr(x, y)

        # Bootstrap
        rng = np.random.RandomState(42)
        boot_ics = []
        for _ in range(n_boot):
            idx = rng.randint(0, len(x), len(x))
            try:
                ic_b, _ = spearmanr(x[idx], y[idx])
                if not np.isnan(ic_b):
                    boot_ics.append(ic_b)
            except:
                pass

        if len(boot_ics) > 10:
            boot_ics = np.array(boot_ics)
            results.append({
                "feature": feat,
                "ic_full": ic_full,
                "ic_mean": np.mean(boot_ics),
                "ic_lo": np.percentile(boot_ics, 2.5),
                "ic_hi": np.percentile(boot_ics, 97.5),
                "ic_std": np.std(boot_ics),
                "ci_crosses_zero": boot_ics.mean() - 1.96 * boot_ics.std() < 0 < boot_ics.mean() + 1.96 * boot_ics.std(),
                "p_value": p_full,
            })

    return pl.DataFrame(results).sort("ic_mean", descending=True)


# ═══════════════════════════════════════════════════════════════
# Walk-Forward ML (LightGBM)
# ═══════════════════════════════════════════════════════════════

def walkforward_lgbm(df: pl.DataFrame, feature_cols: list,
                      target_col: str = "future_ret",
                      label_col: str = "label",
                      train_window: int = 30,  # 训练窗口 (天)
                      test_window: int = 1,     # 测试窗口 (天)
                      min_train_samples: int = 1000) -> dict:
    """Expanding Window Walk-Forward + LightGBM

    Day 1-30 train → Day 31 test
    Day 1-31 train → Day 32 test
    ...

    严格 out-of-sample, 每天只训练一次
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

    dates = df["date"].unique().sort().to_list()
    all_preds = []
    all_true = []
    feature_importances = pl.DataFrame()

    for i in range(train_window, len(dates)):
        test_date = dates[i]
        train_dates = dates[:i]  # expanding

        train = df.filter(pl.col("date").is_in(train_dates))
        test = df.filter(pl.col("date") == test_date)

        if len(train) < min_train_samples or len(test) < 10:
            continue

        X_train = train[feature_cols].to_numpy()
        y_train = train[label_col].to_numpy().astype(int)
        X_test = test[feature_cols].to_numpy()
        y_test = test[label_col].to_numpy().astype(int)

        # LightGBM
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)

        # 预测
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba > 0.5).astype(int)

        all_preds.append({
            "date": test_date, "pred": pred.tolist(), "proba": proba.tolist(),
        })
        all_true.append({
            "date": test_date, "true": y_test.tolist(),
        })

        # Feature importance
        imp = pl.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }).sort("importance", descending=True).with_columns(pl.lit(test_date).alias("date"))
        feature_importances = pl.concat([feature_importances, imp]) if len(feature_importances) > 0 else imp

    # 汇总评估
    if not all_preds:
        return {"valid": False, "msg": "No out-of-sample predictions"}

    flat_pred = np.concatenate([p["pred"] for p in all_preds])
    flat_true = np.concatenate([t["true"] for t in all_true])
    flat_proba = np.concatenate([p["proba"] for p in all_preds])

    results = {
        "valid": True,
        "n_days_tested": len(all_preds),
        "n_samples": len(flat_pred),
        "accuracy": accuracy_score(flat_true, flat_pred),
        "auc": roc_auc_score(flat_true, flat_proba) if len(np.unique(flat_true)) > 1 else np.nan,
        "precision": precision_score(flat_true, flat_pred, zero_division=0),
        "recall": recall_score(flat_true, flat_pred, zero_division=0),
    }

    # 逐日
    daily_metrics = []
    for p, t in zip(all_preds, all_true):
        acc = accuracy_score(t["true"], p["pred"])
        daily_metrics.append({"date": p["date"], "acc": acc, "n": len(p["pred"])})

    results["daily_metrics"] = daily_metrics
    results["feature_importances"] = feature_importances

    # 平均 feature importance
    avg_imp = feature_importances.group_by("feature").agg(
        pl.col("importance").mean().alias("avg_importance")
    ).sort("avg_importance", descending=True)
    results["avg_importance"] = avg_imp

    return results
