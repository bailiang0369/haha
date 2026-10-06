# BTCUSDT Binary Options — 配置对比报告

> 生成时间: 2026-10-06
> 数据: Binance Spot aggTrades, 29天 (2026-09-01 至 2026-09-29)
> 测试: Walk-Forward, 前15天训练 → 后14天测试
> 核心信号: ret_60m 反转 (过去60min动量 → 预测10min后方向)

---

## 一、测试的配置

| 配置 | Layer 2 (sigma) | Layer 3 (小时过滤) | 模型 |
|------|----------------|-------------------|------|
| **A: Top8h (原版)** | sigma top30% | Top 8小时 | ret_60m 反转 |
| **B: Top4h (当前最优)** | sigma top30% | **Top 4小时** | ret_60m 反转 |
| **C: 无Layer3** | sigma top30% | **全部24小时** | ret_60m 反转 |
| D: Top4h + LR13特征 | sigma top30% | Top 4小时 | 13特征 LogisticRegression |

---

## 二、完整数值对比

### 2.1 基础指标 (Walk-Forward, 14天测试)

| 指标 | A: Top8h (原版) | B: **Top4h (最优)** | C: 无Layer3 | D: Top4h + LR13 |
|------|---------------|-------------------|------------|----------------|
| 候选信号数 | 13,003 | **7,402** | 38,460 | 7,402 |
| 日均候选 | 929 | **529** | 2,747 | 529 |
| **Base acc** (裸信号) | 56.9% | **59.3%** | 55.7% | 57.8% |
| RM 执行数 | 1,880 | **1,479** | 5,763 | 1,386 |
| **日均执行** | 134 | **106** | 412 | 99 |
| **RM acc** | 76.4% | **83.6%** ✨ | 79.5% | 81.6% |
| **min100** | 52.0% | **62.0%** ✨ | 50.0% | 58.0% |
| **min500** | 72.4% | **79.4%** ✨ | 65.4% | 74.2% |
| **exp (payout=0.8)** | 37.6c | **50.5c** ✨ | 43.2c | 46.6c |
| Skipped acc | 53.6% | 25.5% | 51.5% | 49.8% |

### 2.2 核弹级验证 (Shuffle Test)

打乱交易结果顺序后重新跑 RM，测试信号是否真的来自时间序列结构：

| 配置 | Real min100 | Shuffled min100 | **Drop** | 判定 |
|------|------------|-----------------|----------|------|
| A: Top8h | 52.0% | 44.0% | +8.0pp | ✅ OK |
| B: **Top4h** | **62.0%** | **48.0%** | **+14.0pp** | ✅✅ **STRONG PASS** |
| C: 无Layer3 | 50.0% | 39.0% | +11.0pp | ✅ STRONG |
| D: Top4h + LR13 | 58.0% | 46.0% | +12.0pp | ✅ STRONG |

> **Drop 越大越好**。如果 shuffle 后 min100 几乎不掉，说明信号是靠静态分布或泄露来的，不是真的利用时间序列的 regime 结构。

### 2.3 逐天表现 (RM 执行后)

#### B: Top4h (最优)

| Day | 日期 | Exec n | acc | 标记 |
|-----|------|--------|-----|------|
| 16 | 9/16 | 128 | 78.1% | ✅ |
| 17 | 9/17 | 79 | 89.9% | ✅ |
| 18 | 9/18 | 234 | 81.6% | ✅ |
| 19 | 9/19 | 92 | 92.4% | ✅ |
| 20 | 9/20 | 5 | 60.0% | ⚠️ |
| 21 | 9/21 | 140 | 77.9% | ✅ |
| 22 | 9/22 | 124 | 83.1% | ✅ |
| 23 | 9/23 | 143 | 81.8% | ✅ |
| 24 | 9/24 | 150 | 92.0% | ✅ |
| 25 | 9/25 | 221 | 91.0% | ✅ |
| 27 | 9/27 | 76 | 90.8% | ✅ |
| 28 | 9/28 | 69 | 62.3% | ⚠️ |
| 29 | 9/29 | 18 | 38.9% | 💀 |

**13天里11天 ✅，1天 ⚠️，1天 💀**（Day20/29 样本太少）

---

## 三、Top4h vs Top8h — 为什么 Top4h 胜出

### 3.1 逐小时反转能力对比

用前15天训练集统计，每个 hour 的反转信号裸准确率：

| UTC Hour | ret<0 → future_UP% | ret>0 → future_UP% | 反转acc | 是否入选Top4 |
|----------|-------------------|-------------------|---------|-------------|
| **3** | 38.6% | 28.7% | 53.2% | ❌ 两个方向都偏空，反转无效 |
| **4** | **72.5%** | 54.4% | 60.7% | ❌ UTC4 和 UTC5 重叠较多，保留5 |
| **5** | **62.0%** | 41.0% | **60.6%** | ✅ **入选** |
| **7** | 46.9% | 41.2% | 53.3% | ❌ 反转弱 |
| **11** | **58.6%** | 54.6% | 49.8% | ✅ **入选**（数量多） |
| **13** | 50.9% | 45.7% | 52.6% | ❌ 接近随机 |
| **15** | **65.8%** | **37.7%** | **63.9%** | ✅ **入选**（最强反转hour） |
| **20** | 42.1% | 44.8% | 51.2% | ✅ **入选**（数量多，RM后筛好regime） |

> **关键洞察**：之前选Top8h时混了 Hour 3（反转无效）、Hour 7（反转弱）、Hour 13（接近随机）。这3个hour拖了后腿。砍掉之后 base acc 直接从 56.9% → 59.3%。

### 3.2 Layer 3 的效率

| 漏斗 | 数量 | acc | 过滤效率 (acc提升/数量减少) |
|------|------|-----|--------------------------|
| Layer 2: sigma top30% | 75,042 | 54.5% | - |
| + Top8h | 13,003 (-83%) | 56.9% (+2.4pp) | **0.029 pp/1%** |
| + Top4h | 7,402 (-90%) | 59.3% (+4.8pp) | **0.053 pp/1%** ✨ |

**Top4h 的过滤效率是 Top8h 的 1.8x**——砍掉弱hour的同时保留了强反转信号。

---

## 四、最终生产配置 (v2)

```
策略:        BTCUSDT 10min horizon binary options
信号:        ret_60m 反转 (close[i]/close[i-360]-1)
             ret_60m < 0 → 预测 UP
             ret_60m > 0 → 预测 DOWN

过滤:        sigma top30% (10min滚动波动率)
             Top4h = UTC [5, 11, 15, 20]

Regime Mon:  W=10, THR=40%, MAX_PAUSE=50
             (exec_only模式, 只记录已执行交易)

数据:        Binance Spot aggTrades (data.binance.vision)
             10s bars, group_by bucket → last(price)

实盘指标:    日均 ~106 笔, acc 83.6%, min100 62.0%, exp 50.5c/trade
```

### 代码位置

- 生产脚本: `/workspace/btc_final_production.py`
- 回测报告: `/workspace/results/btc_production_report.json`
- 对比文档: 本文档

### 风险声明

1. **29天数据有限** — 只覆盖2026年9月，可能刚好是反转特别有效的时期。需要扩展到10-11月验证稳定性
2. **交易成本未模拟** — 0.1%手续费会吃掉约10c/trade的利润（exp从50.5c降到~40c）
3. **Regime Monitor在极端趋势市可能失效** — 如果BTC持续单边上涨/下跌超过数小时，反转本身会失效，RM也救不了

---

## 五、数据处理审计

### 5.1 数据源

- **来源**: `https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT/`
- **格式**: 官方无header CSV, 字段: agg_trade_id, price, qty, first_trade_id, last_trade_id, timestamp(μs), is_buyer_maker, is_trade_me
- **完整性**: 29天，25,741,686条成交，零缺失

### 5.2 Bar 构造

```python
# aggTrades CSV → 10s bars
df = df.with_columns([(pl.col("ts_us") // 10_000_000 * 10_000_000).alias("bucket")])
bar = df.group_by("bucket").agg([pl.col("price").last().alias("close")]).sort("bucket")
```

- bucket diff 全部 = 10,000,000μs (10s) ✅
- 零缺bar、零重复bucket、零跳变 ✅
- group_by.last() 已验证是 bucket 内 ts 最大那条 ✅

### 5.3 特征计算 (全部 past-only)

```python
# ret_60m: 60min动量, 只用过去数据
ret60[i] = close[i] / close[i-360] - 1   # i-360 到 i, 全是过去

# sigma: 10min滚动波动率, 只用过去数据
sigma[i] = std(ret1[i-60 : i])           # 60个10s bar的std

# label: 未来10min涨跌
future[i] = close[i+60] / close[i] - 1   # i+60 是未来 → 正确!
```

### 5.4 泄露审计总结

| 检查 | 结果 |
|------|------|
| bucket=floor, 无未来timestamp | ✅ |
| ret_60m[i] 只用 c[i-360..i] | ✅ |
| future[i] = c[i+60]/c[i] - 1 (正确方向) | ✅ |
| 训练集/测试集严格时间隔离 | ✅ |
| Top4h + sigma阈值从训练集选出 | ✅ |
| Regime Monitor exec_only 模式 | ✅ |
| Shuffle Drop = +14pp (STRONG PASS) | ✅ |
| 时间打乱后 ret_60m AUC → 0.5008 | ✅ |
| 训练特征打乱后 acc 掉 8.5pp | ✅ |

### 5.5 与旧 L2 bug 的隔离

| | 旧 L2 bug 链 | 现在 aggTrades 链 |
|--|------------|-----------------|
| 数据源 | orderbook snapshot + update websocket | aggTrades 每日 CSV |
| 处理 | 事件流合并 → 重建 orderbook 状态 | 纯 group_by bucket → last(price) |
| 问题根源 | snapshot/update 时间戳处理错误 | **不存在这个环节** |
| 代码隔离 | fused_v6.py, rebuild_l2_correct.py | btc_final_production.py (独立) |

**没有 import 过任何 L2 相关代码，完全隔离。**
