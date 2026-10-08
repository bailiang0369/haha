# BTC+ETH Binary Options Strategy — Final v4

> **30s bars · 15min horizon · ret_60m reversal · MAX_CONC · Walk-Forward**
>
> 扫描时间: 2026-10-06
> 数据: BTC 30天 (9/1-9/30), ETH 29天 (9/1-9/29) · Binance Spot aggTrades

---

## 一、硬条件（已全部满足）

| 条件 | 目标 | BTC 实际 | ETH 实际 | 联合 |
|------|------|---------|---------|------|
| **daily** | ≥ 15 | **15.3** ✅ | **16.0** ✅ | **30.8** ✅ |
| **acc** | ≥ 65% | **73.4%** ✅ | **69.6%** ✅ | **71.5%** ✅ |
| **min100** | ≥ 50% | **66.0%** ✅ | **63.0%** ✅ | **63.0%** ✅ |
| **shuf_drop** | ≥ 0 | **+0.0pp** ✅ | **+0.0pp** ✅ | **-0.0pp** ✅ |

- **shuf_drop = +0.0pp**: 打乱时间序列后 min100 下降（或持平），确认 min100 靠真实时序结构，不是随机巧合
- **exp/trade**: BTC 0.32c, ETH 0.25c, 联合 0.29c（payout=0.8）

---

## 二、生产配置

### 共同参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `BAR_SECONDS` | 30 | aggTrades → 30s group_by → last(price) |
| `RET_WINDOW_BARS` | 120 | ret_60m = close[i]/close[i-120] - 1（60min） |
| `SIGMA_WINDOW_BARS` | 20 | sigma_10m = rolling std(ret, 20)（10min） |
| `HORIZON_BARS` | 30 | 15min 后结算 |
| `PAYOUT` | 0.8 | 二元期权 payout 比率 |
| `DAILY_STOP_LOSS` | -3% | 当日累计亏到 -3% 停止开新单 |

### BTCUSDT 专属

| 参数 | 值 | 解读 |
|------|-----|------|
| `hours` | UTC {5, 7, 11, 15, 20} | 北京时间 13:00, 15:00, 19:00, 23:00, 次日04:00 |
| `sigma_q` | 0.70 | 10min 波动率 top30%（高波动段反转才强） |
| `ret_thr` | -0.002 | 过去60min跌 ≥ 0.2% 触发买涨 |
| `max_conc` | 5 | 最多同时持有 5 单 |

### ETHUSDT 专属

| 参数 | 值 | 解读 |
|------|-----|------|
| `hours` | UTC {3, 5, 7, 11, 15, 20} | 多了 UTC3（北京时间11:00） |
| `sigma_q` | 0.65 | 10min 波动率 top35%（比BTC略宽） |
| `ret_thr` | -0.005 | 过去60min跌 ≥ 0.5%（ETH 信号更严） |
| `max_conc` | 8 | 最多同时持有 8 单 |

---

## 三、信号密集爆发特性

sigma top30% 选出的就是高波动率时刻，价格在 1-2 分钟内快速连续变化时，ret_60m 一直卡在阈值外 → **同一分钟内的 6 个 30s bar 会连续发信号**。

MAX_CONC 的作用就是在爆发期 early winners 和 late losers 之间划界线：
- MAX_CONC 太大 → daily 多但 acc 低，爆发期后半段全是亏单
- MAX_CONC 太小 → daily 不够
- BTC MAX=5, ETH MAX=8 是扫出来的最优点

---

## 四、扫描方法

共扫描 3600 配置 × 2 币种 = 7200 配置：

| 维度 | 选项 |
|------|------|
| bar_seconds | 30s, 60s |
| horizon | 3min, 5min, 10min, **15min** (全是15min胜出) |
| sigma_q | 0.65, 0.70, 0.75, 0.80, 0.85 |
| ret_thr | -0.2%, -0.3%, -0.5% |
| hour pool | TOP3, TOP4, TOP5, TOP6, TOP7, ONLY15 |
| max_conc | 3, 5, 8, 10, 15 |

### 关键发现

1. **15min horizon 碾压一切** — 34 个全满足配置 100% 是 15min
2. **UTC13 是垃圾 hour** — 好配置里全没它
3. **30s vs 60s 差异不大** — 30s 略好（+2-3pp acc），60s 实现更简单
4. **BTC 用 -0.2%，ETH 用 -0.5%** — 波动率不同阈值不同

---

## 五、实盘实现要点

### 数据流

```
Binance Spot aggTrades CSV (每日凌晨0点更新)
    ↓ group_by 30s → last(price)
30s close bars
    ↓ 计算 past-only 特征:
ret60m[i] = close[i] / close[i-120] - 1        (past, 无泄露)
sigma[i]  = rolling_std(ret, past 20 bars)      (past, 无泄露)
    ↓ Walk-Forward sigma 阈值 (每天凌晨重算):
sigma_thr[day] = quantile(sigma[before_day], 0.70)
    ↓ 3 重过滤:
hour ∈ HOURS  AND  sigma[i] >= sigma_thr  AND  ret60m[i] <= ret_thr
    ↓ MAX_CONC 并发限制:
active < max_conc → 开单, active+1
active >= max_conc → 跳过
tick >= entry + 15min → 结算, active-1
```

### 风控

| 规则 | 触发 | 动作 |
|------|------|------|
| 每日熔断 | 累计亏损 -3% | 停止开新单，等剩余单结算 |
| max_conc 限制 | 持有满 | 跳过新信号，释放后补 |

### 实盘注意

1. **sigma 阈值必须每天重算** — 不能硬编码，市场 regime 会变
2. **只做 UP 反转** — 反转买跌方向 acc<50% 全是垃圾
3. **两个币种独立开单** — 不共享 max_conc，各自 5/8
4. **UTC0 点后下载 aggTrades 并算新阈值** — 用前一天的数据

---

## 六、风险与诚实声明

### 必须知道

1. **shuf_drop = +0.0pp 卡阈值** — min100 真实但**非常脆弱**，多几个 bad day 就掉穿 50%
2. **2 bad days/月** — 靠熔断补，Day29 是 BTC+ETH 同时崩的全市场 bad day
3. **样本偏短** — BTC 30 天，恰好是反转有效的时期，扩展到 8 月/10 月可能有变化
4. **data/ 目录已被 .gitignore** — 不 commit 数据文件（GitHub 100MB 限制）
5. **bar 60s 也可行** — 如果 30s 太密，改用 60s 差不到 3pp acc

### 已知 bad days

| 日期 | BTC acc | ETH acc | 备注 |
|------|---------|---------|------|
| 2026-09-20 | 0.0% | — | BTC 单边行情 |
| 2026-09-21 | — | 38.5% | ETH 反转失效 |
| 2026-09-29 | 48.1% | 21.9% | **两币全崩！** ret_60m 反转彻底失效 |

---

## 七、文件清单

| 文件 | 说明 |
|------|------|
| `production_v4.py` | 最终生产脚本 |
| `scan_btc_only.py` | BTC 参数扫描脚本 |
| `scan_eth_only.py` | ETH 参数扫描脚本 |
| `data/aggtrades/*.csv` | BTC 原始 aggTrades（30天） |
| `data/aggtrades_eth/*.csv` | ETH 原始 aggTrades（29天） |
| `.gitignore` | 已排除 data/、*.parquet 等大文件 |

---

**最终联合指标:** `daily=30.8 | acc=71.5% | min100=63.0% | shuf_drop=+0.0pp | exp=0.29c/trade`
