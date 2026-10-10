"""全局配置 —— 事件合约预测回测项目."""
from pathlib import Path

# ---------- 路径 ----------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ---------- 标的 & 时间 ----------
SYMBOL = "BTCUSDT"
TIMEFRAME = "1m"                 # 模型特征对齐粒度
PREDICT_HORIZON_MIN = 3          # 预测未来多少分钟
THRESHOLD_PCT = 0.05             # 标签阈值(%)，> 0.05% 记 UP，<-0.05% 记 DOWN

# ---------- 币安 API ----------
# 事件合约属于 U 本位合约，走 fapi；现货走 api，这里我们用合约数据更贴近
BINANCE_BASE = "https://fapi.binance.com"
# 如需拉历史逐笔(historicalTrades)，需在 .env 填 BINANCE_API_KEY（仅 K 线和 depth 免 key）
BINANCE_API_KEY = None            # 运行时从 .env 或环境变量覆盖
REQUEST_INTERVAL = 0.25           # 请求间隔(秒)，避免被限速
REQUEST_TIMEOUT = 15

# ---------- 回测 ----------
INITIAL_CAPITAL = 10000.0
EVENT_ODDS = 1.85                 # 事件合约买涨/买跌赔率
EVENT_FEE_PCT = 0.01              # 开仓手续费 1%
EVENT_EXPIRE_MIN = 3              # 合约到期时间 = PREDICT_HORIZON_MIN
MIN_CONFIDENCE = 0.55             # 模型置信度阈值，低于此不下注
POSITION_SIZE_PCT = 0.05          # 每次下注占总资金比例 (5%)

# ---------- 模型 ----------
MODEL_NAME = "xgb"
MODEL_FILE = MODEL_DIR / "xgb_btc_event.joblib"
TEST_RATIO = 0.20                 # 时间序列拆分，后 20% 做测试
XGB_PARAMS = dict(
    objective="multi:softprob",
    num_class=3,                  # DOWN / NEUTRAL / UP
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)

# ---------- 特征工程 ----------
LOOKBACK_KLINES = 60              # 回看多少根 1m K 线
DEPTH_LEVELS = 20                 # L2 前多少档做 imbalance
TRADES_WINDOWS = [1, 5, 15]       # 逐笔滚动窗口(分钟)

# ---------- 标签类别 ----------
LABELS = ["DOWN", "NEUTRAL", "UP"]
