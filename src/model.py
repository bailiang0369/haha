"""标签生成 + 分类模型训练.

标签定义:
  对每个时刻 t (我们的特征时间戳, 即 K 线 close_time),
  看 PREDICT_HORIZON_MIN 分钟后 close(t + h) 相对 close(t) 的收益率:
    ret >  +threshold  -> UP  (2)
    ret <  -threshold  -> DOWN (0)
    其他               -> NEUTRAL (1)
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import config

log = logging.getLogger("model")


# ---------------------------------------------------------------------------
# 标签
# ---------------------------------------------------------------------------
def make_labels(feature_df: pd.DataFrame, close_series: pd.Series) -> pd.Series:
    """返回一个 Series, 索引与 feature_df 对齐, 值为 0/1/2 (DOWN/NEUTRAL/UP)."""
    h = config.PREDICT_HORIZON_MIN
    # close_series 的 index 是 K 线 open_time. 我们的特征 index 是 close_time = open_time + 1m
    # 所以 close at feature t 对应 close at open_time = t - 1m
    close_at_t = close_series.copy()
    close_at_t.index = close_at_t.index + pd.to_timedelta(1, unit="m")  # 转成 close_time 索引
    close_at_t = close_at_t.reindex(feature_df.index)

    future_close = close_at_t.shift(-h)  # t+h
    ret = future_close / close_at_t - 1  # 未来 h 分钟净收益率

    thr = config.THRESHOLD_PCT / 100.0
    y = pd.Series(1, index=ret.index, dtype=int)  # default NEUTRAL
    y[ret > thr] = 2
    y[ret < -thr] = 0
    # 尾部 h 行没有未来收盘价 -> 丢弃
    y = y.dropna().astype(int)
    # 只保留 y 里也存在于 feature_df 的索引
    return y.reindex(feature_df.index).dropna().astype(int)


# ---------------------------------------------------------------------------
# 训练 / 测试 切分 (时间序列)
# ---------------------------------------------------------------------------
def split_ts(X: pd.DataFrame, y: pd.Series, test_ratio: float = config.TEST_RATIO):
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]
    n = len(X)
    cut = int(n * (1 - test_ratio))
    # 按时间排序 (通常已排好)
    order = X.index.argsort()
    X, y = X.iloc[order], y.iloc[order]
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
def train(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    model = XGBClassifier(**config.XGB_PARAMS)
    model.fit(X_train, y_train)
    log.info(f"model trained. classes={model.classes_}")
    return model


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)
    y_true = y_test.values
    report = classification_report(
        y_true, y_pred, target_names=config.LABELS, zero_division=0, output_dict=True
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "report": report,
    }


def feature_importance(model, X: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return imp.head(top_n).reset_index().rename(columns={"index": "feature", 0: "gain"})


def save_model(model, path: Path = config.MODEL_FILE) -> None:
    joblib.dump(model, path)
    log.info(f"model saved -> {path}")


def load_model(path: Path = config.MODEL_FILE):
    return joblib.load(path)
