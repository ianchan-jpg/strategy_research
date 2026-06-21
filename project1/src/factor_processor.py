"""因子预处理工具。

当前只提供基础函数，后续可以在这里扩展缺失值处理、中性化等逻辑。
"""

from __future__ import annotations

import pandas as pd


def winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """按分位数去极值，降低异常值影响。"""
    low = s.quantile(lower)
    high = s.quantile(upper)
    return s.clip(lower=low, upper=high)


def standardize_series(s: pd.Series) -> pd.Series:
    """标准化为均值 0、标准差 1。"""
    std = s.std()
    if std == 0 or pd.isna(std):
        return s * 0
    return (s - s.mean()) / std


def preprocess_factors(df: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    """对指定因子列做去极值和标准化。"""
    out = df.copy()
    for col in factor_cols:
        out[col] = standardize_series(winsorize_series(out[col]))
    return out
