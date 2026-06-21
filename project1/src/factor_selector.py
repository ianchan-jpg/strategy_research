"""因子相关性和去重工具。

当前只提供查看相关性的函数，不自动删除任何因子。
"""

from __future__ import annotations

import pandas as pd


def factor_correlation(df: pd.DataFrame, factor_cols: list[str], method: str = "spearman") -> pd.DataFrame:
    """计算因子相关性矩阵，用于人工观察冗余程度。"""
    return df[factor_cols].corr(method=method)


def find_high_corr_pairs(corr: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """列出高相关因子对，但不做自动删除。"""
    rows = []
    cols = list(corr.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = corr.loc[left, right]
            if abs(value) >= threshold:
                rows.append({"factor_1": left, "factor_2": right, "corr": value})
    return pd.DataFrame(rows)
