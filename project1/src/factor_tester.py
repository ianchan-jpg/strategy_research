"""单因子检验工具。

这里提供最小闭环需要的统计函数：
Rank IC、全样本 OLS、滚动 OLS、分组收益。
所有函数只接收已经对齐好的因子和目标变量，不在这里读取 Excel。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr


def rank_ic(factor: pd.Series, target: pd.Series) -> float:
    """计算 Rank IC，即因子值与未来相对收益的 Spearman 秩相关。"""
    valid = pd.concat([factor, target], axis=1).dropna()
    if len(valid) < 3:
        return np.nan
    return float(spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]).correlation)


def rank_ic_with_pvalue(factor: pd.Series, target: pd.Series) -> pd.Series:
    """计算 Rank IC 和对应 p 值。"""
    valid = pd.concat([factor, target], axis=1).dropna()
    if len(valid) < 3:
        return pd.Series({"rank_ic": np.nan, "rank_ic_pvalue": np.nan})
    result = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
    return pd.Series({"rank_ic": float(result.correlation), "rank_ic_pvalue": float(result.pvalue)})


def ols_regression(factor: pd.Series, target: pd.Series) -> pd.Series:
    """单因子 OLS 回归：target = alpha + beta * factor。"""
    valid = pd.concat([factor, target], axis=1).dropna()
    valid.columns = ["factor", "target"]
    if len(valid) < 5:
        return pd.Series(
            {
                "n": len(valid),
                "alpha": np.nan,
                "beta": np.nan,
                "t_beta": np.nan,
                "p_beta": np.nan,
                "r2": np.nan,
            }
        )
    x = sm.add_constant(valid["factor"])
    model = sm.OLS(valid["target"], x).fit()
    return pd.Series(
        {
            "n": int(model.nobs),
            "alpha": model.params.get("const", np.nan),
            "beta": model.params.get("factor", np.nan),
            "t_beta": model.tvalues.get("factor", np.nan),
            "p_beta": model.pvalues.get("factor", np.nan),
            "r2": model.rsquared,
        }
    )


def rolling_ols(df: pd.DataFrame, date_col: str, factor_col: str, target_col: str, window: int = 52) -> pd.DataFrame:
    """计算滚动 OLS，每一行使用过去 window 个有效样本估计一次回归。"""
    valid = df[[date_col, factor_col, target_col]].dropna().sort_values(date_col).reset_index(drop=True)
    rows = []

    for end_idx in range(window - 1, len(valid)):
        sample = valid.iloc[end_idx - window + 1 : end_idx + 1]
        result = ols_regression(sample[factor_col], sample[target_col])
        rows.append(
            {
                "date": sample[date_col].iloc[-1],
                "window": window,
                "n": int(result["n"]),
                "alpha": result["alpha"],
                "beta": result["beta"],
                "t_beta": result["t_beta"],
                "p_beta": result["p_beta"],
                "r2": result["r2"],
            }
        )

    return pd.DataFrame(rows)


def grouped_return(df: pd.DataFrame, factor_col: str, target_col: str, n_groups: int = 5) -> tuple[pd.DataFrame, bool]:
    """按因子从低到高分组，计算每组下一期相对收益均值。"""
    valid = df[[factor_col, target_col]].dropna().copy()
    if len(valid) < n_groups * 3:
        return pd.DataFrame(), False

    valid["group"] = pd.qcut(valid[factor_col], q=n_groups, labels=False, duplicates="drop") + 1
    group_table = (
        valid.groupby("group", as_index=False)
        .agg(
            sample_count=(target_col, "size"),
            mean_next_relative_return=(target_col, "mean"),
            median_next_relative_return=(target_col, "median"),
            factor_min=(factor_col, "min"),
            factor_max=(factor_col, "max"),
        )
        .sort_values("group")
    )

    means = group_table["mean_next_relative_return"]
    is_monotonic = bool(means.is_monotonic_increasing or means.is_monotonic_decreasing)
    return group_table, is_monotonic
