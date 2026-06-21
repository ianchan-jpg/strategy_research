"""构造相对收益目标变量。

核心目标：
使用当期因子，预测下一期“量化收益 - 主观收益”。
"""

from __future__ import annotations

import pandas as pd


def build_relative_return(
    df: pd.DataFrame,
    date_col: str,
    subjective_col: str,
    quant_col: str,
    target_name: str,
    shift_periods: int = -1,
) -> pd.DataFrame:
    """构造下一期相对收益。

    参数说明：
    - subjective_col：主观策略收益列
    - quant_col：量化策略收益列
    - shift_periods=-1：把下一期收益移动到当前行，方便用当前因子预测
    """
    out = df[[date_col, subjective_col, quant_col]].copy()
    out = out.sort_values(date_col)
    out[target_name] = (out[quant_col] - out[subjective_col]).shift(shift_periods)
    return out


def build_stock_target(df: pd.DataFrame, date_col: str, subjective_col: str, quant_col: str) -> pd.DataFrame:
    """构造股票组：量化股票收益 - 主观股票收益。"""
    return build_relative_return(df, date_col, subjective_col, quant_col, "next_stock_quant_minus_subjective")


def build_cta_target(df: pd.DataFrame, date_col: str, subjective_col: str, quant_col: str) -> pd.DataFrame:
    """构造 CTA 组：量化 CTA 收益 - 主观 CTA 收益。"""
    return build_relative_return(df, date_col, subjective_col, quant_col, "next_cta_quant_minus_subjective")
