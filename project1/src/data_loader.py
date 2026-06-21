"""数据读取与检查工具。

本模块负责读取 Excel、列出工作表、识别日期列，并输出基础数据质量信息。
这里不做建模，只帮助我们先弄清楚原始数据长什么样。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
import warnings

import pandas as pd


def list_sheets(excel_path: str | Path) -> List[str]:
    """返回 Excel 文件中的所有工作表名称。"""
    excel_path = Path(excel_path)
    workbook = pd.ExcelFile(excel_path)
    return workbook.sheet_names


def read_sheet(excel_path: str | Path, sheet_name: str) -> pd.DataFrame:
    """读取单个工作表，并去掉全空行和全空列。"""
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    return df


def infer_date_column(df: pd.DataFrame) -> str | None:
    """寻找最像日期的列，避免把普通数字误判成 1970 年日期。"""
    columns = list(df.columns)
    date_named_cols = [
        col
        for col in columns
        if any(key in str(col).lower() for key in ["date", "time", "日期", "时间"])
    ]

    for col in date_named_cols + columns[:5]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(df[col], errors="coerce")
        valid = parsed.dropna()
        if valid.empty:
            continue
        year_in_range = valid.dt.year.between(1990, 2100).mean()
        if parsed.notna().mean() >= 0.6 and year_in_range >= 0.8:
            return str(col)
    return None


def infer_frequency(dates: pd.Series) -> str:
    """根据相邻日期的中位数间隔，粗略判断数据频率。"""
    clean_dates = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if clean_dates.empty:
        return "未识别"

    median_days = clean_dates.diff().dt.days.dropna().median()
    if pd.isna(median_days):
        return "单期或无法判断"
    if median_days <= 2:
        return "日频"
    if median_days <= 10:
        return "周频"
    if median_days <= 45:
        return "月频"
    if median_days <= 120:
        return "季频"
    return "低频/不规则"


def infer_sheet_type(sheet_name: str, columns: list[str]) -> str:
    """根据工作表名称和列名，粗略判断工作表用途。"""
    if sheet_name in {"日度序列", "周度序列", "月度序列"}:
        return "指标数据"
    if sheet_name == "策略指数":
        return "策略收益数据"
    if "目录" in sheet_name or "对应" in sheet_name:
        return "指标说明/映射表"

    joined_cols = " ".join(columns)
    if "主观" in joined_cols and "量化" in joined_cols:
        return "策略收益数据"
    return "待人工确认"


def summarize_missing_rate(df: pd.DataFrame) -> dict:
    """汇总缺失率，既保留整体情况，也保留缺失最严重的列。"""
    missing = df.isna().mean().sort_values(ascending=False)
    return {
        "overall_missing_rate": round(float(df.isna().mean().mean()), 4),
        "max_missing_rate": round(float(missing.iloc[0]), 4) if len(missing) else None,
        "max_missing_column": str(missing.index[0]) if len(missing) else None,
        "columns_over_50pct_missing": int((missing > 0.5).sum()),
    }


def inspect_workbook(excel_path: str | Path) -> Dict[str, dict]:
    """检查全部工作表的列名、日期范围、缺失率和大致频率。"""
    result: Dict[str, dict] = {}
    for sheet in list_sheets(excel_path):
        df = read_sheet(excel_path, sheet)
        columns = [str(col) for col in df.columns]
        sheet_type = infer_sheet_type(sheet, columns)
        date_col = infer_date_column(df)

        date_min = None
        date_max = None
        frequency = "未识别"
        if date_col is not None:
            dates = pd.to_datetime(df[date_col], errors="coerce").dropna().sort_values()
            if not dates.empty:
                date_min = dates.min().date().isoformat()
                date_max = dates.max().date().isoformat()
                frequency = infer_frequency(dates)
        if sheet_type == "指标说明/映射表":
            frequency = "不适用"

        missing_summary = summarize_missing_rate(df)

        result[sheet] = {
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "columns": columns,
            "date_column": date_col,
            "date_min": date_min,
            "date_max": date_max,
            "frequency": frequency,
            "sheet_type": sheet_type,
            **missing_summary,
            "missing_rate_by_column": df.isna().mean().round(4).to_dict(),
        }
    return result


def inspection_to_tables(result: Dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把检查结果拆成两个表：工作表概况和列缺失率明细。"""
    sheet_rows = []
    missing_rows = []

    for sheet, info in result.items():
        sheet_rows.append(
            {
                "sheet": sheet,
                "sheet_type": info["sheet_type"],
                "rows": info["rows"],
                "cols": info["cols"],
                "date_column": info["date_column"],
                "date_min": info["date_min"],
                "date_max": info["date_max"],
                "frequency": info["frequency"],
                "overall_missing_rate": info["overall_missing_rate"],
                "max_missing_rate": info["max_missing_rate"],
                "max_missing_column": info["max_missing_column"],
                "columns_over_50pct_missing": info["columns_over_50pct_missing"],
                "columns_preview": " | ".join(info["columns"][:12]),
            }
        )

        for col, missing_rate in info["missing_rate_by_column"].items():
            missing_rows.append(
                {
                    "sheet": sheet,
                    "column": col,
                    "missing_rate": missing_rate,
                }
            )

    return pd.DataFrame(sheet_rows), pd.DataFrame(missing_rows)
