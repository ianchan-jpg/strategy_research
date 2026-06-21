"""报告输出工具。"""

from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    """保存结果表。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_line_plot(df: pd.DataFrame, x_col: str, y_col: str, path: str | Path, title: str) -> None:
    """保存简单折线图。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(df[x_col], df[y_col])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_bar_plot(df: pd.DataFrame, x_col: str, y_col: str, path: str | Path, title: str) -> None:
    """保存柱状图。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.bar(df[x_col].astype(str), df[y_col])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
