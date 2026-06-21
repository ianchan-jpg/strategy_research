"""单因子最小闭环。

本脚本只做一个因子的完整检验，不做批量检验，也不自动删除因子。
使用逻辑：
Factor(t) -> RelativeReturn(t+1)
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PROJECT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))
sys.path.insert(0, str(SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from factor_processor import preprocess_factors  # noqa: E402
from factor_tester import grouped_return, ols_regression, rank_ic_with_pvalue, rolling_ols  # noqa: E402
from report import save_bar_plot, save_line_plot, save_table  # noqa: E402


DATE_COL = "date"
WEEKLY_SHEET = "周度序列"
STRATEGY_SHEET = "策略指数"
SUBJECTIVE_COL = "主观-均衡"
QUANT_COL = "量化-均衡"
TARGET_COL = "next_relative_return"
FACTOR_Z_COL = "factor_z"
ROLLING_WINDOW = 52


def choose_weekly_equity_factor(weekly_df: pd.DataFrame, strategy_df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """自动选择一个完整度较高、逻辑清晰的权益量化因子。"""
    weekly = weekly_df.copy()
    strategy = strategy_df.copy()
    weekly[DATE_COL] = pd.to_datetime(weekly[DATE_COL])
    strategy[DATE_COL] = pd.to_datetime(strategy[DATE_COL])

    start_date = max(weekly[DATE_COL].min(), strategy[DATE_COL].min())
    end_date = min(weekly[DATE_COL].max(), strategy[DATE_COL].max())
    weekly = weekly[(weekly[DATE_COL] >= start_date) & (weekly[DATE_COL] <= end_date)]

    equity_keywords = ["A股", "全A", "沪深300", "中证", "zz2000", "微盘股", "000300", "000852", "000905", "881001"]
    preferred_keywords = ["截面波动"]
    rows = []

    for col in weekly.columns:
        if col == DATE_COL:
            continue
        col_name = str(col)
        is_equity = any(key in col_name for key in equity_keywords)
        is_preferred = any(key in col_name for key in preferred_keywords)
        if not is_equity:
            continue

        values = pd.to_numeric(weekly[col], errors="coerce")
        valid_count = int(values.notna().sum())
        missing_rate = round(float(values.isna().mean()), 4)
        if valid_count < 52:
            continue

        # 优先截面波动；再优先全 A；再优先非“变化”类原始指标；最后看完整度。
        score = 0
        score += 1000 if is_preferred else 0
        score += 100 if ("全A" in col_name or "881001" in col_name) else 0
        score += 20 if "变化" not in col_name else 0
        score += valid_count
        rows.append(
            {
                "factor": col_name,
                "is_preferred_cross_section_vol": is_preferred,
                "valid_count": valid_count,
                "missing_rate": missing_rate,
                "score": score,
            }
        )

    candidates = pd.DataFrame(rows).sort_values(["score", "valid_count", "factor"], ascending=[False, False, True])
    if candidates.empty:
        raise ValueError("周度序列中没有找到可用的权益因子候选。")

    return str(candidates.iloc[0]["factor"]), candidates


def build_next_relative_return(strategy_df: pd.DataFrame) -> pd.DataFrame:
    """构造下一期相对收益：下一期(量化-均衡 - 主观-均衡)。"""
    out = strategy_df[[DATE_COL, SUBJECTIVE_COL, QUANT_COL]].copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL])
    out = out.sort_values(DATE_COL)
    out["current_relative_return"] = out[QUANT_COL] - out[SUBJECTIVE_COL]
    out[TARGET_COL] = out["current_relative_return"].shift(-1)
    return out


def save_scatter_plot(df: pd.DataFrame, path: Path, factor_col: str, target_col: str) -> None:
    """保存因子与下一期相对收益的散点图。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = df[[factor_col, target_col]].dropna()
    plt.figure(figsize=(7, 5))
    plt.scatter(valid[factor_col], valid[target_col], alpha=0.65)
    plt.xlabel(factor_col)
    plt.ylabel(target_col)
    plt.title("Factor vs Next Relative Return")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_report(
    path: Path,
    factor_name: str,
    data: pd.DataFrame,
    ic_value: float,
    ic_pvalue: float,
    ols_result: pd.Series,
    rolling_result: pd.DataFrame,
    group_result: pd.DataFrame,
    is_monotonic: bool,
    notes: list[str],
) -> None:
    """生成中文 Markdown 报告。"""
    rolling_beta_positive_ratio = rolling_result["beta"].gt(0).mean() if not rolling_result.empty else float("nan")
    rolling_beta_mean = rolling_result["beta"].mean() if not rolling_result.empty else float("nan")
    rolling_beta_std = rolling_result["beta"].std() if not rolling_result.empty else float("nan")

    lines = [
        "# 单因子最小闭环报告",
        "",
        "## 基本设置",
        "",
        f"- 使用因子：`{factor_name}`",
        f"- 因子来源：`factors.xlsx` 的 `{WEEKLY_SHEET}` 工作表",
        f"- 目标变量：下一期 `{QUANT_COL} - {SUBJECTIVE_COL}`",
        "- 预测方向：`Factor(t) -> RelativeReturn(t+1)`",
        f"- 数据区间：{data[DATE_COL].min().date()} 至 {data[DATE_COL].max().date()}",
        f"- 有效样本数：{len(data)}",
        "",
        "## Rank IC",
        "",
        f"- Rank IC：{ic_value:.6f}",
        f"- Rank IC p值：{ic_pvalue:.6f}",
        "",
        "## 全样本 OLS 回归",
        "",
        "- 回归形式：`next_relative_return = alpha + beta * standardized_factor + error`",
        f"- alpha：{ols_result['alpha']:.6f}",
        f"- beta：{ols_result['beta']:.6f}",
        f"- beta t值：{ols_result['t_beta']:.6f}",
        f"- beta p值：{ols_result['p_beta']:.6f}",
        f"- R²：{ols_result['r2']:.6f}",
        "",
        "## 52周滚动回归稳定性",
        "",
        f"- 滚动窗口数：{len(rolling_result)}",
        f"- 滚动 beta 均值：{rolling_beta_mean:.6f}",
        f"- 滚动 beta 标准差：{rolling_beta_std:.6f}",
        f"- 滚动 beta 为正比例：{rolling_beta_positive_ratio:.2%}",
        "",
        "## 分组收益",
        "",
        f"- 分组数量：{len(group_result)}",
        f"- 分组收益是否单调：{'是' if is_monotonic else '否'}",
        "",
        "分组收益表已保存到 `outputs/tables/single_factor_group_returns.csv`。",
        "52周滚动 beta 明细已保存到 `outputs/tables/single_factor_rolling_ols_52w.csv`。",
        "对齐数据前20行已保存到 `outputs/tables/single_factor_aligned_head20.csv`。",
        "",
        "## 记录的问题",
        "",
    ]

    if notes:
        lines.extend([f"- {note}" for note in notes])
    else:
        lines.append("- 暂无。")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """运行单因子最小闭环。"""
    excel_path = ROOT_DIR / "data" / "factors.xlsx"
    tables_dir = PROJECT_DIR / "outputs" / "tables"
    figures_dir = PROJECT_DIR / "outputs" / "figures"
    reports_dir = PROJECT_DIR / "outputs" / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    weekly_df = pd.read_excel(excel_path, sheet_name=WEEKLY_SHEET)
    strategy_df = pd.read_excel(excel_path, sheet_name=STRATEGY_SHEET)

    factor_name, candidates = choose_weekly_equity_factor(weekly_df, strategy_df)
    save_table(candidates, tables_dir / "single_factor_candidates.csv")

    factor_df = weekly_df[[DATE_COL, factor_name]].copy()
    factor_df[DATE_COL] = pd.to_datetime(factor_df[DATE_COL])
    factor_df[factor_name] = pd.to_numeric(factor_df[factor_name], errors="coerce")

    target_df = build_next_relative_return(strategy_df)
    merged = pd.merge(factor_df, target_df[[DATE_COL, TARGET_COL]], on=DATE_COL, how="inner").sort_values(DATE_COL)
    merged = merged.dropna(subset=[factor_name, TARGET_COL]).reset_index(drop=True)

    processed = preprocess_factors(merged.rename(columns={factor_name: FACTOR_Z_COL}), [FACTOR_Z_COL])
    processed["factor_raw"] = merged[factor_name]
    processed["factor_name"] = factor_name

    ic_result = rank_ic_with_pvalue(processed[FACTOR_Z_COL], processed[TARGET_COL])
    ic_value = ic_result["rank_ic"]
    ic_pvalue = ic_result["rank_ic_pvalue"]
    ols_result = ols_regression(processed[FACTOR_Z_COL], processed[TARGET_COL])
    rolling_result = rolling_ols(processed, DATE_COL, FACTOR_Z_COL, TARGET_COL, window=ROLLING_WINDOW)
    group_result, is_monotonic = grouped_return(processed, FACTOR_Z_COL, TARGET_COL, n_groups=5)

    summary = pd.DataFrame(
        [
            {
                "factor": factor_name,
                "date_start": processed[DATE_COL].min().date().isoformat(),
                "date_end": processed[DATE_COL].max().date().isoformat(),
                "sample_count": len(processed),
                "rank_ic": ic_value,
                "rank_ic_pvalue": ic_pvalue,
                "alpha": ols_result["alpha"],
                "beta": ols_result["beta"],
                "t_beta": ols_result["t_beta"],
                "p_beta": ols_result["p_beta"],
                "r2": ols_result["r2"],
                "rolling_window": ROLLING_WINDOW,
                "rolling_beta_mean": rolling_result["beta"].mean() if not rolling_result.empty else None,
                "rolling_beta_std": rolling_result["beta"].std() if not rolling_result.empty else None,
                "group_return_monotonic": is_monotonic,
            }
        ]
    )

    save_table(processed[[DATE_COL, "factor_name", "factor_raw", FACTOR_Z_COL, TARGET_COL]], tables_dir / "single_factor_dataset.csv")
    save_table(processed[[DATE_COL, "factor_name", "factor_raw", FACTOR_Z_COL, TARGET_COL]].head(20), tables_dir / "single_factor_aligned_head20.csv")
    save_table(summary, tables_dir / "single_factor_summary.csv")
    save_table(rolling_result, tables_dir / "single_factor_rolling_ols_52w.csv")
    save_table(group_result, tables_dir / "single_factor_group_returns.csv")

    save_scatter_plot(processed, figures_dir / "single_factor_scatter.png", FACTOR_Z_COL, TARGET_COL)
    save_line_plot(rolling_result, "date", "beta", figures_dir / "single_factor_rolling_beta.png", "52W Rolling OLS Beta")
    save_line_plot(rolling_result, "date", "r2", figures_dir / "single_factor_rolling_r2.png", "52W Rolling OLS R2")
    save_bar_plot(group_result, "group", "mean_next_relative_return", figures_dir / "single_factor_group_returns.png", "Grouped Next Relative Return")

    notes = []
    if "截面波动" in factor_name and "881001" in factor_name:
        notes.append("自动选择结果为 881001.WI 相关的全 A 权益截面波动指标，符合优先使用截面波动类指标的要求。")
    elif "截面波动" in factor_name:
        notes.append("自动选择结果为截面波动类指标，但不是 881001.WI 全 A 指标；可能是候选完整度或列名排序导致。")
    else:
        notes.append("未找到可用的截面波动类指标，脚本改用其他权益相关周度因子。")
    notes.append("策略指数中的 `主观-均衡` 和 `量化-均衡` 数值表现为周收益率，因此本脚本直接相减，没有再做净值收益率转换。")
    notes.append("目标变量通过 shift(-1) 构造，仅用当期因子解释下一期相对收益，未使用未来因子。")

    write_report(
        reports_dir / "single_factor_report.md",
        factor_name,
        processed,
        ic_value,
        ic_pvalue,
        ols_result,
        rolling_result,
        group_result,
        is_monotonic,
        notes,
    )

    print("单因子最小闭环完成。")
    print(f"使用因子：{factor_name}")
    print(f"有效样本数：{len(processed)}")
    print(f"Rank IC：{ic_value:.6f}，p值：{ic_pvalue:.6f}")
    print(f"OLS beta：{ols_result['beta']:.6f}，t值：{ols_result['t_beta']:.6f}，p值：{ols_result['p_beta']:.6f}，R2：{ols_result['r2']:.6f}")
    print(f"报告：{reports_dir / 'single_factor_report.md'}")


if __name__ == "__main__":
    main()
