"""周度权益候选因子初步批量测试。

本脚本在单因子闭环基础上扩展到 10-20 个候选因子。
只输出测试和筛选标记，不删除原始数据、不删除因子。
预测方向保持为：
Factor(t) -> RelativeReturn(t+1)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PROJECT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
SCRIPT_DIR = PROJECT_DIR / "scripts"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from factor_processor import preprocess_factors  # noqa: E402
from factor_selector import factor_correlation, find_high_corr_pairs  # noqa: E402
from factor_tester import grouped_return, ols_regression, rank_ic_with_pvalue, rolling_ols  # noqa: E402
from report import save_table  # noqa: E402
from run_single_factor import (  # noqa: E402
    DATE_COL,
    FACTOR_Z_COL,
    ROLLING_WINDOW,
    STRATEGY_SHEET,
    TARGET_COL,
    WEEKLY_SHEET,
    build_next_relative_return,
    choose_weekly_equity_factor,
)


BATCH_FACTOR_COUNT = 15
HIGH_CORR_THRESHOLD = 0.7


def test_one_factor(weekly_df: pd.DataFrame, target_df: pd.DataFrame, factor_name: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """对单个候选因子做完整检验，返回汇总、滚动结果和分组结果。"""
    factor_df = weekly_df[[DATE_COL, factor_name]].copy()
    factor_df[DATE_COL] = pd.to_datetime(factor_df[DATE_COL])
    factor_df[factor_name] = pd.to_numeric(factor_df[factor_name], errors="coerce")

    merged = pd.merge(factor_df, target_df[[DATE_COL, TARGET_COL]], on=DATE_COL, how="inner").sort_values(DATE_COL)
    merged = merged.dropna(subset=[factor_name, TARGET_COL]).reset_index(drop=True)
    if merged.empty:
        summary = {
            "factor": factor_name,
            "sample_count": 0,
            "rank_ic": None,
            "rank_ic_pvalue": None,
            "ols_beta": None,
            "ols_t_beta": None,
            "ols_p_beta": None,
            "ols_r2": None,
            "rolling_beta_positive_ratio": None,
            "group_return_monotonic": False,
        }
        return summary, pd.DataFrame(), pd.DataFrame()

    processed = preprocess_factors(merged.rename(columns={factor_name: FACTOR_Z_COL}), [FACTOR_Z_COL])
    processed["factor_raw"] = merged[factor_name]
    processed["factor"] = factor_name

    ic_result = rank_ic_with_pvalue(processed[FACTOR_Z_COL], processed[TARGET_COL])
    ols_result = ols_regression(processed[FACTOR_Z_COL], processed[TARGET_COL])
    rolling_result = rolling_ols(processed, DATE_COL, FACTOR_Z_COL, TARGET_COL, window=ROLLING_WINDOW)
    group_result, is_monotonic = grouped_return(processed, FACTOR_Z_COL, TARGET_COL, n_groups=5)

    rolling_positive_ratio = rolling_result["beta"].gt(0).mean() if not rolling_result.empty else None
    summary = {
        "factor": factor_name,
        "sample_count": len(processed),
        "date_start": processed[DATE_COL].min().date().isoformat(),
        "date_end": processed[DATE_COL].max().date().isoformat(),
        "rank_ic": ic_result["rank_ic"],
        "rank_ic_pvalue": ic_result["rank_ic_pvalue"],
        "ols_beta": ols_result["beta"],
        "ols_t_beta": ols_result["t_beta"],
        "ols_p_beta": ols_result["p_beta"],
        "ols_r2": ols_result["r2"],
        "rolling_beta_positive_ratio": rolling_positive_ratio,
        "group_return_monotonic": is_monotonic,
    }

    rolling_result.insert(0, "factor", factor_name)
    group_result.insert(0, "factor", factor_name)
    return summary, rolling_result, group_result


def add_screening_label(summary: pd.DataFrame, high_corr_pairs: pd.DataFrame) -> pd.DataFrame:
    """给候选因子打初步筛选标签，不自动删除任何因子。"""
    out = summary.copy()
    out["abs_rank_ic"] = out["rank_ic"].abs()
    out["screening_label"] = "证据不足"
    out["screening_reason"] = "Rank IC 或回归证据较弱"

    useful = (
        (out["sample_count"] >= 200)
        & (out["rank_ic_pvalue"] <= 0.1)
        & (out["ols_p_beta"] <= 0.1)
        & (out["rolling_beta_positive_ratio"].between(0.6, 1.0) | out["rolling_beta_positive_ratio"].between(0.0, 0.4))
    )
    review = (
        (out["sample_count"] >= 200)
        & ((out["rank_ic_pvalue"] <= 0.2) | (out["ols_p_beta"] <= 0.2) | out["group_return_monotonic"])
        & ~useful
    )
    out.loc[useful, "screening_label"] = "保留"
    out.loc[useful, "screening_reason"] = "统计证据相对较强，进入后续复核"
    out.loc[review, "screening_label"] = "待复核"
    out.loc[review, "screening_reason"] = "存在部分有效信号，但证据不够完整"

    duplicate_factors: set[str] = set()
    if not high_corr_pairs.empty:
        ranked = out.sort_values(["abs_rank_ic", "sample_count"], ascending=[False, False]).reset_index(drop=True)
        rank_map = {factor: idx for idx, factor in enumerate(ranked["factor"])}
        for _, row in high_corr_pairs.iterrows():
            left = row["factor_1"]
            right = row["factor_2"]
            if rank_map.get(left, 10**9) <= rank_map.get(right, 10**9):
                duplicate_factors.add(right)
            else:
                duplicate_factors.add(left)

    mask_duplicate = out["factor"].isin(duplicate_factors) & (out["screening_label"] != "保留")
    out.loc[mask_duplicate, "screening_label"] = "重复"
    out.loc[mask_duplicate, "screening_reason"] = f"与其他候选因子 Spearman 相关性绝对值超过 {HIGH_CORR_THRESHOLD}"

    return out.drop(columns=["abs_rank_ic"]).sort_values(["screening_label", "rank_ic_pvalue", "ols_p_beta"])


def save_heatmap(corr: pd.DataFrame, path: Path) -> None:
    """保存候选因子相关性热力图。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Spearman corr")
    plt.xticks(range(len(corr.columns)), range(1, len(corr.columns) + 1), rotation=0)
    plt.yticks(range(len(corr.index)), range(1, len(corr.index) + 1))
    plt.title("Candidate Factor Spearman Correlation")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_rank_ic_bar(summary: pd.DataFrame, path: Path) -> None:
    """保存 Rank IC 柱状图。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = summary.sort_values("rank_ic")
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(plot_df)), plot_df["rank_ic"])
    plt.yticks(range(len(plot_df)), range(1, len(plot_df) + 1))
    plt.xlabel("Rank IC")
    plt.title("Batch Candidate Rank IC")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_batch_report(path: Path, selected_factors: list[str], summary: pd.DataFrame, high_corr_pairs: pd.DataFrame) -> None:
    """生成批量初筛中文报告。"""
    label_counts = summary["screening_label"].value_counts().to_dict()
    lines = [
        "# 周度权益候选因子批量初筛报告",
        "",
        "## 测试口径",
        "",
        f"- 因子来源：`factors.xlsx` 的 `{WEEKLY_SHEET}` 工作表",
        "- 目标变量：下一期 `量化-均衡 - 主观-均衡`",
        "- 预测方向：`Factor(t) -> RelativeReturn(t+1)`",
        f"- 候选因子数量：{len(selected_factors)}",
        f"- 滚动回归窗口：{ROLLING_WINDOW} 周",
        f"- 高相关阈值：Spearman 相关系数绝对值 > {HIGH_CORR_THRESHOLD}",
        "",
        "## 初步筛选标签统计",
        "",
    ]
    for label in ["保留", "重复", "证据不足", "待复核"]:
        lines.append(f"- {label}：{label_counts.get(label, 0)}")

    best = summary.sort_values("rank_ic_pvalue").head(5)
    lines.extend(["", "## Rank IC p值较低的前5个因子", ""])
    for _, row in best.iterrows():
        lines.append(
            f"- `{row['factor']}`：Rank IC={row['rank_ic']:.6f}，p值={row['rank_ic_pvalue']:.6f}，标签={row['screening_label']}"
        )

    lines.extend(["", "## 高相关因子对", ""])
    if high_corr_pairs.empty:
        lines.append("- 未发现绝对相关系数大于阈值的因子对。")
    else:
        for _, row in high_corr_pairs.head(30).iterrows():
            lines.append(f"- `{row['factor_1']}` vs `{row['factor_2']}`：corr={row['corr']:.6f}")

    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `outputs/tables/batch_factor_summary.csv`",
            "- `outputs/tables/batch_factor_screening.csv`",
            "- `outputs/tables/batch_factor_corr_spearman.csv`",
            "- `outputs/tables/batch_factor_high_corr_pairs.csv`",
            "- `outputs/tables/batch_factor_rolling_beta.csv`",
            "- `outputs/tables/batch_factor_group_returns.csv`",
            "- `outputs/figures/batch_factor_rank_ic.png`",
            "- `outputs/figures/batch_factor_corr_heatmap.png`",
            "",
            "本次只做初步测试和标记，不自动删除原始数据或因子。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """运行 10-20 个周度权益候选因子的初步批量测试。"""
    excel_path = ROOT_DIR / "data" / "factors.xlsx"
    tables_dir = PROJECT_DIR / "outputs" / "tables"
    figures_dir = PROJECT_DIR / "outputs" / "figures"
    reports_dir = PROJECT_DIR / "outputs" / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    weekly_df = pd.read_excel(excel_path, sheet_name=WEEKLY_SHEET)
    strategy_df = pd.read_excel(excel_path, sheet_name=STRATEGY_SHEET)
    target_df = build_next_relative_return(strategy_df)

    _, candidates = choose_weekly_equity_factor(weekly_df, strategy_df)
    selected_factors = candidates.head(BATCH_FACTOR_COUNT)["factor"].tolist()
    save_table(candidates, tables_dir / "batch_factor_candidates_all.csv")
    save_table(candidates.head(BATCH_FACTOR_COUNT), tables_dir / "batch_factor_candidates_selected.csv")

    summary_rows = []
    rolling_tables = []
    group_tables = []
    aligned_factors = []
    for factor_name in selected_factors:
        summary, rolling_result, group_result = test_one_factor(weekly_df, target_df, factor_name)
        summary_rows.append(summary)
        if not rolling_result.empty:
            rolling_tables.append(rolling_result)
        if not group_result.empty:
            group_tables.append(group_result)

        aligned = weekly_df[[DATE_COL, factor_name]].copy()
        aligned[DATE_COL] = pd.to_datetime(aligned[DATE_COL])
        aligned[factor_name] = pd.to_numeric(aligned[factor_name], errors="coerce")
        aligned_factors.append(aligned)

    summary_df = pd.DataFrame(summary_rows)

    corr_base = aligned_factors[0]
    for aligned in aligned_factors[1:]:
        corr_base = pd.merge(corr_base, aligned, on=DATE_COL, how="outer")
    corr = factor_correlation(corr_base, selected_factors, method="spearman")
    high_corr_pairs = find_high_corr_pairs(corr, threshold=HIGH_CORR_THRESHOLD)

    screening_df = add_screening_label(summary_df, high_corr_pairs)

    save_table(summary_df, tables_dir / "batch_factor_summary.csv")
    save_table(screening_df, tables_dir / "batch_factor_screening.csv")
    save_table(corr.reset_index().rename(columns={"index": "factor"}), tables_dir / "batch_factor_corr_spearman.csv")
    save_table(high_corr_pairs, tables_dir / "batch_factor_high_corr_pairs.csv")
    save_table(pd.concat(rolling_tables, ignore_index=True) if rolling_tables else pd.DataFrame(), tables_dir / "batch_factor_rolling_beta.csv")
    save_table(pd.concat(group_tables, ignore_index=True) if group_tables else pd.DataFrame(), tables_dir / "batch_factor_group_returns.csv")

    save_rank_ic_bar(summary_df, figures_dir / "batch_factor_rank_ic.png")
    save_heatmap(corr, figures_dir / "batch_factor_corr_heatmap.png")
    write_batch_report(reports_dir / "batch_factor_report.md", selected_factors, screening_df, high_corr_pairs)

    print("周度权益候选因子批量初筛完成。")
    print(f"测试因子数量：{len(selected_factors)}")
    print(f"高相关因子对数量：{len(high_corr_pairs)}")
    print(f"报告：{reports_dir / 'batch_factor_report.md'}")


if __name__ == "__main__":
    main()
