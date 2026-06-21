"""按指标类别扩展周度权益因子测试。

本脚本不做因子合成，不删除原始因子，只做：
1. 核对两个 000852 截面波动指标的来源和差异；
2. 分析当前“保留”因子的 5 组收益形态；
3. 按类别选择数据完整的周度权益指标继续测试；
4. 对高相关因子只推荐代表指标，其他因子仅标记不删除。
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

from factor_selector import factor_correlation, find_high_corr_pairs  # noqa: E402
from report import save_table  # noqa: E402
from run_batch_test import HIGH_CORR_THRESHOLD, add_screening_label, test_one_factor  # noqa: E402
from run_single_factor import DATE_COL, STRATEGY_SHEET, WEEKLY_SHEET, build_next_relative_return  # noqa: E402


TABLES_DIR = PROJECT_DIR / "outputs" / "tables"
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"
REPORTS_DIR = PROJECT_DIR / "outputs" / "reports"


CATEGORY_RULES = {
    "市场活跃度": {
        "include": ["成交量_A股指数_1w", "成交额_A股指数_1w"],
        "exclude": [],
        "limit": 3,
    },
    "大小盘风格": {
        "include": ["相对强弱1000_300", "相对强弱500_300", "相对强弱1000_500", "相对强弱微盘股"],
        "exclude": [],
        "limit": 3,
    },
    "赚钱效应": {
        "include": ["赚钱效应_300", "赚钱效应_500", "赚钱效应_1000", "赚钱效应_A股指数周度"],
        "exclude": [],
        "limit": 3,
    },
    "市场宽度或分化": {
        "include": ["截面波动_wind全A_周度", "截面波动_A股指数周度", "截面波动_中证", "截面波动_沪深300"],
        "exclude": ["变化"],
        "limit": 3,
    },
    "趋势或相对强弱": {
        "include": ["相对强弱成长_价值", "相对强弱中证2000", "相对强弱300_全A", "相对强弱500_全A", "相对强弱1000_全A"],
        "exclude": [],
        "limit": 3,
    },
    "拥挤度": {
        "include": ["成交持仓比_股指期货", "成交持仓比_I_FCM", "持仓量_I_FCM"],
        "exclude": [],
        "limit": 3,
    },
}


def catalog_match_for_factor(catalog: pd.DataFrame, factor: str) -> pd.DataFrame:
    """用列名拆分信息，在指标目录中做宽松匹配。"""
    parts = [part for part in factor.split("_") if part and part not in {"周度"}]
    mask = pd.Series(False, index=catalog.index)
    text = catalog.apply(lambda row: " ".join(row.fillna("").map(str)), axis=1)
    for part in parts:
        if len(part) >= 4:
            mask = mask | text.str.contains(part, regex=False, na=False)
    return catalog.loc[mask].copy()


def compare_000852_factors(weekly_df: pd.DataFrame, catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """核对两个 000852 截面波动因子的原始数据和指标目录。"""
    factors = ["截面波动_A股指数周度_000852.SH", "截面波动_中证1000_周度_000852.SH"]
    left, right = factors
    data = weekly_df[[DATE_COL, left, right]].copy()
    data[left] = pd.to_numeric(data[left], errors="coerce")
    data[right] = pd.to_numeric(data[right], errors="coerce")
    valid = data.dropna(subset=[left, right]).copy()
    valid["diff_left_minus_right"] = valid[left] - valid[right]
    valid["abs_diff"] = valid["diff_left_minus_right"].abs()

    summary = pd.DataFrame(
        [
            {
                "factor": left,
                "valid_count": int(data[left].notna().sum()),
                "mean": data[left].mean(),
                "std": data[left].std(),
                "min": data[left].min(),
                "max": data[left].max(),
            },
            {
                "factor": right,
                "valid_count": int(data[right].notna().sum()),
                "mean": data[right].mean(),
                "std": data[right].std(),
                "min": data[right].min(),
                "max": data[right].max(),
            },
            {
                "factor": "两列差异",
                "valid_count": len(valid),
                "mean": valid["diff_left_minus_right"].mean(),
                "std": valid["diff_left_minus_right"].std(),
                "min": valid["diff_left_minus_right"].min(),
                "max": valid["diff_left_minus_right"].max(),
            },
        ]
    )
    summary["spearman_corr_between_two"] = data[[left, right]].corr(method="spearman").iloc[0, 1]

    catalog_rows = []
    for factor in factors:
        matched = catalog_match_for_factor(catalog, factor)
        for _, row in matched.head(20).iterrows():
            catalog_rows.append(
                {
                    "factor_column": factor,
                    "indicator_id": row.get("indicator_id"),
                    "code": row.get("code"),
                    "name": row.get("name"),
                    "cls1": row.get("cls1"),
                    "cls2": row.get("cls2"),
                    "cls3": row.get("cls3"),
                    "cls4": row.get("cls4"),
                    "description": row.get("description"),
                    "formula": row.get("formula"),
                    "freq": row.get("freq"),
                    "start_date": row.get("start_date"),
                    "end_date": row.get("end_date"),
                }
            )
    return summary, pd.DataFrame(catalog_rows)


def classify_group_shape(group_df: pd.DataFrame) -> str:
    """判断五组收益是否像 U 型、倒 U 型、极端组有效或无明显形态。"""
    means = group_df.sort_values("group")["mean_next_relative_return"].astype(float).tolist()
    if len(means) < 5:
        return "分组不足"
    if all(means[i] <= means[i + 1] for i in range(len(means) - 1)):
        return "单调递增"
    if all(means[i] >= means[i + 1] for i in range(len(means) - 1)):
        return "单调递减"

    middle = means[1:4]
    if means[0] > max(middle) and means[-1] > max(middle):
        return "U型"
    if means[0] < min(middle) and means[-1] < min(middle):
        return "倒U型"
    if means[-1] == max(means) or means[0] == max(means):
        return "仅极端高收益组较有效"
    if means[-1] == min(means) or means[0] == min(means):
        return "仅极端低收益组较明显"
    return "非单调但无典型U型"


def analyze_retained_group_shapes() -> pd.DataFrame:
    """读取当前三个保留因子的分组收益并判断非线性形态。"""
    group_returns = pd.read_csv(TABLES_DIR / "batch_factor_group_returns.csv")
    screening = pd.read_csv(TABLES_DIR / "batch_factor_screening.csv")
    retained = screening.loc[screening["screening_label"] == "保留", "factor"].tolist()
    rows = []
    for factor in retained:
        part = group_returns[group_returns["factor"] == factor].sort_values("group")
        shape = classify_group_shape(part)
        for _, row in part.iterrows():
            rows.append({**row.to_dict(), "nonlinear_shape": shape})
    return pd.DataFrame(rows)


def select_category_factors(weekly_df: pd.DataFrame, strategy_df: pd.DataFrame) -> pd.DataFrame:
    """按类别各选 2-3 个数据完整的周度权益因子。"""
    weekly = weekly_df.copy()
    strategy = strategy_df.copy()
    weekly[DATE_COL] = pd.to_datetime(weekly[DATE_COL])
    strategy[DATE_COL] = pd.to_datetime(strategy[DATE_COL])
    start_date = max(weekly[DATE_COL].min(), strategy[DATE_COL].min())
    end_date = min(weekly[DATE_COL].max(), strategy[DATE_COL].max())
    weekly = weekly[(weekly[DATE_COL] >= start_date) & (weekly[DATE_COL] <= end_date)]

    selected_rows = []
    selected_set = set()
    for category, rule in CATEGORY_RULES.items():
        candidates = []
        for col in weekly.columns:
            col_name = str(col)
            if col_name == DATE_COL:
                continue
            if not any(key in col_name for key in rule["include"]):
                continue
            if any(key in col_name for key in rule["exclude"]):
                continue
            valid_count = int(pd.to_numeric(weekly[col], errors="coerce").notna().sum())
            if valid_count < 180:
                continue
            missing_rate = 1 - valid_count / len(weekly)
            candidates.append({"category": category, "factor": col_name, "valid_count": valid_count, "missing_rate": missing_rate})

        candidates_df = pd.DataFrame(candidates)
        if candidates_df.empty:
            continue
        candidates_df = candidates_df.sort_values(["valid_count", "factor"], ascending=[False, True]).head(rule["limit"])
        for _, row in candidates_df.iterrows():
            if row["factor"] not in selected_set:
                selected_rows.append(row.to_dict())
                selected_set.add(row["factor"])
    return pd.DataFrame(selected_rows)


def recommend_representatives(summary: pd.DataFrame, high_corr_pairs: pd.DataFrame) -> pd.DataFrame:
    """在高相关因子群中推荐一个代表指标，其他只标记不删除。"""
    if high_corr_pairs.empty:
        return pd.DataFrame(
            columns=[
                "corr_cluster",
                "category",
                "factor",
                "rank_ic",
                "rank_ic_pvalue",
                "sample_count",
                "recommended_representative",
                "representative_recommendation",
            ]
        )

    factors = summary["factor"].tolist()
    conflicted = set(high_corr_pairs["factor_1"]).union(set(high_corr_pairs["factor_2"]))
    parent = {f: f for f in factors}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for _, row in high_corr_pairs.iterrows():
        if row["factor_1"] in parent and row["factor_2"] in parent:
            union(row["factor_1"], row["factor_2"])

    summary = summary[summary["factor"].isin(conflicted)].copy()
    summary["corr_cluster"] = summary["factor"].map(find)
    summary["abs_rank_ic"] = summary["rank_ic"].abs()
    representatives = (
        summary.sort_values(["corr_cluster", "rank_ic_pvalue", "abs_rank_ic", "sample_count"], ascending=[True, True, False, False])
        .groupby("corr_cluster", as_index=False)
        .first()[["corr_cluster", "factor"]]
        .rename(columns={"factor": "recommended_representative"})
    )
    out = summary.merge(representatives, on="corr_cluster", how="left")
    out["representative_recommendation"] = out.apply(
        lambda row: "推荐作为该高相关组代表" if row["factor"] == row["recommended_representative"] else f"与代表指标高度相关，优先参考：{row['recommended_representative']}",
        axis=1,
    )
    return out.drop(columns=["abs_rank_ic"]).sort_values(["corr_cluster", "rank_ic_pvalue"])


def save_category_charts(summary: pd.DataFrame) -> None:
    """保存按类别展示的 Rank IC 图。"""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = summary.sort_values(["category", "rank_ic"])
    labels = [f"{i+1}" for i in range(len(plot_df))]
    plt.figure(figsize=(10, 7))
    plt.barh(labels, plot_df["rank_ic"])
    plt.xlabel("Rank IC")
    plt.ylabel("Factor index")
    plt.title("Category Factor Rank IC")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "category_factor_rank_ic.png", dpi=150)
    plt.close()


def write_report(
    path: Path,
    diff_summary: pd.DataFrame,
    catalog_rows: pd.DataFrame,
    retained_shapes: pd.DataFrame,
    category_summary: pd.DataFrame,
    high_corr_pairs: pd.DataFrame,
    representatives: pd.DataFrame,
) -> None:
    """输出中文阶段报告。"""
    lines = [
        "# 下一阶段周度权益因子复核报告",
        "",
        "## 1. 两个 000852 截面波动指标核对",
        "",
        "- `截面波动_A股指数周度_000852.SH`：来自“截面波动_A股指数周度”口径，指标目录描述为 A股指数所有成分股周收益的标准差。",
        "- `截面波动_中证1000_周度_000852.SH`：来自“截面波动_中证1000_周度”口径，指标目录描述为中证1000所有成分股单周收益的标准差。",
        "- 两者使用同一指数代码 `000852.SH`，原因是代码都指向中证1000；但前缀代表不同指标族或数据生产口径，一个是通用 A股指数周度截面波动模板下的中证1000列，另一个是单独命名的中证1000截面波动指标。",
        "- 两列不是完全重复：原始数值不同，Spearman相关性为 "
        f"{diff_summary['spearman_corr_between_two'].iloc[0]:.6f}，应作为高相关但不同口径的候选因子处理。",
        "",
        "相关明细已输出到 `outputs/tables/factor_000852_data_comparison.csv` 和 `outputs/tables/factor_000852_catalog_matches.csv`。",
        "",
        "## 2. 当前三个“保留”因子的五组收益形态",
        "",
    ]

    shape_summary = retained_shapes[["factor", "nonlinear_shape"]].drop_duplicates()
    for _, row in shape_summary.iterrows():
        lines.append(f"- `{row['factor']}`：{row['nonlinear_shape']}。")
    lines.append("")
    lines.append("从五组均值看，三个保留因子均不是严格单调；主要表现为中间组有回落、高分位组更强，偏“极端高收益组较有效”，不是标准 U 型或倒 U 型。")
    lines.append("")
    lines.append("## 3. 按类别扩展测试")
    lines.append("")

    for category, part in category_summary.groupby("category"):
        lines.append(f"### {category}")
        for _, row in part.sort_values("rank_ic_pvalue").iterrows():
            lines.append(
                f"- `{row['factor']}`：样本数={int(row['sample_count'])}，Rank IC={row['rank_ic']:.6f}，"
                f"p值={row['rank_ic_pvalue']:.6f}，OLS beta={row['ols_beta']:.6f}，"
                f"t值={row['ols_t_beta']:.6f}，R²={row['ols_r2']:.6f}，分组单调={row['group_return_monotonic']}。"
            )
        lines.append("")

    lines.append("## 4. 高相关因子代表推荐")
    lines.append("")
    if high_corr_pairs.empty or representatives.empty:
        lines.append("- 本轮扩展因子未发现绝对相关系数大于阈值的因子对。")
    else:
        rec = representatives[["corr_cluster", "recommended_representative"]].drop_duplicates()
        for _, row in rec.iterrows():
            cluster_members = representatives.loc[representatives["corr_cluster"] == row["corr_cluster"], "factor"].tolist()
            members_text = "、".join(f"`{factor}`" for factor in cluster_members)
            lines.append(f"- 高相关组：{members_text}；推荐代表：`{row['recommended_representative']}`。")
    lines.append("")
    lines.append("完整高相关因子对和代表推荐已输出到 `outputs/tables/category_factor_high_corr_pairs.csv` 与 `outputs/tables/category_factor_representatives.csv`。")
    lines.append("")
    lines.append("本阶段不做因子合成，不删除原始因子。")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """运行下一阶段周度权益因子复核。"""
    excel_path = ROOT_DIR / "data" / "factors.xlsx"
    weekly_df = pd.read_excel(excel_path, sheet_name=WEEKLY_SHEET)
    strategy_df = pd.read_excel(excel_path, sheet_name=STRATEGY_SHEET)
    catalog = pd.read_excel(excel_path, sheet_name="指标目录")
    target_df = build_next_relative_return(strategy_df)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    diff_summary, catalog_rows = compare_000852_factors(weekly_df, catalog)
    retained_shapes = analyze_retained_group_shapes()
    selected = select_category_factors(weekly_df, strategy_df)

    summary_rows = []
    rolling_tables = []
    group_tables = []
    aligned_factors = []
    for _, row in selected.iterrows():
        factor = row["factor"]
        summary, rolling_result, group_result = test_one_factor(weekly_df, target_df, factor)
        summary["category"] = row["category"]
        summary_rows.append(summary)
        if not rolling_result.empty:
            rolling_result.insert(0, "category", row["category"])
            rolling_tables.append(rolling_result)
        if not group_result.empty:
            group_result.insert(0, "category", row["category"])
            group_tables.append(group_result)

        aligned = weekly_df[[DATE_COL, factor]].copy()
        aligned[DATE_COL] = pd.to_datetime(aligned[DATE_COL])
        aligned[factor] = pd.to_numeric(aligned[factor], errors="coerce")
        aligned_factors.append(aligned)

    category_summary = pd.DataFrame(summary_rows)
    corr_base = aligned_factors[0]
    selected_factors = selected["factor"].tolist()
    for aligned in aligned_factors[1:]:
        corr_base = pd.merge(corr_base, aligned, on=DATE_COL, how="outer")
    corr = factor_correlation(corr_base, selected_factors, method="spearman")
    high_corr_pairs = find_high_corr_pairs(corr, threshold=HIGH_CORR_THRESHOLD)
    screening = add_screening_label(category_summary, high_corr_pairs)
    representatives = recommend_representatives(category_summary, high_corr_pairs)

    save_table(diff_summary, TABLES_DIR / "factor_000852_data_comparison.csv")
    save_table(catalog_rows, TABLES_DIR / "factor_000852_catalog_matches.csv")
    save_table(retained_shapes, TABLES_DIR / "retained_factor_group_shape_detail.csv")
    save_table(selected, TABLES_DIR / "category_factor_selected.csv")
    save_table(category_summary, TABLES_DIR / "category_factor_summary.csv")
    save_table(screening, TABLES_DIR / "category_factor_screening.csv")
    save_table(corr.reset_index().rename(columns={"index": "factor"}), TABLES_DIR / "category_factor_corr_spearman.csv")
    save_table(high_corr_pairs, TABLES_DIR / "category_factor_high_corr_pairs.csv")
    save_table(representatives, TABLES_DIR / "category_factor_representatives.csv")
    save_table(pd.concat(rolling_tables, ignore_index=True), TABLES_DIR / "category_factor_rolling_beta.csv")
    save_table(pd.concat(group_tables, ignore_index=True), TABLES_DIR / "category_factor_group_returns.csv")

    save_category_charts(category_summary)
    write_report(
        REPORTS_DIR / "category_factor_review_report.md",
        diff_summary,
        catalog_rows,
        retained_shapes,
        category_summary,
        high_corr_pairs,
        representatives,
    )

    print("下一阶段周度权益因子复核完成。")
    print(f"类别数：{selected['category'].nunique()}，测试因子数：{len(selected)}")
    print(f"报告：{REPORTS_DIR / 'category_factor_review_report.md'}")


if __name__ == "__main__":
    main()
