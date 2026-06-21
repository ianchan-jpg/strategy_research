"""股票组周度权益量化因子全量批量检验。

口径：
- 只使用 `周度序列`；
- 目标变量为下一周 `量化-均衡 - 主观-均衡`；
- 候选因子来自指标目录中 `cls1=市场`、`cls2=权益量化`、`freq=W` 的股票市场环境相关指标；
- 不做因子合成，不做机器学习，不删除任何原始因子。
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
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
)


TABLES_DIR = PROJECT_DIR / "outputs" / "tables"
FIGURES_DIR = PROJECT_DIR / "outputs" / "figures"
REPORTS_DIR = PROJECT_DIR / "outputs" / "reports"

MIN_SAMPLE = 104
MAX_MISSING_RATE = 0.6
HIGH_CORR_THRESHOLD = 0.7


def normalize_text(value) -> str:
    """把目录字段转成便于匹配的字符串。"""
    if pd.isna(value):
        return ""
    return str(value).strip()


def get_eligible_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    """筛选指标目录中的周度权益量化股票市场环境指标。"""
    cat = catalog.copy()
    for col in ["cls1", "cls2", "freq", "code", "name", "cls3", "cls4", "description", "formula"]:
        if col in cat.columns:
            cat[col] = cat[col].map(normalize_text)

    eligible = cat[
        (cat["cls1"] == "市场")
        & (cat["cls2"] == "权益量化")
        & (cat["freq"].str.upper() == "W")
        & (cat["code"] != "")
    ].copy()
    return eligible


def match_catalog_row(column: str, eligible_catalog: pd.DataFrame) -> pd.Series | None:
    """用周度列名匹配指标目录中的 code。"""
    matches = []
    for _, row in eligible_catalog.iterrows():
        code = row["code"]
        if not code:
            continue
        if column == code or column.startswith(code + "_") or code in column:
            matches.append((len(code), row))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def build_candidate_universe(weekly_df: pd.DataFrame, target_df: pd.DataFrame, eligible_catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成测试候选和数据排除清单。"""
    weekly = weekly_df.copy()
    weekly[DATE_COL] = pd.to_datetime(weekly[DATE_COL], errors="coerce")
    target_dates = set(pd.to_datetime(target_df[DATE_COL], errors="coerce").dropna())

    candidates = []
    exclusions = []
    for col in weekly.columns:
        col_name = str(col)
        if col_name == DATE_COL:
            continue

        row = match_catalog_row(col_name, eligible_catalog)
        if row is None:
            exclusions.append(
                {
                    "factor": col_name,
                    "screening_label": "数据不足",
                    "exclude_reason": "不属于指标目录中的周度权益量化股票市场环境指标",
                }
            )
            continue

        raw_notna_count = int(weekly[col_name].notna().sum())
        values = pd.to_numeric(weekly[col_name], errors="coerce")
        numeric_notna_count = int(values.notna().sum())
        valid_mask = values.notna() & weekly[DATE_COL].isin(target_dates)
        valid_count = int(valid_mask.sum())
        missing_rate = round(float(1 - values.notna().mean()), 4)
        unique_count = int(values[valid_mask].nunique(dropna=True))

        base_info = {
            "factor": col_name,
            "indicator_id": row.get("indicator_id"),
            "code": row.get("code"),
            "name": row.get("name"),
            "cls1": row.get("cls1"),
            "cls2": row.get("cls2"),
            "cls3": row.get("cls3"),
            "cls4": row.get("cls4"),
            "description": row.get("description"),
            "formula": row.get("formula"),
            "valid_count": valid_count,
            "missing_rate": missing_rate,
            "unique_count": unique_count,
            "screening_label": "数据不足",
        }

        if raw_notna_count > 0 and numeric_notna_count == 0:
            exclusions.append({**base_info, "exclude_reason": "非数值列或无法转换为数值"})
        elif valid_count == 0:
            exclusions.append({**base_info, "exclude_reason": "无法与策略指数日期对齐"})
        elif valid_count < MIN_SAMPLE:
            exclusions.append({**base_info, "exclude_reason": f"样本不足，有效样本数低于{MIN_SAMPLE}"})
        elif missing_rate > MAX_MISSING_RATE:
            exclusions.append({**base_info, "exclude_reason": f"缺失率过高，超过{MAX_MISSING_RATE:.0%}"})
        elif unique_count <= 1:
            exclusions.append({**base_info, "exclude_reason": "常数列或有效取值过少"})
        else:
            candidates.append(base_info)

    return pd.DataFrame(candidates), pd.DataFrame(exclusions)


def test_one_factor(weekly_df: pd.DataFrame, target_df: pd.DataFrame, factor_name: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """运行单个因子的 Rank IC、OLS、滚动回归和分组收益。"""
    factor_df = weekly_df[[DATE_COL, factor_name]].copy()
    factor_df[DATE_COL] = pd.to_datetime(factor_df[DATE_COL], errors="coerce")
    factor_df[factor_name] = pd.to_numeric(factor_df[factor_name], errors="coerce")
    merged = pd.merge(factor_df, target_df[[DATE_COL, TARGET_COL]], on=DATE_COL, how="inner").sort_values(DATE_COL)
    merged = merged.dropna(subset=[factor_name, TARGET_COL]).reset_index(drop=True)

    processed = preprocess_factors(merged.rename(columns={factor_name: FACTOR_Z_COL}), [FACTOR_Z_COL])
    processed["factor_raw"] = merged[factor_name]
    processed["factor"] = factor_name

    ic = rank_ic_with_pvalue(processed[FACTOR_Z_COL], processed[TARGET_COL])
    ols = ols_regression(processed[FACTOR_Z_COL], processed[TARGET_COL])
    rolling = rolling_ols(processed, DATE_COL, FACTOR_Z_COL, TARGET_COL, window=ROLLING_WINDOW)
    groups, monotonic = grouped_return(processed, FACTOR_Z_COL, TARGET_COL, n_groups=5)

    rolling_positive_ratio = rolling["beta"].gt(0).mean() if not rolling.empty else None
    summary = {
        "factor": factor_name,
        "sample_count": len(processed),
        "date_start": processed[DATE_COL].min().date().isoformat(),
        "date_end": processed[DATE_COL].max().date().isoformat(),
        "rank_ic": ic["rank_ic"],
        "rank_ic_pvalue": ic["rank_ic_pvalue"],
        "ols_alpha": ols["alpha"],
        "ols_beta": ols["beta"],
        "ols_t_beta": ols["t_beta"],
        "ols_p_beta": ols["p_beta"],
        "ols_r2": ols["r2"],
        "rolling_beta_positive_ratio": rolling_positive_ratio,
        "rolling_beta_mean": rolling["beta"].mean() if not rolling.empty else None,
        "rolling_beta_std": rolling["beta"].std() if not rolling.empty else None,
        "group_return_monotonic": monotonic,
    }
    rolling.insert(0, "factor", factor_name)
    groups.insert(0, "factor", factor_name)
    return summary, rolling, groups, processed[[DATE_COL, "factor", "factor_raw", FACTOR_Z_COL, TARGET_COL]]


def stable_direction(row: pd.Series) -> bool:
    """滚动 beta 方向是否稳定。"""
    ratio = row["rolling_beta_positive_ratio"]
    return pd.notna(ratio) and (ratio >= 0.6 or ratio <= 0.4)


def assign_initial_label(summary: pd.DataFrame) -> pd.DataFrame:
    """先根据单因子统计结果打标签，不考虑高相关组。"""
    out = summary.copy()
    rank_sig = out["rank_ic_pvalue"] <= 0.1
    ols_sig = out["ols_p_beta"] <= 0.1
    stable = out.apply(stable_direction, axis=1)

    out["screening_label"] = "证据不足"
    out["screening_reason"] = "Rank IC、OLS或滚动稳定性证据不足"

    core = rank_sig & ols_sig & stable
    weak = (~core) & ((rank_sig ^ ols_sig) | ((out["rank_ic_pvalue"] <= 0.2) & stable) | ((out["ols_p_beta"] <= 0.2) & stable))

    out.loc[core, "screening_label"] = "核心候选"
    out.loc[core, "screening_reason"] = "Rank IC和OLS均显著，且滚动方向较稳定"
    out.loc[weak, "screening_label"] = "弱候选"
    out.loc[weak, "screening_reason"] = "存在部分统计证据，但未同时满足核心候选条件"
    return out


def high_corr_clusters(factors: list[str], high_corr_pairs: pd.DataFrame) -> dict[str, int]:
    """把高相关因子对合并成相关性组。"""
    parent = {factor: factor for factor in factors}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        if a not in parent or b not in parent:
            return
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for _, row in high_corr_pairs.iterrows():
        union(row["factor_1"], row["factor_2"])

    roots = {}
    result = {}
    next_id = 1
    for factor in factors:
        root = find(factor)
        if root not in roots:
            roots[root] = next_id
            next_id += 1
        result[factor] = roots[root]
    return result


def choose_group_representatives(screening: pd.DataFrame, cluster_map: dict[str, int]) -> pd.DataFrame:
    """在高相关组中推荐代表指标，但不删除任何因子。"""
    out = screening.copy()
    out["corr_group"] = out["factor"].map(cluster_map)
    out["abs_rank_ic"] = out["rank_ic"].abs()
    out["is_high_corr_group"] = out.groupby("corr_group")["factor"].transform("size") > 1
    out["is_group_representative"] = False
    out["group_representative"] = None

    level_order = {"核心候选": 1, "弱候选": 2, "证据不足": 3, "组内代表": 4}
    ranked = out.copy()
    ranked["level_order"] = ranked["screening_label"].map(level_order).fillna(99)
    ranked = ranked.sort_values(
        ["corr_group", "level_order", "rank_ic_pvalue", "ols_p_beta", "abs_rank_ic", "sample_count"],
        ascending=[True, True, True, True, False, False],
    )
    reps = ranked.groupby("corr_group", as_index=False).first()[["corr_group", "factor"]].rename(columns={"factor": "representative"})
    rep_map = dict(zip(reps["corr_group"], reps["representative"]))
    out["group_representative"] = out["corr_group"].map(rep_map)
    out["is_group_representative"] = out["factor"] == out["group_representative"]

    nonsig_rep = (
        out["is_high_corr_group"]
        & out["is_group_representative"]
        & (out["rank_ic_pvalue"] > 0.1)
        & (out["ols_p_beta"] > 0.1)
    )
    out.loc[nonsig_rep, "screening_label"] = "组内代表"
    out.loc[nonsig_rep, "screening_reason"] = "自身不显著，但在高相关组内相对更适合作为观察代表"
    return out.drop(columns=["abs_rank_ic"])


def save_rank_ic_chart(summary: pd.DataFrame, path: Path) -> None:
    """保存 Rank IC 绝对值前30的图。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = summary.assign(abs_rank_ic=summary["rank_ic"].abs()).sort_values("abs_rank_ic", ascending=False).head(30)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(plot_df)), plot_df["rank_ic"])
    plt.yticks(range(len(plot_df)), range(1, len(plot_df) + 1))
    plt.xlabel("Rank IC")
    plt.title("Full Stock Factor Test - Top 30 Rank IC")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_report(path: Path, tested: pd.DataFrame, exclusions: pd.DataFrame, high_corr_pairs: pd.DataFrame) -> None:
    """输出简要中文报告。"""
    counts = tested["screening_label"].value_counts().to_dict()
    lines = [
        "# 股票组周度权益因子全量批量检验报告",
        "",
        "## 测试口径",
        "",
        "- 数据来源：`factors.xlsx` 的 `周度序列`。",
        "- 目标变量：下一周 `量化-均衡 - 主观-均衡`。",
        "- 候选范围：指标目录中 `cls1=市场`、`cls2=权益量化`、`freq=W`，并能匹配到周度序列的数值因子。",
        "- 预测方向：`Factor(t) -> RelativeReturn(t+1)`，不使用未来因子。",
        "- 暂不做因子合成和机器学习，不自动删除因子。",
        "",
        "## 样本处理",
        "",
        f"- 已测试因子数：{len(tested)}",
        f"- 排除因子数：{len(exclusions)}",
        f"- 高相关因子对数量：{len(high_corr_pairs)}",
        "",
        "## 分类统计",
        "",
    ]
    for label in ["核心候选", "弱候选", "组内代表", "证据不足", "数据不足"]:
        value = counts.get(label, 0)
        if label == "数据不足":
            value = len(exclusions)
        lines.append(f"- {label}：{value}")

    top = tested.assign(abs_rank_ic=tested["rank_ic"].abs()).sort_values("abs_rank_ic", ascending=False).head(10)
    lines.extend(["", "## |Rank IC| 前10因子", ""])
    for _, row in top.iterrows():
        lines.append(
            f"- `{row['factor']}`：Rank IC={row['rank_ic']:.6f}，Rank IC p值={row['rank_ic_pvalue']:.6f}，"
            f"OLS p值={row['ols_p_beta']:.6f}，标签={row['screening_label']}。"
        )

    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            "- `outputs/tables/full_stock_factor_candidates.csv`",
            "- `outputs/tables/full_stock_factor_exclusion_list.csv`",
            "- `outputs/tables/full_stock_factor_summary.csv`",
            "- `outputs/tables/full_stock_factor_screening.csv`",
            "- `outputs/tables/full_stock_factor_corr_spearman.csv`",
            "- `outputs/tables/full_stock_factor_high_corr_pairs.csv`",
            "- `outputs/tables/full_stock_factor_rolling_beta.csv`",
            "- `outputs/tables/full_stock_factor_group_returns.csv`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """运行股票组全量周度因子批量检验。"""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    excel_path = ROOT_DIR / "data" / "factors.xlsx"
    weekly_df = pd.read_excel(excel_path, sheet_name=WEEKLY_SHEET)
    strategy_df = pd.read_excel(excel_path, sheet_name=STRATEGY_SHEET)
    catalog = pd.read_excel(excel_path, sheet_name="指标目录")
    target_df = build_next_relative_return(strategy_df)
    target_df[DATE_COL] = pd.to_datetime(target_df[DATE_COL], errors="coerce")

    eligible_catalog = get_eligible_catalog(catalog)
    candidates, exclusions = build_candidate_universe(weekly_df, target_df, eligible_catalog)
    save_table(eligible_catalog, TABLES_DIR / "full_stock_factor_eligible_catalog.csv")
    save_table(candidates, TABLES_DIR / "full_stock_factor_candidates.csv")
    save_table(exclusions, TABLES_DIR / "full_stock_factor_exclusion_list.csv")

    summary_rows = []
    rolling_tables = []
    group_tables = []
    aligned_tables = []
    corr_base = weekly_df[[DATE_COL]].copy()
    corr_base[DATE_COL] = pd.to_datetime(corr_base[DATE_COL], errors="coerce")

    for _, candidate in candidates.iterrows():
        factor = candidate["factor"]
        summary, rolling, groups, aligned = test_one_factor(weekly_df, target_df, factor)
        for meta_col in ["indicator_id", "code", "name", "cls3", "cls4", "description", "formula"]:
            summary[meta_col] = candidate.get(meta_col)
        summary_rows.append(summary)
        rolling_tables.append(rolling)
        group_tables.append(groups)
        aligned_tables.append(aligned)

        series = weekly_df[[DATE_COL, factor]].copy()
        series[DATE_COL] = pd.to_datetime(series[DATE_COL], errors="coerce")
        series[factor] = pd.to_numeric(series[factor], errors="coerce")
        corr_base = pd.merge(corr_base, series, on=DATE_COL, how="outer")

    summary_df = pd.DataFrame(summary_rows)
    tested_factors = summary_df["factor"].tolist()
    corr = factor_correlation(corr_base, tested_factors, method="spearman")
    high_corr_pairs = find_high_corr_pairs(corr, threshold=HIGH_CORR_THRESHOLD)
    cluster_map = high_corr_clusters(tested_factors, high_corr_pairs)
    screening = assign_initial_label(summary_df)
    screening = choose_group_representatives(screening, cluster_map)
    screening = screening.sort_values(["screening_label", "rank_ic_pvalue", "ols_p_beta", "factor"])

    save_table(summary_df, TABLES_DIR / "full_stock_factor_summary.csv")
    save_table(screening, TABLES_DIR / "full_stock_factor_screening.csv")
    save_table(corr.reset_index().rename(columns={"index": "factor"}), TABLES_DIR / "full_stock_factor_corr_spearman.csv")
    save_table(high_corr_pairs, TABLES_DIR / "full_stock_factor_high_corr_pairs.csv")
    save_table(pd.concat(rolling_tables, ignore_index=True), TABLES_DIR / "full_stock_factor_rolling_beta.csv")
    save_table(pd.concat(group_tables, ignore_index=True), TABLES_DIR / "full_stock_factor_group_returns.csv")
    save_table(pd.concat(aligned_tables, ignore_index=True), TABLES_DIR / "full_stock_factor_aligned_data.csv")

    save_rank_ic_chart(screening, FIGURES_DIR / "full_stock_factor_top_rank_ic.png")
    write_report(REPORTS_DIR / "full_stock_factor_batch_report.md", screening, exclusions, high_corr_pairs)

    print("股票组周度权益因子全量批量检验完成。")
    print(f"测试因子数：{len(screening)}")
    print(f"排除因子数：{len(exclusions)}")
    print(f"高相关因子对：{len(high_corr_pairs)}")
    print(f"报告：{REPORTS_DIR / 'full_stock_factor_batch_report.md'}")


if __name__ == "__main__":
    main()
