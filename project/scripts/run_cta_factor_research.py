"""CTA组周度因子全量研究。

复用股票组已验证的统计框架，但不复用股票组最终因子。
目标变量：
    cta_relative_return = 量化期货 - 主观期货
预测方向：
    Factor(t) -> CTA RelativeReturn(t+1)

本阶段不做因子合成、机器学习或PCA。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PROJECT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_DIR / ".matplotlib"))
sys.path.insert(0, str(SRC_DIR))

from factor_processor import preprocess_factors  # noqa: E402
from factor_selector import factor_correlation, find_high_corr_pairs  # noqa: E402
from factor_tester import grouped_return, ols_regression, rank_ic_with_pvalue, rolling_ols  # noqa: E402
from report import save_table  # noqa: E402


DATE_COL = "date"
WEEKLY_SHEET = "周度序列"
STRATEGY_SHEET = "策略指数"
CTA_SUBJECTIVE_COL = "主观期货"
CTA_QUANT_COL = "量化期货"
TARGET_COL = "next_cta_relative_return"
FACTOR_Z_COL = "factor_z"
ROLLING_WINDOW = 52
MIN_SAMPLE = 104
MAX_MISSING_RATE = 0.6
HIGH_CORR_THRESHOLD = 0.7

TABLES_DIR = PROJECT_DIR / "outputs" / "tables"
REPORTS_DIR = PROJECT_DIR / "outputs" / "reports"

CTA_INCLUDE_KEYWORDS = [
    "南华",
    "wind大类商品",
    "wind商品",
    "期货因子",
    "基差",
    "展期",
    "期限结构",
    "成交持仓比",
    "持仓量",
    "成交量",
    "波动率",
    "相关系数_wind大类",
    "秩相关系数_wind大类",
    "周度TR",
    "商品",
]
STOCK_EXCLUDE_KEYWORDS = [
    "A股",
    "中信风格",
    "沪深300",
    "中证",
    "微盘股",
    "全A",
    "000300.SH",
    "000852.SH",
    "000905.SH",
    "932000.CSI",
    "881001.WI",
    "I_FCM",
    "股指期货",
]

DIMENSIONS = [
    "趋势强度",
    "时序动量",
    "波动率",
    "品种分化",
    "板块相关性",
    "成交量",
    "持仓量",
    "成交持仓比",
    "基差",
    "期限结构",
    "拥挤度",
    "跳空或反转",
]


def normalize(value) -> str:
    """转成字符串用于匹配。"""
    if pd.isna(value):
        return ""
    return str(value).strip()


def build_cta_target(strategy_df: pd.DataFrame) -> pd.DataFrame:
    """构造下一期CTA相对收益：下一期(量化期货 - 主观期货)。"""
    out = strategy_df[[DATE_COL, CTA_SUBJECTIVE_COL, CTA_QUANT_COL]].copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    out = out.sort_values(DATE_COL)
    out["cta_relative_return"] = out[CTA_QUANT_COL] - out[CTA_SUBJECTIVE_COL]
    out[TARGET_COL] = out["cta_relative_return"].shift(-1)
    return out[[DATE_COL, TARGET_COL]]


def catalog_match(column: str, catalog: pd.DataFrame) -> pd.Series | None:
    """按code匹配指标目录。"""
    matches = []
    for _, row in catalog.iterrows():
        code = normalize(row.get("code"))
        if not code:
            continue
        if column == code or column.startswith(code + "_") or code in column:
            matches.append((len(code), row))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def is_cta_related(column: str, row: pd.Series | None) -> bool:
    """判断是否与CTA、期货、商品相关。"""
    text = column
    if row is not None:
        text += " " + " ".join(normalize(row.get(col)) for col in ["name", "cls1", "cls2", "cls3", "cls4", "description", "formula"])
    if any(key in text for key in STOCK_EXCLUDE_KEYWORDS) and not any(key in text for key in ["南华", "wind大类商品", "期货因子", "商品", "基差", "展期"]):
        return False
    if row is not None and normalize(row.get("cls2")) == "期货":
        return True
    return any(key in text for key in CTA_INCLUDE_KEYWORDS)


def infer_dimension(row: pd.Series) -> str:
    """映射CTA研究维度。"""
    text = " ".join(normalize(row.get(col)) for col in ["factor", "name", "cls3", "cls4", "description", "formula", "code"])
    if "成交持仓比" in text:
        return "成交持仓比"
    if "持仓量" in text or "持仓变化" in text:
        return "持仓量"
    if "成交量" in text or "成交额" in text:
        return "成交量"
    if "基差" in text:
        return "基差"
    if "展期" in text or "期限结构" in text:
        return "期限结构"
    if "波动" in text or "周度TR" in text:
        return "波动率"
    if "截面" in text or "品种" in text:
        return "品种分化"
    if "相关系数" in text or "秩相关" in text or "板块趋同性" in text:
        return "板块相关性"
    if "时间序列动量" in text or "时序动量" in text:
        return "时序动量"
    if "动量" in text or "趋势" in text or "涨跌幅" in text or "均价突破" in text:
        return "趋势强度"
    if "拥挤" in text or "市场热度" in text or "投机" in text:
        return "拥挤度"
    if "跳空" in text or "反转" in text or "偏度" in text:
        return "跳空或反转"
    return "待人工确认"


def build_candidate_universe(weekly_df: pd.DataFrame, target_df: pd.DataFrame, catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成CTA候选清单和排除清单。"""
    weekly = weekly_df.copy()
    weekly[DATE_COL] = pd.to_datetime(weekly[DATE_COL], errors="coerce")
    target_dates = set(pd.to_datetime(target_df[DATE_COL], errors="coerce").dropna())

    candidates = []
    exclusions = []
    for col in weekly.columns:
        factor = str(col)
        if factor == DATE_COL:
            continue
        row = catalog_match(factor, catalog)
        if not is_cta_related(factor, row):
            exclusions.append(
                {
                    "factor": factor,
                    "screening_label": "数据不足",
                    "exclude_reason": "明显与CTA无关的股票因子或非期货商品指标",
                }
            )
            continue

        values = pd.to_numeric(weekly[factor], errors="coerce")
        valid_mask = values.notna() & weekly[DATE_COL].isin(target_dates)
        valid_count = int(valid_mask.sum())
        missing_rate = round(float(1 - values.notna().mean()), 4)
        unique_count = int(values[valid_mask].nunique(dropna=True))
        base = {
            "factor": factor,
            "indicator_id": row.get("indicator_id") if row is not None else None,
            "code": row.get("code") if row is not None else None,
            "name": row.get("name") if row is not None else None,
            "cls1": row.get("cls1") if row is not None else None,
            "cls2": row.get("cls2") if row is not None else None,
            "cls3": row.get("cls3") if row is not None else None,
            "cls4": row.get("cls4") if row is not None else None,
            "description": row.get("description") if row is not None else None,
            "formula": row.get("formula") if row is not None else None,
            "valid_count": valid_count,
            "missing_rate": missing_rate,
            "unique_count": unique_count,
        }
        base["dimension"] = infer_dimension(pd.Series(base))
        if valid_count == 0:
            exclusions.append({**base, "screening_label": "数据不足", "exclude_reason": "无法与CTA策略日期对齐"})
        elif valid_count < MIN_SAMPLE:
            exclusions.append({**base, "screening_label": "数据不足", "exclude_reason": f"样本不足，有效样本数低于{MIN_SAMPLE}"})
        elif missing_rate > MAX_MISSING_RATE:
            exclusions.append({**base, "screening_label": "数据不足", "exclude_reason": f"缺失率过高，超过{MAX_MISSING_RATE:.0%}"})
        elif unique_count <= 1:
            exclusions.append({**base, "screening_label": "数据不足", "exclude_reason": "常数列或有效取值过少"})
        else:
            candidates.append(base)
    return pd.DataFrame(candidates), pd.DataFrame(exclusions)


def test_one_factor(weekly_df: pd.DataFrame, target_df: pd.DataFrame, factor: str) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """运行单个CTA候选因子的完整检验。"""
    factor_df = weekly_df[[DATE_COL, factor]].copy()
    factor_df[DATE_COL] = pd.to_datetime(factor_df[DATE_COL], errors="coerce")
    factor_df[factor] = pd.to_numeric(factor_df[factor], errors="coerce")
    merged = pd.merge(factor_df, target_df, on=DATE_COL, how="inner").sort_values(DATE_COL)
    merged = merged.dropna(subset=[factor, TARGET_COL]).reset_index(drop=True)
    processed = preprocess_factors(merged.rename(columns={factor: FACTOR_Z_COL}), [FACTOR_Z_COL])
    processed["factor_raw"] = merged[factor]
    processed["factor"] = factor

    ic = rank_ic_with_pvalue(processed[FACTOR_Z_COL], processed[TARGET_COL])
    ols = ols_regression(processed[FACTOR_Z_COL], processed[TARGET_COL])
    rolling = rolling_ols(processed, DATE_COL, FACTOR_Z_COL, TARGET_COL, window=ROLLING_WINDOW)
    groups, monotonic = grouped_return(processed, FACTOR_Z_COL, TARGET_COL, n_groups=5)
    rolling_positive = rolling["beta"].gt(0).mean() if not rolling.empty else None
    summary = {
        "factor": factor,
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
        "rolling_beta_positive_ratio": rolling_positive,
        "rolling_beta_mean": rolling["beta"].mean() if not rolling.empty else None,
        "rolling_beta_std": rolling["beta"].std() if not rolling.empty else None,
        "group_return_monotonic": monotonic,
    }
    rolling.insert(0, "factor", factor)
    groups.insert(0, "factor", factor)
    return summary, rolling, groups, processed[[DATE_COL, "factor", "factor_raw", FACTOR_Z_COL, TARGET_COL]]


def add_labels(summary: pd.DataFrame, high_corr: pd.DataFrame) -> pd.DataFrame:
    """给CTA因子打核心候选、弱候选、组内代表、证据不足标签。"""
    out = summary.copy()
    stable = out["rolling_beta_positive_ratio"].ge(0.6) | out["rolling_beta_positive_ratio"].le(0.4)
    rank_sig = out["rank_ic_pvalue"].le(0.1)
    ols_sig = out["ols_p_beta"].le(0.1)
    out["screening_label"] = "证据不足"
    out["screening_reason"] = "Rank IC、OLS或滚动稳定性证据不足"
    core = rank_sig & ols_sig & stable
    weak = (~core) & ((rank_sig | ols_sig) | out["group_return_monotonic"])
    out.loc[core, "screening_label"] = "核心候选"
    out.loc[core, "screening_reason"] = "Rank IC和OLS均显著，且滚动方向较稳定"
    out.loc[weak, "screening_label"] = "弱候选"
    out.loc[weak, "screening_reason"] = "存在部分统计证据，但未同时满足核心候选条件"

    if not high_corr.empty:
        out["abs_rank_ic"] = out["rank_ic"].abs()
        ranked = out.sort_values(["screening_label", "rank_ic_pvalue", "ols_p_beta", "abs_rank_ic"], ascending=[True, True, True, False])
        priority = {factor: idx for idx, factor in enumerate(ranked["factor"])}
        duplicate = set()
        for _, row in high_corr.iterrows():
            left, right = row["factor_1"], row["factor_2"]
            duplicate.add(right if priority.get(left, 10**9) <= priority.get(right, 10**9) else left)
        mask = out["factor"].isin(duplicate) & (out["screening_label"] == "证据不足")
        out.loc[mask, "screening_label"] = "组内代表"
        out.loc[mask, "screening_reason"] = "自身证据不强，但与高相关组内其他指标信息接近，作为组内观察代表"
        out = out.drop(columns=["abs_rank_ic"])
    return out


def dimension_coverage(candidates: pd.DataFrame, exclusions: pd.DataFrame, screening: pd.DataFrame) -> pd.DataFrame:
    """CTA研究维度覆盖审计。"""
    rows = []
    for dim in DIMENSIONS:
        if dim == "拥挤度":
            crowding_dims = ["成交量", "持仓量", "成交持仓比", "拥挤度"]
            cand = candidates[candidates["dimension"].isin(crowding_dims)]
            excl = exclusions[exclusions["dimension"].isin(crowding_dims)]
            tested = screening[screening["dimension"].isin(crowding_dims)]
        else:
            cand = candidates[candidates["dimension"] == dim]
            excl = exclusions[exclusions["dimension"] == dim]
            tested = screening[screening["dimension"] == dim]
        core = tested[tested["screening_label"] == "核心候选"]
        if len(cand) + len(excl) == 0:
            status = "未找到合适指标"
        elif len(tested) == 0:
            status = "数据不足"
        elif len(core) > 0:
            status = "有核心候选"
        elif (tested["screening_label"] == "弱候选").any():
            status = "有弱候选"
        else:
            status = "已测试但未通过"
        rows.append(
            {
                "dimension": dim,
                "raw_candidate_count": len(cand) + len(excl),
                "tested_count": len(tested),
                "excluded_count": len(excl),
                "core_count": len(core),
                "weak_count": int((tested["screening_label"] == "弱候选").sum()),
                "coverage_status": status,
                "core_examples": " | ".join(core["factor"].head(5).tolist()),
            }
        )
    return pd.DataFrame(rows)


def representative_factors(screening: pd.DataFrame) -> pd.DataFrame:
    """每个维度选一个分类代表因子，不删除其他因子。"""
    rows = []
    priority = {"核心候选": 1, "弱候选": 2, "组内代表": 3, "证据不足": 4}
    tmp = screening.copy()
    tmp["priority"] = tmp["screening_label"].map(priority).fillna(99)
    tmp["abs_rank_ic"] = tmp["rank_ic"].abs()
    for dim, part in tmp.groupby("dimension"):
        best = part.sort_values(["priority", "rank_ic_pvalue", "ols_p_beta", "abs_rank_ic"], ascending=[True, True, True, False]).iloc[0]
        rows.append(
            {
                "dimension": dim,
                "representative_factor": best["factor"],
                "screening_label": best["screening_label"],
                "rank_ic": best["rank_ic"],
                "rank_ic_pvalue": best["rank_ic_pvalue"],
                "ols_p_beta": best["ols_p_beta"],
                "reason": "按标签优先级、Rank IC p值、OLS p值和|Rank IC|综合排序后选为该维度代表。",
            }
        )
    return pd.DataFrame(rows).sort_values(["dimension"])


def write_report(path: Path, coverage: pd.DataFrame, screening: pd.DataFrame, high_corr: pd.DataFrame) -> None:
    """中文报告。"""
    counts = screening["screening_label"].value_counts().to_dict()
    lines = [
        "# CTA组周度因子全量研究报告",
        "",
        "## 研究口径",
        "",
        "- 目标变量：下一期 `量化期货 - 主观期货`。",
        "- 预测方向：`Factor(t) -> CTA RelativeReturn(t+1)`。",
        "- 候选范围：周度序列中期货、商品、CTA相关的数值因子。",
        "- 本阶段不做因子合成、机器学习或PCA。",
        "",
        "## 检验结果概览",
        "",
        f"- 核心候选：{counts.get('核心候选', 0)}",
        f"- 弱候选：{counts.get('弱候选', 0)}",
        f"- 组内代表：{counts.get('组内代表', 0)}",
        f"- 证据不足：{counts.get('证据不足', 0)}",
        f"- 高相关因子对：{len(high_corr)}",
        "",
        "## 维度覆盖",
        "",
    ]
    for _, row in coverage.iterrows():
        lines.append(
            f"- {row['dimension']}：测试{row['tested_count']}个，排除{row['excluded_count']}个，核心{row['core_count']}个，状态={row['coverage_status']}。"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """运行CTA组全量因子研究。"""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    excel = ROOT_DIR / "data" / "factors.xlsx"
    weekly = pd.read_excel(excel, sheet_name=WEEKLY_SHEET)
    strategy = pd.read_excel(excel, sheet_name=STRATEGY_SHEET)
    catalog = pd.read_excel(excel, sheet_name="指标目录")
    target = build_cta_target(strategy)

    candidates, exclusions = build_candidate_universe(weekly, target, catalog)
    summary_rows, rolling_rows, group_rows, aligned_rows = [], [], [], []
    corr_base = weekly[[DATE_COL]].copy()
    corr_base[DATE_COL] = pd.to_datetime(corr_base[DATE_COL], errors="coerce")

    for _, cand in candidates.iterrows():
        factor = cand["factor"]
        summary, rolling, groups, aligned = test_one_factor(weekly, target, factor)
        for col in ["indicator_id", "code", "name", "cls2", "cls3", "cls4", "description", "formula", "dimension"]:
            summary[col] = cand.get(col)
        summary_rows.append(summary)
        rolling_rows.append(rolling)
        group_rows.append(groups)
        aligned_rows.append(aligned)

        series = weekly[[DATE_COL, factor]].copy()
        series[DATE_COL] = pd.to_datetime(series[DATE_COL], errors="coerce")
        series[factor] = pd.to_numeric(series[factor], errors="coerce")
        corr_base = pd.merge(corr_base, series, on=DATE_COL, how="outer")

    summary = pd.DataFrame(summary_rows)
    factors = summary["factor"].tolist()
    corr = factor_correlation(corr_base, factors, method="spearman") if factors else pd.DataFrame()
    high_corr = find_high_corr_pairs(corr, threshold=HIGH_CORR_THRESHOLD) if not corr.empty else pd.DataFrame()
    screening = add_labels(summary, high_corr) if not summary.empty else pd.DataFrame()
    coverage = dimension_coverage(candidates, exclusions, screening)
    reps = representative_factors(screening) if not screening.empty else pd.DataFrame()

    save_table(candidates, TABLES_DIR / "cta_factor_candidates.csv")
    save_table(exclusions, TABLES_DIR / "cta_factor_exclusion_list.csv")
    save_table(summary, TABLES_DIR / "cta_factor_summary.csv")
    save_table(screening, TABLES_DIR / "cta_factor_screening.csv")
    save_table(corr.reset_index().rename(columns={"index": "factor"}) if not corr.empty else pd.DataFrame(), TABLES_DIR / "cta_factor_corr_spearman.csv")
    save_table(high_corr, TABLES_DIR / "cta_factor_high_corr_pairs.csv")
    save_table(pd.concat(rolling_rows, ignore_index=True) if rolling_rows else pd.DataFrame(), TABLES_DIR / "cta_factor_rolling_beta.csv")
    save_table(pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame(), TABLES_DIR / "cta_factor_group_returns.csv")
    save_table(pd.concat(aligned_rows, ignore_index=True) if aligned_rows else pd.DataFrame(), TABLES_DIR / "cta_factor_aligned_data.csv")
    save_table(coverage, TABLES_DIR / "cta_dimension_coverage_audit.csv")
    save_table(reps, TABLES_DIR / "cta_factor_representatives.csv")
    write_report(REPORTS_DIR / "cta_factor_research_report.md", coverage, screening, high_corr)

    with pd.ExcelWriter(REPORTS_DIR / "cta_factor_research_report.xlsx", engine="openpyxl") as writer:
        for sheet, df in {
            "维度覆盖": coverage,
            "候选因子": candidates,
            "排除清单": exclusions,
            "全量检验": screening,
            "核心候选": screening[screening["screening_label"] == "核心候选"] if not screening.empty else pd.DataFrame(),
            "高相关因子对": high_corr,
            "分类代表因子": reps,
        }.items():
            df.to_excel(writer, sheet_name=sheet, index=False)

    print("CTA组周度因子全量研究完成。")
    print(f"候选因子数：{len(candidates)}")
    print(f"排除因子数：{len(exclusions)}")
    print(f"核心候选数：{int((screening['screening_label'] == '核心候选').sum()) if not screening.empty else 0}")
    print(f"报告：{REPORTS_DIR / 'cta_factor_research_report.md'}")


if __name__ == "__main__":
    main()
