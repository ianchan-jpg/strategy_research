"""检查 factors.xlsx 的工作表、列名、日期范围、缺失率和频率。

运行方式：
    python project/scripts/inspect_data.py

本脚本只读取原始 Excel，并把检查结果保存到 outputs 目录。
不会覆盖 data/factors.xlsx。
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PROJECT_DIR.parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from data_loader import inspect_workbook, inspection_to_tables  # noqa: E402


def dataframe_to_markdown(df) -> str:
    """把 DataFrame 转成简单 Markdown 表格，避免额外依赖 tabulate。"""
    if df.empty:
        return "空表"

    columns = [str(col) for col in df.columns]
    rows = []
    rows.append("| " + " | ".join(columns) + " |")
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")

    for _, row in df.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in df.columns]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join(rows)


def write_markdown_report(sheet_summary, output_path: Path) -> None:
    """保存一份适合人工阅读的 Markdown 检查报告。"""
    lines = [
        "# factors.xlsx 数据检查报告",
        "",
        "本报告由 `project/scripts/inspect_data.py` 生成，只做数据盘点，不做建模。",
        "",
        "## 工作表概况",
        "",
        dataframe_to_markdown(sheet_summary),
        "",
        "## 初步判断",
        "",
        "- `指标目录`、`大类指标名称对应`：指标说明/映射表。",
        "- `日度序列`、`周度序列`、`月度序列`：指标数据，后续作为因子来源。",
        "- `策略指数`：策略收益数据，包含主观股票、量化股票、主观CTA、量化CTA相关列。",
        "",
        "## 下一步",
        "",
        "读取数据 → 构造相对收益 → 因子预处理 → 单因子检验 → 批量检验 → 因子去重 → 输出报告",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """执行 Excel 数据检查。"""
    excel_path = ROOT_DIR / "data" / "factors.xlsx"
    if not excel_path.exists():
        raise FileNotFoundError(f"没有找到原始文件：{excel_path}")

    result = inspect_workbook(excel_path)
    sheet_summary, missing_detail = inspection_to_tables(result)

    tables_dir = PROJECT_DIR / "outputs" / "tables"
    reports_dir = PROJECT_DIR / "outputs" / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    sheet_summary_path = tables_dir / "data_inspection_sheet_summary.csv"
    missing_detail_path = tables_dir / "data_inspection_missing_detail.csv"
    report_path = reports_dir / "data_inspection_report.md"

    sheet_summary.to_csv(sheet_summary_path, index=False, encoding="utf-8-sig")
    missing_detail.to_csv(missing_detail_path, index=False, encoding="utf-8-sig")
    write_markdown_report(sheet_summary, report_path)

    print("数据检查完成。")
    print(f"工作表概况：{sheet_summary_path}")
    print(f"缺失率明细：{missing_detail_path}")
    print(f"Markdown报告：{report_path}")
    print("")
    print(sheet_summary[["sheet", "sheet_type", "rows", "cols", "date_min", "date_max", "frequency"]])


if __name__ == "__main__":
    main()
