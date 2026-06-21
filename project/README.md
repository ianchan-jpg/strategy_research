# 主观策略 vs 量化策略比价项目

本项目用于比较主观策略和量化策略的相对表现，并研究哪些因子可以预测下一期“量化收益 - 主观收益”。

当前阶段只完成基础工程框架和原始数据检查，不做复杂建模，不批量回归，也不自动删除因子。

## 项目目标

1. 股票组：比较主观股票与量化股票。
2. CTA组：比较主观CTA与量化CTA。
3. 使用当期因子预测下一期“量化收益 - 主观收益”。
4. 后续完成 Rank IC、ICIR、回归、滚动回归和因子相关性筛选。

## 目录说明

```text
project/
├── data/
│   ├── raw/                 # 原始数据备份或外部输入；当前不移动、不覆盖根目录 data/factors.xlsx
│   └── processed/           # 清洗、对齐、频率统一后的中间数据
├── src/
│   ├── data_loader.py       # 读取Excel、检查工作表、列名、日期范围、缺失率和频率
│   ├── target_builder.py    # 构造股票和CTA相对收益：量化收益 - 主观收益
│   ├── factor_processor.py  # 因子清洗、去极值、标准化
│   ├── factor_tester.py     # Rank IC、回归、分组检验等基础函数
│   ├── factor_selector.py   # 因子相关性检查和人工去重辅助，不自动删除因子
│   └── report.py            # 输出结果表和图
├── models/
│   └── results/             # 回归参数、检验结果和模型输出
├── config/
│   └── config.yaml          # 工作表、列名、窗口、阈值等参数
├── scripts/
│   ├── inspect_data.py      # 数据检查入口
│   ├── run_single_factor.py # 单因子检验入口，当前为占位说明
│   └── run_batch_test.py    # 批量检验入口，当前为占位说明
├── outputs/
│   ├── tables/              # CSV结果表
│   ├── figures/             # 图片
│   └── reports/             # Markdown或HTML报告
├── tests/                   # 单元测试
├── README.md
└── requirements.txt
```

## 原始数据

原始 Excel 文件保留在：

```text
data/factors.xlsx
```

不要覆盖该文件。后续如果需要生成清洗后的数据，应保存到 `project/data/processed/`。

## 已识别的数据表

根据初步检查：

- `指标目录`：指标说明表。
- `大类指标名称对应`：指标分类或名称映射表。
- `日度序列`：日频指标数据。
- `周度序列`：周频指标数据。
- `月度序列`：月频指标数据。
- `策略指数`：策略收益数据，包含 `主观-均衡`、`量化-均衡`、`主观期货`、`量化期货` 等列。

## 后续运行顺序

```text
读取数据
→ 构造相对收益
→ 因子预处理
→ 单因子检验
→ 批量检验
→ 因子去重
→ 输出报告
```

## 当前可运行命令

在项目根目录运行：

```bash
python project/scripts/inspect_data.py
```

输出文件：

- `project/outputs/tables/data_inspection_sheet_summary.csv`
- `project/outputs/tables/data_inspection_missing_detail.csv`
- `project/outputs/reports/data_inspection_report.md`

## 依赖

本项目使用 Python、pandas、numpy、scipy、statsmodels 和 matplotlib。
