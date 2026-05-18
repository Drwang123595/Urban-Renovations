# 全量城市更新二分类分析结果归档（2026-05-07）

本文件夹集中保存 `40276_full_no_llm_policy_v2_complete_20260429` 的全量 no-LLM 分析结果。

## 核心结果

- 输入/输出总量：40276 篇
- 正例数：30770
- 负例数：9506
- 正例率：76.40%
- 负例率：23.60%
- `llm_used_sum`：0
- `llm_attempted_sum`：0
- `llm_adjudication_required_sum`：16939
- 标签一致性错误：0

## 文件结构

- `predictions/`：全量逐篇预测工作簿。
- `reports/`：全量概览报告工作簿。
- `logs/`：本次全量运行 stdout/stderr 日志。
- `summaries/`：全量实验说明和筛选后的摘要/分布 CSV。
- `indices/manifest.csv`：复制文件与原始工件路径索引。

## 注意

本归档复制了全量预测工作簿，原始运行目录仍保留在 `Data/Urban Renovation V2.0/runs/research_matrix/20260429_policy_v2_full_no_llm_full_on_complete`。
