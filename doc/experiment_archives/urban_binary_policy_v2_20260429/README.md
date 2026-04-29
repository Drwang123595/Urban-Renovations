# 城市更新二分类规则 V2 实验归档

归档时间：2026-04-29 00:17:20

本文件夹整理城市更新二分类规则 V2 的关键实验记录。大型预测工作簿和运行报告保留在各自 `Data/.../runs/...` 目录，本归档只保存可提交的摘要、命令、分布统计和工件索引。

## 核心结论

- 1000 篇带标签样本 no-LLM：Accuracy 84.1%，达到 `>= 80%` 目标。
- 1000 篇带标签样本 LLM 困难样本裁决：Accuracy 90.4%，达到 `>= 85%` 目标。
- 第一轮 10000 篇 no-LLM：正例率 76.65%，落在 `75%-85%` 目标区间。
- 重新抽样 10000 篇 no-LLM：正例率 75.86%，落在 `75%-85%` 目标区间。
- 全量 40276 篇 no-LLM：正例率 76.40%，`llm_used_sum == 0` 且 `llm_attempted_sum == 0`。
- no-LLM 大样本运行均保持 `llm_used_sum == 0` 且 `llm_attempted_sum == 0`。

## 文件说明

- `experiment_summary.csv`：各轮实验的样本量、正负例比例、LLM 使用计数和有标签集指标。
- `policy_action_distribution.csv`：各轮实验的 `binary_policy_action` 分布。
- `topic_group_distribution.csv`：各轮实验的 `topic_final_group` 分布。
- `dynamic_mapping_distribution.csv`：动态主题映射状态分布。
- `evidence_balance_distribution.csv`：证据倾向分布。
- `taxonomy_coverage_distribution.csv`：taxonomy 覆盖状态分布。
- `artifact_manifest.csv`：大型预测文件、评估文件和概览报告的原始路径索引。
- `run_commands.md`：主要复现实验命令。
- `full_no_llm_complete_20260429.md`：2026-04-29 全量 no-LLM 完整运行记录。
- `systematic_workflow_documentation.md`：城市更新指标提取任务的体系化执行流程文档。
- `paper_experiment_section_binary_classification.md`：论文实验部分草稿，聚焦规则-LLM协同二分类。
- `urban_renewal_full_workflow_taxonomy_status_20260429.md`：完整流程与 `taxonomy_coverage_status` 解释的 Markdown 源文件。
- `urban_renewal_full_workflow_taxonomy_status_20260429.docx`：完整流程与 `taxonomy_coverage_status` 解释的 Word 文档。

## 稳定版代码说明

稳定版接入 `UrbanBinaryPolicyV2` 作为最终二分类策略层。该策略保留 `topic_final` 作为主题解释字段，最终 `final_label/urban_flag/是否属于城市更新研究` 由二分类证据、hard negative 保护、动态主题证据和可选 LLM 困难样本裁决共同决定。

无 LLM 与稳定发布路径仍保持 `llm_used == 0` 合同；LLM 只在 `research_matrix + hybrid_llm_assist on` 的困难样本裁决路径中生效。
