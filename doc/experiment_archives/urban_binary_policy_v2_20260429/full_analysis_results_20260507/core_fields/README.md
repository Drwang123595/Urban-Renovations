# 全量分析核心字段精简版（2026-05-07）

本文件夹从全量预测工作簿中抽取人工分析最常用的核心字段，方便筛选、复核和统计。

## 文件

- `urban_renewal_full_core_fields_20260507.xlsx`：核心字段精简工作簿。
- `core_field_dictionary_20260507.csv`：字段说明表。
- `manifest.csv`：生成文件与来源索引。

## 精简结果

- 原始字段数：174
- 保留字段数：62
- 保留数据行数：40276
- 正例数：30770
- 负例数：9506
- 正例率：76.40%
- 标签一致性错误：0

## 工作簿 sheet

- `核心结果`：保留全量 40276 行和核心字段。
- `分析摘要`：正负例、LLM 计数、标签一致性等核心汇总。
- `字段分布`：关键字段的取值分布。
- `字段说明`：每个保留字段的用途说明。

## 推荐分析字段

- 最终二分类：`final_label`、`urban_flag`、`是否属于城市更新研究`
- 策略动作：`binary_policy_action`、`binary_policy_conflict_type`
- 固定主题：`topic_final`、`topic_final_group`、`taxonomy_coverage_status`
- 动态主题：`dynamic_topic_name_zh`、`dynamic_mapping_status`
- 解释证据：`decision_explanation`、`primary_positive_evidence`、`primary_negative_evidence`、`evidence_balance`

## 中文字段审阅版

- `urban_renewal_full_core_fields_chinese_review_20260507.xlsx`：主表字段已改为中文，便于人工筛选和审阅。
- `chinese_field_mapping_20260507.csv`：中文字段名与原始机器字段名映射。

中文审阅版不覆盖原始机器字段版；如后续需要程序读取，请仍优先使用 `urban_renewal_full_core_fields_20260507.xlsx`。
