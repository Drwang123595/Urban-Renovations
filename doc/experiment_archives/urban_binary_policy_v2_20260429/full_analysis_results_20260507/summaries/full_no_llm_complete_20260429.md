# 全量 no-LLM 实验记录（2026-04-29）

## 运行设置

- run_id: `40276_full_no_llm_policy_v2_complete_20260429`
- 输入数据：`Data/Urban Renovation V2.0/input/labels/Urban Renovation V2.0.xlsx`
- 输出预测：`Data/Urban Renovation V2.0/runs/research_matrix/20260429_policy_v2_full_no_llm_full_on_complete/predictions/urban_renewal_three_stage_hybrid_policy_v2_full_no_llm_full_on_complete_20260429.xlsx`
- 概览报告：`Data/Urban Renovation V2.0/runs/research_matrix/20260429_policy_v2_full_no_llm_full_on_complete/reports/Policy_V2_Full_NoLLM_Overview_20260429.xlsx`
- 任务方法：`three_stage_hybrid`
- LLM 设置：`--hybrid-llm-assist off`
- 本地增强：`--dynamic-topics on`、`--dynamic-topics-full-corpus`、`--dynamic-binary-refine on`、`--dynamic-binary-allow-flip`

## 核心结果

| 指标 | 数值 |
|---|---:|
| 全量输入/输出 | 40276 |
| 正例数 | 30770 |
| 负例数 | 9506 |
| 正例率 | 76.40% |
| 负例率 | 23.60% |
| `llm_used_sum` | 0 |
| `llm_attempted_sum` | 0 |
| `llm_adjudication_required_sum` | 16939 |
| 标签一致性错误 | 0 |

## 策略动作分布

| `binary_policy_action` | 数量 | 占比 |
|---|---:|---:|
| `conflict_review` | 16939 | 42.06% |
| `accept_positive` | 13831 | 34.34% |
| `accept_negative` | 9322 | 23.15% |
| `protected_negative` | 184 | 0.46% |

## 主题与动态映射

| `topic_final_group` | 数量 | 占比 |
|---|---:|---:|
| `nonurban` | 17706 | 43.96% |
| `urban` | 11945 | 29.66% |
| `unknown` | 10625 | 26.38% |

| `dynamic_mapping_status` | 数量 | 占比 |
|---|---:|---:|
| `mapped_to_fixed` | 30254 | 75.12% |
| `needs_review` | 7457 | 18.51% |
| `candidate_new_nonurban_topic` | 2378 | 5.90% |
| `candidate_new_urban_topic` | 187 | 0.46% |

## 结论

本轮是完整全量 no-LLM 测试，不调用大模型/API。预测结果共 `40276` 行，正例率 `76.40%`，落在前置目标 `75%-85%` 区间内。日志中未检出 `DeepSeek`、`HTTP`、`https://`、`API call`、`llm_used=1`、`llm_attempted=1` 等调用痕迹；运行 stdout 明确记录 `Hybrid LLM Assist: off`。
