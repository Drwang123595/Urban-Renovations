# Probe100 Pipeline Report (2026-04-07)

## 1. 测试目标

- 打通当前城市更新识别整套流程。
- 选取 100 篇有标签样本进行端到端测试。
- 在不改 prompt 和主题体系的前提下，优先通过第一阶段轻量负筛与 hybrid 路由优化，把最终 hybrid 准确率提升到 `>= 85%`。

## 2. 测试数据

- 数据文件:
  - `C:\Users\26409\Desktop\Urban Renovation\output\spreadsheet\Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407.xlsx`
- 标签列:
  - `是否属于城市更新研究_local_v2`
- 取样方式:
  - 直接取前 `100` 条有标签样本

## 3. 运行命令

```powershell
& 'C:\Users\26409\Desktop\Urban Renovation\.venv-bertopic313\Scripts\python.exe' scripts\benchmark_api_vs_classifier.py `
  --input "C:\Users\26409\Desktop\Urban Renovation\output\spreadsheet\Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407.xlsx" `
  --truth-column "是否属于城市更新研究_local_v2" `
  --limit 100 `
  --shot-mode few `
  --max-workers 4 `
  --tag route_tuned_probe100_20260407 `
  --reuse-pure-llm "C:\Users\26409\Desktop\Urban Renovation\output\spreadsheet\urban_sample1000_pure_llm_api_result_probe100_stage3b_20260407.xlsx"
```

说明:
- pure LLM 结果复用了已有 100 篇结果，避免重复调用 API。
- classifier 与 hybrid 使用当前代码重新运行。

## 4. 本轮代码改动

### 4.1 Stage1 轻量负筛

文件:
- `C:\Users\26409\Desktop\Urban Renovation\src\urban_rule_filter.py`

改动:
- 在 `R4_rural_nonurban` 中补充 `rural gentrification`
- 继续保持 Stage1 只做轻量排除，不做重判断

目的:
- 把明确的 rural/non-urban 样本更早挡住
- 避免这类样本进入 stage2/stage3 被误判为城市更新

### 4.2 Hybrid LLM 路由收紧

文件:
- `C:\Users\26409\Desktop\Urban Renovation\src\urban_hybrid_classifier.py`

改动:
- 保留原有 hard-case reason 生成逻辑
- 新增更严格的 LLM 介入门槛:
  - `topic_group != urban` 时，允许 LLM 介入
  - `bertopic_high_purity_conflict` 时，允许 LLM 介入
  - `topic_label == U4` 且 **没有** `low_margin` 时，允许 LLM 介入
  - 其他 urban 正类桶默认直接信任 stage2 classifier，不再送 LLM 翻案

目的:
- 减少 LLM 对已经较稳定的 urban 正例进行误翻
- 保留其在不确定 nonurban 样本和部分 `U4` 社会影响类样本中的修正价值

### 4.3 单元测试更新

文件:
- `C:\Users\26409\Desktop\Urban Renovation\tests\test_urban_hybrid_pipeline.py`

新增/调整:
- `rural gentrification` Stage1 排除测试
- low-confidence urban + margin collapse 不再进 LLM
- `U4` 低置信但 margin 仍稳定时允许进 LLM
- noisy `U2` 样本默认回退到 stage2 classifier
- BERTopic outlier 测试改为 nonurban 场景

验证:
- `27 passed`

## 5. 基线与优化后结果

### 5.1 基线结果

评估文件:
- `C:\Users\26409\Desktop\Urban Renovation\output\spreadsheet\urban_sample1000_llm_classifier_hybrid_evaluation_post_stage1_probe100_20260407.xlsx`

| model | accuracy | precision | recall | specificity | llm_call_rate |
|---|---:|---:|---:|---:|---:|
| pure_llm_api | 0.73 | 0.9024 | 0.6167 | 0.9000 | 1.00 |
| local_topic_classifier | 0.85 | 0.8169 | 0.9667 | 0.6750 | 0.00 |
| three_stage_hybrid | 0.80 | 0.8846 | 0.7667 | 0.8500 | 0.54 |

问题定位:
- hybrid 主问题不在 Stage1，而在 `llm_assist`
- 基线里 `llm_assist` 占比 `54%`
- 其准确率只有 `75.93%`
- 典型伤害来自:
  - `U1 low_confidence`
  - `U2 low_confidence + low_margin`
  - 一部分 `U3/U4` margin collapse 样本

### 5.2 优化后结果

评估文件:
- `C:\Users\26409\Desktop\Urban Renovation\output\spreadsheet\urban_sample1000_llm_classifier_hybrid_evaluation_route_tuned_probe100_20260407.xlsx`

| model | accuracy | precision | recall | specificity | llm_call_rate |
|---|---:|---:|---:|---:|---:|
| pure_llm_api | 0.73 | 0.9024 | 0.6167 | 0.9000 | 1.00 |
| local_topic_classifier | 0.85 | 0.8169 | 0.9667 | 0.6750 | 0.00 |
| three_stage_hybrid | 0.87 | 0.8615 | 0.9333 | 0.7750 | 0.30 |

结论:
- hybrid 准确率从 `0.80` 提升到 `0.87`
- 已超过目标 `0.85`

## 6. Hybrid 结构性变化

优化后 error breakdown:

| metric | value |
|---|---:|
| stage1_reject_error | 2 |
| stage2_only_error | 9 |
| llm_corrected_cases | 4 |
| residual_hard_cases | 2 |
| hybrid_llm_calls | 30 |
| hybrid_llm_call_rate | 0.30 |

优化后 decision source:

| decision_source | count | share | accuracy |
|---|---:|---:|---:|
| stage2_classifier | 66 | 0.66 | 0.8636 |
| llm_assist | 30 | 0.30 | 0.9333 |
| stage1_rule | 4 | 0.04 | 0.5000 |

解读:
- LLM 调用率从 `54%` 降到 `30%`
- LLM 介入后的准确率反而升到 `93.33%`
- 说明当前改法的核心收益是:
  - 减少无效 LLM 介入
  - 只让 LLM 处理它真正擅长的难例

## 7. 仍然存在的问题

1. Stage1 仍有 `2` 个误杀
   - 都来自 `greenfield_expansion` 规则
   - 说明这条规则还需要再做一层“更新语境保护”

2. Stage2 仍是主要剩余误差来源
   - `stage2_only_error = 9`
   - 说明下一轮如果要继续提到 `88%-90%`，重点应放在 topic taxonomy 或训练样本，而不是继续加大 Stage1

3. 当前 benchmark 只覆盖 100 篇
   - 已达到本轮目标
   - 但仍建议在 1000 篇标准集上再做一次完整回归

## 8. 输出文件

本轮主要输出:

- `C:\Users\26409\Desktop\Urban Renovation\output\spreadsheet\urban_sample1000_pure_llm_api_result_route_tuned_probe100_20260407.xlsx`
- `C:\Users\26409\Desktop\Urban Renovation\output\spreadsheet\urban_sample1000_classifier_result_route_tuned_probe100_20260407.xlsx`
- `C:\Users\26409\Desktop\Urban Renovation\output\spreadsheet\urban_sample1000_three_stage_hybrid_result_route_tuned_probe100_20260407.xlsx`
- `C:\Users\26409\Desktop\Urban Renovation\output\spreadsheet\urban_sample1000_llm_classifier_hybrid_evaluation_route_tuned_probe100_20260407.xlsx`

## 9. 本轮结论

- 本轮目标已完成:
  - 100 篇有标签样本已打通全流程
  - hybrid 准确率提升到 `87%`
  - 超过目标 `85%`

- 当前最有效的策略不是继续加重第一阶段，而是:
  - Stage1 保持轻量负筛
  - Stage2 继续承担主判定
  - Stage3/LLM 只处理少量真正高价值难例
