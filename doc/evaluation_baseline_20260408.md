# 稳定发布与测试计划

本文档替代早期的 2026-04-08 基线解释，并锁定截至 2026-04-17 的当前性能最优稳定版。

## 治理更新

当前实验治理分为三条轨道。

1. `stable_release`
   - 仅包含当前混合主线
   - 数据集固定为 `Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407`
   - 主入口固定为 `scripts/pipeline/main_py313.py`
2. `research_matrix`
   - 承载方法对比
   - 承载长上下文顺序稳定性协议
   - `spatial` 与稳定门禁分开报告
3. `legacy_archive`
   - 仅保留历史脚本、历史报告和启发式真值匹配

当前评估合同如下。

- 只有 `scripts/evaluation/evaluate.py` 可以生成官方验收汇总
- `Accuracy` 保持 `0-100` 标尺
- `Precision`、`Recall`、`F1` 保持 `0-1` 标尺
- 任何 `Accuracy > 100` 均为无效结果
- 严格轨道只允许通过显式 `--truth` 或唯一标签工作簿绑定真值
- `scripts/main.py` 仅作为历史兼容入口

## 标准目录布局

所有新的稳定发布和研究输出都应按数据集、实验轨道和运行标签分别落入独立运行目录。

```text
Data/<dataset_id>/
  input/labels/<dataset_id>.xlsx
  runs/<experiment_track>/<run_tag>/
    predictions/
    reports/
    reviews/
    logs/
    Stable_Run_Summary.json
```

标准标注输入路径为 `Data/<dataset_id>/input/labels/<dataset_id>.xlsx`。

旧目录仅用于历史比较。

- `Data/<dataset_id>/labels`
- `Data/<dataset_id>/output`
- `Data/<dataset_id>/Result`

## 锁定稳定版

- 运行环境：Python `3.13`
- 主入口：`scripts/pipeline/main_py313.py`
- 稳定管线入口：`scripts/pipeline/run_stable_release.py`
- 稳定模式：`three_stage_hybrid --hybrid-llm-assist on`
- 稳定结果目录：
  - `Data/Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407/runs/stable_release/20260417_unknown_recovery/reports`
- 稳定输出：
  - `Data/Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407/runs/stable_release/20260417_unknown_recovery/predictions/urban_renewal_three_stage_hybrid_few_llm_on_20260417_unknown_recovery.xlsx`
- 稳定复核池：
  - `Data/Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407/runs/stable_release/20260417_unknown_recovery/reviews/Unknown_Review_hybrid_llm_on_20260417_unknown_recovery.xlsx`

## 锁定指标

稳定版城市更新识别指标如下。

| 运行 | Accuracy | Precision | Recall | F1 | TP | TN | FP | FN | Unknown |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `three_stage_hybrid + LLM on` | 88.0% | 0.956743 | 0.940000 | 0.948298 | 752 | 128 | 34 | 48 | 38 |

稳定版 Unknown 恢复诊断如下。

| 决策来源 | 总数 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| `rule_model_fusion` | 846 | 91.1348 | 0.960452 | 0.935351 | 0.947735 |
| `stage1_rule` | 17 | 100.0000 | 0.000000 | 0.000000 | 0.000000 |
| `unknown_hint_resolution` | 99 | 92.9293 | 0.923077 | 0.986301 | 0.953642 |
| `unknown_review` | 38 | 0.0000 | 0.000000 | 0.000000 | 0.000000 |

参考对比基线如下。

| 基线目录 | 运行 | Precision | Recall | F1 | FP | Unknown |
|---|---|---:|---:|---:|---:|---:|
| `baseline_20260409_finalstable` | `three_stage_hybrid + LLM on` | 0.949266 | 0.913882 | 0.931238 | 38 | 56 |
| `baseline_20260408_precision_round2` | `three_stage_hybrid + LLM on` | 0.947441 | 0.910904 | 0.928814 | 38 | 88 |
| `baseline_20260408_precision` | `three_stage_hybrid + LLM on` | 0.946839 | 0.901505 | 0.923616 | 37 | 111 |

## 发布门禁

任何新版本要被称为稳定版，必须先满足以下条件。

1. 运行 `python -m pytest -q`
   - 先在干净 Python `3.13` 环境中执行 `python -m pip install -e .[dev]`
   - 测试必须全部通过
2. 在 Python `3.13` 下重新运行固定实验矩阵
   - `local_topic_classifier`
   - `three_stage_hybrid --hybrid-llm-assist off`
   - `three_stage_hybrid --hybrid-llm-assist on`
3. 使用 `scripts/evaluation/evaluate.py` 评估全部输出
   - 必须包含以下工作表：
     - `All Metrics`
     - `Run Metadata`
     - `Protocol`
     - `Comparability`
     - `Long Context Stability`
     - `Decision Source Metrics`
     - `Unknown Rate`
     - `Theme Metrics`
     - `Theme Confusion`
     - `U-N Family Metrics`
     - `Topic Distribution`
     - `Explainability Quality`
     - `Evidence Balance Metrics`

单次稳定管线命令如下。

```powershell
.venv-bertopic313\Scripts\python.exe scripts\pipeline\run_stable_release.py --skip-classification
```

只有在明确需要重新运行并覆盖锁定的 1000 篇预测工作簿时，才使用 `--force`。

长上下文比较规则如下。

- 任何 `stepwise_long` 风格的长上下文结论都必须聚合三个固定顺序
- 主结论表只能引用聚合均值
- 如果 `max_delta_accuracy > 1.5` 或 `max_delta_f1 > 0.015`，则该方法判定为 `order_sensitive`

稳定运行验收阈值如下。

- `Accuracy >= 88.0`
- `Precision >= 0.956`
- `Recall >= 0.940`
- `F1 >= 0.948`
- `FP <= 34`
- `FN <= 48`
- `Predicted Unknown Count <= 38`
- `llm_used == 0`
- `unknown_hint_resolution` accuracy `>= 92.0%`
- 解释字段覆盖率 `>= 100%`
- 决策规则链覆盖率 `>= 100%`
- 二分类证据覆盖率 `>= 100%`

## 功能回归覆盖

必需回归覆盖如下。

- `stage1_rule`
  - 数学术语误用
  - 农村非城市场景
  - 绿地开发 / 新城建设
- `rule_model_fusion`
  - 同标签一致
  - 同组优先
  - 跨组强规则覆盖
  - 跨组强本地分类覆盖
- `Unknown` 保守恢复
  - `N3/N8 -> U*`
  - `U12/U4/U9/U1 -> N*`
  - `local Unknown + llm family hint`
- BERTopic 辅助合同
  - BERTopic 可以输出提示
  - BERTopic 不得改写 `topic_final`
- 输出合同
  - `urban_flag` 必须由 `topic_final` 派生
  - 每一行都必须填充确定性解释字段
  - `decision_source` 必须限定在以下集合：
    - `rule_model_fusion`
    - `stage1_rule`
    - `stage2_classifier`
    - `unknown_hint_resolution`
    - `unknown_review`

必需主题边界样本如下。

- 一般治理 / 政策 / 话语样本不得因弱更新词汇被判为正类
- `TIF / PPP / compensation / redevelopment finance` 必须保持正类
- `brownfield redevelopment / adaptive reuse / urban village / inner-city regeneration` 必须保持正类
- 一般 `gentrification / neighborhood change` 不得自动判为正类
- 明确绑定更新过程或更新后果的 gentrification 样本必须保持正类

## 已知限制

- `theme_gold` 尚未大规模填充
- 锁定稳定版中的 `Theme Metrics` 与 `Theme Confusion` 预期保持结构化空表
- 下一阶段工作仅限于：
  - 填充 `theme_gold`
  - 对剩余 `Unknown` 池进行高精度恢复
