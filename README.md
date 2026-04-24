# 城市更新文献识别系统

截至 2026-04-17，本项目的稳定版合同如下。

- 运行环境：Python `3.13`
- 主运行入口：`scripts/pipeline/main_py313.py`
- 稳定发布入口：`scripts/pipeline/run_stable_release.py`
- 兼容入口：`scripts/main.py` 以及根目录 `scripts/*.py` 包装脚本
- 稳定配置：`three_stage_hybrid --hybrid-llm-assist on`
- 主任务输出形态：
  - `topic_final` 是主输出字段
  - `urban_flag` / `final_label` 均由 `topic_final` 派生
  - 主题空间固定为 `U1-U15 / N1-N10 / Unknown`
- BERTopic 只作为辅助信号：
  - 支持动态主题发现
  - 支持 `Unknown` 复核
  - 支持规则与标签迭代
  - 不作为在线主判定来源
- LLM 受精度约束：
  - 仅用于为 `Unknown` 样本收集家族提示
  - 不覆盖 `topic_final`
  - 稳定发布中 `llm_used` 必须保持为 `0`

## 实验治理

当前项目固定分为三条实验轨道。

- `stable_release`
  - 仅包含当前混合主线
  - 数据集固定为 `Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407`
  - 只承载主任务 `urban_renewal`
- `research_matrix`
  - 承载主任务的方法对比
  - 承载长上下文顺序敏感性对比
  - `spatial` 空间属性任务应在该轨道中单独报告
- `legacy_archive`
  - 仅保存历史脚本、历史报告和启发式真值绑定结果
  - 新的稳定结论不得引用该轨道作为依据

## 目录结构

标准项目数据结构如下。

```text
Data/
  <dataset_id>/
    input/
      labels/              # 只读真值与已标注输入工作簿
    runs/
      <experiment_track>/
        <run_tag>/
          predictions/     # 模型或管线预测工作簿
          reports/         # Eval_*.xlsx 与 Eval_Summary.xlsx
          reviews/         # Unknown_Review 与人工复核工作簿
          logs/            # 运行日志
          Stable_Run_Summary.json
  train/                   # 研究和开发训练工作簿
output/
  models/                  # 当前代码使用的本地模型工件
history/
  sessions/                # 可选的提示词/会话审计轨迹
```

兼容说明：旧版 `Data/<dataset_id>/labels`、`Data/<dataset_id>/output` 和 `Data/<dataset_id>/Result` 目录仅作为历史归档保留。新的稳定运行必须使用 `Data/<dataset_id>/runs/<track>/<tag>/...` 布局。

真值与数据合同如下。

- 稳定发布只使用 `Data/Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407/input/labels`
- `test1-test7_merged` 仅作为历史基线
- 标签工作簿是只读真值源
- 官方汇总结论必须来自 `scripts/evaluation/evaluate.py` 和 `Eval_Summary.xlsx`

指标标尺合同如下。

- `Accuracy` 按 `0-100` 标尺存储和报告
- `Precision`、`Recall`、`F1` 保持 `0-1` 标尺
- 任何 `Accuracy > 100` 都视为错误

长上下文比较合同如下。

- 长上下文结果属于 `research_matrix`，不进入稳定发布门禁
- 固定使用以下顺序：
  - `canonical_title_order`
  - `shuffle_seed_20260415_a`
  - `shuffle_seed_20260415_b`
- 只能引用聚合结果，不得引用单次长上下文运行作为结论
- `Eval_Summary.xlsx` 中的 `Long Context Stability` 是顺序敏感性判断依据

## 稳定发布锁定版本

当前锁定的性能最优稳定版如下。

- 输出文件：
  - `Data/Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407/runs/stable_release/20260417_unknown_recovery/predictions/urban_renewal_three_stage_hybrid_few_llm_on_20260417_unknown_recovery.xlsx`
- 汇总文件：
  - `Data/Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407/runs/stable_release/20260417_unknown_recovery/reports/Eval_Summary.xlsx`
- Unknown 复核池：
  - `Data/Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407/runs/stable_release/20260417_unknown_recovery/reviews/Unknown_Review_hybrid_llm_on_20260417_unknown_recovery.xlsx`

稳定发布指标如下。

- `Accuracy = 88.0`
- `Precision = 0.956743`
- `Recall = 0.940000`
- `F1 = 0.948298`
- `FP = 34`
- `FN = 48`
- `Predicted Unknown Count = 38`
- `unknown_hint_resolution Accuracy = 92.9293`
- `llm_attempted = 137`
- `llm_used = 0`

稳定管线命令如下。

```powershell
.venv-bertopic313\Scripts\python.exe scripts\pipeline\run_stable_release.py --skip-classification
```

只有在明确需要重新运行 1000 篇实时分类并覆盖锁定预测工作簿时，才使用 `--force`。

用于对比的完整矩阵历史基线如下。

- 历史归档：`Data/Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407/Result/baseline_20260409_finalstable`

## 发布门禁

每次发布检查都必须使用同一标注数据集、同一提示词家族和同一评估器。

必跑方法矩阵如下。

1. `local_topic_classifier`
2. `three_stage_hybrid --hybrid-llm-assist off`
3. `three_stage_hybrid --hybrid-llm-assist on`

必检项目如下。

- 先初始化环境：
  - `python -m pip install -e .[dev]`
  - `python -m pytest -q`
- `evaluate.py` 输出必须包含：
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
- 官方叙事报告由 `scripts/reporting/generate_stage_report.py` 生成

稳定发布验收阈值如下。

- `hybrid + LLM on` 的 accuracy `>= 88.0`
- `hybrid + LLM on` 的 precision `>= 0.956`
- `hybrid + LLM on` 的 recall `>= 0.940`
- `hybrid + LLM on` 的 F1 `>= 0.948`
- `FP <= 34`
- `FN <= 48`
- `Predicted Unknown Count <= 38`
- `llm_used == 0`
- `unknown_hint_resolution` 子集 accuracy `>= 92.0%`
- 解释字段覆盖率 `>= 100%`
- 决策规则链覆盖率 `>= 100%`
- 二分类证据覆盖率 `>= 100%`

## 标签输入

二分类评估使用当前标签工作簿。

存在以下列时，系统支持可选主题复核。

- `theme_gold`
- `theme_gold_source`
- `review_status`

只有 `theme_gold` 非空的样本才会进入主题评估。当前稳定发布的 `Theme Metrics` 仍为空，因此 25 类主题准确率尚未闭环。
