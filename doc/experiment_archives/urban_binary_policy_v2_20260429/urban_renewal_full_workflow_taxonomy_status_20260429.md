# 城市更新指标提取完整流程说明

本文档系统描述城市更新指标提取任务从输入文献到最终二分类结果的完整执行过程，重点说明固定 `topic taxonomy`、`taxonomy_coverage_status`、动态主题发现、动态二分类 `refine` 与 `UrbanBinaryPolicyV2` 如何共同服务最终的城市更新二分类。

本文档对应当前稳定实验链路：`research_matrix / three_stage_hybrid / Policy V2`。默认运行模式为 `no-LLM`，即不调用大模型/API；在允许大模型的研究实验中，LLM 只用于困难样本裁决，不替代全量规则流程。

## 1. 任务目标与边界

本任务的最终目标是判断一篇文献是否属于城市更新研究。形式上，这是一个二分类任务：

- `1`：属于城市更新研究。
- `0`：不属于城市更新研究。

最终落地字段包括：

- `final_label`
- `urban_flag`
- `是否属于城市更新研究`

这三列必须保持一致。其他字段如 `topic_final`、`topic_final_group`、`taxonomy_coverage_status`、`dynamic_topic_*`、`binary_policy_action` 是解释字段、治理字段或中间证据字段，不应被单独等同为最终二分类标签。

本流程遵循三个原则：

- 规则主导：边界清晰的样本优先由确定性规则和固定主题体系处理。
- 本地增强：动态主题发现和动态二分类 `refine` 只使用本地算法，不调用 LLM/API。
- LLM 局部裁决：只有在 `research_matrix` 且显式启用 LLM 时，才对困难样本进行结构化裁决。

## 2. 总体流程

完整流程由八个阶段组成：

1. 输入数据与运行配置。
2. 逐篇本地混合分类。
3. 固定 `topic taxonomy` 覆盖诊断。
4. 动态主题发现。
5. 动态二分类 `refine`。
6. `UrbanBinaryPolicyV2` 最终二分类。
7. 评估与结构诊断。
8. 实验记录归档。

| 阶段 | 输入 | 核心分析 | 输出 |
|---|---|---|---|
| 运行配置 | `dataset_id`、`experiment_track`、输入路径、方法参数 | 确定运行目录、是否启用动态主题、动态二分类、LLM 裁决 | 运行上下文、输出路径、日志路径 |
| 数据读取 | Excel 输入工作簿 | 读取标题、摘要、关键词、学科背景，逐行构造文献记录 | 待分类 records、checkpoint |
| 本地混合分类 | 单篇文献记录 | 规则预筛、固定 taxonomy、本地主题判断、Unknown recovery、基础二分类 | `topic_final`、`urban_probability_score`、解释字段 |
| 覆盖诊断 | `topic_final`、route reason、open-set 证据 | 判断样本与固定 taxonomy 的关系 | `taxonomy_coverage_status` |
| 动态主题 | 基础预测 DataFrame | 候选池构建、本地聚类、关键词命名、固定主题映射 | `dynamic_topic_*`、`dynamic_mapping_status` |
| 动态 refine | 动态主题与基础二分类 | 用高置信动态主题修复漏召或冲突样本 | `dynamic_binary_override_*`、`binary_resolved` |
| V2 策略 | 全部证据字段 | 统一最终二分类，标记冲突和 LLM 裁决需求 | `final_label`、`binary_policy_action` |
| 评估/诊断 | 预测结果与可选真值 | 有标签算指标，无标签看结构分布 | `Eval_Summary.xlsx` 或 overview report |

## 3. 输入数据与运行配置

标准输入是 Excel 工作簿。当前全量实验输入为：

`Data/Urban Renovation V2.0/input/labels/Urban Renovation V2.0.xlsx`

核心输入字段如下：

| 字段 | 作用 |
|---|---|
| `Article Title` | 标题证据。标题权重较高，用于规则命中、主题识别、动态主题关键词和 LLM 困难样本裁决。 |
| `Abstract` | 摘要证据。用于判断研究对象、更新动作、城市建成环境语境以及非目标风险。 |
| `Author Keywords` | 作者关键词。用于固定 taxonomy 命中、动态主题聚类和主题命名。 |
| `Keywords Plus` | 扩展关键词。用于补充主题证据。 |
| `WoS Categories` | 学科背景。用于识别泛城市、方法、生态、交通、农村等非目标风险。 |
| `Research Areas` | 研究领域背景证据。 |

典型 no-LLM 全流程配置如下：

```powershell
.\.venv-bertopic313\Scripts\python.exe scripts\pipeline\main_py313.py `
  --task urban_renewal `
  --experiment-track research_matrix `
  --input "Data\Urban Renovation V2.0\input\labels\Urban Renovation V2.0.xlsx" `
  --urban-method three_stage_hybrid `
  --hybrid-llm-assist off `
  --dynamic-topics on `
  --dynamic-topics-full-corpus `
  --dynamic-binary-refine on `
  --dynamic-binary-allow-flip `
  --non-interactive
```

其中 `--hybrid-llm-assist off` 表示大模型完全不介入；`--dynamic-topics on`、`--dynamic-topics-full-corpus`、`--dynamic-binary-refine on`、`--dynamic-binary-allow-flip` 表示本地增强流程全部打开。

## 4. 本地混合分类 three_stage_hybrid

`three_stage_hybrid` 是逐篇执行的基础分类链。它不是单一关键词规则，也不是单一主题模型，而是将规则、固定主题体系、本地主题判断、Unknown recovery 和二分类证据合成放在同一条链路中。

### 4.1 Stage 1 规则预筛

Stage 1 负责识别非常明确的正向证据和强负向风险。

正向证据主要包括两类：

- 更新动作锚点：`renewal`、`regeneration`、`redevelopment`、`revitalization`、`rehabilitation`、`retrofit`、`adaptive reuse`、`gentrification` 等。
- 既有城市对象锚点：`built environment`、`neighborhood`、`community`、`brownfield`、`old district`、`housing estate`、`industrial heritage`、`public space`、`informal settlement` 等。

负向风险主要包括：

- 农村或农业语境：`rural`、`village`、`agriculture` 等。
- 纯方法语境：`algorithm`、`model`、`simulation`、纯技术预测等。
- 非目标主题：生态修复、交通可达性、旅游开发、新城/绿地开发等。
- 泛城市语境但缺少更新对象：只谈城市治理、城市化、房地产市场或住房市场，而没有既有建成环境更新过程。

如果样本命中强负类规则，系统会优先进入 `hard_negative`，不再把它强行交给固定 taxonomy 正常吸收。

### 4.2 固定 topic taxonomy

固定 `topic taxonomy` 是预先定义好的主题标签体系。它包括三类标签：

- `U1-U15`：城市更新主题。
- `N1-N10`：非城市更新但容易混淆的高风险主题。
- `Unknown`、`Urban_Renewal_Other`、`Nonurban_Other`：兜底标签。

固定 taxonomy 的作用是把二分类之前的主题证据结构化，而不是直接替代最终二分类。每个主题由 `seeds`、`context_terms`、`combo_rules`、`exclude_terms`、`requires_anchor`、`missing_anchor_penalty` 等规则定义。系统会结合标题、摘要、关键词和学科背景做加权打分，并生成：

- `topic_rule`
- `topic_local_label`
- `topic_final`
- `topic_final_group`
- `topic_final_name`

`topic_final_group` 只分为 `urban`、`nonurban`、`unknown`。它是重要证据，但不是最终标签。

### 4.3 基础二分类与解释字段

主题融合完成后，系统会生成基础二分类分数和解释字段：

- `urban_probability_score`
- `binary_decision_threshold`
- `binary_decision_source`
- `binary_decision_evidence`
- `decision_explanation`
- `primary_positive_evidence`
- `primary_negative_evidence`
- `evidence_balance`
- `decision_rule_stack`

这些字段共同构成可解释链路。基础二分类会先写入 `final_label`、`urban_flag` 和 `是否属于城市更新研究`，但后续动态主题、动态二分类 `refine` 和 `UrbanBinaryPolicyV2` 仍会统一校准最终标签。

## 5. taxonomy_coverage_status 的完整含义

`taxonomy_coverage_status` 描述样本与固定 taxonomy 的关系。它不是置信度，也不是最终二分类标签。

| 状态 | 定义 | 触发条件 | 与最终二分类的关系 | 后续分析重点 |
|---|---|---|---|---|
| `covered` | 样本被现有固定 taxonomy 直接解释。 | `topic_final` 落入 `U1-U15` 或 `N1-N10`，`topic_final_group` 为 `urban` 或 `nonurban`，且不是 `Urban_Renewal_Other` / `Nonurban_Other`。 | 不等于正例。`U` 主题通常支持正例，`N` 主题通常支持负例，但最终仍由 V2 策略统一决定。 | 衡量固定 taxonomy 的原生覆盖能力。 |
| `unknown` | 固定 taxonomy 不能安全吸收样本。 | 最终 `topic_final=Unknown`，且缺少足够证据进入 `open_set`。 | 通常是复核、Unknown recovery、动态主题和 LLM 困难样本裁决的重点来源。 | 用于发现 taxonomy 漏洞和语义覆盖不足。 |
| `open_set` | 系统知道大类方向，但现有细粒度主题没有合适标签。 | 样本被识别为 `Urban_Renewal_Other` 或 `Nonurban_Other`。 | 可能支持正例或负例，取决于 open-set 方向和 V2 策略证据。 | 用于识别后续是否需要扩展 U/N 主题体系。 |
| `hard_negative` | 样本在正常 taxonomy 覆盖前被强负类规则拦截。 | 命中 `math_term_misuse`、`rural_nonurban` 等硬负类 route reason，或二分类 hard negative override。 | 通常进入 `protected_negative`，最终标签为 `0`。 | 用于检查负类保护是否过强或是否误伤真实城市更新。 |
| `binary_resolved` | 样本最初没有被固定 taxonomy 直接覆盖，但后续动态二分类 `refine` 将其修复到明确主题。 | `DynamicBinaryRefiner` 选出的 `candidate_topic` 不是 `Unknown`，也不是 open-set 标签，并且允许更新最终字段。 | 表示后处理修复成功，不等同于 `covered`；最终仍由 `UrbanBinaryPolicyV2` 同步标签。 | 用于衡量动态主题和二分类 refine 对漏召修复的贡献。 |

最容易混淆的是 `covered` 与 `binary_resolved`：

- `covered` 表示固定 taxonomy 在原始主题阶段就能直接解释样本。
- `binary_resolved` 表示固定 taxonomy 原本没有稳定吸收样本，后来通过动态主题和二分类 `refine` 修复到了一个明确主题。

同样需要区分 `unknown` 与 `open_set`：

- `unknown` 表示系统无法安全判断主题方向。
- `open_set` 表示系统已经知道样本大致属于城市更新侧或非城市更新侧，但当前 `U1-U15` / `N1-N10` 没有合适细类。

因此，`open_set` 更像 taxonomy 扩展候选，`unknown` 更像覆盖失败和复核压力来源。

## 6. 动态主题发现层

动态主题发现层是固定 taxonomy 的旁路增强。它不直接覆盖最终二分类，而是把 `unknown`、`open_set`、`binary_resolved`、复核、近阈值、冲突样本组织成可解释主题簇，为后续二分类修复、复核排序和 taxonomy 迭代提供证据。

### 6.1 候选池构建

默认候选池包括：

- `topic_final=Unknown` 或 `topic_final_group=unknown`。
- `taxonomy_coverage_status in {unknown, open_set, binary_resolved}`。
- `review_flag_raw > 0`。
- `review_reason` 包含 `unknown`、`open_set`、`near_threshold`、`conflict`、`inconsistency`、`uncertain`。
- `binary_decision_source` 包含 `unknown`、`review`、`uncertain`、`anchor_guard`。

若启用 `--dynamic-topics-full-corpus`，非候选样本也会进入 `full_corpus_pool` 作为背景聚类，用于判断某个动态主题是否只是局部噪声。

### 6.2 本地聚类与命名

动态主题默认使用本地算法，不调用 LLM/API。可用路径包括：

- TF-IDF 或本地 embedding 表征。
- MiniBatchKMeans、HDBSCAN 或 BERTopic 风格聚类。
- c-TF-IDF、KeyBERT 或高频短语抽取。
- 基于关键词模板的中文主题命名。

主题名不是由 LLM 生成，而是由关键词和规则模板生成。例如：

- `brownfield` -> 棕地再开发与土地利用转型。
- `neighborhood` / `community` -> 社区更新与社区治理。
- `heritage` / `historic` -> 历史街区保护与遗产活化。
- `gentrification` -> 绅士化与社区变化。

### 6.3 与固定 taxonomy 的映射

动态主题关键词会与固定 `TOPIC_DEFINITIONS` 的 `seeds`、`context_terms`、正负向词和组合规则做匹配。

| `dynamic_mapping_status` | 含义 |
|---|---|
| `mapped_to_fixed` | 动态主题可以高置信映射回既有固定主题。 |
| `candidate_new_urban_topic` | 具备城市更新动作和对象，但不能映射到现有 `U1-U15`，可能需要扩展城市更新主题。 |
| `candidate_new_nonurban_topic` | 主题明显偏农村、交通、方法、生态、旅游等非城市更新方向。 |
| `needs_review` | 证据不足，必须进入人工或 LLM 困难样本复核。 |

动态主题输出字段包括：

- `dynamic_topic_id`
- `dynamic_topic_name_zh`
- `dynamic_topic_keywords`
- `dynamic_topic_size`
- `dynamic_topic_confidence`
- `dynamic_topic_source_pool`
- `dynamic_to_fixed_topic_candidate`
- `dynamic_mapping_status`

## 7. 动态二分类 refine

动态二分类 `refine` 是本文档选中文本的核心位置：

> 这条样本最初并不是被固定 taxonomy 直接覆盖住的，但后续动态二分类 refine 把它修复到了一个明确主题。

这句话对应的状态就是 `binary_resolved`。

其含义不是“系统随意把样本翻成正例”，而是：

1. 该样本最初可能落入 `unknown`、`open_set` 或冲突复核池。
2. 动态主题发现层将它归入一个高置信主题簇。
3. 该主题簇与固定 taxonomy 存在明确映射，或者具备清晰的城市更新动作和既有城市对象证据。
4. 动态二分类 `refine` 通过门槛后，才允许更新 `topic_final`、`topic_final_group` 和最终二分类字段。
5. 系统将 `taxonomy_coverage_status` 标记为 `binary_resolved`，说明这是后处理修复成功，而不是固定 taxonomy 原生覆盖成功。

动态二分类 `refine` 的基本约束包括：

| 判断环节 | 规则含义 |
|---|---|
| 基本门槛 | `dynamic_binary_candidate_label` 必须为 `0` 或 `1`，`dynamic_topic_confidence` 需要达到高置信阈值，`dynamic_topic_size` 不能过小。 |
| 正例修复 | Unknown 或冲突样本若同时具备核心更新动作和既有城市对象，且没有农村或纯方法风险，可以被修复为正例候选。 |
| 负例保护 | 高置信非城市动态簇可以把疑似假阳性送入 `protected_negative` 或 `conflict_review`。 |
| 字段同步 | 若允许 `mutate_final_fields`，refine 会同步更新 `final_label`、`urban_flag`、`是否属于城市更新研究`、`topic_final`、`topic_final_group`、`taxonomy_coverage_status` 和解释字段。 |

因此，`binary_resolved` 的论文含义是：固定主题体系覆盖不足被动态主题证据和二分类后处理修复。它可以作为衡量动态主题层贡献的变量。

## 8. UrbanBinaryPolicyV2 最终二分类

`UrbanBinaryPolicyV2` 是最终落点层。它接收基础分类、固定 taxonomy、动态主题、动态 `refine`、解释字段和风险信号，统一决定最终标签。

该层解决两个问题：

- 不能让 `topic_final_group` 粗暴决定最终二分类。
- 不能让动态主题单独覆盖最终标签。

最终策略动作包括：

| `binary_policy_action` | 含义 | 典型触发 |
|---|---|---|
| `accept_positive` | 接受正例。 | 城市更新主题、更新动作、既有城市对象与分数证据一致，且没有强负类风险。 |
| `accept_negative` | 接受负例。 | 缺少更新动作或城市对象，或固定非城市主题/风险证据足以支持负例。 |
| `protected_negative` | 硬负类保护。 | 命中 `rural_nonurban`、`math_term_misuse` 或其他 hard negative。 |
| `conflict_review` | 保留为冲突样本。 | 当前二分类为正但 `topic_final_group` 为 `nonurban/unknown`，或 `evidence_balance=conflict_positive`，或近阈值/动态主题冲突。 |

no-LLM 路径中，`conflict_review` 不会触发 API 调用，只会记录 `llm_adjudication_required=1`。LLM 路径只在 `research_matrix` 且 `--hybrid-llm-assist on` 时，对这些困难样本执行结构化二分类裁决。

LLM 裁决必须满足以下边界：

- 只处理困难样本，不处理全量样本。
- 输入包括标题、摘要、关键词、规则证据、固定主题、动态主题和冲突类型。
- 输出必须是结构化二分类裁决。
- 只有解析成功且置信度达到阈值时，才允许覆盖规则结果。
- no-LLM 稳定路径必须保持 `llm_used_sum=0` 且 `llm_attempted_sum=0`。

## 9. 最终输出与质量检查

最终预测工作簿必须包含三类字段：

| 类别 | 代表字段 | 检查重点 |
|---|---|---|
| 最终标签 | `final_label`、`urban_flag`、`是否属于城市更新研究` | 三列必须一致。 |
| 固定主题 | `topic_final`、`topic_final_group`、`topic_final_name`、`taxonomy_coverage_status` | 说明固定 taxonomy 如何解释或未解释样本。 |
| 二分类解释 | `urban_probability_score`、`binary_decision_source`、`decision_explanation`、`evidence_balance` | 说明为什么判为正例或负例。 |
| 动态主题 | `dynamic_topic_id`、`dynamic_topic_name_zh`、`dynamic_mapping_status` | 说明动态主题对复核和 taxonomy 迭代的贡献。 |
| V2 策略 | `binary_policy_action`、`binary_policy_reason`、`binary_policy_conflict_type`、`llm_adjudication_required` | 说明最终二分类动作与困难样本来源。 |
| LLM 审计 | `llm_used`、`llm_attempted`、`llm_adjudication_label`、`llm_adjudication_confidence` | no-LLM 必须全部为 `0`；LLM 模式只允许困难样本裁决。 |

输出后必须检查：

- 行数是否等于输入行数。
- `final_label`、`urban_flag`、`是否属于城市更新研究` 是否完全一致。
- no-LLM 模式下 `llm_used_sum` 和 `llm_attempted_sum` 是否都为 `0`。
- 是否包含 `binary_policy_action`、`binary_policy_reason`、`llm_adjudication_required`。
- 是否包含 `dynamic_topic_id`、`dynamic_mapping_status`。

## 10. 有标签评估与无标签结构诊断

有标签数据用于计算标准二分类指标：

- Accuracy
- Precision
- Recall
- F1
- TP / TN / FP / FN

无标签大样本不计算 Accuracy，而是进行结构诊断：

- 样本总数。
- 正例数、负例数、正例率。
- `binary_policy_action` 分布。
- `topic_final_group` 分布。
- `dynamic_mapping_status` 分布。
- `taxonomy_coverage_status` 分布。
- `llm_used_sum` 与 `llm_attempted_sum`。
- 标签一致性错误数量。

这种双重评估是必要的。1000 篇带标签样本能验证局部精度，但不能保证大规模真实语料中的输出比例合理。10000 篇和全量 40276 篇无标签结构诊断可以检查规则是否过度保守或过度宽松。

## 11. 已归档实验结果

当前已归档的核心实验结果如下：

| 实验 | 样本量 | LLM 设置 | 正例率 | Accuracy | 说明 |
|---|---:|---|---:|---:|---|
| `1000_labeled_no_llm_stable_policy_v2` | 1000 | off | 96.70% | 84.10% | 达到 no-LLM Accuracy >= 80% 目标，高召回但 FP 较多。 |
| `1000_labeled_llm_stable_policy_v2` | 1000 | conflict only | 89.40% | 90.40% | LLM 只裁决困难样本后达到 Accuracy >= 85% 目标。 |
| `10000_no_llm_stable_policy_v2_seed20260428` | 10000 | off | 76.65% | 不适用 | 无标签结构诊断，正例率落在 75%-85%。 |
| `10000_no_llm_resample_seed2026042802` | 10000 | off | 75.86% | 不适用 | 排除上一轮标题后重抽样，比例稳定。 |
| `40276_full_no_llm_policy_v2_complete_20260429` | 40276 | off | 76.40% | 不适用 | 全量运行，`llm_used_sum=0`，`llm_attempted_sum=0`，标签一致性错误为 0。 |

全量 no-LLM 结果：

| 指标 | 数值 |
|---|---:|
| 输出行数 | 40276 |
| 正例数 | 30770 |
| 负例数 | 9506 |
| 正例率 | 76.40% |
| `llm_used_sum` | 0 |
| `llm_attempted_sum` | 0 |
| `llm_adjudication_required_sum` | 16939 |
| 标签一致性错误 | 0 |

全量 no-LLM 的 `binary_policy_action` 分布：

| `binary_policy_action` | 数量 | 占比 |
|---|---:|---:|
| `conflict_review` | 16939 | 42.06% |
| `accept_positive` | 13831 | 34.34% |
| `accept_negative` | 9322 | 23.15% |
| `protected_negative` | 184 | 0.46% |

全量动态映射结果：

| `dynamic_mapping_status` | 数量 | 占比 |
|---|---:|---:|
| `mapped_to_fixed` | 30254 | 75.12% |
| `needs_review` | 7457 | 18.51% |
| `candidate_new_nonurban_topic` | 2378 | 5.90% |
| `candidate_new_urban_topic` | 187 | 0.46% |

这些结果说明：当前 no-LLM 策略在全量语料上保持了约 76% 的正例率，满足 75%-85% 的大样本结构目标。同时，系统将 42.06% 的样本显式标记为 `conflict_review`，说明困难样本没有被隐藏，而是被保留下来供 LLM 或人工复核。

## 12. 论文写作中的解释方式

在论文中，`taxonomy_coverage_status` 可以作为主题体系覆盖能力的诊断变量，而不是模型性能指标。

推荐表述如下：

固定 taxonomy 提供可解释的主题证据，但其覆盖能力并非无限。为避免将新兴主题、边界样本或跨主题样本强行塞入既有标签体系，本文引入 `taxonomy_coverage_status` 记录样本与固定主题体系的关系。其中，`covered` 表示固定 taxonomy 直接覆盖成功，`unknown` 表示固定 taxonomy 无法安全解释，`open_set` 表示样本大类方向已知但细粒度主题缺位，`hard_negative` 表示样本被强负类规则提前拦截，`binary_resolved` 表示样本通过动态主题和二分类后处理被修复到明确主题。

对于 `binary_resolved`，应特别说明：

`binary_resolved` 并不是硬编码翻转标签，而是后处理证据链的显式记录。只有当动态主题簇具备足够置信度、样本量、关键词一致性，并同时满足城市更新动作锚点、既有城市对象锚点和风险控制条件时，系统才将原本未被固定 taxonomy 直接覆盖的样本修复到明确主题。该字段使动态主题层的贡献可以被追踪、复核和量化。

## 13. 后续分析建议

后续应重点分析以下样本池：

- `binary_policy_action=conflict_review`：按 `binary_policy_conflict_type` 分层抽检，判断假阳性风险。
- `taxonomy_coverage_status=binary_resolved`：确认动态 refine 是否真正修复了固定 taxonomy 盲区。
- `dynamic_mapping_status=candidate_new_urban_topic`：判断是否需要扩展 `U` 主题或补充 anchor/seed。
- `dynamic_mapping_status=needs_review`：做 Top 动态主题合并，识别尚未被固定 taxonomy 覆盖的新兴研究方向。
- `llm_adjudication_required=1`：在允许 LLM 的 `research_matrix` 实验中进行困难样本裁决，并对比 no-LLM 与 LLM 的 Accuracy、FP、FN 变化。

## 14. 实现位置

| 模块 | 作用 |
|---|---|
| `src/urban/urban_topic_taxonomy.py` | 定义 `Unknown`、`Urban_Renewal_Other`、`Nonurban_Other`、`U/N` 主题体系和主题分组逻辑。 |
| `src/urban/urban_hybrid_classifier.py` | 执行 `three_stage_hybrid` 基础分类，并设置 `covered`、`unknown`、`open_set`、`hard_negative` 等 taxonomy 覆盖状态。 |
| `src/urban/dynamic_topic_discovery.py` | 构建动态主题候选池、聚类、命名和固定 taxonomy 映射。 |
| `src/urban/dynamic_binary_refinement.py` | 根据高置信动态主题修复二分类与主题字段，并写入 `binary_resolved`。 |
| `src/urban/urban_binary_policy_v2.py` | 统一最终二分类策略动作，生成 `binary_policy_action`、`llm_adjudication_required` 和最终标签。 |
| `scripts/evaluation/evaluate.py` | 生成有标签评估和结构诊断报告。 |

本流程文档应与以下归档材料配套使用：

- `experiment_summary.csv`
- `run_commands.md`
- `full_no_llm_complete_20260429.md`
- `systematic_workflow_documentation.md`
- `paper_experiment_section_binary_classification.md`
