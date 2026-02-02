# 城市更新文献自动化标注研究报告

**日期**: 2026-02-02  
**版本**: v1.0

## 1. 项目背景与目标
本项目旨在构建一个基于大语言模型（LLM）的自动化文献标注系统，用于从海量学术文献的标题和摘要中提取关键信息。核心任务是识别该文献是否属于“城市更新”领域、是否包含“空间研究”、其“空间等级”及“具体空间描述”。通过引入多策略提示工程（Single, Stepwise, CoT, Reflection）和混合并发调度架构，实现了高精度、高效率的自动化处理。

## 2. 系统架构与技术实现

### 2.1 核心架构：混合调度引擎 (Hybrid Scheduler)
为了平衡处理速度与实验科学性，我们设计了独特的混合调度逻辑：

*   **并行处理组 (Parallel Group)**: 
    *   **策略**: `single`, `stepwise`, `cot`, `reflection`
    *   **机制**: 采用 `ThreadPoolExecutor` 线程池并发执行。每个任务拥有独立隔离的 `session_path`，确保线程安全。
    *   **收益**: 将处理时间缩短为原来的 1/N（N为并发数），极大提升了吞吐量。

*   **串行处理组 (Serial Group)**:
    *   **策略**: `stepwise_long` (长上下文)
    *   **机制**: 在主线程中严格串行执行。通过传递 `session_path=None` 触发内存复用机制，使模型能够“记住”之前处理过的论文（跨论文 Context）。
    *   **收益**: 确保了长上下文实验的逻辑正确性，避免了并发导致的上下文乱序。

### 2.2 提示词策略体系 (Prompt Engineering)
针对复杂的指标体系，设计了四层递进的策略，均支持 Zero-shot, One-shot, Few-shot 模式：

1.  **Single (单步策略)**: 
    *   一次性将所有规则输入模型，要求一次输出所有字段。效率最高，适合基准测试。
2.  **Stepwise (分步策略)**: 
    *   将任务拆解为 Step 1 (更新判定) -> Step 2 (空间判定) -> Step 3 (空间提取)。降低模型单次认知负荷，提升逻辑链条的稳定性。
3.  **CoT (思维链策略)**: 
    *   强制模型在输出结果前，先在 `<thinking>` 标签内进行推理。显式检查“是否新建”、“是否仅为背景”、“空间层级对齐”等关键点。
4.  **Reflection (反思策略)**: 
    *   引入 Critique 环节。模型先给出初步答案，然后根据特定的“检查清单”自我审查，最后给出修正后的答案。

### 2.3 指标体系重构 (9级空间分类)
根据最新的研究需求，建立了严格的判定标准：

*   **城市更新 (Urban Renewal)**: 
    *   **Yes**: 针对**既有建成区 (Existing built-up areas)** 的干预/改进。
    *   **No**: 新城开发、一般性城市化研究、纯房地产市场分析。
*   **空间研究 (Spatial Study)**: 
    *   **Yes**: 空间是**研究对象 (Subject)**（如案例研究、数据来源）。
    *   **No**: 空间仅为**背景 (Context)**（如“以中国为例...”但实际做理论模型）。
*   **空间等级 (Spatial Level)**: 
    *   Global / Multi-country / Country / Multi-province / Province / Urban Agglomeration / Single City / Multi-neighborhood / Single Neighborhood。

## 3. 当前效果评估

基于 `Data/test2` 数据集的初步运行结果，系统表现如下：

*   **准确性**: 
    *   在“是否属于城市更新”和“空间研究”的二分类任务上，Stepwise Few-shot 策略表现出了极高的准确率。
    *   空间等级判定能够准确区分 `Single City` (如 Barcelona) 和 `Multi-country`，且能识别出具体的空间描述。
*   **稳定性**:
    *   混合调度逻辑运行稳定，未出现上下文污染或线程冲突。
    *   API 调用通过 OpenAI SDK 适配，支持自动重试和速率限制，鲁棒性强。
*   **灵活性**:
    *   支持通过 `.env` 文件无缝切换 DeepSeek, Qwen, Moonshot 等任意 OpenAI 兼容模型。
    *   提供 `scripts/merge_results.py` 工具，可一键生成多策略对比报表。

## 4. 结论与展望
本项目成功搭建了一套工业级的文献自动化标注流水线，兼顾了学术实验的严谨性（Long Context）与工程实现的效率（Concurrency）。

**后续建议**:
1.  **大规模验证**: 运行 1000+ 篇量级的数据集，验证系统的长期稳定性。
2.  **策略优选**: 利用对比报表量化各策略的 F1-Score，选出性价比最高的策略组合（如 CoT Zero-shot 是否能打败 Stepwise Few-shot）。
3.  **Bad Case 分析**: 针对 Reflection 策略修正错误的案例进行深入分析，进一步微调提示词中的“负样本排除”规则。
