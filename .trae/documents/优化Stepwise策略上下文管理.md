# 代码优化计划：修复 Stepwise 策略上下文污染问题

在审查代码时，我发现 `src/strategies/stepwise.py` 和 `src/strategies/stepwise_long.py` 的处理逻辑存在一个潜在的**上下文污染**风险，这可能会影响实验的准确性。

## 1. 发现的问题

### 1.1 Stepwise 策略 (Short Context)
目前 `stepwise.py` 的逻辑是：
*   每次处理一篇新论文，创建一个**新** Memory 实例。
*   Step 1 -> Add Prompt -> Add Response
*   Step 2 -> Add Prompt -> Add Response...
*   这部分逻辑是正确的，每篇论文互不干扰。

### 1.2 Stepwise Long 策略 (Long Context)
*   **问题所在**: `stepwise_long.py` 旨在保留**跨论文**的记忆，但它在 Step 2 和 Step 3 中构造 `prompt2` 和 `prompt3` 时，直接调用了 `get_step_prompt`。
*   `get_step_prompt` 的实现是：`[TITLE]... [ABSTRACT]... Step N: Question...`。
*   这意味着：在同一个 Memory 中，处理同一篇论文时，Title 和 Abstract 会被**重复输入 3 次**（Step 1 一次，Step 2 一次，Step 3 一次）。
*   **后果**:
    1.  **Token 浪费**: 摘要被重复输入，迅速消耗 Token Window，导致长上下文效果打折。
    2.  **模型困惑**: 模型在一个会话中看到三次同样的 Title/Abstract，虽然可能不会报错，但这不符合人类对话习惯，可能引起模型对“当前任务是什么”的混淆。

## 2. 优化方案

### 2.1 修改 `PromptGenerator`
增加一个参数 `include_context` (默认为 True)。
*   对于 Step 1，`include_context=True`，输出完整的 `[TITLE]... [ABSTRACT]... Question`。
*   对于 Step 2 和 Step 3，`include_context=False`，仅输出 `Step 2: Spatial study? Output only 1 or 0.`。

### 2.2 修改 `stepwise.py` 和 `stepwise_long.py`
*   **Step 1**: 发送完整 Prompt。
*   **Step 2 & 3**: 发送简化的 Prompt（仅包含问题），因为模型在 Step 1 的上下文中已经有了 Title 和 Abstract。

## 3. 实施步骤

1.  **修改 `src/prompts.py`**: 更新 `get_step_prompt` 方法，支持 `include_context` 参数。
2.  **修改 `src/strategies/stepwise.py`**: 在 Step 2/3 调用时传入 `include_context=False`。
3.  **修改 `src/strategies/stepwise_long.py`**: 同上。

这不仅能节省大量 Token，还能让对话流更自然，符合 Stepwise 的设计初衷。
