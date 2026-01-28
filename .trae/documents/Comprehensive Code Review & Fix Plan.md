# 代码审阅与优化方案 (V2)

我已收到您的反馈。除了解决并发冲突外，我们还将优化会话历史（Session History）的存储结构，使其更易于人工查阅和管理。

## 1. 核心改进点

### 1.1 解决 `index.json` 并发写冲突 (高优先级)
*   **问题**: 多线程同时读写全局索引文件导致崩溃。
*   **方案**: 在批量任务中**禁用**全局索引更新 (`skip_index=True`)。索引功能仅在单次调试模式下保留。

### 1.2 优化 `history/sessions` 存储结构 (新需求)
*   **现状**: 所有会话文件都堆积在 `history/sessions/` 根目录下，文件名是随机 UUID (`uuid.json`)，无法直观分辨属于哪篇论文或哪个策略。
*   **改进目标**: 建立层级化的文件夹结构，文件命名具备语义。
*   **新结构设计**:
    ```text
    history/
      sessions/
        {task_name}/             <-- 任务名 (如 test1)
          {paper_index}_Title/   <-- 论文标识 (序号+截断标题)
            single.json          <-- 策略名作为文件名
            cot.json
            stepwise.json
            ...
    ```
*   **优势**:
    1.  **直观**: 打开文件夹就能看到这篇论文在不同策略下的所有对话记录。
    2.  **整洁**: 避免根目录数千个文件堆积。
    3.  **可溯源**: 轻松定位特定论文的 AI 推理过程。

## 2. 实施步骤

### 步骤 1: 修改 `ConversationMemory` (`src/memory.py`)
*   **构造函数升级**:
    *   增加 `file_path` 参数：允许外部直接指定保存路径，而不仅是依赖 UUID。
    *   增加 `skip_index` 参数：用于关闭索引更新。
*   **逻辑调整**: 如果提供了 `file_path`，则保存到该路径；否则回退到默认的 UUID 逻辑。

### 步骤 2: 修改 `ExtractionStrategy` 及其子类 (`src/strategies/*.py`)
*   **接口调整**: `process` 方法需要接收 `task_name` 和 `paper_index` (或行号) 等上下文信息，以便生成路径。
    *   *注意*: 这需要修改 `process` 的签名，或者在 `__init__` 时传入任务信息。
    *   *更优方案*: 保持 `process` 签名简洁，但在 `DataProcessor` 调用时，通过 `Strategy` 的某个方法设置“当前上下文”或在 `process` 中增加可选参数 `session_path`。
*   **实现**:
    *   修改 `DataProcessor`，在调用 `process` 前，计算出该论文的目标会话路径：`history/sessions/{task}/{index}_{title_snippet}/{strategy}.json`。
    *   将此路径传递给策略对象，策略对象在初始化 `ConversationMemory` 时使用该路径。

### 步骤 3: 修改 `DataProcessor` (`src/data_processor.py`)
*   在 `run_batch` 循环中，生成语义化的会话路径。
*   调用策略时传入路径参数。

### 步骤 4: 禁用 `index.json`
*   在批量运行时，彻底关闭索引更新功能，避免性能损耗和并发锁。

## 3. 预期效果
执行后，您的 `history` 目录将变得井井有条，您可以直接点开 `test1/001_Urban_Renewal/cot.json` 查看 AI 是如何一步步推理出结果的，极大方便了调试和效果对比。
