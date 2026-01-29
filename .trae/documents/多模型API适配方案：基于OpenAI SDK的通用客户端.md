# 多模型 API 适配方案

您希望将 API 调用从硬编码的 DeepSeek 迁移到支持任意厂家（如 OpenAI, Anthropic, DashScope 等）的通用模式。

目前业界最通用的标准是 **OpenAI 兼容接口**。几乎所有主流模型服务商（包括 DeepSeek, DashScope, Moonshot, SiliconFlow 等）都提供了兼容 OpenAI SDK 的接口。

因此，最简单的方案是**引入 `openai` 官方 SDK**，并通过**配置文件**来动态指定 `base_url`, `api_key` 和 `model_name`。

## 1. 核心设计

### 1.1 配置文件 (`src/config.py` & `.env`)
不再硬编码 DeepSeek，而是使用通用的环境变量：
*   `LLM_API_KEY`
*   `LLM_BASE_URL` (例如 DeepSeek 是 `https://api.deepseek.com`, DashScope 是 `https://dashscope.aliyuncs.com/compatible-mode/v1`)
*   `LLM_MODEL_NAME` (例如 `deepseek-chat`, `qwen-turbo`, `gpt-4o`)

### 1.2 客户端重构 (`src/llm_client.py`)
*   **移除** `requests` 直接调用。
*   **引入** `openai` Python 库。
*   **实现**：
    ```python
    from openai import OpenAI
    
    class LLMClient:
        def __init__(self):
            self.client = OpenAI(
                api_key=Config.LLM_API_KEY,
                base_url=Config.LLM_BASE_URL
            )
    ```
*   这样，无论您用哪家模型，只要它支持 OpenAI 格式（绝大多数都支持），您只需要改 `.env` 文件即可。

## 2. 实施步骤

1.  **更新依赖**: 在 `requirements.txt` 中添加 `openai>=1.0.0`。
2.  **更新配置**: 修改 `src/config.py`，将 `DEEPSEEK_` 前缀改为通用的 `LLM_` 前缀（同时为了兼容旧 `.env`，可以保留读取逻辑，但优先读取新变量）。
3.  **重构客户端**: 重写 `src/llm_client.py`，使用 `OpenAI` 类替换原有的 `requests` 逻辑。
4.  **更新 `.env`**: 创建或更新 `.env` 文件，提供模板。

## 3. 优势
*   **零代码切换**: 切换模型只需改 `.env`。
*   **标准统一**: 利用 OpenAI SDK 的成熟特性（自动重试、类型检查）。
*   **兼容性强**: 支持所有 OpenAI Compatible 的服务商。
