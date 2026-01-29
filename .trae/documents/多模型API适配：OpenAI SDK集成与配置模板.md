# 多模型 API 适配方案

我们将把硬编码的 DeepSeek 客户端替换为基于 `openai` SDK 的通用客户端，并通过 `.env` 文件实现灵活配置。这样您只需修改配置文件即可切换任何支持 OpenAI 接口规范的模型服务商（如 DeepSeek, DashScope, Moonshot, SiliconFlow 等）。

## 1. 核心变更

### 1.1 配置文件 (`.env` 模板)
我们将创建一个 `.env.example` 文件，展示如何配置不同厂商的 API。
核心变量：
*   `LLM_API_KEY`: 您的 API 密钥
*   `LLM_BASE_URL`: 模型服务的接口地址
*   `LLM_MODEL_NAME`: 模型名称

### 1.2 代码重构
1.  **`src/config.py`**: 更新配置加载逻辑，优先读取 `LLM_` 前缀的变量，同时向下兼容 `DEEPSEEK_` 前缀。
2.  **`src/llm_client.py`**: 引入 `openai` 库，重写 `DeepSeekClient` 类（建议重命名为 `LLMClient`），使用 `OpenAI(base_url=..., api_key=...)` 初始化。

## 2. 实施步骤

1.  **添加依赖**: 在 `requirements.txt` 中加入 `openai`。
2.  **更新 Config**: 修改 `src/config.py`。
3.  **重构 Client**: 修改 `src/llm_client.py`。
4.  **创建模板**: 创建 `.env` 文件（包含您的具体配置模板）。

## 3. `.env` 模板预览

```ini
# ==========================================
# LLM Configuration (OpenAI Compatible)
# ==========================================

# 1. DeepSeek (Default)
LLM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL_NAME=deepseek-chat

# 2. Aliyun DashScope (Qwen)
# LLM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# LLM_MODEL_NAME=qwen-plus

# 3. Moonshot (Kimi)
# LLM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# LLM_BASE_URL=https://api.moonshot.cn/v1
# LLM_MODEL_NAME=moonshot-v1-8k

# ==========================================
# System Settings
# ==========================================
MAX_WORKERS=5       # For multi-thread processing
MAX_TOKENS=1000     # Max output tokens
TIMEOUT=60          # Request timeout in seconds
```

我将为您直接创建这个 `.env` 文件，并完成代码的适配工作。
