# -*- coding: utf-8 -*-
import argparse
import json
import os
import random
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    # 如果没有安装 tqdm，使用一个简单的替代品
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, unit=None, **kwargs):
            self.iterable = iterable
            self.total = total or (len(iterable) if iterable else None)
            self.desc = desc or ""
            self.unit = unit or "it"
            self.n = 0
            self.iter = iter(iterable) if iterable else None
        
        def __iter__(self):
            return self
        
        def __next__(self):
            if self.iter is None:
                raise StopIteration
            try:
                item = next(self.iter)
                self.n += 1
                if self.n % max(1, (self.total or 100) // 20) == 0 or self.n == self.total:
                    print(f"\r{self.desc}: {self.n}/{self.total or '?'} {self.unit}", end="", flush=True)
                return item
            except StopIteration:
                print()  # 换行
                raise
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            if hasattr(self, 'n') and self.total:
                print(f"\r{self.desc}: {self.n}/{self.total} {self.unit} 完成", flush=True)
            return False
        
        def update(self, n=1):
            self.n += n
            if self.total:
                if self.n % max(1, self.total // 20) == 0 or self.n == self.total:
                    print(f"\r{self.desc}: {self.n}/{self.total} {self.unit}", end="", flush=True)
        
        def set_postfix(self, postfix=None, **kwargs):
            # 简单实现，仅用于兼容性
            pass


DEFAULT_INPUT_XLSX = r"data/raw/V3.0.xlsx"
DEFAULT_SHEET = "清理汇总后数据"
DEFAULT_CONFIG_PATH = r"scripts/llm_extraction/python/study_area_llm_config.json"


def _find_project_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "data").exists() and (p / "scripts").exists():
            return p
    return start.resolve()


def _env_get(name: str) -> str:
    return os.environ.get(name, "").strip()


def _env_require(name: str, hint: str) -> str:
    v = _env_get(name)
    if v and "在这里填" not in v:
        return v
    raise RuntimeError(f"缺少环境变量: {name}\n请在 scripts/.env 中配置：\n{hint}")


def _load_env_from_file(path: str) -> None:
    p = Path(path)
    if not p.exists():
        return
    try:
        text = p.read_text(encoding="utf-8")
    except Exception:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if (len(v) >= 2) and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
            v = v[1:-1].strip()
        if not k:
            continue
        if k not in os.environ:
            os.environ[k] = v


def _now_ms() -> int:
    return int(time.time() * 1000)


def _compact_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    s = str(v)
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?。！？])\s+", text.strip())
    return [p.strip() for p in parts if p and p.strip()]


def _extract_trigger_sentences(abstract: str, max_sentences: int = 6) -> List[str]:
    triggers = [
        r"\bcase study\b",
        r"\bstudy (area|site)\b",
        r"\bin the city of\b",
        r"\bin (?:the )?(?:metropolitan area|metro area|municipality|county|province|state|region) of\b",
        r"\bwe (?:examine|study|analy[sz]e|investigate|focus on)\b",
        r"\bconducted in\b",
        r"\bin \b[A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+){0,4}\b",
    ]
    rx = re.compile("|".join(triggers), re.IGNORECASE)
    sents = _split_sentences(_compact_spaces(abstract))
    hits = [s for s in sents if rx.search(s)]
    if len(hits) >= max_sentences:
        return hits[:max_sentences]
    remaining = [s for s in sents if s not in hits]
    for s in remaining:
        if len(hits) >= max_sentences:
            break
        if len(s) >= 40:
            hits.append(s)
    return hits[:max_sentences]


def _hash_key(row: pd.Series) -> str:
    doi = _safe_str(row.get("DOI", ""))
    ut = _safe_str(row.get("UT (Unique WOS ID)", "")) or _safe_str(row.get("UT", ""))
    openalex = _safe_str(row.get("OpenAlex ID", ""))
    if doi:
        return f"doi:{doi}"
    if ut:
        return f"ut:{ut}"
    if openalex:
        return f"openalex:{openalex}"
    return f"idx:{int(row.name)}"


@dataclass
class ModelConfig:
    provider: str
    model: str
    api_key_env: str
    base_url: str


def _load_json_file(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) or {}



SCHEMA_EXTRACT = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "is_city_specific": {"type": "boolean"},
        "primary_area": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "city": {"type": "string"},
                        "admin1": {"type": "string"},
                        "country": {"type": "string"},
                        "raw_mention": {"type": "string"},
                    },
                    "required": ["city", "admin1", "country", "raw_mention"],
                },
            ]
        },
        "secondary_areas": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "city": {"type": "string"},
                    "admin1": {"type": "string"},
                    "country": {"type": "string"},
                    "raw_mention": {"type": "string"},
                },
                "required": ["city", "admin1", "country", "raw_mention"],
            },
        },
        "evidence_sentence": {"type": "string"},
        "reason": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "flags": {"type": "array", "items": {"type": "string"}},
        "is_urban_renewal": {"type": "boolean"},
        "urban_renewal_confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "is_city_specific",
        "primary_area",
        "secondary_areas",
        "evidence_sentence",
        "reason",
        "confidence",
        "flags",
        "is_urban_renewal",
        "urban_renewal_confidence",
    ],
}


def _build_prompt(abstract: str, triggers: List[str]) -> str:
    abstract = _compact_spaces(abstract)
    triggers_text = "\n".join([f"- {t}" for t in triggers]) if triggers else ""
    return _compact_spaces(
        f"""
你是一位学术地理信息分析专家。请分析给定摘要，判断这篇论文是否针对一个具体的城市展开研究，并提取详细的地理信息。

**分析步骤：**

**第一步：提取所有地理名称**
仔细识别摘要中出现的所有地理名称，包括：
- 国家名称（如：China, United States, Germany）
- 省/州/地区名称（如：Guangdong, California, Bavaria）
- 城市名称（如：Shenzhen, New York, Munich）
- 其他具体地点（如：特定区域、流域、行政区）

**第二步：判断研究的地理尺度**
- **城市级研究**：如果研究的核心结论、主要数据、政策建议或应用场景明确指向**一个具体城市**（例如：为该城市制定规划、评估该城市的风险、模拟该城市的系统、分析该城市的数据），则判定为城市级研究。
- **非城市级研究**：如果研究是在国家、跨国区域或全球尺度上描述普遍规律、进行多城市比较、提出宏观政策，或仅将城市作为数据点/案例，则判定为非城市级研究。

**第三步：提取详细信息（仅当为城市级研究时）**
如果判定为城市级研究，请从摘要中提取：
1. **城市名称**：作为核心研究对象的城市名（必须是明确的行政或功能城市名）
2. **国家名称**：该城市所属的国家
3. **省/州名称**：该城市所属的省、州或一级行政区（如果有明确提及）
4. **原始提及**：摘要中提及该城市的原始文本
5. **证据句子**：从摘要中**原封不动地摘取**最能证明该城市是研究对象的完整句子（必须是从摘要中直接截取的原文，不要改写）

**第四步：判断是否为城市更新研究领域**
请判断这篇论文是否属于"城市更新"（Urban Renewal / Urban Regeneration / Urban Redevelopment）研究领域。

**城市更新研究的核心特征：**
1. **研究对象**：针对现有建成区、旧城区、衰败区域的改造、更新、再生
2. **核心议题**：
   - 旧城改造、旧区更新、城市再生
   - 棚户区改造、城中村改造、老旧小区更新
   - 工业区转型、废弃地再利用
   - 历史街区保护与更新、文化遗产保护
   - 城市空间重构、功能置换
   - 社区更新、邻里改造
   - 城市存量用地盘活、土地再利用
3. **关键词识别**：
   - 英文：urban renewal, urban regeneration, urban redevelopment, urban rehabilitation, urban revitalization, slum clearance, gentrification (如果是改造性质), brownfield redevelopment, heritage preservation/restoration (如果是更新性质)
   - 中文：城市更新、城市改造、旧城改造、城市再生、城市复兴、棚户区改造、城中村改造、老旧小区、历史街区更新、工业区改造
4. **排除领域**（不属于城市更新）：
   - 纯粹的新城建设、新区开发（非更新性质）
   - 新城市主义、城市规划设计（非针对现有建成区）
   - 纯房地产开发（无更新改造性质）
   - 基础设施建设（非城市更新）
   - 单纯的智慧城市、可持续发展（除非明确涉及更新改造）

**置信度评分规则（urban_renewal_confidence）：**
- **0.90-1.00**：摘要中明确提及城市更新相关术语，研究焦点明确为城市更新（如："This study examines urban renewal in..."）
- **0.75-0.89**：研究主题高度相关但未明确使用"城市更新"术语（如：研究棚户区改造、旧城改造但未用"urban renewal"）
- **0.60-0.74**：研究部分涉及城市更新，但不是核心主题（如：多主题研究中包含更新内容）
- **0.40-0.59**：研究主题模糊，可能涉及也可能不涉及城市更新
- **0.20-0.39**：研究主题与城市更新关联性较低，但存在某些相关元素
- **0.00-0.19**：明确不属于城市更新研究领域

**输出要求：**
请输出一个合法的JSON对象，包含以下字段：
{{
  "is_city_specific": true/false,
  "primary_area": {{
    "city": "城市名称（如果是城市级研究，否则为null）",
    "admin1": "省/州名称（如果有，否则为空字符串）",
    "country": "国家名称（如果是城市级研究，否则为空字符串）",
    "raw_mention": "摘要中提及城市的原始文本（如果是城市级研究，否则为空字符串）"
  }},
  "secondary_areas": [],
  "evidence_sentence": "从摘要中直接截取的证据句子（必须原封不动，不要改写）",
  "reason": "判断理由的简要说明",
  "confidence": 0.0-1.0之间的数值（表示地理判断的置信度）,
  "flags": [],
  "is_urban_renewal": true/false（是否为城市更新研究）,
  "urban_renewal_confidence": 0.0-1.0之间的数值（城市更新判断的置信度，需根据上述评分规则）
}}

**重要提示：**
- 只输出JSON，不要添加任何解释文字或markdown格式
- evidence_sentence必须是摘要中的原句，不要改写或总结
- 如果无法确定国家或省/州，使用空字符串""，不要猜测
- confidence应反映地理判断的确定性（0.9+表示非常确定，0.7-0.9表示较确定，0.5-0.7表示不确定）
- urban_renewal_confidence需严格按照上述评分规则，结合摘要中城市更新相关关键词的明确程度和研究的核心焦点进行评分

You are an expert in academic geospatial analysis. Analyze the given abstract to determine if this paper focuses on a specific city, and extract detailed geographic information.

**Analysis Steps:**

**Step 1: Extract All Geographic Names**
Identify all geographic names in the abstract, including:
- Country names (e.g., China, United States, Germany)
- Province/State/Region names (e.g., Guangdong, California, Bavaria)
- City names (e.g., Shenzhen, New York, Munich)
- Other specific locations (e.g., specific regions, watersheds, administrative areas)

**Step 2: Determine Geographic Scale**
- **City-specific study**: If the core findings, primary data, policy implications, or application scenarios explicitly target **a specific city** (e.g., planning for the city, assessing its risks, modeling its systems, analyzing its data), classify as city-specific.
- **Non-city-specific study**: If the study describes general patterns, compares multiple cities, proposes macro-level policies at national/cross-national/global scales, or uses cities only as data points/cases, classify as non-city-specific.

**Step 3: Extract Detailed Information (only if city-specific)**
If classified as city-specific, extract from the abstract:
1. **City name**: The city serving as the core research object (must be a distinct administrative or functional city name)
2. **Country name**: The country where the city is located
3. **Province/State name**: The province, state, or first-level administrative division (if explicitly mentioned)
4. **Raw mention**: The original text mentioning the city in the abstract
5. **Evidence sentence**: A **verbatim sentence** from the abstract that best proves the city is the research object (must be directly extracted, not paraphrased)

**Step 4: Determine if the Study Belongs to Urban Renewal Research**
Please determine if this paper belongs to the "Urban Renewal / Urban Regeneration / Urban Redevelopment" research field.

**Core Characteristics of Urban Renewal Research:**
1. **Research Object**: Focus on the transformation, renewal, and regeneration of existing built-up areas, old urban areas, and declining regions
2. **Core Topics**:
   - Old city renovation, old district renewal, urban regeneration
   - Slum clearance, urban village redevelopment, old residential area renewal
   - Industrial zone transformation, brownfield redevelopment
   - Historic district preservation and renewal, cultural heritage protection
   - Urban spatial restructuring, functional replacement
   - Community renewal, neighborhood renovation
   - Urban land reuse, land reutilization
3. **Keywords**:
   - English: urban renewal, urban regeneration, urban redevelopment, urban rehabilitation, urban revitalization, slum clearance, gentrification (if renovation-oriented), brownfield redevelopment, heritage preservation/restoration (if renewal-oriented)
   - Chinese: 城市更新、城市改造、旧城改造、城市再生、城市复兴、棚户区改造、城中村改造、老旧小区、历史街区更新、工业区改造
4. **Excluded Fields** (NOT urban renewal):
   - Pure new town construction, new district development (non-renewal nature)
   - New urbanism, urban planning design (not targeting existing built-up areas)
   - Pure real estate development (no renewal/renovation nature)
   - Infrastructure construction (not urban renewal)
   - Pure smart city, sustainable development (unless explicitly involving renewal/renovation)

**Confidence Scoring Rules (urban_renewal_confidence):**
- **0.90-1.00**: Abstract explicitly mentions urban renewal-related terms, research focus is clearly urban renewal (e.g., "This study examines urban renewal in...")
- **0.75-0.89**: Research topic is highly relevant but doesn't explicitly use "urban renewal" term (e.g., studies slum clearance, old city renovation but doesn't use "urban renewal")
- **0.60-0.74**: Research partially involves urban renewal but it's not the core theme (e.g., multi-topic research including renewal content)
- **0.40-0.59**: Research topic is ambiguous, may or may not involve urban renewal
- **0.20-0.39**: Research topic has low relevance to urban renewal but contains some related elements
- **0.00-0.19**: Clearly does not belong to urban renewal research field

**Output Requirements:**
Output a valid JSON object with the following fields:
{{
  "is_city_specific": true/false,
  "primary_area": {{
    "city": "City name (if city-specific, otherwise null)",
    "admin1": "Province/State name (if available, otherwise empty string)",
    "country": "Country name (if city-specific, otherwise empty string)",
    "raw_mention": "Original text mentioning the city (if city-specific, otherwise empty string)"
  }},
  "secondary_areas": [],
  "evidence_sentence": "Verbatim evidence sentence from abstract (must be directly extracted, not paraphrased)",
  "reason": "Brief explanation of the judgment",
  "confidence": 0.0-1.0 (confidence level of geographic judgment),
  "flags": [],
  "is_urban_renewal": true/false (whether it's urban renewal research),
  "urban_renewal_confidence": 0.0-1.0 (confidence of urban renewal judgment, follow scoring rules above)
}}

**Important Notes:**
- Output only JSON, no explanatory text or markdown formatting
- evidence_sentence must be a verbatim sentence from the abstract, not paraphrased or summarized
- If country or province/state cannot be determined, use empty string "", do not guess
- confidence should reflect certainty of geographic judgment (0.9+ = very certain, 0.7-0.9 = fairly certain, 0.5-0.7 = uncertain)
- urban_renewal_confidence must strictly follow the scoring rules above, based on the explicitness of urban renewal-related keywords and the research's core focus

摘要 / Abstract:
{abstract}

候选关键句（供你定位研究区域，可能为空）/ Candidate Key Sentences (may be empty):
{triggers_text}
"""
    )


def _post_validate(obj: Dict[str, Any]) -> Tuple[bool, str]:
    required = [
        "is_city_specific",
        "primary_area",
        "secondary_areas",
        "evidence_sentence",
        "reason",
        "confidence",
        "flags",
    ]
    for k in required:
        if k not in obj:
            return False, f"missing:{k}"
    if not isinstance(obj["is_city_specific"], bool):
        return False, "type:is_city_specific"
    if obj["primary_area"] is not None and not isinstance(obj["primary_area"], dict):
        return False, "type:primary_area"
    if not isinstance(obj["secondary_areas"], list):
        return False, "type:secondary_areas"
    conf = obj["confidence"]
    if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
        return False, "range:confidence"
    return True, ""


def _http_post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout_s: int = 90) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        # 提供更详细的403错误信息
        if e.code == 403:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(
                f"HTTP 403 Forbidden - API密钥认证失败\n"
                f"可能的原因：\n"
                f"1. API密钥无效或已过期\n"
                f"2. API密钥格式错误（确保没有多余的空格或换行）\n"
                f"3. API密钥没有访问该模型的权限\n"
                f"4. 使用了错误的API端点URL\n"
                f"\n当前配置：\n"
                f"  URL: {url}\n"
                f"  环境变量: {headers.get('Authorization', '').split(' ')[0] if 'Authorization' in headers else '未设置'}\n"
                f"  响应: {error_body[:200] if error_body else '无响应内容'}\n"
                f"\n解决方法：\n"
                f"1. 检查API密钥是否正确设置在环境变量中\n"
                f"2. 访问API提供商的网站验证密钥是否有效\n"
                f"3. 确认密钥有访问所使用模型的权限"
            ) from e
        raise


def call_openai_compat_chat(
    cfg: ModelConfig,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_s: int = 90,
) -> str:
    api_key = os.environ.get(cfg.api_key_env, "").strip()
    if not api_key or "在这里填" in api_key:
        # 尝试确定 .env 文件的路径
        script_dir = Path(__file__).resolve().parent
        # 假设脚本在 scripts/llm_extraction/python/，而 .env 在 scripts/.env
        env_file = script_dir.parent.parent / ".env"
        if not env_file.exists():
             project_root = _find_project_root(script_dir)
             env_file = project_root / "scripts" / ".env"
        
        raise RuntimeError(
            f"缺少 API 密钥环境变量: {cfg.api_key_env}\n"
            f"请设置该环境变量，方式如下（任选一种）：\n"
            f"  方式1: 在项目根目录创建 .env 文件（{env_file}），添加以下内容：\n"
            f"    {cfg.api_key_env}=你的API密钥\n"
            f"  方式2: 在命令行中设置环境变量：\n"
            f"    Windows PowerShell: $env:{cfg.api_key_env}='你的API密钥'\n"
            f"    Windows CMD: set {cfg.api_key_env}=你的API密钥\n"
            f"    Linux/Mac: export {cfg.api_key_env}='你的API密钥'"
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = _http_post_json(cfg.base_url, headers, payload, timeout_s=timeout_s)
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"Empty choices from {cfg.provider}")
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    finish_reason = choices[0].get("finish_reason")
    
    # 针对 DeepSeek Reasoner 的特殊处理：如果 content 为空但有 reasoning_content
    if not content and msg.get("reasoning_content"):
        # 如果只有推理过程没有最终内容，这是一个异常情况，但我们可以尝试记录
        print(f"Warning: {cfg.model} returned reasoning_content but empty content. Finish reason: {finish_reason}")
    
    if not content:
        # 记录更多调试信息
        debug_info = f"Keys: {list(msg.keys())}"
        if "reasoning_content" in msg:
            debug_info += f", reasoning_len={len(msg['reasoning_content'])}"
        if finish_reason:
            debug_info += f", finish_reason={finish_reason}"
            
        if finish_reason == "length":
             raise RuntimeError(f"Output truncated (max_tokens={max_tokens} too small?). {debug_info}")
             
        raise RuntimeError(f"Empty content from {cfg.provider} ({cfg.model}). {debug_info}")
    return content


def build_openai_compat_payload(model: str, messages: List[Dict[str, str]], max_tokens: int, temperature: float) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def _estimate_messages_chars(messages: List[Dict[str, str]]) -> int:
    total = 0
    for m in messages:
        total += len(_safe_str(m.get("role", ""))) + 8
        total += len(_safe_str(m.get("content", ""))) + 8
    return total


def _build_experiment1_system_prompt(shot: str) -> str:
    base = _compact_spaces(
        """
你是学术论文信息抽取与标注助手。你将对一批论文进行结构化标注。每篇论文会分三轮对话依次提取结果：第1轮仅判断“是否城市更新研究”；第2轮仅判断“空间/非空间研究”；第3轮仅提取“空间等级”和“具体空间描述”。你必须严格保持轮次任务边界，不要在某一轮输出其他轮次的字段。

全局要求：
1) 输入包括：标题（Title）、摘要（Abstract）、关键词（Keywords）。
2) 你必须在每一轮只输出一段严格合法 JSON（不要markdown，不要额外解释文字）。
3) evidence 必须直接摘取输入中的原文片段（标题/摘要/关键词），不要改写。
4) basis 只允许写最多3条“可复核的短依据”，不要写长推理过程。
5) 你将持续收到多篇论文，务必保持前后一致的判定口径；在不确定时给出较低置信度。
"""
    )
    if shot == "zero":
        return base
    if shot == "one":
        ex = _compact_spaces(
            """
示例（仅用于学习格式与口径）：
Title: Urban renewal of public housing estates in Murcia
Abstract: This study analyses and evaluates three estates that make up a new urban axis in the city of Murcia.
Keywords: urban renewal; public housing

Round1 输出:
{"UR_IsUrbanRenewal":1,"confidence":0.90,"evidence":"urban renewal; public housing","basis":["关键词包含urban renewal/public housing","摘要明确讨论urban renewal对象"],"notes":""}
Round2 输出:
{"Spatial_IsSpatial":1,"confidence":0.85,"evidence":"in the city of Murcia","basis":["研究对象是具体城市案例"],"notes":""}
Round3 输出:
{"Spatial_Level":"市级","Spatial_Location":"Spain-Region of Murcia-Murcia-Murcia","confidence":0.80,"evidence":"city of Murcia","basis":["摘要明确指向Murcia城市"],"notes":""}
"""
        )
        return base + "\n\n" + ex
    ex = _compact_spaces(
        """
示例（仅用于学习格式与口径）：
Example A
Title: Urban regeneration and gentrification in London
Abstract: We examine regeneration policies and their impacts in London neighbourhoods.
Keywords: urban regeneration; gentrification
Round1: {"UR_IsUrbanRenewal":1,"confidence":0.85,"evidence":"urban regeneration; regeneration policies","basis":["关键词/摘要明确urban regeneration"],"notes":""}
Round2: {"Spatial_IsSpatial":1,"confidence":0.80,"evidence":"in London neighbourhoods","basis":["研究对象包含明确地点"],"notes":""}
Round3: {"Spatial_Level":"市级","Spatial_Location":"United Kingdom-England-London-London","confidence":0.75,"evidence":"in London neighbourhoods","basis":["摘要明确London"],"notes":""}

Example B
Title: A review of urban renewal financing instruments
Abstract: This paper reviews financing instruments and proposes a conceptual framework. No case location is analysed.
Keywords: urban renewal; financing
Round1: {"UR_IsUrbanRenewal":1,"confidence":0.70,"evidence":"urban renewal","basis":["主题为城市更新融资综述"],"notes":"偏综述，城市更新方向仍成立"}
Round2: {"Spatial_IsSpatial":0,"confidence":0.85,"evidence":"No case location is analysed","basis":["未分析具体空间对象"],"notes":""}
Round3: {"Spatial_Level":"","Spatial_Location":"","confidence":0.90,"evidence":"No case location is analysed","basis":["非空间研究无需空间层级"],"notes":""}

Example C
Title: National housing policy reform and sustainability
Abstract: We discuss national policy reforms and outcomes across countries without focusing on any particular city.
Keywords: housing policy; sustainability
Round1: {"UR_IsUrbanRenewal":0,"confidence":0.70,"evidence":"national policy reforms","basis":["主题为国家政策改革，未体现城市更新核心对象"],"notes":""}
Round2: {"Spatial_IsSpatial":0,"confidence":0.80,"evidence":"without focusing on any particular city","basis":["无具体空间研究对象"],"notes":""}
Round3: {"Spatial_Level":"","Spatial_Location":"","confidence":0.90,"evidence":"without focusing on any particular city","basis":["非空间研究无需空间层级"],"notes":""}
"""
    )
    return base + "\n\n" + ex


def _build_experiment1_round_prompt(round_no: int, key: str, title: str, abstract: str, keywords: str) -> str:
    title = _compact_spaces(title)
    abstract = _compact_spaces(abstract)
    keywords = _compact_spaces(keywords)
    if round_no == 1:
        return _compact_spaces(
            f"""
【论文ID】{key}
请进行第1轮标注：判断该论文是否属于“城市更新研究”（1是/0否）。
只输出JSON，字段必须为：UR_IsUrbanRenewal(0/1), confidence(0-1), evidence, basis(数组), notes(字符串)。

Title: {title}
Abstract: {abstract}
Keywords: {keywords}
"""
        )
    if round_no == 2:
        return _compact_spaces(
            f"""
【论文ID】{key}
请进行第2轮标注：判断该论文是否属于“空间研究”（1是/0否）。
空间研究：研究对象/结论显著依赖空间位置、空间分布、特定地区/城市/区域案例、地理差异、地图/空间模型等；否则为非空间研究。
只输出JSON，字段必须为：Spatial_IsSpatial(0/1), confidence(0-1), evidence, basis(数组), notes(字符串)。

Title: {title}
Abstract: {abstract}
Keywords: {keywords}
"""
        )
    return _compact_spaces(
        f"""
【论文ID】{key}
请进行第3轮标注：提取空间等级与具体空间描述。
Spatial_Level 只能取：国家/省级（州）/市级/城市群/。
Spatial_Location 输出格式：国家-省份/州-市级-具体城市；无法确定则留空字符串。
如果第2轮你判定为非空间研究，请两个字段都输出空字符串。
只输出JSON，字段必须为：Spatial_Level, Spatial_Location, confidence(0-1), evidence, basis(数组), notes(字符串)。

Title: {title}
Abstract: {abstract}
Keywords: {keywords}
"""
    )


def _try_parse_json_obj(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    t = t[start : end + 1]
    try:
        return json.loads(t)
    except Exception:
        return None


def _ensure_list_str(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if _safe_str(x)]
    return [str(v)]


@dataclass
class Experiment1Session:
    cfg: ModelConfig
    shot: str
    session_id: int = 0
    context_char_budget: int = 250000
    trace_jsonl: str = ""
    trace_level: str = "tail"
    trace_tail_messages: int = 30
    memory_max_examples: int = 20
    dry_run: bool = False

    def __post_init__(self) -> None:
        self.messages: List[Dict[str, str]] = [{"role": "system", "content": _build_experiment1_system_prompt(self.shot)}]
        self.memory_examples: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def _write_trace(self, record: Dict[str, Any]) -> None:
        if not self.trace_jsonl:
            return
        out_dir = os.path.dirname(self.trace_jsonl)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with self.lock:
            with open(self.trace_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _messages_for_trace(self) -> List[Dict[str, str]]:
        if self.trace_level == "none":
            return []
        if self.trace_level == "full":
            return self.messages
        tail_n = max(0, int(self.trace_tail_messages))
        if tail_n <= 0:
            return []
        return self.messages[-tail_n:]

    def _build_memory_summary(self) -> str:
        if not self.memory_examples:
            return ""
        items = self.memory_examples[-self.memory_max_examples :]
        lines = ["以下为已完成标注的近期样例（用于保持一致口径）："]
        for ex in items:
            lines.append(f"- ID={ex.get('key','')}")
            if ex.get("title"):
                lines.append(f"  Title: {ex.get('title')}")
            if ex.get("evidence"):
                lines.append(f"  evidence: {ex.get('evidence')}")
            lines.append(f"  UR_IsUrbanRenewal={ex.get('UR_IsUrbanRenewal','')}, Spatial_IsSpatial={ex.get('Spatial_IsSpatial','')}, Spatial_Level={ex.get('Spatial_Level','')}, Spatial_Location={ex.get('Spatial_Location','')}")
        return _compact_spaces("\n".join(lines))

    def _reset_with_memory(self, reason: str) -> Dict[str, Any]:
        self.session_id += 1
        memory = self._build_memory_summary()
        self.messages = [{"role": "system", "content": _build_experiment1_system_prompt(self.shot)}]
        if memory:
            self.messages.append({"role": "user", "content": f"CONTEXT_RESET=true; reason={reason}\n{memory}"})
        return {"ContextReset": 1, "ContextReset_Reason": reason, "SessionId": self.session_id}

    def ensure_budget(self, upcoming_chars: int = 0) -> Dict[str, Any]:
        chars = _estimate_messages_chars(self.messages) + max(0, int(upcoming_chars))
        if chars <= int(self.context_char_budget):
            return {"ContextReset": 0, "ContextReset_Reason": "", "SessionId": self.session_id}
        return self._reset_with_memory("exceed_budget")

    def ask_json(self, prompt: str, max_tokens: int, temperature: float, timeout_s: int, meta: Optional[Dict[str, Any]] = None, max_repairs: int = 2) -> Tuple[str, Optional[Dict[str, Any]], str, Dict[str, Any]]:
        reset_info = self.ensure_budget(upcoming_chars=len(prompt) + 2000)
        before_chars = _estimate_messages_chars(self.messages)
        self.messages.append({"role": "user", "content": prompt})
        if self.dry_run:
            rno = int((meta or {}).get("round") or 0)
            if rno == 1:
                obj = {"UR_IsUrbanRenewal": 0, "confidence": 0.0, "evidence": "", "basis": [], "notes": "dry_run"}
            elif rno == 2:
                obj = {"Spatial_IsSpatial": 0, "confidence": 0.0, "evidence": "", "basis": [], "notes": "dry_run"}
            else:
                obj = {"Spatial_Level": "", "Spatial_Location": "", "confidence": 0.0, "evidence": "", "basis": [], "notes": "dry_run"}
            raw = json.dumps(obj, ensure_ascii=False)
            self.messages.append({"role": "assistant", "content": raw})
            parsed = obj
            err = ""
            repairs = 0
            after_chars = _estimate_messages_chars(self.messages)
            record = {
                "session_id": self.session_id,
                "context_chars_before": before_chars,
                "context_chars_after": after_chars,
                "context_reset": reset_info.get("ContextReset", 0),
                "context_reset_reason": reset_info.get("ContextReset_Reason", ""),
                "prompt": prompt,
                "raw": raw,
                "parsed_ok": True,
                "repairs": 0,
                "messages": self._messages_for_trace(),
                "dry_run": True,
            }
            if meta:
                record.update(meta)
            self._write_trace(record)
            return raw, parsed, err, reset_info

        try:
            raw = call_openai_compat_chat(self.cfg, self.messages, max_tokens=max_tokens, temperature=temperature, timeout_s=timeout_s)
        except Exception as e:
            # 捕获 API 调用错误，避免中断整个批处理
            err_msg = f"api_error:{type(e).__name__}:{str(e)}"
            print(f"\n[Error] Session {self.session_id} failed: {err_msg}")
            
            # 记录失败的 trace
            record = {
                "session_id": self.session_id,
                "context_chars_before": before_chars,
                "context_chars_after": before_chars,
                "context_reset": reset_info.get("ContextReset", 0),
                "context_reset_reason": reset_info.get("ContextReset_Reason", ""),
                "prompt": prompt,
                "raw": "",
                "parsed_ok": False,
                "error": err_msg,
                "messages": self._messages_for_trace(),
            }
            if meta:
                record.update(meta)
            self._write_trace(record)
            
            return "", None, err_msg, reset_info

        self.messages.append({"role": "assistant", "content": raw})

        parsed = _try_parse_json_obj(raw)
        err = ""
        repairs = 0
        while parsed is None and repairs < max_repairs:
            repairs += 1
            fix = _compact_spaces(
                f"""
                你上一轮输出无法被解析为JSON。请仅输出一段严格合法JSON，不要输出任何其它文字。
                上一轮输出：
                {raw}
                """
            )
            self.messages.append({"role": "user", "content": fix})
            try:
                raw = call_openai_compat_chat(self.cfg, self.messages, max_tokens=max_tokens, temperature=0, timeout_s=timeout_s)
                self.messages.append({"role": "assistant", "content": raw})
                parsed = _try_parse_json_obj(raw)
            except Exception as e:
                # 修复过程中如果再次报错，停止修复，保留上次的错误信息
                print(f"\n[Error] Repair failed: {e}")
                break

        if parsed is None:
            err = "parse_failed"

        after_chars = _estimate_messages_chars(self.messages)
        record = {
                "session_id": self.session_id,
                "context_chars_before": before_chars,
                "context_chars_after": after_chars,
                "context_reset": reset_info.get("ContextReset", 0),
                "context_reset_reason": reset_info.get("ContextReset_Reason", ""),
                "prompt": prompt,
                "raw": raw,
                "parsed_ok": parsed is not None,
                "repairs": repairs,
                "messages": self._messages_for_trace(),
        }
        if meta:
            record.update(meta)
        self._write_trace(record)
        return raw, parsed, err, reset_info

    def remember_example(self, key: str, title: str, evidence: str, ur: Any, spatial: Any, level: Any, location: Any) -> None:
        rec = {
            "key": key,
            "title": _compact_spaces(title)[:200],
            "evidence": _compact_spaces(evidence)[:240],
            "UR_IsUrbanRenewal": ur,
            "Spatial_IsSpatial": spatial,
            "Spatial_Level": level,
            "Spatial_Location": location,
        }
        self.memory_examples.append(rec)
        if len(self.memory_examples) > max(50, self.memory_max_examples * 5):
            self.memory_examples = self.memory_examples[-max(50, self.memory_max_examples * 5) :]


def extract_one(
    abstract: str,
    primary_cfg: ModelConfig,
    fallback_cfg: Optional[ModelConfig],
    max_tokens: int,
    temperature: float,
    fallback_temperature: float,
    conf_threshold: float,
    timeout_s: int,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    triggers = _extract_trigger_sentences(abstract)
    prompt = _build_prompt(abstract, triggers)
    messages = [{"role": "user", "content": prompt}]
    raw = call_openai_compat_chat(primary_cfg, messages, max_tokens=max_tokens, temperature=temperature, timeout_s=timeout_s)
    parsed = _try_parse_json(raw)
    # 如果JSON解析失败，尝试解析新的文本格式
    if parsed is None:
        parsed = _parse_text_format_response(raw, abstract)
    # 如果文本格式也失败，尝试修复JSON
    if parsed is None:
        parsed = _repair_json_with_model(primary_cfg, prompt, raw, max_tokens=max_tokens, temperature=0, timeout_s=timeout_s)

    if parsed is None:
        return None, primary_cfg.model, "parse_failed"

    ok, err = _post_validate(parsed)
    if not ok:
        return parsed, primary_cfg.model, f"invalid:{err}"

    needs_fallback = False
    if float(parsed.get("confidence", 0)) < conf_threshold:
        needs_fallback = True
    flags = parsed.get("flags") or []
    if any(f in {"multi_city", "ambiguous_top1", "no_explicit_location"} for f in flags):
        needs_fallback = True
    if parsed.get("is_city_specific") and (not parsed.get("primary_area") or not _safe_str(parsed["primary_area"].get("country", ""))):
        needs_fallback = True

    if needs_fallback and fallback_cfg is not None:
        fallback_prompt = _compact_spaces(
            f"""
你是一位学术地理信息分析专家。请基于给定摘要和初步分析结果，进行复核并纠正判断。

**任务：** 复核并纠正初步分析结果，确保准确提取城市、国家、省/州信息和证据句子，以及准确判断是否为城市更新研究。

**复核重点：**
1. 确认是否为城市级研究（核心结论是否服务于一个具体城市）
2. 如果为城市级研究，确保提取完整的城市、国家、省/州信息
3. 确保evidence_sentence是从摘要中直接截取的原句（不要改写）
4. 如果初步结果中缺少国家或省/州信息，请从摘要中仔细查找并补充
5. 复核是否为城市更新研究，并给出合理的置信度评分（参考主提示词中的评分规则）

**输出要求：**
请输出一个合法的JSON对象，包含以下字段：
{{
  "is_city_specific": true/false,
  "primary_area": {{
    "city": "城市名称（如果是城市级研究，否则为null）",
    "admin1": "省/州名称（如果有，否则为空字符串）",
    "country": "国家名称（如果是城市级研究，否则为空字符串）",
    "raw_mention": "摘要中提及城市的原始文本（如果是城市级研究，否则为空字符串）"
  }},
  "secondary_areas": [],
  "evidence_sentence": "从摘要中直接截取的证据句子（必须原封不动）",
  "reason": "判断理由的简要说明",
  "confidence": 0.0-1.0之间的数值,
  "flags": [],
  "is_urban_renewal": true/false,
  "urban_renewal_confidence": 0.0-1.0之间的数值（需严格按照主提示词中的评分规则）
}}

**重要提示：**
- 只输出JSON，不要添加任何解释文字
- evidence_sentence必须是摘要中的原句，不要改写
- 如果无法确定国家或省/州，使用空字符串""，不要猜测
- 请仔细检查并补充初步结果中缺失的信息
- 城市更新判断需基于摘要中是否涉及旧城改造、城市再生、棚户区改造等核心特征

摘要：
{_compact_spaces(abstract)}

初步分析结果（请复核并纠正）：
{json.dumps(parsed, ensure_ascii=False)}
"""
        )
        raw2 = call_openai_compat_chat(
            fallback_cfg,
            [{"role": "user", "content": fallback_prompt}],
            max_tokens=max_tokens,
            temperature=fallback_temperature,
            timeout_s=timeout_s,
        )
        parsed2 = _try_parse_json(raw2)
        # 如果JSON解析失败，尝试解析新的文本格式
        if parsed2 is None:
            parsed2 = _parse_text_format_response(raw2, abstract)
        # 如果文本格式也失败，尝试修复JSON
        if parsed2 is None:
            parsed2 = _repair_json_with_model(fallback_cfg, fallback_prompt, raw2, max_tokens=max_tokens, temperature=0, timeout_s=timeout_s)
        if parsed2 is not None:
            ok2, err2 = _post_validate(parsed2)
            if ok2:
                return parsed2, fallback_cfg.model, ""
            return parsed2, fallback_cfg.model, f"invalid:{err2}"

    return parsed, primary_cfg.model, ""


def _parse_text_format_response(text: str, abstract: str) -> Optional[Dict[str, Any]]:
    """
    解析新格式的文本响应，转换为JSON格式
    
    支持的格式：
    - 中文："这篇论文是针对 **[城市名]** 进行的研究。"
    - 中文："这篇论文**没有针对某个具体的城市**进行研究。"
    - 英文："This paper focuses on the city of **[City Name]**."
    - 英文："This paper does not focus on a specific city."
    """
    if not text:
        return None
    
    text = text.strip()
    
    # 检查中文格式：有具体城市
    chinese_city_pattern = r'这篇论文是针对\s*\*\*?([^*]+)\*\*?\s*进行的研究'
    match = re.search(chinese_city_pattern, text)
    if match:
        city_name = match.group(1).strip()
        # 从摘要中提取证据句子
        evidence = _extract_evidence_sentence(abstract, city_name)
        return {
            "is_city_specific": True,
            "primary_area": {
                "city": city_name,
                "admin1": "",
                "country": "",
                "raw_mention": city_name,
            },
            "secondary_areas": [],
            "evidence_sentence": evidence,
            "reason": f"根据分析，这篇论文针对 {city_name} 进行的研究",
            "confidence": 0.85,
            "flags": [],
        }
    
    # 检查中文格式：无具体城市
    chinese_no_city_pattern = r'这篇论文\*\*?没有针对某个具体的城市\*\*?进行研究'
    if re.search(chinese_no_city_pattern, text):
        evidence = _extract_evidence_sentence(abstract, None)
        return {
            "is_city_specific": False,
            "primary_area": None,
            "secondary_areas": [],
            "evidence_sentence": evidence,
            "reason": "根据分析，这篇论文没有针对某个具体的城市进行研究",
            "confidence": 0.85,
            "flags": [],
        }
    
    # 检查英文格式：有具体城市
    english_city_pattern = r'This paper focuses on the city of\s*\*\*?([^*]+)\*\*?\.'
    match = re.search(english_city_pattern, text, re.IGNORECASE)
    if match:
        city_name = match.group(1).strip()
        evidence = _extract_evidence_sentence(abstract, city_name)
        return {
            "is_city_specific": True,
            "primary_area": {
                "city": city_name,
                "admin1": "",
                "country": "",
                "raw_mention": city_name,
            },
            "secondary_areas": [],
            "evidence_sentence": evidence,
            "reason": f"According to analysis, this paper focuses on the city of {city_name}",
            "confidence": 0.85,
            "flags": [],
        }
    
    # 检查英文格式：无具体城市
    english_no_city_pattern = r'This paper does not focus on a specific city'
    if re.search(english_no_city_pattern, text, re.IGNORECASE):
        evidence = _extract_evidence_sentence(abstract, None)
        return {
            "is_city_specific": False,
            "primary_area": None,
            "secondary_areas": [],
            "evidence_sentence": evidence,
            "reason": "According to analysis, this paper does not focus on a specific city",
            "confidence": 0.85,
            "flags": [],
        }
    
    return None


def _extract_evidence_sentence(abstract: str, city_name: Optional[str]) -> str:
    """从摘要中提取证据句子"""
    if not abstract:
        return ""
    
    sentences = re.split(r'[.!?。！？]\s+', abstract)
    if city_name:
        # 查找包含城市名的句子
        for sent in sentences:
            if city_name.lower() in sent.lower():
                return sent.strip() + "."
        # 如果没有找到，返回第一句
        if sentences:
            return sentences[0].strip() + "."
    else:
        # 如果没有城市，返回第一句作为证据
        if sentences:
            return sentences[0].strip() + "."
    
    return abstract[:100] + "..." if len(abstract) > 100 else abstract


def _try_parse_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    t = text.strip()
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    t = t[start : end + 1]
    try:
        return json.loads(t)
    except Exception:
        return None


def _repair_json_with_model(
    cfg: ModelConfig,
    original_prompt: str,
    bad_output: str,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
) -> Optional[Dict[str, Any]]:
    repair_prompt = _compact_spaces(
        f"""
你上一次输出的 JSON 无法被解析。请基于原任务重新输出一段严格合法 JSON，不要添加任何额外文本。

原任务：
{original_prompt}

错误输出：
{bad_output}
"""
    )
    raw = call_openai_compat_chat(cfg, [{"role": "user", "content": repair_prompt}], max_tokens=max_tokens, temperature=temperature, timeout_s=timeout_s)
    return _try_parse_json(raw)


def _should_retry_http_error(e: Exception) -> bool:
    if hasattr(e, "code"):
        try:
            code = int(getattr(e, "code"))
            return code in {408, 409, 429, 500, 502, 503, 504}
        except Exception:
            return False
    return True


def extract_with_retry(
    abstract: str,
    primary_cfg: ModelConfig,
    fallback_cfg: Optional[ModelConfig],
    max_tokens: int,
    temperature: float,
    fallback_temperature: float,
    conf_threshold: float,
    timeout_s: int,
    max_retries: int,
    base_sleep_s: float,
) -> Tuple[Optional[Dict[str, Any]], str, str]:
    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            return extract_one(
                abstract=abstract,
                primary_cfg=primary_cfg,
                fallback_cfg=fallback_cfg,
                max_tokens=max_tokens,
                temperature=temperature,
                fallback_temperature=fallback_temperature,
                conf_threshold=conf_threshold,
                timeout_s=timeout_s,
            )
        except Exception as e:
            last_err = f"{type(e).__name__}:{str(e)[:200]}"
            if attempt >= max_retries or not _should_retry_http_error(e):
                break
            sleep_s = base_sleep_s * (2 ** attempt) + random.random() * 0.25
            time.sleep(sleep_s)
    return None, primary_cfg.model, last_err


def _flatten_result(res: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not res:
        return {
            "StudyArea_IsCitySpecific": "",
            "StudyArea_PrimaryCity": "",
            "StudyArea_PrimaryAdmin1": "",
            "StudyArea_PrimaryCountry": "",
            "StudyArea_PrimaryRaw": "",
            "StudyArea_Secondary": "",
            "StudyArea_Evidence": "",
            "StudyArea_Reason": "",
            "StudyArea_Confidence": "",
            "StudyArea_Flags": "",
            "StudyArea_IsUrbanRenewal": "",
            "StudyArea_UrbanRenewalConfidence": "",
        }
    primary = res.get("primary_area") or {}
    secondary = res.get("secondary_areas") or []
    secondary_text = "; ".join(
        [
            _compact_spaces(
                ", ".join(
                    [p.get("city", ""), p.get("admin1", ""), p.get("country", "")]
                ).strip(", ")
            )
            for p in secondary
            if isinstance(p, dict)
        ]
    )
    return {
        "StudyArea_IsCitySpecific": bool(res.get("is_city_specific")),
        "StudyArea_PrimaryCity": _safe_str(primary.get("city", "")),
        "StudyArea_PrimaryAdmin1": _safe_str(primary.get("admin1", "")),
        "StudyArea_PrimaryCountry": _safe_str(primary.get("country", "")),
        "StudyArea_PrimaryRaw": _safe_str(primary.get("raw_mention", "")),
        "StudyArea_Secondary": secondary_text,
        "StudyArea_Evidence": _safe_str(res.get("evidence_sentence", "")),
        "StudyArea_Reason": _safe_str(res.get("reason", "")),
        "StudyArea_Confidence": res.get("confidence", ""),
        "StudyArea_Flags": ";".join(res.get("flags") or []),
        "StudyArea_IsUrbanRenewal": bool(res.get("is_urban_renewal", False)),
        "StudyArea_UrbanRenewalConfidence": res.get("urban_renewal_confidence", 0.0),
    }


def _result_column_names() -> List[str]:
    cols = ["StudyArea_Key"]
    cols.extend(list(_flatten_result({}).keys()))
    cols.extend(["StudyArea_Model", "StudyArea_Error", "StudyArea_TimestampMs"])
    return cols


def _create_simplified_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    创建简化的输出格式，包含八个字段：
    1. 文献标题
    2. 文献摘要内容
    3. 研究国家
    4. 研究具体城市
    5. 是否为城市级研究
    6. 判断具体研究区域的摘要原文（截取的摘要内容原文）
    7. 是否为城市更新研究
    8. 城市更新研究置信度
    """
    # 确定标题列名（优先使用 Article Title，如果没有则使用 normalized_title）
    title_col = None
    for col in ["Article Title", "normalized_title", "title"]:
        if col in df.columns:
            title_col = col
            break
    
    # 确定摘要列名
    abstract_col = "Abstract" if "Abstract" in df.columns else None
    
    # 构建结果DataFrame
    result = pd.DataFrame()
    
    # 1. 文献标题
    if title_col:
        result["文献标题"] = df[title_col].fillna("")
    else:
        result["文献标题"] = ""
    
    # 2. 文献摘要内容
    if abstract_col:
        result["文献摘要内容"] = df[abstract_col].fillna("")
    else:
        result["文献摘要内容"] = ""
    
    # 3. 研究国家
    country_list = []
    # 4. 研究具体城市
    city_list = []
    
    for idx, row in df.iterrows():
        is_city_specific = row.get("StudyArea_IsCitySpecific", False)
        if is_city_specific:
            city = _safe_str(row.get("StudyArea_PrimaryCity", ""))
            country = _safe_str(row.get("StudyArea_PrimaryCountry", ""))
            raw_mention = _safe_str(row.get("StudyArea_PrimaryRaw", ""))
            
            # 如果既没有城市也没有原始提及，可能是误判，标记为信息不完整
            has_city_info = bool(city or raw_mention)
            
            # 研究国家
            if country:
                country_list.append(country)
            elif has_city_info:
                # 有城市信息但缺少国家，标记为"国家未识别"
                country_list.append("国家未识别")
            else:
                # 完全没有地理信息，可能是误判
                country_list.append("信息不完整")
            
            # 研究具体城市
            if city:
                city_list.append(city)
            elif raw_mention:
                # 使用原始提及作为城市信息
                city_list.append(raw_mention)
            else:
                # 既没有城市也没有原始提及
                city_list.append("城市未识别")
        else:
            country_list.append("非城市级研究")
            city_list.append("非城市级研究")
    
    result["研究国家"] = country_list
    result["研究具体城市"] = city_list
    
    # 5. 是否为城市级研究
    is_city_series = df.get("StudyArea_IsCitySpecific", pd.Series([False] * len(df), dtype=bool))
    # 确保是布尔类型，然后转换为字符串（修复FutureWarning）
    if is_city_series.dtype == 'object':
        is_city_series = is_city_series.fillna(False)
        is_city_series = is_city_series.infer_objects(copy=False)
    else:
        is_city_series = is_city_series.fillna(False)
    is_city_series = is_city_series.astype(bool)
    result["是否为城市级研究"] = is_city_series.map({True: "是", False: "否"})
    
    # 6. 判断具体研究区域的摘要原文（从摘要中截取的相关原文）
    evidence_list = []
    for idx, row in df.iterrows():
        evidence = _safe_str(row.get("StudyArea_Evidence", ""))
        if evidence:
            evidence_list.append(evidence)
        else:
            # 如果没有证据句子，尝试从摘要中提取包含城市名的句子
            is_city_specific = row.get("StudyArea_IsCitySpecific", False)
            if is_city_specific and abstract_col:
                city = _safe_str(row.get("StudyArea_PrimaryCity", ""))
                if city:
                    abstract_text = _safe_str(row.get(abstract_col, ""))
                    # 查找包含城市名的句子
                    sentences = re.split(r'[.!?。！？]\s+', abstract_text)
                    for sent in sentences:
                        if city.lower() in sent.lower():
                            evidence_list.append(sent.strip() + ".")
                            break
                    else:
                        evidence_list.append("")
                else:
                    evidence_list.append("")
            else:
                evidence_list.append("")
    
    result["判断具体研究区域的摘要原文"] = evidence_list
    
    # 7. 是否为城市更新研究
    is_urban_renewal_series = df.get("StudyArea_IsUrbanRenewal", pd.Series([False] * len(df), dtype=bool))
    # 确保是布尔类型，然后转换为字符串（修复FutureWarning）
    if is_urban_renewal_series.dtype == 'object':
        is_urban_renewal_series = is_urban_renewal_series.fillna(False)
        is_urban_renewal_series = is_urban_renewal_series.infer_objects(copy=False)
    else:
        is_urban_renewal_series = is_urban_renewal_series.fillna(False)
    is_urban_renewal_series = is_urban_renewal_series.astype(bool)
    result["是否为城市更新研究"] = is_urban_renewal_series.map({True: "是", False: "否"})
    
    # 8. 城市更新研究置信度
    urban_renewal_conf_list = []
    for idx, row in df.iterrows():
        conf = row.get("StudyArea_UrbanRenewalConfidence", None)
        if conf is not None:
            try:
                conf_val = float(conf)
                # 格式化为两位小数
                urban_renewal_conf_list.append(f"{conf_val:.2f}")
            except (ValueError, TypeError):
                urban_renewal_conf_list.append("0.00")
        else:
            urban_renewal_conf_list.append("0.00")
    result["城市更新研究置信度"] = urban_renewal_conf_list
    
    # 检查并报告错误（如果有）
    error_col = "StudyArea_Error" if "StudyArea_Error" in df.columns else None
    if error_col:
        errors = df[error_col].fillna("")
        error_count = (errors != "").sum()
        if error_count > 0:
            print(f"\n⚠ 警告：发现 {error_count} 条记录存在错误，详细信息：")
            error_summary = errors[errors != ""].value_counts().head(10)
            for err, count in error_summary.items():
                print(f"  - {err}: {count} 条")
            print("\n提示：如果看到 'parse_failed' 或 'invalid' 错误，可能是API返回格式不正确")
            print("如果看到网络错误（如 'URLError', 'timeout'），请检查网络连接和API密钥")
    
    return result


def run_experiment1_batch(
    input_xlsx: str,
    sheet_name: str,
    output_csv: str,
    output_xlsx: str,
    chat_cfg: ModelConfig,
    reasoner_cfg: Optional[ModelConfig],
    compare_models: bool,
    shot: str,
    max_rows: Optional[int],
    resume: bool,
    title_col: str,
    abstract_col: str,
    keywords_col: str,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    context_char_budget: int,
    trace_jsonl: str,
    trace_level: str,
    trace_tail_messages: int,
    debug_columns: bool,
    dry_run: bool,
) -> None:
    out_csv_dir = os.path.dirname(output_csv)
    out_xlsx_dir = os.path.dirname(output_xlsx)
    if out_csv_dir:
        os.makedirs(out_csv_dir, exist_ok=True)
    if out_xlsx_dir:
        os.makedirs(out_xlsx_dir, exist_ok=True)

    try:
        df = pd.read_excel(input_xlsx, sheet_name=sheet_name, engine="openpyxl")
    except Exception:
        excel_file = pd.ExcelFile(input_xlsx, engine="openpyxl")
        sheets = excel_file.sheet_names
        if not sheets:
            raise RuntimeError("Excel文件中没有可用的工作表")
        df = pd.read_excel(input_xlsx, sheet_name=sheets[0], engine="openpyxl")

    if max_rows is not None and int(max_rows) > 0:
        df = df.head(int(max_rows)).copy()

    keys = [_hash_key(df.loc[i]) for i in df.index]
    df["StudyArea_Key"] = keys

    def ensure_cols(prefix: str) -> None:
        base_cols = [
            f"{prefix}UR_IsUrbanRenewal",
            f"{prefix}Spatial_IsSpatial",
            f"{prefix}Spatial_Level",
            f"{prefix}Spatial_Location",
        ]
        for c in base_cols:
            if c not in df.columns:
                df[c] = ""
        if debug_columns:
            for c in [
                f"{prefix}Exp1_Step1_JSON",
                f"{prefix}Exp1_Step2_JSON",
                f"{prefix}Exp1_Step3_JSON",
                f"{prefix}Exp1_SessionId",
                f"{prefix}Exp1_ContextChars",
                f"{prefix}Exp1_ContextReset",
                f"{prefix}Exp1_ContextReset_Reason",
                f"{prefix}Exp1_Model",
                f"{prefix}Exp1_Shot",
                f"{prefix}Exp1_Error",
            ]:
                if c not in df.columns:
                    df[c] = ""

    if compare_models:
        ensure_cols("chat_")
        ensure_cols("reasoner_")
    else:
        ensure_cols("")

    done_keys: set = set()
    if resume and os.path.exists(output_csv):
        try:
            prev = pd.read_csv(output_csv, encoding="utf-8-sig")
            if "StudyArea_Key" in prev.columns:
                for k in prev["StudyArea_Key"].dropna().astype(str).tolist():
                    if k:
                        done_keys.add(k)
        except Exception:
            done_keys = set()

    sessions: Dict[str, Experiment1Session] = {}
    sessions["chat"] = Experiment1Session(
        cfg=chat_cfg,
        shot=shot,
        context_char_budget=int(context_char_budget),
        trace_jsonl=trace_jsonl,
        trace_level=trace_level,
        trace_tail_messages=int(trace_tail_messages),
        dry_run=bool(dry_run),
    )
    if compare_models:
        if reasoner_cfg is None:
            raise RuntimeError("compare_models=true 需要提供 reasoner_cfg")
        sessions["reasoner"] = Experiment1Session(
            cfg=reasoner_cfg,
            shot=shot,
            context_char_budget=int(context_char_budget),
            trace_jsonl=trace_jsonl,
            trace_level=trace_level,
            trace_tail_messages=int(trace_tail_messages),
            dry_run=bool(dry_run),
        )

    def process_one(sess: Experiment1Session, model_name: str, prefix: str, idx: int, key: str, title: str, abstract: str, keywords: str) -> None:
        step_errs: List[str] = []
        reset_any = 0
        reset_reason = ""

        p1 = _build_experiment1_round_prompt(1, key, title, abstract, keywords)
        raw1, obj1, err1, r1 = sess.ask_json(
            p1,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
            meta={"paper_key": key, "round": 1, "model": model_name, "shot": shot},
        )
        if r1.get("ContextReset"):
            reset_any = 1
            reset_reason = r1.get("ContextReset_Reason", "")
        if err1:
            step_errs.append(f"step1:{err1}")
        ur = ""
        ev1 = ""
        if obj1 is not None:
            try:
                ur = int(obj1.get("UR_IsUrbanRenewal", 0))
            except Exception:
                ur = ""
            ev1 = _safe_str(obj1.get("evidence", ""))

        p2 = _build_experiment1_round_prompt(2, key, title, abstract, keywords)
        raw2, obj2, err2, r2 = sess.ask_json(
            p2,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
            meta={"paper_key": key, "round": 2, "model": model_name, "shot": shot},
        )
        if r2.get("ContextReset"):
            reset_any = 1
            reset_reason = r2.get("ContextReset_Reason", "")
        if err2:
            step_errs.append(f"step2:{err2}")
        spatial = ""
        ev2 = ""
        if obj2 is not None:
            try:
                spatial = int(obj2.get("Spatial_IsSpatial", 0))
            except Exception:
                spatial = ""
            ev2 = _safe_str(obj2.get("evidence", ""))

        p3 = _build_experiment1_round_prompt(3, key, title, abstract, keywords)
        raw3, obj3, err3, r3 = sess.ask_json(
            p3,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
            meta={"paper_key": key, "round": 3, "model": model_name, "shot": shot},
        )
        if r3.get("ContextReset"):
            reset_any = 1
            reset_reason = r3.get("ContextReset_Reason", "")
        if err3:
            step_errs.append(f"step3:{err3}")
        level = ""
        location = ""
        ev3 = ""
        if obj3 is not None:
            level = _safe_str(obj3.get("Spatial_Level", ""))
            location = _safe_str(obj3.get("Spatial_Location", ""))
            ev3 = _safe_str(obj3.get("evidence", ""))

        df.at[idx, f"{prefix}UR_IsUrbanRenewal"] = ur
        df.at[idx, f"{prefix}Spatial_IsSpatial"] = spatial
        df.at[idx, f"{prefix}Spatial_Level"] = level
        df.at[idx, f"{prefix}Spatial_Location"] = location

        if debug_columns:
            df.at[idx, f"{prefix}Exp1_Step1_JSON"] = raw1
            df.at[idx, f"{prefix}Exp1_Step2_JSON"] = raw2
            df.at[idx, f"{prefix}Exp1_Step3_JSON"] = raw3
            df.at[idx, f"{prefix}Exp1_SessionId"] = sess.session_id
            df.at[idx, f"{prefix}Exp1_ContextChars"] = _estimate_messages_chars(sess.messages)
            df.at[idx, f"{prefix}Exp1_ContextReset"] = reset_any
            df.at[idx, f"{prefix}Exp1_ContextReset_Reason"] = reset_reason
            df.at[idx, f"{prefix}Exp1_Model"] = model_name
            df.at[idx, f"{prefix}Exp1_Shot"] = shot
            df.at[idx, f"{prefix}Exp1_Error"] = ";".join(step_errs)

        evidence_for_memory = ev3 or ev2 or ev1
        sess.remember_example(key, title, evidence_for_memory, ur, spatial, level, location)

    total = len(df)
    processed = 0
    for idx, row in tqdm(df.iterrows(), total=total, desc="Experiment1 标注", unit="条"):
        key = _safe_str(row.get("StudyArea_Key", ""))
        if not key:
            continue
        if key in done_keys:
            continue
        title = _safe_str(row.get(title_col, ""))
        abstract = _safe_str(row.get(abstract_col, ""))
        keywords = _safe_str(row.get(keywords_col, ""))
        if not title and not abstract and not keywords:
            continue

        if compare_models:
            process_one(sessions["chat"], "deepseek-chat", "chat_", idx, key, title, abstract, keywords)
            process_one(sessions["reasoner"], "deepseek-reasoner", "reasoner_", idx, key, title, abstract, keywords)
        else:
            process_one(sessions["chat"], sessions["chat"].cfg.model, "", idx, key, title, abstract, keywords)

        processed += 1
        if processed % 10 == 0:
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    df.to_excel(output_xlsx, index=False, engine="openpyxl")


def run_batch(
    input_xlsx: str,
    sheet_name: str,
    output_csv: str,
    output_xlsx: str,
    primary_cfg: ModelConfig,
    fallback_cfg: Optional[ModelConfig],
    max_rows: Optional[int],
    max_workers: int,
    resume: bool,
    conf_threshold: float,
    max_tokens: int,
    temperature: float,
    fallback_temperature: float,
    timeout_s: int,
    max_retries: int,
    base_sleep_s: float,
    dry_run: bool,
    dry_run_requests_jsonl: Optional[str],
) -> None:
    out_csv_dir = os.path.dirname(output_csv)
    out_xlsx_dir = os.path.dirname(output_xlsx)
    if out_csv_dir:
        os.makedirs(out_csv_dir, exist_ok=True)
    if out_xlsx_dir:
        os.makedirs(out_xlsx_dir, exist_ok=True)

    print(f"正在读取输入文件: {input_xlsx}")
    print(f"工作表: {sheet_name}")
    
    try:
        df = pd.read_excel(input_xlsx, sheet_name=sheet_name, engine="openpyxl")
        original_count = len(df)
    except ValueError as e:
        if "Worksheet named" in str(e) and "not found" in str(e):
            print(f"⚠ 警告: {str(e)}")
            
            # 尝试智能选择工作表
            try:
                excel_file = pd.ExcelFile(input_xlsx, engine="openpyxl")
                available_sheets = excel_file.sheet_names
                print(f"可用工作表: {available_sheets}")
                
                # 优先查找包含关键词的工作表
                preferred_keywords = ["清理", "汇总", "数据"]
                selected_sheet = None
                
                for keyword in preferred_keywords:
                    for sheet in available_sheets:
                        if keyword in sheet:
                            selected_sheet = sheet
                            print(f"自动选择匹配工作表: {selected_sheet}")
                            break
                    if selected_sheet:
                        break
                
                # 如果没有找到匹配的，使用第一个工作表
                if not selected_sheet and available_sheets:
                    selected_sheet = available_sheets[0]
                    print(f"自动选择第一个工作表: {selected_sheet}")
                
                if selected_sheet:
                    df = pd.read_excel(input_xlsx, sheet_name=selected_sheet, engine="openpyxl")
                    original_count = len(df)
                    # 更新 sheet_name 以便后续使用
                    sheet_name = selected_sheet
                else:
                    raise ValueError("未找到可用的工作表")
            except Exception as e2:
                print(f"❌ 错误: 无法加载数据 - {str(e2)}")
                raise
        else:
            raise
    
    # 交互式选择处理记录数量（默认行为：如果max_rows为None或-1，则提示用户输入）
    if max_rows is None or max_rows == -1:
        print("\n" + "="*60)
        print("=== 交互式选择处理记录数量 ===")
        print("="*60)
        print(f"数据文件中共有 {original_count} 条记录")
        try:
            user_input = input(f"\n请输入要处理的记录数量（1-{original_count}，直接回车处理全部）: ").strip()
            if user_input:
                max_rows = int(user_input)
                if max_rows <= 0 or max_rows > original_count:
                    print(f"⚠ 无效输入，将处理全部 {original_count} 条记录")
                    max_rows = None
                else:
                    print(f"✓ 将处理前 {max_rows} 条记录")
            else:
                max_rows = None
                print(f"✓ 将处理全部 {original_count} 条记录")
            print("="*60 + "\n")
        except (ValueError, KeyboardInterrupt) as e:
            if isinstance(e, KeyboardInterrupt):
                print("\n\n⚠ 用户取消操作")
                return
            print(f"⚠ 输入无效: {e}，将处理全部记录")
            max_rows = None
    
    if max_rows is not None:
        df = df.head(max_rows).copy()
        print(f"✓ 已限制处理前 {max_rows} 行（原始数据共 {original_count} 行）")
    else:
        print(f"✓ 读取到 {len(df)} 条记录，将处理全部记录")

    if "Abstract" not in df.columns:
        raise RuntimeError("Missing column: Abstract")

    done_map: Dict[str, Dict[str, Any]] = {}
    # 自动恢复：如果 resume=True 且输出文件存在，自动加载已处理记录（永久保存功能）
    # 这样每次启动程序，已经分析过的数据不会被重新分析
    if resume and os.path.exists(output_csv):
        print(f"正在自动加载已处理记录: {output_csv}")
        try:
            prev = pd.read_csv(output_csv, encoding="utf-8-sig")
            keep_cols = [c for c in _result_column_names() if c in prev.columns]
            if "StudyArea_Key" in prev.columns:
                prev2 = prev[keep_cols].copy()
                # 只保留有有效结果的记录（不是空结果）
                for _, r in prev2.iterrows():
                    k = _safe_str(r.get("StudyArea_Key", ""))
                    if k:
                        # 检查是否有有效的分析结果（至少有一个非空字段）
                        has_result = False
                        for col in keep_cols:
                            if col != "StudyArea_Key" and col not in ["StudyArea_Model", "StudyArea_Error", "StudyArea_TimestampMs"]:
                                val = r.get(col)
                                if val is not None and str(val).strip() and str(val) != "nan":
                                    has_result = True
                                    break
                        if has_result:
                            done_map[k] = r.to_dict()
                if done_map:
                    print(f"✓ 已加载 {len(done_map)} 条已处理记录，将自动跳过这些记录（永久保存功能已启用）")
                else:
                    print(f"输出文件存在但没有有效的已处理记录")
        except Exception as e:
            print(f"警告：加载已处理记录时出错: {e}，将重新处理所有记录")

    keys = [_hash_key(df.loc[i]) for i in df.index]
    df["StudyArea_Key"] = keys

    tasks: List[Tuple[int, str, str]] = []
    total_records = len(df)
    print(f"正在准备任务（共 {total_records} 条记录）...")
    with tqdm(total=total_records, desc="准备任务", unit="条") as pbar:
        for i, row in df.iterrows():
            key = row["StudyArea_Key"]
            if key in done_map:
                pbar.update(1)
                continue
            abstract = _safe_str(row.get("Abstract", ""))
            if not abstract.strip():
                pbar.update(1)
                continue
            tasks.append((i, key, abstract))
            pbar.update(1)
    
    print(f"准备完成：需要处理 {len(tasks)} 条任务（跳过 {len(df) - len(tasks)} 条已处理或空记录）")

    if dry_run:
        if not dry_run_requests_jsonl:
            raise RuntimeError("dry_run_requests_jsonl is required when dry_run=true")
        out_dir = os.path.dirname(dry_run_requests_jsonl)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        print(f"正在生成请求文件（dry-run 模式）...")
        with open(dry_run_requests_jsonl, "w", encoding="utf-8") as f:
            for i, key, abstract in tqdm(tasks, desc="生成请求", unit="条"):
                triggers = _extract_trigger_sentences(abstract)
                prompt = _build_prompt(abstract, triggers)
                messages = [{"role": "user", "content": prompt}]
                record = {
                    "StudyArea_Key": key,
                    "primary": {
                        "base_url": primary_cfg.base_url,
                        "api_key_env": primary_cfg.api_key_env,
                        "payload": build_openai_compat_payload(primary_cfg.model, messages, max_tokens=max_tokens, temperature=temperature),
                    },
                }
                if fallback_cfg is not None:
                    record["fallback"] = {
                        "base_url": fallback_cfg.base_url,
                        "api_key_env": fallback_cfg.api_key_env,
                        "payload": build_openai_compat_payload(fallback_cfg.model, messages, max_tokens=max_tokens, temperature=fallback_temperature),
                    }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"请求文件已生成: {dry_run_requests_jsonl}")
        print("正在保存结果文件...")
        empty_cols = {"StudyArea_Key": []}
        empty_cols.update(_flatten_result(None))
        empty_cols["StudyArea_Model"] = []
        empty_cols["StudyArea_Error"] = []
        empty_cols["StudyArea_TimestampMs"] = []
        out_df = pd.DataFrame(empty_cols)
        df2 = df.merge(out_df, on="StudyArea_Key", how="left")
        df2.to_csv(output_csv, index=False, encoding="utf-8-sig")
        try:
            df2.to_excel(output_xlsx, index=False, engine="openpyxl")
            print(f"\nDry-run 模式完成！")
            print(f"  - 请求文件: {dry_run_requests_jsonl}")
            print(f"  - CSV: {output_csv}")
            print(f"  - Excel: {output_xlsx}")
        except PermissionError as e:
            raise RuntimeError(
                f"无法写入 Excel 文件 {output_xlsx}。"
                "请确保该文件没有被 Excel 或其他程序打开，然后重试。"
            ) from e
        return

    # 实时保存相关：使用线程锁确保文件写入的线程安全
    file_lock = threading.Lock()
    new_results_cache: Dict[str, Dict[str, Any]] = {}
    
    # 初始化新结果缓存（从done_map中复制，避免修改原始数据）
    if done_map:
        new_results_cache = {k: v.copy() for k, v in done_map.items()}
    
    def save_results_realtime(result_dict: Dict[str, Dict[str, Any]], source_df: pd.DataFrame) -> None:
        """
        实时保存结果到CSV文件（线程安全）
        每次调用都会将当前所有结果（包括新处理的）保存到文件
        """
        with file_lock:
            try:
                # 合并所有结果
                all_results_list = []
                all_cols = set()
                for result in result_dict.values():
                    all_cols.update(result.keys())
                
                # 确保StudyArea_Key存在
                if "StudyArea_Key" not in all_cols:
                    all_cols.add("StudyArea_Key")
                
                # 构建结果列表
                for key, result in result_dict.items():
                    row = {"StudyArea_Key": key}
                    for col in all_cols:
                        row[col] = result.get(col)
                    all_results_list.append(row)
                
                if not all_results_list:
                    return
                
                # 创建结果DataFrame
                results_df = pd.DataFrame(all_results_list)
                
                # 合并到源数据
                merged_df = source_df.merge(results_df, on="StudyArea_Key", how="left")
                
                # 保存到CSV（实时保存，边处理边保存）
                merged_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            except Exception as e:
                # 保存失败不应该中断处理，只记录警告
                print(f"\n警告：实时保存失败: {e}")

    def worker(payload: Tuple[int, str, str]) -> Tuple[int, str, Dict[str, Any]]:
        idx, key, abstract = payload
        res, used_model, err = extract_with_retry(
            abstract=abstract,
            primary_cfg=primary_cfg,
            fallback_cfg=fallback_cfg,
            max_tokens=max_tokens,
            temperature=temperature,
            fallback_temperature=fallback_temperature,
            conf_threshold=conf_threshold,
            timeout_s=timeout_s,
            max_retries=max_retries,
            base_sleep_s=base_sleep_s,
        )
        flat = _flatten_result(res)
        flat["StudyArea_Key"] = key
        flat["StudyArea_Model"] = used_model
        flat["StudyArea_Error"] = err
        flat["StudyArea_TimestampMs"] = _now_ms()
        return idx, key, flat

    if tasks:
        max_workers = max(1, int(max_workers))
        print(f"开始处理任务（使用 {max_workers} 个工作线程）...")
        print(f"✓ 实时保存功能已启用：每条记录处理完成后立即保存，确保数据不丢失")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(worker, t) for t in tasks]
            with tqdm(total=len(futures), desc="处理任务", unit="条") as pbar:
                for fut in as_completed(futures):
                    try:
                        idx, key, flat = fut.result()
                        # 立即更新缓存
                        new_results_cache[key] = flat
                        # 检查是否有错误
                        error = flat.get("StudyArea_Error", "")
                        if error:
                            pbar.set_postfix({"已保存": len(new_results_cache), "错误": f"{error[:20]}..." if len(error) > 20 else error})
                        else:
                            pbar.set_postfix({"已保存": len(new_results_cache)})
                        # 立即保存到文件（实时保存，边处理边保存）
                        save_results_realtime(new_results_cache, df)
                        pbar.update(1)
                    except Exception as e:
                        pbar.update(1)
                        error_msg = str(e)[:50]
                        pbar.set_postfix({"错误": error_msg})
                        print(f"\n⚠ 处理任务时出错: {error_msg}")
        
        # 处理完成后，将所有新结果添加到out_rows（用于后续的Excel保存）
        out_rows = list(new_results_cache.values())
    else:
        out_rows = []
    
    print("\n正在保存最终结果...")
    # CSV文件已经在实时保存中完成了，这里只需要保存Excel文件
    # 读取最新保存的CSV文件（包含所有结果）
    if os.path.exists(output_csv):
        try:
            df2 = pd.read_csv(output_csv, encoding="utf-8-sig")
            print(f"CSV 文件已实时保存，共 {len(df2)} 条记录")
        except Exception as e:
            print(f"警告：无法读取CSV文件: {e}，将重新生成")
            # 如果读取失败，重新生成
            all_results = {}
            if done_map:
                all_results.update(done_map)
            if out_rows:
                for row in out_rows:
                    key = row.get("StudyArea_Key")
                    if key:
                        all_results[key] = row
            
            if all_results:
                all_cols = set()
                for v in all_results.values():
                    all_cols.update(v.keys())
                result_list = []
                for key, result in all_results.items():
                    row = {"StudyArea_Key": key}
                    for col in all_cols:
                        row[col] = result.get(col)
                    result_list.append(row)
                out_df = pd.DataFrame(result_list)
                df2 = df.merge(out_df, on="StudyArea_Key", how="left")
                df2.to_csv(output_csv, index=False, encoding="utf-8-sig")
            else:
                empty_cols = {"StudyArea_Key": []}
                empty_cols.update(_flatten_result(None))
                empty_cols["StudyArea_Model"] = []
                empty_cols["StudyArea_Error"] = []
                empty_cols["StudyArea_TimestampMs"] = []
                out_df = pd.DataFrame(empty_cols)
                df2 = df.merge(out_df, on="StudyArea_Key", how="left")
                df2.to_csv(output_csv, index=False, encoding="utf-8-sig")
    else:
        # CSV文件不存在，创建新的
        all_results = {}
        if done_map:
            all_results.update(done_map)
        if out_rows:
            for row in out_rows:
                key = row.get("StudyArea_Key")
                if key:
                    all_results[key] = row
        
        if all_results:
            all_cols = set()
            for v in all_results.values():
                all_cols.update(v.keys())
            result_list = []
            for key, result in all_results.items():
                row = {"StudyArea_Key": key}
                for col in all_cols:
                    row[col] = result.get(col)
                result_list.append(row)
            out_df = pd.DataFrame(result_list)
            df2 = df.merge(out_df, on="StudyArea_Key", how="left")
        else:
            empty_cols = {"StudyArea_Key": []}
            empty_cols.update(_flatten_result(None))
            empty_cols["StudyArea_Model"] = []
            empty_cols["StudyArea_Error"] = []
            empty_cols["StudyArea_TimestampMs"] = []
            out_df = pd.DataFrame(empty_cols)
            df2 = df.merge(out_df, on="StudyArea_Key", how="left")
        df2.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"CSV 文件已创建，共 {len(df2)} 条记录")
    # 诊断：检查API调用错误
    print("\n" + "="*60)
    print("正在检查处理结果...")
    print("="*60)
    
    if "StudyArea_Error" in df2.columns:
        errors = df2["StudyArea_Error"].fillna("")
        total_records = len(df2)
        success_count = (errors == "").sum()
        error_count = (errors != "").sum()
        
        print(f"总记录数: {total_records}")
        print(f"成功处理: {success_count} 条 ({success_count/total_records*100:.1f}%)")
        print(f"处理失败: {error_count} 条 ({error_count/total_records*100:.1f}%)")
        
        if error_count > 0:
            print(f"\n错误类型统计：")
            error_summary = errors[errors != ""].value_counts().head(10)
            for err, count in error_summary.items():
                percentage = count / error_count * 100
                print(f"  - {err}: {count} 条 ({percentage:.1f}%)")
            
            # 提供解决建议
            print(f"\n常见错误及解决方法：")
            if "parse_failed" in errors.values:
                print(f"  ❌ parse_failed: API返回格式无法解析")
                print(f"     解决方法：检查API响应格式，可能需要调整提示词")
            if any("URLError" in str(e) or "timeout" in str(e).lower() for e in errors.values):
                print(f"  ❌ 网络错误: 可能是网络连接问题或API超时")
                print(f"     解决方法：检查网络连接，或增加timeout_s参数")
            if any("401" in str(e) or "403" in str(e) or "Forbidden" in str(e) for e in errors.values):
                print(f"  ❌ 认证错误 (HTTP 403/401): API密钥认证失败")
                print(f"     可能的原因：")
                print(f"      1. API密钥无效或已过期")
                print(f"      2. API密钥格式错误（包含多余空格或换行符）")
                print(f"      3. API密钥没有访问该模型的权限")
                print(f"      4. 使用了错误的API端点URL")
                print(f"     解决方法：")
                print(f"      1. 检查环境变量是否正确设置：")
                print(f"         Windows PowerShell: echo $env:DASHSCOPE_API_KEY")
                print(f"         Windows CMD: echo %DASHSCOPE_API_KEY%")
                print(f"      2. 重新设置API密钥（确保没有多余的空格）")
                print(f"      3. 访问API提供商网站验证密钥是否有效")
                print(f"      4. 确认API密钥有访问所使用模型的权限")
            if any("429" in str(e) for e in errors.values):
                print(f"  ❌ 429错误: API调用频率超限")
                print(f"     解决方法：减少并发数（--max-workers）或等待后重试")
        else:
            print(f"✓ 所有记录都成功处理！")
    print("="*60 + "\n")
    
    # 创建简化输出（包含八个字段）
    print("正在生成简化输出格式（八个字段）...")
    simplified_df = _create_simplified_output(df2)
    print(f"简化输出包含以下字段：{', '.join(simplified_df.columns.tolist())}")
    
    try:
        print(f"\n正在保存到 CSV 文件: {output_csv}")
        simplified_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"CSV 文件已保存")
        
        print(f"正在保存到 Excel 文件: {output_xlsx}")
        simplified_df.to_excel(output_xlsx, index=False, engine="openpyxl")
        print(f"Excel 文件已保存")
        print(f"\n处理完成！共处理 {len(simplified_df)} 条记录，结果已保存到：")
        print(f"  - CSV: {output_csv}")
        print(f"  - Excel: {output_xlsx}")
        print(f"\n输出字段说明：")
        print(f"  1. 文献标题：论文的标题")
        print(f"  2. 文献摘要内容：论文的完整摘要")
        print(f"  3. 研究国家：研究所在的国家（如果是城市级研究），否则显示'非城市级研究'")
        print(f"     提示：如果缺少国家信息但确认为城市级研究，显示'国家未识别'；如果信息完全不完整，显示'信息不完整'")
        print(f"  4. 研究具体城市：研究的具体城市名称（如果是城市级研究），否则显示'非城市级研究'")
        print(f"     提示：如果缺少城市名称但确认为城市级研究，显示'城市未识别'")
        print(f"  5. 是否为城市级研究：是/否")
        print(f"  6. 判断具体研究区域的摘要原文：从摘要中截取的、用于判断研究区域的相关原文句子")
        print(f"  7. 是否为城市更新研究：是/否（基于摘要判断该论文是否属于城市更新研究领域）")
        print(f"  8. 城市更新研究置信度：0.00-1.00之间的数值（表示城市更新判断的置信度，0.90+表示非常确定，0.75-0.89表示较确定，0.60-0.74表示部分相关，0.40-0.59表示模糊，0.20-0.39表示低相关，0.00-0.19表示不相关）")
    except PermissionError as e:
        raise RuntimeError(
            f"无法写入 Excel 文件 {output_xlsx}。"
            "请确保该文件没有被 Excel 或其他程序打开，然后重试。"
        ) from e


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    # 脚本位于 scripts/llm_extraction/python/
    # .env 位于 scripts/.env
    # 所以是 script_dir.parent.parent / ".env"
    env_path = script_dir.parent.parent / ".env"
    
    if env_path.exists():
        print(f"正在加载环境变量: {env_path}")
        _load_env_from_file(str(env_path))
    else:
        # 回退逻辑
        project_root = _find_project_root(script_dir)
        env_path = project_root / "scripts" / ".env"
        print(f"尝试加载环境变量: {env_path}")
        _load_env_from_file(str(env_path))
    
    # 重新确定 project_root 用于后续路径解析
    project_root = script_dir.parent.parent.parent
    if not (project_root / "data").exists():
         project_root = _find_project_root(script_dir)

    def _resolve_path(path: str) -> str:
        """将相对路径转换为基于项目根目录的绝对路径"""
        p = Path(path)
        if p.is_absolute():
            return str(p)
        return str(project_root / p)

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    pre_args, _ = pre.parse_known_args()
    config_path = _resolve_path(pre_args.config)
    cfg = _load_json_file(config_path)

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["study_area", "experiment1"], default="study_area")
    ap.add_argument("--config", default=config_path)
    ap.add_argument("--input", default=_resolve_path(cfg.get("input_xlsx", DEFAULT_INPUT_XLSX)))
    ap.add_argument("--sheet", default=cfg.get("sheet_name", DEFAULT_SHEET))
    # 如果配置文件中没有指定输出路径，则基于输入文件名自动生成
    default_output_csv = cfg.get("output_csv")
    default_output_xlsx = cfg.get("output_xlsx")
    
    # 如果配置文件中没有指定，或者用户没有通过命令行覆盖，则基于输入文件自动生成
    if not default_output_csv or default_output_csv == r"data/processed/study_area_extracted.csv":
        # 从输入文件路径提取文件名（不含扩展名）
        input_path = Path(cfg.get("input_xlsx", DEFAULT_INPUT_XLSX))
        input_stem = input_path.stem
        default_output_csv = f"data/processed/{input_stem}/study_area_extracted_{input_stem}.csv"
    
    if not default_output_xlsx or default_output_xlsx == r"data/processed/study_area_extracted.xlsx":
        input_path = Path(cfg.get("input_xlsx", DEFAULT_INPUT_XLSX))
        input_stem = input_path.stem
        default_output_xlsx = f"data/processed/{input_stem}/study_area_extracted_{input_stem}.xlsx"
    
    ap.add_argument("--output-csv", default=_resolve_path(default_output_csv))
    ap.add_argument("--output-xlsx", default=_resolve_path(default_output_xlsx))

    ap.add_argument("--primary-model", default=None)
    ap.add_argument("--primary-base-url", default=None)
    ap.add_argument("--primary-key-env", default=None)

    ap.add_argument("--fallback-model", default=None)
    ap.add_argument("--fallback-base-url", default=None)
    ap.add_argument("--fallback-key-env", default=None)
    # 从 .env 读取 ENABLE_FALLBACK，如果没有则从配置文件读取
    enable_fallback_env = os.environ.get("ENABLE_FALLBACK", "").strip().lower()
    if enable_fallback_env:
        # 如果 ENABLE_FALLBACK=false/0/no/off，则禁用fallback
        disable_fallback_default = enable_fallback_env in ("false", "0", "no", "off")
    else:
        disable_fallback_default = bool(cfg.get("disable_fallback", False))
    ap.add_argument("--disable-fallback", action="store_true", default=disable_fallback_default)

    ap.add_argument("--max-rows", type=int, default=cfg.get("max_rows"), 
                    help="限制处理的记录数量。如果不指定，将进入交互式模式提示用户输入。设置为 -1 也会进入交互式模式")
    ap.add_argument("--max-workers", type=int, default=int(cfg.get("max_workers", 2)))
    # 默认启用自动恢复（永久保存功能）
    resume_config = cfg.get("resume", True)  # 默认启用
    ap.add_argument("--resume", action="store_true", default=resume_config,
                    help="启用自动恢复已处理记录（永久保存功能）。默认已启用，此参数用于在配置中禁用时强制启用")
    ap.add_argument("--no-resume", action="store_true", dest="disable_resume",
                    help="禁用自动恢复已处理记录，强制重新处理所有记录")

    ap.add_argument("--conf-threshold", type=float, default=float(cfg.get("conf_threshold", 0.75)))
    ap.add_argument("--max-tokens", type=int, default=int(cfg.get("max_tokens", 450)))
    ap.add_argument("--temperature", type=float, default=float(cfg.get("temperature", 0.1)))
    ap.add_argument("--fallback-temperature", type=float, default=float(cfg.get("fallback_temperature", 0.1)))
    ap.add_argument("--timeout-s", type=int, default=int(cfg.get("timeout_s", 120)))
    ap.add_argument("--max-retries", type=int, default=int(cfg.get("max_retries", 3)))
    ap.add_argument("--base-sleep-s", type=float, default=float(cfg.get("base_sleep_s", 1.0)))
    ap.add_argument("--dry-run", action="store_true", default=bool(cfg.get("dry_run", False)))
    ap.add_argument("--dry-run-requests-jsonl", default=_resolve_path(cfg.get("dry_run_requests_jsonl", r"data/processed/study_area_requests.jsonl")))

    ap.add_argument("--title-col", default=cfg.get("title_col", "Article Title"))
    ap.add_argument("--abstract-col", default=cfg.get("abstract_col", "Abstract"))
    ap.add_argument("--keywords-col", default=cfg.get("keywords_col", "Author Keywords"))
    ap.add_argument("--shot", choices=["zero", "one", "few"], default=cfg.get("shot", "zero"))
    ap.add_argument("--compare-models", action="store_true", default=bool(cfg.get("compare_models", False)))
    ap.add_argument("--chat-model", default=None)
    ap.add_argument("--chat-base-url", default=None)
    ap.add_argument("--chat-key-env", default=None)
    ap.add_argument("--reasoner-model", default=None)
    ap.add_argument("--reasoner-base-url", default=None)
    ap.add_argument("--reasoner-key-env", default=None)
    ap.add_argument("--context-char-budget", type=int, default=int(cfg.get("context_char_budget", 250000)))
    ap.add_argument("--trace-jsonl", default=_resolve_path(cfg.get("trace_jsonl", r"data/processed/experiment1_trace.jsonl")))
    ap.add_argument("--trace-level", choices=["full", "tail", "none"], default=cfg.get("trace_level", "tail"))
    ap.add_argument("--trace-tail-messages", type=int, default=int(cfg.get("trace_tail_messages", 30)))
    ap.add_argument("--exp-dry-run", action="store_true", default=bool(cfg.get("exp_dry_run", False)))
    ap.add_argument("--exp-debug-columns", action="store_true", default=True)
    ap.add_argument("--exp-no-debug-columns", action="store_true", default=False)

    args = ap.parse_args()
    
    # 确保所有路径参数都被解析为绝对路径
    args.input = _resolve_path(args.input) if args.input else args.input
    # 处理输出路径：如果用户通过命令行指定了--input但没有指定输出，则基于输入文件名自动生成
    if args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = Path(_resolve_path(args.input))
        input_stem = input_path.stem
        
        # 检查输出路径是否是默认路径（如果是，则基于输入文件名生成新路径）
        default_csv = _resolve_path(r"data/processed/study_area_extracted.csv")
        default_xlsx = _resolve_path(r"data/processed/study_area_extracted.xlsx")
        
        # 如果输出路径是默认路径，则替换为基于输入文件名的路径
        if args.output_csv == default_csv or (not args.output_csv) or (args.output_csv.endswith("study_area_extracted.csv") and input_stem not in args.output_csv):
            output_dir = project_root / "data" / "processed" / input_stem
            output_dir.mkdir(parents=True, exist_ok=True)
            args.output_csv = str(output_dir / f"study_area_extracted_{input_stem}.csv")
        
        if args.output_xlsx == default_xlsx or (not args.output_xlsx) or (args.output_xlsx.endswith("study_area_extracted.xlsx") and input_stem not in args.output_xlsx):
            if 'output_dir' not in locals():
                output_dir = project_root / "data" / "processed" / input_stem
                output_dir.mkdir(parents=True, exist_ok=True)
            args.output_xlsx = str(output_dir / f"study_area_extracted_{input_stem}.xlsx")
    
    args.output_csv = _resolve_path(args.output_csv) if args.output_csv else args.output_csv
    args.output_xlsx = _resolve_path(args.output_xlsx) if args.output_xlsx else args.output_xlsx
    args.dry_run_requests_jsonl = _resolve_path(args.dry_run_requests_jsonl) if args.dry_run_requests_jsonl else args.dry_run_requests_jsonl
    args.config = _resolve_path(args.config) if args.config else args.config
    
    # 处理 resume 参数：如果指定了 --no-resume，则禁用自动恢复
    if hasattr(args, 'disable_resume') and args.disable_resume:
        args.resume = False
    # 否则使用配置的默认值（通常是 True，表示默认启用永久保存功能）
    
    # 处理 max_rows 参数：如果为 -1，则交互式选择
    # 注意：这里先不读取完整文件，在 run_batch 中会读取，避免重复读取

    if args.exp_no_debug_columns:
        args.exp_debug_columns = False

    if args.task == "experiment1":
        if args.exp_dry_run:
            chat_model = args.chat_model or _env_get("CHAT_MODEL") or "dry_run"
            chat_base_url = args.chat_base_url or _env_get("CHAT_BASE_URL") or "http://localhost"
        else:
            chat_model = args.chat_model or _env_require(
                "CHAT_MODEL",
                "CHAT_MODEL=deepseek-chat\nCHAT_BASE_URL=https://api.deepseek.com/v1/chat/completions\nCHAT_API_KEY_ENV=DEEPSEEK_API_KEY\nDEEPSEEK_API_KEY=你的key",
            )
            chat_base_url = args.chat_base_url or _env_require(
                "CHAT_BASE_URL",
                "CHAT_MODEL=deepseek-chat\nCHAT_BASE_URL=https://api.deepseek.com/v1/chat/completions\nCHAT_API_KEY_ENV=DEEPSEEK_API_KEY\nDEEPSEEK_API_KEY=你的key",
            )
        chat_key_env = args.chat_key_env or _env_get("CHAT_API_KEY_ENV") or "CHAT_API_KEY"
        chat_key_direct = _env_get("CHAT_API_KEY")
        if chat_key_direct:
            os.environ[chat_key_env] = chat_key_direct

        chat_cfg = ModelConfig(
            provider="deepseek",
            model=chat_model,
            api_key_env=chat_key_env,
            base_url=chat_base_url,
        )
        reasoner_cfg = None
        if args.compare_models:
            if args.exp_dry_run:
                reasoner_model = args.reasoner_model or _env_get("REASONER_MODEL") or "dry_run"
                reasoner_base_url = args.reasoner_base_url or _env_get("REASONER_BASE_URL") or "http://localhost"
            else:
                reasoner_model = args.reasoner_model or _env_require(
                    "REASONER_MODEL",
                    "REASONER_MODEL=deepseek-reasoner\nREASONER_BASE_URL=https://api.deepseek.com/v1/chat/completions\nREASONER_API_KEY_ENV=DEEPSEEK_API_KEY\nDEEPSEEK_API_KEY=你的key",
                )
                reasoner_base_url = args.reasoner_base_url or _env_require(
                    "REASONER_BASE_URL",
                    "REASONER_MODEL=deepseek-reasoner\nREASONER_BASE_URL=https://api.deepseek.com/v1/chat/completions\nREASONER_API_KEY_ENV=DEEPSEEK_API_KEY\nDEEPSEEK_API_KEY=你的key",
                )
            reasoner_key_env = args.reasoner_key_env or _env_get("REASONER_API_KEY_ENV") or "REASONER_API_KEY"
            reasoner_key_direct = _env_get("REASONER_API_KEY")
            if reasoner_key_direct:
                os.environ[reasoner_key_env] = reasoner_key_direct
            reasoner_cfg = ModelConfig(
                provider="deepseek",
                model=reasoner_model,
                api_key_env=reasoner_key_env,
                base_url=reasoner_base_url,
            )

        if not args.exp_dry_run:
            _env_require(chat_cfg.api_key_env, f"{chat_cfg.api_key_env}=你的key")
            if args.compare_models and reasoner_cfg is not None:
                _env_require(reasoner_cfg.api_key_env, f"{reasoner_cfg.api_key_env}=你的key")

        run_experiment1_batch(
            input_xlsx=args.input,
            sheet_name=args.sheet,
            output_csv=args.output_csv,
            output_xlsx=args.output_xlsx,
            chat_cfg=chat_cfg,
            reasoner_cfg=reasoner_cfg,
            compare_models=args.compare_models,
            shot=args.shot,
            max_rows=args.max_rows,
            resume=args.resume,
            title_col=args.title_col,
            abstract_col=args.abstract_col,
            keywords_col=args.keywords_col,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
            context_char_budget=args.context_char_budget,
            trace_jsonl=args.trace_jsonl,
            trace_level=args.trace_level,
            trace_tail_messages=args.trace_tail_messages,
            debug_columns=bool(args.exp_debug_columns),
            dry_run=bool(args.exp_dry_run),
        )
        return

    if args.dry_run:
        primary_model = args.primary_model or _env_get("PRIMARY_MODEL") or "dry_run"
        primary_base_url = args.primary_base_url or _env_get("PRIMARY_BASE_URL") or "http://localhost"
    else:
        primary_model = args.primary_model or _env_require(
            "PRIMARY_MODEL",
            "PRIMARY_MODEL=你的模型名称\nPRIMARY_BASE_URL=你的API端点URL\nPRIMARY_API_KEY_ENV=DEEPSEEK_API_KEY\nDEEPSEEK_API_KEY=你的key",
        )
        primary_base_url = args.primary_base_url or _env_require(
            "PRIMARY_BASE_URL",
            "PRIMARY_MODEL=你的模型名称\nPRIMARY_BASE_URL=你的API端点URL\nPRIMARY_API_KEY_ENV=DEEPSEEK_API_KEY\nDEEPSEEK_API_KEY=你的key",
        )
    primary_key_env = args.primary_key_env or _env_get("PRIMARY_API_KEY_ENV") or "PRIMARY_API_KEY"
    primary_key_direct = _env_get("PRIMARY_API_KEY")
    if primary_key_direct:
        os.environ[primary_key_env] = primary_key_direct
    if not args.dry_run:
        _env_require(primary_key_env, f"{primary_key_env}=你的key")
    primary_cfg = ModelConfig(
        provider=_env_get("PRIMARY_PROVIDER") or "custom",
        model=primary_model,
        api_key_env=primary_key_env,
        base_url=primary_base_url,
    )

    enable_fallback_env = _env_get("ENABLE_FALLBACK").lower()
    enable_fallback = (not args.disable_fallback) and (enable_fallback_env not in {"false", "0", "no", "off"})
    fallback_cfg = None
    if enable_fallback:
        fb_model = args.fallback_model or _env_get("FALLBACK_MODEL")
        fb_url = args.fallback_base_url or _env_get("FALLBACK_BASE_URL")
        if fb_model and fb_url:
            fb_key_env = args.fallback_key_env or _env_get("FALLBACK_API_KEY_ENV") or "FALLBACK_API_KEY"
            fb_key_direct = _env_get("FALLBACK_API_KEY")
            if fb_key_direct:
                os.environ[fb_key_env] = fb_key_direct
            if not args.dry_run:
                _env_require(fb_key_env, f"{fb_key_env}=你的key")
            fallback_cfg = ModelConfig(
                provider=_env_get("FALLBACK_PROVIDER") or "custom",
                model=fb_model,
                api_key_env=fb_key_env,
                base_url=fb_url,
            )

    run_batch(
        input_xlsx=args.input,
        sheet_name=args.sheet,
        output_csv=args.output_csv,
        output_xlsx=args.output_xlsx,
        primary_cfg=primary_cfg,
        fallback_cfg=fallback_cfg,
        max_rows=args.max_rows,
        max_workers=args.max_workers,
        resume=args.resume,
        conf_threshold=args.conf_threshold,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        fallback_temperature=args.fallback_temperature,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
        base_sleep_s=args.base_sleep_s,
        dry_run=args.dry_run,
        dry_run_requests_jsonl=args.dry_run_requests_jsonl,
    )


if __name__ == "__main__":
    main()
