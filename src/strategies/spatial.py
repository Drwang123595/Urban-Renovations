import json
import re
from typing import Dict, Any, Optional, Union
from pathlib import Path
from .base import ExtractionStrategy
from ..prompting.generator import PromptGenerator
from ..runtime.llm_client import DeepSeekClient
from ..runtime.memory import ConversationMemory

class SpatialExtractionStrategy(ExtractionStrategy):
    _FALSEY_SPATIAL_VALUES = {
        "",
        "0",
        "false",
        "no",
        "n",
        "null",
        "none",
        "not mentioned",
        "n/a",
        "na",
        "nan",
    }
    _TRUEY_SPATIAL_VALUES = {"1", "true", "yes", "y"}
    _EMPTY_AREA_VALUES = {
        "",
        "null",
        "none",
        "not mentioned",
        "n/a",
        "na",
        "nan",
    }
    _PLACEHOLDER_AREA_TERMS = (
        "unspecified",
        "unknown",
        "unnamed",
        "not specified",
        "case study context",
    )
    _GENERIC_AREA_PATTERN = re.compile(
        r"^(?:an?\s+|the\s+)?(?:"
        r"city|site|case study|study area|urban area|project area|municipality|"
        r"neighbou?rhood|block"
        r")$",
        re.IGNORECASE,
    )
    _IMPLICIT_GENERIC_TERMS = (
        "city",
        "site",
        "context",
        "municipal",
        "municipality",
        "neighborhood",
        "neighbourhood",
        "project area",
        "case study",
    )

    def __init__(self, client: DeepSeekClient, prompt_gen: PromptGenerator):
        super().__init__(client, prompt_gen)
        self.memory: Optional[ConversationMemory] = None

    def _get_or_create_memory(
        self,
        system_prompt: str,
        session_path: Optional[Union[str, Path]] = None,
        audit_metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationMemory:
        """
        Get existing memory (for long context) or create new one (for isolated/first run).
        """
        if session_path:
            return self._create_memory(system_prompt, session_path, audit_metadata=audit_metadata)

        if self.memory is None:
            self.memory = self._create_memory(system_prompt, audit_metadata=audit_metadata)
        elif audit_metadata:
            self.memory.update_audit_metadata(audit_metadata)

        if self.memory.is_context_full():
            print(f"[INFO] Memory full. Resetting context for SpatialExtractionStrategy.")
            self.memory.save()
            self.memory = self._create_memory(system_prompt, audit_metadata=audit_metadata)

        return self.memory

    def process(
        self,
        title: str,
        abstract: str,
        session_path: Optional[Union[str, Path]] = None,
        audit_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract spatial attributes using the specialized spatial.yaml prompt.
        Returns: Is_Spatial_Research, Spatial_Scale_Level, Specific_Study_Area, Reasoning, Confidence
        """
        system_prompt = self.prompt_gen.get_spatial_system_prompt()
        user_prompt = self.prompt_gen.get_spatial_user_prompt(title, abstract)

        memory = self._get_or_create_memory(system_prompt, session_path, audit_metadata=audit_metadata)
        memory.add_user_message(user_prompt)

        assistant_msg = self.client.chat_completion(memory.get_messages())
        if not assistant_msg:
            self._safe_save(memory, "spatial_empty_response")
            return {}
        memory.add_assistant_message(assistant_msg)
        self._safe_save(memory, "spatial_sample_completed")

        result = self.parse_json_output(assistant_msg)

        result["raw_response"] = assistant_msg
        return result

    def _normalize_spatial_flag(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if value != value:  # NaN guard without adding a dependency.
                return False
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().strip('"').strip("'").lower()
            if normalized in self._TRUEY_SPATIAL_VALUES:
                return True
            if normalized in self._FALSEY_SPATIAL_VALUES:
                return False
        return False

    def _is_placeholder_area(self, value: Any) -> bool:
        if value is None:
            return True

        text = str(value).strip().strip('"').strip("'")
        normalized = re.sub(r"\s+", " ", text.lower()).strip(" .;:")
        if normalized in self._EMPTY_AREA_VALUES:
            return True

        if self._GENERIC_AREA_PATTERN.fullmatch(normalized):
            return True

        if any(term in normalized for term in self._PLACEHOLDER_AREA_TERMS):
            return True

        if "implicit" in normalized and any(
            term in normalized for term in self._IMPLICIT_GENERIC_TERMS
        ):
            return True

        return False

    def _clean_text_field(self, value: Any, default: str = "Not mentioned") -> str:
        if value is None:
            return default
        text = str(value).strip()
        if text.lower().strip(" .;:") in self._EMPTY_AREA_VALUES:
            return default
        return text or default

    def parse_json_output(self, text: str) -> Dict[str, Any]:
        """
        Parse JSON output from the LLM.
        Expected format:
        {
          "Reasoning": "...",
          "Is_Spatial_Research": true / false,
          "Spatial_Scale_Level": "3. National / Single-country Scale" or null,
          "Specific_Study_Area": "Beijing and Shanghai" or null,
          "Confidence": "High / Medium / Low"
        }
        """
        default_result = {
            "空间研究/非空间研究": "0",
            "空间等级": "Not mentioned",
            "具体空间描述": "Not mentioned",
            "Reasoning": "",
            "Confidence": "Low"
        }

        try:
            start = text.find('{')
            if start == -1:
                return default_result
            decoder = json.JSONDecoder()
            data, end = decoder.raw_decode(text[start:])
            json_str = text[start:start + end]
            data = json.loads(json_str)

            is_spatial = self._normalize_spatial_flag(data.get("Is_Spatial_Research", False))
            area = data.get("Specific_Study_Area")

            default_result["Reasoning"] = data.get("Reasoning", "")
            default_result["Confidence"] = data.get("Confidence", "Low")

            if not is_spatial or self._is_placeholder_area(area):
                default_result["空间研究/非空间研究"] = "0"
                default_result["空间等级"] = "Not mentioned"
                default_result["具体空间描述"] = "Not mentioned"
                return default_result

            default_result["空间研究/非空间研究"] = "1"
            default_result["空间等级"] = self._clean_text_field(data.get("Spatial_Scale_Level"))
            default_result["具体空间描述"] = self._clean_text_field(area)

        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            print(f"[WARN] Failed to parse JSON output: {e}. Raw response: {text[:200]}")

        return default_result

    def _safe_save(self, memory: ConversationMemory, scene: str):
        try:
            memory.set_last_event(scene)
            memory.set_error_code("empty_response" if "empty_response" in scene else None)
            memory.save()
        except Exception as error:
            print(f"[WARN] Failed to persist spatial session in {scene}: {error}")
