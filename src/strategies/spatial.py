import json
import re
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path
from .base import ExtractionStrategy
from ..prompting.generator import PromptGenerator
from ..runtime.config import Schema
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
        r"^(?:an?\s+|the\s+)?"
        r"(?:(?:selected|local|urban|brownfield|ecologically sensitive|contentious|"
        r"case study|study)\s+)*"
        r"(?:city|site|case study|study area|urban area|project area|municipality|"
        r"neighbou?rhood|district|block|corridor|development|area)"
        r"(?:\s+(?:under study|in\s+(?:an?\s+|the\s+)?"
        r"(?:city|municipality|site|study area|case study context|urban context)))?$",
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
    _GENERIC_ANCHOR_STOPWORDS = {
        "a",
        "an",
        "the",
        "selected",
        "local",
        "urban",
        "brownfield",
        "ecologically",
        "sensitive",
        "contentious",
        "case",
        "study",
        "city",
        "site",
        "area",
        "municipality",
        "neighborhood",
        "neighbourhood",
        "district",
        "block",
        "corridor",
        "development",
        "project",
    }
    _SCALE_LEVELS = {
        "1": "1. Global Scale",
        "2": "2. Multi-national / Continental Scale",
        "3": "3. National / Single-country Scale",
        "4": "4. Multi-provincial / Sub-national Regional Scale",
        "5": "5. Single-provincial / State Scale",
        "6": "6. Multi-city / Megaregion Scale",
        "7": "7. Single-city / Municipal Scale",
        "8": "8. District / County Scale",
        "9": "9. Micro / Neighborhood / Block Scale",
    }
    _COUNTRY_REGION_ALIASES = {
        "united kingdom": ("united kingdom", "u.k.", "uk", "british", "england", "scotland", "wales"),
        "united states": ("united states", "u.s.", "american", "federal"),
        "china": ("china", "chinese", "prc"),
        "european union": ("european union", "e.u.", "eu", "european commission", "european"),
        "hong kong": ("hong kong", "hksar"),
    }
    _IMPLICIT_POLICY_TERMS = re.compile(
        r"\b(government|ministry|department|agency|authority|commission|policy|"
        r"plan|planning|programme|program|act|law|regulation|national|federal)\b",
        re.IGNORECASE,
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

        result = self.parse_json_output(assistant_msg, title=title, abstract=abstract)

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

        if self._looks_like_generic_boundary(text):
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

    def _default_result(
        self,
        reasoning: str = "",
        confidence: str = "Low",
        validation_status: str = "not_spatial",
        validation_reason: str = "default_non_spatial",
        evidence: str = "",
    ) -> Dict[str, Any]:
        return {
            Schema.IS_SPATIAL: "0",
            Schema.SPATIAL_LEVEL: "Not mentioned",
            Schema.SPATIAL_DESC: "Not mentioned",
            "Reasoning": reasoning,
            "Confidence": confidence,
            Schema.SPATIAL_VALIDATION_STATUS: validation_status,
            Schema.SPATIAL_VALIDATION_REASON: validation_reason,
            Schema.SPATIAL_AREA_EVIDENCE: evidence,
        }

    def _normalize_for_match(self, value: Any) -> str:
        text = "" if value is None else str(value)
        text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip().lower()

    def _strip_implicit_suffix(self, value: str) -> str:
        text = re.sub(r"\([^)]*\bimplicit\b[^)]*\)", "", value, flags=re.IGNORECASE)
        text = re.sub(r"\bimplicit(?:ly)?\b", "", text, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", text).strip(" .;:,")

    def _has_named_anchor(self, value: str) -> bool:
        tokens = re.findall(r"\b[A-Z][A-Za-z-]+\b", value)
        for token in tokens:
            if token.lower() not in self._GENERIC_ANCHOR_STOPWORDS:
                return True
        return False

    def _looks_like_generic_boundary(self, value: str) -> bool:
        text = str(value).strip()
        normalized = self._normalize_for_match(text).strip(" .;:")
        if self._GENERIC_AREA_PATTERN.fullmatch(normalized):
            return True
        boundary_match = re.search(
            r"\b(city|site|case study|study area|urban area|project area|"
            r"municipality|neighbou?rhood|district|block|corridor|development)\b",
            normalized,
            flags=re.IGNORECASE,
        )
        return bool(boundary_match and not self._has_named_anchor(text))

    def _normalize_scale_level(self, value: Any) -> Optional[str]:
        text = self._clean_text_field(value, default="")
        normalized = self._normalize_for_match(text)
        if not normalized:
            return None
        match = re.match(r"^([1-9])(?:\.|\b)", normalized)
        if match:
            return self._SCALE_LEVELS.get(match.group(1))
        for level in self._SCALE_LEVELS.values():
            label = level.split(".", 1)[1].strip().lower()
            if normalized == label or label in normalized:
                return level
        return None

    def _scale_number(self, scale_level: str) -> Optional[int]:
        match = re.match(r"^([1-9])\.", str(scale_level).strip())
        if not match:
            return None
        return int(match.group(1))

    def _is_country_or_region_area(self, area: str) -> bool:
        core = self._normalize_for_match(self._strip_implicit_suffix(area)).strip(" .;:,")
        return core in self._COUNTRY_REGION_ALIASES

    def _area_scale_mismatch(self, area: str, scale_level: str) -> bool:
        scale_number = self._scale_number(scale_level)
        if scale_number is None:
            return True
        normalized_area = self._normalize_for_match(area)
        if "implicit" in normalized_area and scale_number >= 6:
            return True
        if self._is_country_or_region_area(area) and scale_number >= 6:
            return True
        if normalized_area in {"global", "world", "worldwide"} and scale_number != 1:
            return True
        return False

    def _area_fragments(self, area: str) -> list[str]:
        core = self._strip_implicit_suffix(area)
        core = re.sub(r"\b(?:in|within|of)\b", ",", core, flags=re.IGNORECASE)
        fragments = re.split(r"\s*(?:,|;|/|&|\band\b)\s*", core, flags=re.IGNORECASE)
        return [fragment.strip(" .;:,") for fragment in fragments if fragment.strip(" .;:,")]

    def _source_supports_implicit_country(
        self,
        area: str,
        source_text: str,
    ) -> Tuple[bool, str]:
        core = self._normalize_for_match(self._strip_implicit_suffix(area)).strip(" .;:,")
        aliases = self._COUNTRY_REGION_ALIASES.get(core)
        if not aliases:
            return False, ""
        source = self._normalize_for_match(source_text)
        if not self._IMPLICIT_POLICY_TERMS.search(source):
            return False, ""
        for alias in aliases:
            if self._normalize_for_match(alias) in source:
                return True, alias
        return False, ""

    def _source_supports_area(
        self,
        area: str,
        title: str = "",
        abstract: str = "",
    ) -> Tuple[bool, str, str]:
        source_text = f"{title or ''} {abstract or ''}".strip()
        if not source_text:
            return False, "missing_source_text", ""

        source = self._normalize_for_match(source_text)
        core_area = self._strip_implicit_suffix(str(area))
        area_norm = self._normalize_for_match(core_area).strip(" .;:,")
        if area_norm and area_norm in source:
            return True, "explicit_area_evidence", core_area

        fragments = self._area_fragments(core_area)
        if len(fragments) >= 2 and all(self._normalize_for_match(fragment) in source for fragment in fragments):
            return True, "explicit_area_fragment_evidence", "; ".join(fragments)

        if "implicit" in self._normalize_for_match(str(area)):
            ok, evidence = self._source_supports_implicit_country(str(area), source_text)
            if ok:
                return True, "implicit_country_region_evidence", evidence

        return False, "area_not_supported_by_title_or_abstract", core_area

    def parse_json_output(
        self,
        text: str,
        title: str = "",
        abstract: str = "",
    ) -> Dict[str, Any]:
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
        default_result = self._default_result()

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

            if not is_spatial:
                default_result[Schema.SPATIAL_VALIDATION_STATUS] = "not_spatial"
                default_result[Schema.SPATIAL_VALIDATION_REASON] = "model_non_spatial"
                return default_result

            cleaned_area = self._clean_text_field(area)
            if self._is_placeholder_area(cleaned_area):
                default_result[Schema.SPATIAL_VALIDATION_STATUS] = "rejected"
                default_result[Schema.SPATIAL_VALIDATION_REASON] = "placeholder_or_generic_area"
                default_result[Schema.SPATIAL_AREA_EVIDENCE] = cleaned_area
                return default_result

            scale_level = self._normalize_scale_level(data.get("Spatial_Scale_Level"))
            if not scale_level:
                default_result[Schema.SPATIAL_VALIDATION_STATUS] = "rejected"
                default_result[Schema.SPATIAL_VALIDATION_REASON] = "missing_or_invalid_scale"
                default_result[Schema.SPATIAL_AREA_EVIDENCE] = cleaned_area
                return default_result

            if self._area_scale_mismatch(cleaned_area, scale_level):
                default_result[Schema.SPATIAL_VALIDATION_STATUS] = "rejected"
                default_result[Schema.SPATIAL_VALIDATION_REASON] = "scale_area_mismatch"
                default_result[Schema.SPATIAL_AREA_EVIDENCE] = cleaned_area
                return default_result

            supported, validation_reason, evidence = self._source_supports_area(
                cleaned_area,
                title=title,
                abstract=abstract,
            )
            if not supported:
                default_result[Schema.SPATIAL_VALIDATION_STATUS] = "rejected"
                default_result[Schema.SPATIAL_VALIDATION_REASON] = validation_reason
                default_result[Schema.SPATIAL_AREA_EVIDENCE] = evidence or cleaned_area
                return default_result

            default_result[Schema.IS_SPATIAL] = "1"
            default_result[Schema.SPATIAL_LEVEL] = scale_level
            default_result[Schema.SPATIAL_DESC] = cleaned_area
            default_result[Schema.SPATIAL_VALIDATION_STATUS] = "accepted"
            default_result[Schema.SPATIAL_VALIDATION_REASON] = validation_reason
            default_result[Schema.SPATIAL_AREA_EVIDENCE] = evidence

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
