import json
import html
import re
import unicodedata
from typing import Dict, Any, Optional, Union, Tuple
from pathlib import Path
from ..strategies.base import ExtractionStrategy
from .geo_resolver import GeoResolver
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
    }
    _IMPLICIT_POLICY_TERMS = re.compile(
        r"\b(government|ministry|department|agency|authority|commission|policy|"
        r"plan|planning|programme|program|act|law|regulation|national|federal)\b",
        re.IGNORECASE,
    )
    _AREA_NEAR_MATCH_STOPWORDS = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "in",
        "on",
        "at",
        "to",
        "from",
        "for",
        "with",
        "within",
        "area",
        "areas",
        "city",
        "cities",
        "region",
        "regions",
        "district",
        "town",
        "towns",
        "community",
        "communities",
        "study",
        "case",
        "project",
        "program",
        "programme",
        "site",
        "sites",
        "urban",
        "municipal",
        "municipality",
        "metropolitan",
        "neighborhood",
        "neighbourhood",
    }
    _AREA_EXTRACTION_MODES = {
        "",
        "named_place",
        "aggregate_places",
        "implicit_country_region",
        "no_identifiable_area",
    }

    def __init__(self, client: DeepSeekClient, prompt_gen: PromptGenerator):
        super().__init__(client, prompt_gen)
        self.memory: Optional[ConversationMemory] = None
        self.geo_resolver = GeoResolver()

    def _get_geo_resolver(self) -> GeoResolver:
        resolver = getattr(self, "geo_resolver", None)
        if resolver is None:
            resolver = GeoResolver()
            self.geo_resolver = resolver
        return resolver

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
            Schema.LLM_SPATIAL_SCALE_LEVEL_RAW: "",
            Schema.RESOLVED_STUDY_AREA: "",
            Schema.RESOLVED_GEO_ID: "",
            Schema.AREA_HIERARCHY_PATH: "",
            Schema.MAPPED_SPATIAL_SCALE_LEVEL: "",
            Schema.SCALE_DECISION_SOURCE: "",
            Schema.GEO_RESOLUTION_STATUS: "not_applicable",
            Schema.GEO_RESOLUTION_REASON: "",
            Schema.GEO_RESOLUTION_CONFIDENCE: "",
            Schema.GEO_SOURCE: "",
        }

    def _normalize_for_match(self, value: Any) -> str:
        text = "" if value is None else str(value)
        text = self._decode_html_entities(text)
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
        text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip().lower()

    def _decode_html_entities(self, value: Any) -> str:
        text = "" if value is None else str(value)
        text = re.sub(r"&\s*#\s*(\d+)\s*;", r"&#\1;", text)
        text = re.sub(r"&\s*#x\s*([0-9A-Fa-f]+)\s*;", r"&#x\1;", text)
        text = re.sub(r"&\s*([A-Za-z][A-Za-z0-9]+)\s*;", r"&\1;", text)
        return html.unescape(text)

    def _compact_for_match(self, value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "", self._normalize_for_match(value))

    def _source_contains_phrase(self, source: str, phrase: str) -> bool:
        normalized_phrase = self._normalize_for_match(phrase).strip(" .;:,")
        if not normalized_phrase:
            return False
        if normalized_phrase in source:
            return True
        compact_phrase = self._compact_for_match(phrase)
        if len(compact_phrase) < 4:
            return False
        return compact_phrase in self._compact_for_match(source)

    def _area_match_tokens(self, value: Any) -> set[str]:
        normalized = self._normalize_for_match(value)
        return {
            token
            for token in re.findall(r"[a-z0-9]+", normalized)
            if len(token) > 2 and token not in self._AREA_NEAR_MATCH_STOPWORDS
        }

    def _source_supports_area_near_match(self, area: str, source_text: str) -> Tuple[bool, str]:
        area_tokens = self._area_match_tokens(area)
        if len(area_tokens) < 3:
            return False, ""
        source_tokens = self._area_match_tokens(source_text)
        overlap = area_tokens & source_tokens
        if len(overlap) / len(area_tokens) >= 0.8:
            return True, " ".join(sorted(overlap))
        return False, ""

    def _source_supports_structured_area(self, area: str, source_text: str) -> Tuple[bool, str]:
        normalized_area = self._normalize_for_match(area)
        patterns = [
            r"\b(.+?)\s+district\s+of\s+(.+)\b",
            r"\b(.+?)\s+towns?\s+(?:on|in|of)\s+(.+)\b",
            r"\b(.+?)\s+communities\s+in\s+(.+)\b",
            r"\b(.+?)\s+project\s+in\s+(.+)\b",
        ]
        source_tokens = self._area_match_tokens(source_text)
        for pattern in patterns:
            match = re.search(pattern, normalized_area, flags=re.IGNORECASE)
            if not match:
                continue
            anchor_tokens = self._area_match_tokens(" ".join(match.groups()))
            if anchor_tokens and anchor_tokens.issubset(source_tokens):
                return True, " ".join(sorted(anchor_tokens))
        return False, ""

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
        core = re.sub(r"\([^)]*\)", ",", core)
        core = re.sub(r"\b(?:in|within|of)\b", ",", core, flags=re.IGNORECASE)
        fragments = re.split(r"\s*(?:,|;|/|&|\band\b)\s*", core, flags=re.IGNORECASE)
        return [fragment.strip(" .;:,") for fragment in fragments if fragment.strip(" .;:,")]

    def _area_primary_anchor(self, area: str) -> str:
        core = self._strip_implicit_suffix(area)
        core = re.sub(r"\([^)]*\)", "", core).strip(" .;:,")
        core = re.sub(r"'s\b", "", core, flags=re.IGNORECASE)
        core = re.sub(r"\bmetropolitan area\b", "", core, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", core).strip(" .;:,")

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
        evidence_text: str = "",
    ) -> Tuple[bool, str, str]:
        source_text = f"{title or ''} {abstract or ''}".strip()
        if not source_text:
            return False, "missing_source_text", ""

        source = self._normalize_for_match(source_text)
        clean_evidence_text = self._clean_text_field(evidence_text, default="")
        if clean_evidence_text and self._source_contains_phrase(source, clean_evidence_text):
            return True, "explicit_area_evidence_text", clean_evidence_text

        core_area = self._strip_implicit_suffix(str(area))
        area_norm = self._normalize_for_match(core_area).strip(" .;:,")
        if area_norm and self._source_contains_phrase(source, area_norm):
            return True, "explicit_area_evidence", core_area

        if "(" in core_area and ")" in core_area:
            near_ok, near_evidence = self._source_supports_area_near_match(core_area, source_text)
            if near_ok:
                return True, "explicit_area_near_match_evidence", near_evidence or core_area

        primary_anchor = self._area_primary_anchor(core_area)
        primary_norm = self._normalize_for_match(primary_anchor).strip(" .;:,")
        if primary_norm and self._source_contains_phrase(source, primary_norm):
            return True, "explicit_area_primary_anchor_evidence", primary_anchor

        fragments = self._area_fragments(core_area)
        if len(fragments) >= 2 and all(self._source_contains_phrase(source, fragment) for fragment in fragments):
            return True, "explicit_area_fragment_evidence", "; ".join(fragments)

        if "implicit" in self._normalize_for_match(str(area)):
            ok, evidence = self._source_supports_implicit_country(str(area), source_text)
            if ok:
                return True, "implicit_country_region_evidence", evidence

        structured_ok, structured_evidence = self._source_supports_structured_area(core_area, source_text)
        if structured_ok:
            return True, "explicit_area_near_match_evidence", structured_evidence or core_area

        near_ok, near_evidence = self._source_supports_area_near_match(core_area, source_text)
        if near_ok:
            return True, "explicit_area_near_match_evidence", near_evidence or core_area

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
            canonical_area = self._clean_text_field(data.get("Canonical_Study_Area"), default="")
            evidence_text = self._clean_text_field(data.get("Area_Evidence_Text"), default="")
            extraction_mode = self._clean_text_field(data.get("Area_Extraction_Mode"), default="")
            extraction_mode = self._normalize_for_match(extraction_mode).replace(" ", "_")
            raw_scale_level = data.get("Spatial_Scale_Level")
            scale_level = self._normalize_scale_level(raw_scale_level)

            default_result["Reasoning"] = data.get("Reasoning", "")
            default_result["Confidence"] = data.get("Confidence", "Low")
            default_result[Schema.LLM_SPATIAL_SCALE_LEVEL_RAW] = self._clean_text_field(
                raw_scale_level,
                default="",
            )

            if not is_spatial:
                default_result[Schema.SPATIAL_VALIDATION_STATUS] = "not_spatial"
                default_result[Schema.SPATIAL_VALIDATION_REASON] = "model_non_spatial"
                return default_result

            if extraction_mode and extraction_mode not in self._AREA_EXTRACTION_MODES:
                extraction_mode = ""
            if extraction_mode == "no_identifiable_area":
                default_result[Schema.SPATIAL_VALIDATION_STATUS] = "rejected"
                default_result[Schema.SPATIAL_VALIDATION_REASON] = "no_identifiable_area_mode"
                return default_result

            cleaned_area = self._clean_text_field(area)
            if self._is_placeholder_area(cleaned_area):
                default_result[Schema.SPATIAL_VALIDATION_STATUS] = "rejected"
                default_result[Schema.SPATIAL_VALIDATION_REASON] = "placeholder_or_generic_area"
                default_result[Schema.SPATIAL_AREA_EVIDENCE] = cleaned_area
                return default_result

            supported, validation_reason, evidence = self._source_supports_area(
                cleaned_area,
                title=title,
                abstract=abstract,
                evidence_text=evidence_text,
            )
            if not supported:
                default_result[Schema.SPATIAL_VALIDATION_STATUS] = "rejected"
                default_result[Schema.SPATIAL_VALIDATION_REASON] = validation_reason
                default_result[Schema.SPATIAL_AREA_EVIDENCE] = evidence or cleaned_area
                return default_result

            resolution_area = canonical_area or cleaned_area
            geo_resolution = self._get_geo_resolver().resolve(
                resolution_area,
                llm_scale_level=scale_level,
            )
            default_result.update(geo_resolution.as_dict())
            mapped_scale_level = geo_resolution.mapped_spatial_scale_level
            if not mapped_scale_level:
                default_result[Schema.SPATIAL_VALIDATION_STATUS] = "rejected"
                default_result[Schema.SPATIAL_VALIDATION_REASON] = "missing_or_invalid_scale"
                default_result[Schema.SPATIAL_AREA_EVIDENCE] = evidence or cleaned_area
                return default_result

            if (
                geo_resolution.scale_decision_source == "llm_fallback_unresolved"
                and self._area_scale_mismatch(cleaned_area, mapped_scale_level)
            ):
                default_result[Schema.SPATIAL_VALIDATION_STATUS] = "rejected"
                default_result[Schema.SPATIAL_VALIDATION_REASON] = "scale_area_mismatch"
                default_result[Schema.SPATIAL_AREA_EVIDENCE] = evidence or cleaned_area
                return default_result

            default_result[Schema.IS_SPATIAL] = "1"
            default_result[Schema.SPATIAL_LEVEL] = mapped_scale_level
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
