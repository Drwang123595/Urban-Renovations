"""Structured LLM semantic evidence for boundary urban-renewal cases."""

from __future__ import annotations

import json
import re
from typing import Any

from .evidence import ArticleEvidenceInput, LLMSemanticEvidence, RuleEvidence, TopicEvidence


class LLMSemanticAnalyzer:
    """Ask the LLM to explain semantic boundary cases as structured evidence."""

    def __init__(self, llm_strategy: Any = None, *, enabled: bool = True, confidence_floor: float = 0.75):
        self.llm_strategy = llm_strategy
        self.enabled = bool(enabled)
        self.confidence_floor = float(confidence_floor)

    def should_call(self, *, rule: RuleEvidence, topic: TopicEvidence) -> bool:
        if not self.enabled or self.llm_strategy is None:
            return False
        if rule.hard_exclusion_reason:
            return False
        if topic.topic_group == "unknown":
            return True
        if topic.conflict_flag:
            return True
        if topic.topic_group == "nonurban" and rule.renewal_action_hits:
            return True
        if rule.policy_project_hits and not rule.existing_urban_object_hits:
            return True
        if rule.risk_hits and rule.renewal_action_hits:
            return True
        return False

    def analyze(
        self,
        article: ArticleEvidenceInput,
        *,
        rule: RuleEvidence,
        topic: TopicEvidence,
        session_path: Any = None,
        audit_metadata: dict[str, Any] | None = None,
    ) -> LLMSemanticEvidence:
        if not self.should_call(rule=rule, topic=topic):
            return LLMSemanticEvidence(attempted=False, used=False)

        auxiliary_context = {
            "task": "urban_renewal_semantic_evidence",
            "rule_topic": rule.rule_topic_candidate,
            "topic_candidate": topic.topic_candidate,
            "topic_group": topic.topic_group,
            "renewal_action_hits": list(rule.renewal_action_hits),
            "existing_urban_object_hits": list(rule.existing_urban_object_hits),
            "policy_project_hits": list(rule.policy_project_hits),
            "risk_hits": list(rule.risk_hits),
            "instruction": (
                "Return JSON with research_object, object_is_existing_urban, existing_urban_object, "
                "renewal_action_present, renewal_action, action_is_main_subject, "
                "policy_or_governance_context, is_background_only, exclusion_risk, suggested_topic, "
                "label_hint, confidence, reason."
            ),
        }
        try:
            result = self._process_with_supported_signature(
                article,
                auxiliary_context=auxiliary_context,
                session_path=session_path,
                audit_metadata=audit_metadata,
            )
        except Exception as exc:
            return LLMSemanticEvidence(attempted=True, used=False, reason=f"exception:{type(exc).__name__}")

        payload = result if isinstance(result, dict) else _parse_json_payload(str(result or ""))
        if not isinstance(payload, dict):
            return LLMSemanticEvidence(attempted=True, used=False, reason="parse_failure")

        confidence = _safe_float(payload.get("confidence"), 0.0)
        label_hint = _normalize_label(payload.get("label_hint", payload.get("label", "")))
        return LLMSemanticEvidence(
            attempted=True,
            used=bool(label_hint in {"0", "1"} and confidence >= self.confidence_floor),
            research_object=_text(payload.get("research_object", "")),
            object_is_existing_urban=_safe_bool(payload.get("object_is_existing_urban")),
            existing_urban_object=_text(payload.get("existing_urban_object", "")),
            renewal_action_present=_safe_bool(payload.get("renewal_action_present")),
            renewal_action=_text(payload.get("renewal_action", "")),
            action_is_main_subject=_safe_bool(payload.get("action_is_main_subject")),
            policy_or_governance_context=_safe_bool(payload.get("policy_or_governance_context")),
            is_background_only=_safe_bool(payload.get("is_background_only")),
            exclusion_risk=_text(payload.get("exclusion_risk", "")),
            suggested_topic=_text(payload.get("suggested_topic", "")),
            label_hint=label_hint,
            confidence=confidence,
            reason=_text(payload.get("reason", "")),
        )

    def _process_with_supported_signature(
        self,
        article: ArticleEvidenceInput,
        *,
        auxiliary_context: dict[str, Any],
        session_path: Any = None,
        audit_metadata: dict[str, Any] | None = None,
    ) -> Any:
        try:
            return self.llm_strategy.process(
                article.title,
                article.abstract,
                session_path=session_path,
                metadata={
                    "author_keywords": article.author_keywords,
                    "keywords_plus": article.keywords_plus,
                    "keywords": article.keywords,
                    "wos_categories": article.wos_categories,
                    "research_areas": article.research_areas,
                },
                auxiliary_context=auxiliary_context,
                audit_metadata=audit_metadata,
            )
        except TypeError:
            return self.llm_strategy.process(
                article.title,
                _legacy_semantic_prompt(article, auxiliary_context),
                session_path=session_path,
            )


def _parse_json_payload(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _normalize_label(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text if text in {"0", "1"} else ""


def _safe_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _legacy_semantic_prompt(article: ArticleEvidenceInput, auxiliary_context: dict[str, Any]) -> str:
    return (
        f"{article.abstract}\n\n"
        "Semantic evidence task: decide whether the article's main research object is an existing "
        "urban space/community/building and whether a renewal/regeneration/redevelopment action is "
        "the main subject. Return JSON with label_hint, confidence, object_is_existing_urban, "
        "renewal_action_present, action_is_main_subject, is_background_only, exclusion_risk, "
        "suggested_topic, and reason.\n"
        f"Rule evidence: {json.dumps(auxiliary_context, ensure_ascii=True)}"
    )


def _safe_float(value: Any, default: float) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return min(max(float(value), 0.0), 1.0)
    except Exception:
        return float(default)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
