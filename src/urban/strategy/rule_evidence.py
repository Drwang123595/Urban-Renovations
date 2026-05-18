"""Rule-evidence extraction for the stable urban-renewal strategy."""

from __future__ import annotations

from typing import Iterable

from ..rules.metadata_filter import (
    RISK_BACKGROUND_SUPPORT,
    RISK_EXPLICIT_RENEWAL_OTHER_OBJECT,
    RISK_GENERIC_TECHNICAL,
    RISK_GREENFIELD_EXPANSION,
    RISK_SOCIAL_HISTORY_MEDIA,
)
from ..taxonomy.core import (
    COMMON_EXISTING_URBAN_OBJECTS,
    COMMON_METHOD_ANCHORS,
    COMMON_RENEWAL_ANCHORS,
    COMMON_RURAL_ANCHORS,
    CORE_RENEWAL_ANCHORS,
    UNKNOWN_TOPIC_LABEL,
)
from ..urban_metadata import normalize_phrase
from .evidence import ArticleEvidenceInput, RuleEvidence


HARD_EXCLUSION_REASONS = {"math_term_misuse", "rural_nonurban"}
POLICY_PROJECT_TERMS = (
    "policy",
    "program",
    "programme",
    "project",
    "strategy",
    "intervention",
    "initiative",
    "scheme",
    "plan",
    "planning",
    "implementation",
    "governance",
    "participation",
)
EXISTING_OBJECT_EXTENSIONS = (
    "existing district",
    "existing districts",
    "existing neighborhood",
    "existing neighbourhood",
    "existing community",
    "existing communities",
    "existing building",
    "existing buildings",
    "old building",
    "old buildings",
    "older building",
    "older buildings",
    "building stock",
    "historic urban fabric",
    "urban fabric",
    "built environment",
    "aging district",
    "ageing district",
    "older district",
    "older neighborhood",
    "older neighbourhood",
    "industrial district",
    "old industrial district",
    "inner-city district",
)
RISK_TAGS = {
    RISK_BACKGROUND_SUPPORT,
    RISK_EXPLICIT_RENEWAL_OTHER_OBJECT,
    RISK_GENERIC_TECHNICAL,
    RISK_GREENFIELD_EXPANSION,
    RISK_SOCIAL_HISTORY_MEDIA,
}


class RuleEvidenceExtractor:
    def extract_from_article(
        self,
        article: ArticleEvidenceInput,
        *,
        route_reason: str = "",
        risk_tags: Iterable[str] = (),
        rule_topic_candidate: str = UNKNOWN_TOPIC_LABEL,
        rule_confidence: float = 0.0,
        explanation: str = "",
    ) -> RuleEvidence:
        text = article.normalized_text
        renewal_hits = _hits(text, CORE_RENEWAL_ANCHORS + COMMON_RENEWAL_ANCHORS)
        existing_hits = _hits(text, COMMON_EXISTING_URBAN_OBJECTS + EXISTING_OBJECT_EXTENSIONS)
        policy_hits = _hits(text, POLICY_PROJECT_TERMS)
        risks = set(_normalize_tokens(risk_tags))
        if _hits(text, COMMON_RURAL_ANCHORS):
            risks.add("rural_risk")
        method_hits = _hits(text, COMMON_METHOD_ANCHORS)
        if method_hits and not (renewal_hits and existing_hits):
            risks.add("method_only_risk")
        if "greenfield" in text or "new town" in text or "urban expansion" in text:
            risks.add("greenfield_risk")

        hard_exclusion = route_reason if route_reason in HARD_EXCLUSION_REASONS else ""
        if "rural_risk" in risks and any(term in text for term in ("rural regeneration", "rural renewal")):
            hard_exclusion = hard_exclusion or "rural_nonurban"

        return RuleEvidence(
            renewal_action_hits=tuple(renewal_hits),
            existing_urban_object_hits=tuple(existing_hits),
            policy_project_hits=tuple(policy_hits),
            risk_hits=tuple(sorted(risks)),
            hard_exclusion_reason=hard_exclusion,
            rule_topic_candidate=rule_topic_candidate or UNKNOWN_TOPIC_LABEL,
            rule_confidence=float(rule_confidence or 0.0),
            explanation=explanation or route_reason,
        )


def _hits(text: str, phrases: Iterable[str]) -> list[str]:
    seen: dict[str, None] = {}
    normalized_text = normalize_phrase(text).replace("-", " ")
    for phrase in phrases:
        normalized = normalize_phrase(str(phrase or "")).replace("-", " ")
        if normalized and normalized in normalized_text:
            seen.setdefault(normalized, None)
    return list(seen.keys())


def _normalize_tokens(value: Iterable[str]) -> list[str]:
    tokens: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        if text in RISK_TAGS:
            tokens.append(text)
        else:
            tokens.append(text)
    return tokens
