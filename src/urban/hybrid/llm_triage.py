from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ...runtime.config import Config
from ..core.metadata import normalize_phrase
from ..taxonomy.core import COMMON_EXISTING_URBAN_OBJECTS, CORE_RENEWAL_ANCHORS


HARD_NEGATIVE_REASONS = {"math_term_misuse", "rural_nonurban"}
NEAR_THRESHOLD_MARGIN = 0.12
HIGH_CONFIDENCE_POSITIVE_SCORE = 0.75
HIGH_CONFIDENCE_NEGATIVE_SCORE = 0.25
BOUNDARY_BUCKET_TOKENS = (
    "governance",
    "policy",
    "finance",
    "method",
    "brownfield",
    "environment",
    "social",
    "impact",
    "rural",
    "greenfield",
)
HIGH_RISK_TOPIC_LABELS = {"N1", "N2", "N3", "N4", "N5", "N7", "N8", "N9", "N10", "U9", "U10", "U12", "U15"}


@dataclass(frozen=True)
class LlmTriageDecision:
    should_call: bool
    action: str
    reasons: list[str] = field(default_factory=list)


class LlmTriagePolicy:
    """Select rows that benefit from structured LLM adjudication."""

    def __init__(
        self,
        *,
        threshold_margin: float = NEAR_THRESHOLD_MARGIN,
        high_positive_score: float = HIGH_CONFIDENCE_POSITIVE_SCORE,
        high_negative_score: float = HIGH_CONFIDENCE_NEGATIVE_SCORE,
    ):
        self.threshold_margin = float(threshold_margin)
        self.high_positive_score = float(high_positive_score)
        self.high_negative_score = float(high_negative_score)

    def evaluate(self, row: pd.Series) -> LlmTriageDecision:
        route_reason = _text(row.get("metadata_route_reason"))
        if route_reason in HARD_NEGATIVE_REASONS:
            return LlmTriageDecision(False, "protected_hard_negative", [f"hard_negative:{route_reason}"])

        label = _label(row.get("final_label", row.get("urban_flag", "")))
        score = _float(row.get("urban_probability_score"), default=None)
        threshold = _float(row.get("binary_decision_threshold"), default=float(Config.URBAN_BINARY_DECISION_THRESHOLD))
        topic_group = _text(row.get("topic_final_group")).lower()
        rule_group = _text(row.get("topic_rule_group")).lower()
        local_group = _text(row.get("topic_local_group")).lower()
        family = _text(row.get("family_predicted_family")).lower()
        risk_tags = _text(row.get("stage1_risk_tags")).lower()
        boundary_bucket = _text(row.get("boundary_bucket")).lower()
        unknown_path = _text(row.get("unknown_recovery_path")).lower()
        review_flag = _truthy(row.get("review_flag"))
        reasons: list[str] = []

        if score is not None and abs(score - threshold) <= self.threshold_margin:
            reasons.append("near_threshold")
        if label == "1" and topic_group in {"nonurban", "unknown"}:
            reasons.append(f"positive_{topic_group}_topic")
        if label == "0" and _has_positive_text_anchor(row):
            reasons.append("negative_with_renewal_anchor")
        if _cross_group_conflict(rule_group, local_group):
            reasons.append("rule_local_cross_group_conflict")
        if review_flag:
            reasons.append("review_flag")
        if any(token in unknown_path for token in ("pending", "review", "retained_unknown")):
            reasons.append(f"unknown_recovery:{unknown_path}")
        if _has_boundary_risk(row, risk_tags=risk_tags, boundary_bucket=boundary_bucket):
            reasons.append("high_risk_boundary")

        if reasons:
            return LlmTriageDecision(True, "call_llm", list(dict.fromkeys(reasons)))

        if _high_confidence_positive(label, score, topic_group, family, risk_tags, self.high_positive_score):
            return LlmTriageDecision(False, "skip_high_confidence_positive", [])
        if _high_confidence_negative(label, score, row, self.high_negative_score):
            return LlmTriageDecision(False, "skip_high_confidence_negative", [])

        return LlmTriageDecision(False, "skip_not_in_triage_scope", [])


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _float(value: Any, *, default: float | None = 0.0) -> float | None:
    try:
        if value in (None, ""):
            return default
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed


def _label(value: Any) -> str:
    text = _text(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text if text in {"0", "1"} else ""


def _truthy(value: Any) -> bool:
    text = _text(value).lower()
    return text in {"1", "1.0", "true", "yes", "on"}


def _normalized_title_abstract(row: pd.Series) -> str:
    title = _text(row.get("Article Title"))
    abstract = _text(row.get("Abstract"))
    return normalize_phrase(f"{title} {abstract}").replace("-", " ")


def _has_positive_text_anchor(row: pd.Series) -> bool:
    text = _normalized_title_abstract(row)
    renewal_hit = any(normalize_phrase(anchor).replace("-", " ") in text for anchor in CORE_RENEWAL_ANCHORS)
    object_hit = any(normalize_phrase(anchor).replace("-", " ") in text for anchor in COMMON_EXISTING_URBAN_OBJECTS)
    return bool(renewal_hit or object_hit)


def _cross_group_conflict(rule_group: str, local_group: str) -> bool:
    return rule_group in {"urban", "nonurban"} and local_group in {"urban", "nonurban"} and rule_group != local_group


def _has_boundary_risk(row: pd.Series, *, risk_tags: str, boundary_bucket: str) -> bool:
    final_topic = _text(row.get("topic_final"))
    rule_topic = _text(row.get("topic_rule"))
    local_topic = _text(row.get("topic_local_label"))
    if final_topic in HIGH_RISK_TOPIC_LABELS or rule_topic in HIGH_RISK_TOPIC_LABELS or local_topic in HIGH_RISK_TOPIC_LABELS:
        return True
    if risk_tags:
        return True
    return any(token in boundary_bucket for token in BOUNDARY_BUCKET_TOKENS)


def _high_confidence_positive(
    label: str,
    score: float | None,
    topic_group: str,
    family: str,
    risk_tags: str,
    score_floor: float,
) -> bool:
    return bool(
        label == "1"
        and score is not None
        and score >= score_floor
        and topic_group == "urban"
        and family == "urban"
        and not risk_tags
    )


def _high_confidence_negative(label: str, score: float | None, row: pd.Series, score_ceiling: float) -> bool:
    return bool(
        label == "0"
        and score is not None
        and score <= score_ceiling
        and not _has_positive_text_anchor(row)
    )
