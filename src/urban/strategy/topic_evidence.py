"""Topic and family evidence adapters for stable urban-renewal decisions."""

from __future__ import annotations

from typing import Any, Mapping

from ..taxonomy.core import UNKNOWN_TOPIC_GROUP, UNKNOWN_TOPIC_LABEL, topic_group_for_label
from .evidence import ClusterEvidence, FamilyConsistencyEvidence, TopicEvidence


def build_topic_evidence_from_row(row: Mapping[str, Any]) -> TopicEvidence:
    topic = _text(row.get("topic_final", row.get("topic_label", UNKNOWN_TOPIC_LABEL))) or UNKNOWN_TOPIC_LABEL
    group = _text(row.get("topic_final_group", row.get("topic_group", ""))).lower()
    if not group:
        group = topic_group_for_label(topic)
    confidence = _safe_float(row.get("topic_confidence_effective", row.get("topic_confidence", row.get("confidence", 0.0))))
    margin = _safe_float(row.get("topic_margin_effective", row.get("topic_margin", 0.0)))
    top3 = tuple(part.strip() for part in _text(row.get("topic_local_top3", row.get("topic_rule_top3", ""))).split(";") if part.strip())
    conflict = _safe_int(row.get("binary_topic_consistency_flag", row.get("stage1_conflict_flag", 0)))
    return TopicEvidence(
        topic_candidate=topic,
        topic_group=group or UNKNOWN_TOPIC_GROUP,
        confidence=confidence,
        margin=margin,
        top3=top3,
        evidence=_text(row.get("topic_matches", row.get("topic_rule_matches", ""))),
        conflict_flag=conflict,
    )


def build_family_evidence_from_row(row: Mapping[str, Any]) -> FamilyConsistencyEvidence:
    rule_group = _text(row.get("topic_family_rule", row.get("topic_rule_group", ""))).lower() or UNKNOWN_TOPIC_GROUP
    model_group = _text(row.get("topic_family_local", row.get("topic_local_group", ""))).lower() or UNKNOWN_TOPIC_GROUP
    if rule_group == model_group and rule_group != UNKNOWN_TOPIC_GROUP:
        status = "consistent"
    elif rule_group == UNKNOWN_TOPIC_GROUP or model_group == UNKNOWN_TOPIC_GROUP:
        status = "unknown"
    else:
        status = "conflict"
    return FamilyConsistencyEvidence(
        rule_group=rule_group,
        model_group=model_group,
        consistency_status=_text(row.get("family_decision_source", "")) or status,
        conflict_pattern=_text(row.get("family_conflict_pattern", "")),
        boundary_bucket=_text(row.get("boundary_bucket", "")),
        family_probability_urban=_safe_float(row.get("family_probability_urban", 0.0)),
    )


def build_cluster_evidence_from_row(row: Mapping[str, Any]) -> ClusterEvidence:
    return ClusterEvidence(
        cluster_id=_text(row.get("bertopic_topic_id", row.get("bertopic_dynamic_topic_id", ""))),
        cluster_label_hint=_text(row.get("bertopic_mapped_label", row.get("bertopic_hint_label", ""))),
        cluster_positive_rate=_safe_float(row.get("bertopic_pos_rate", 0.0)),
        cluster_topic_words=_text(row.get("bertopic_top_terms", row.get("bertopic_dynamic_topic_words", ""))),
        support=_text(row.get("bertopic_primary_reason", "")),
        conflict=_text(row.get("bertopic_hint_conflict_flag", "")),
    )


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)
