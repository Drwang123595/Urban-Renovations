"""Result construction helpers for the urban hybrid classifier."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ...runtime.config import Schema
from ..taxonomy.core import (
    OPEN_SET_NONURBAN_LABEL,
    OPEN_SET_URBAN_LABEL,
    UNKNOWN_TOPIC_GROUP,
    UNKNOWN_TOPIC_LABEL,
    legacy_topic_for_label,
    topic_group_for_label,
    topic_name_for_label,
    urban_flag_for_topic_label,
)
from .binary_scoring import normalize_final_binary_label, resolve_binary_audit_topic


def build_base_output(
    _classifier: Any,
    *,
    record: Any,
    route_result: Any,
) -> Dict[str, Any]:
    """Build the base output row from metadata and stage-1 rule evidence."""

    return {
        **record.to_output_dict(),
        "urban_flag": "",
        "metadata_route": route_result.route,
        "metadata_route_reason": route_result.reason,
        "metadata_candidate_topic_buckets": "; ".join(route_result.candidate_topic_buckets),
        "metadata_candidate_matches": "; ".join(route_result.matched_candidate_terms),
        "metadata_negative_domains": "; ".join(route_result.matched_negative_domains),
        "metadata_negative_keywords": "; ".join(route_result.matched_negative_keywords),
        "metadata_related_domains": "; ".join(route_result.matched_related_domains),
        "metadata_filter_result": route_result.route,
        "metadata_filter_reason": route_result.reason,
        "metadata_positive_signals": "; ".join(route_result.matched_candidate_terms),
        "stage1_decision": route_result.stage1_decision,
        "stage1_reason_tag": route_result.stage1_reason_tag,
        "stage1_hit_signals": "; ".join(route_result.stage1_hit_signals),
        "stage1_risk_tags": "; ".join(route_result.stage1_risk_tags),
        "stage1_conflict_flag": route_result.stage1_conflict_flag,
        "topic_rule": route_result.topic_rule,
        "topic_rule_group": route_result.topic_rule_group,
        "topic_rule_name": route_result.topic_rule_name,
        "topic_rule_score": route_result.topic_rule_score,
        "topic_rule_margin": route_result.topic_rule_margin,
        "topic_rule_top3": "; ".join(route_result.topic_rule_top3),
        "topic_rule_matches": "; ".join(route_result.topic_rule_matches),
        "review_flag_rule": int(route_result.review_flag_rule),
        "review_reason_rule": route_result.review_reason_rule,
        "decision_source": "",
        "decision_reason": "",
        "llm_used": 0,
        "llm_attempted": 0,
        "llm_failure_reason": "",
        "llm_family_hint": "",
        "llm_family_hint_reason": "",
        "topic_family_rule": route_result.topic_rule_group or UNKNOWN_TOPIC_GROUP,
        "topic_family_local": UNKNOWN_TOPIC_GROUP,
        "topic_family_final": UNKNOWN_TOPIC_GROUP,
        "family_predicted_family": UNKNOWN_TOPIC_GROUP,
        "family_decision_source": "",
        "family_confidence": 0.0,
        "family_probability_urban": 0.0,
        "topic_within_family_label": "",
        "topic_family_within_score": 0.0,
        "topic_family_within_margin": 0.0,
        "boundary_bucket": "",
        "family_conflict_pattern": "",
        "unknown_recovery_path": "not_triggered",
        "unknown_recovery_evidence": "",
        "review_flag": 0,
        "review_reason": "",
        "anchor_guard_flag": 0,
        "anchor_guard_action": "none",
        "anchor_guard_reason": "",
        "anchor_guard_hits": "",
        "uncertain_nonurban_guard_flag": 0,
        "uncertain_nonurban_guard_action": "none",
        "uncertain_nonurban_guard_reason": "",
        "uncertain_nonurban_guard_evidence": "",
        "urban_probability_score": "",
        "binary_decision_threshold": "",
        "binary_decision_source": "",
        "binary_decision_evidence": "",
        "binary_topic_consistency_flag": 0,
        "binary_recall_calibration_flag": 0,
        "binary_recall_calibration_tier": "none",
        "binary_recall_calibration_reason": "",
        "binary_audit_resolution_flag": 0,
        "binary_audit_resolution_action": "none",
        "binary_audit_resolution_reason": "",
        "binary_audit_resolution_evidence": "",
        "review_flag_raw": 0,
        "review_reason_raw": "",
        "open_set_flag": 0,
        "open_set_topic": "",
        "open_set_reason": "",
        "open_set_evidence": "",
        "taxonomy_coverage_status": "unknown",
        "decision_explanation": "",
        "primary_positive_evidence": "",
        "primary_negative_evidence": "",
        "evidence_balance": "",
        "decision_rule_stack": "",
        "legacy_topic_label": "",
        "legacy_topic_group": "",
        "legacy_topic_name": "",
        "topic_local_label": "",
        "topic_local_group": "",
        "topic_local_name": "",
        "topic_local_confidence": 0.0,
        "topic_local_margin": 0.0,
        "topic_local_top3": "",
        "topic_label": "",
        "topic_group": "",
        "topic_name": "",
        "topic_final": "",
        "topic_final_group": "",
        "topic_final_name": "",
        "topic_confidence": 0.0,
        "topic_margin": 0.0,
        "topic_confidence_effective": 0.0,
        "topic_margin_effective": 0.0,
        "topic_matches": "",
        "topic_binary_score": 0.0,
        "topic_binary_probability": 0.0,
        "bertopic_status": "",
        "bertopic_topic_id": -1,
        "bertopic_topic_name": "",
        "bertopic_probability": 0.0,
        "bertopic_is_outlier": 0,
        "bertopic_count": 0,
        "bertopic_pos_rate": "",
        "bertopic_mapped_label": "",
        "bertopic_mapped_group": "",
        "bertopic_mapped_name": "",
        "bertopic_label_purity": 0.0,
        "bertopic_mapped_label_share": 0.0,
        "bertopic_top_terms": "",
        "bertopic_sample_titles": "",
        "bertopic_source_split": "",
        "bertopic_high_purity": 0,
        "bertopic_true_outlier": 0,
        "bertopic_prior_mode": "auxiliary_only",
        "bertopic_confidence_delta": 0.0,
        "bertopic_margin_delta": 0.0,
        "bertopic_hint_label": "",
        "bertopic_hint_group": "",
        "bertopic_hint_name": "",
        "bertopic_hint_conflict_flag": 0,
        "bertopic_cluster_quality": "",
        "bertopic_dynamic_topic_id": -1,
        "bertopic_dynamic_topic_words": "",
        "bertopic_primary_label": "",
        "bertopic_primary_group": "",
        "bertopic_primary_name": "",
        "bertopic_primary_probability": 0.0,
        "bertopic_primary_support": 0,
        "bertopic_primary_purity": 0.0,
        "bertopic_primary_mapped_share": 0.0,
        "bertopic_primary_override": 0,
        "bertopic_primary_reason": "",
    }


def build_final_result(
    classifier: Any,
    base: Dict[str, Any],
    *,
    final_topic: str,
    decision_source: str,
    decision_reason: str,
    confidence: float,
    review_flag: int,
    review_reason: str,
    binary_label: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the final prediction row while preserving the output contract."""

    audit_binary_label = normalize_final_binary_label(
        base,
        final_topic=final_topic,
        binary_label=binary_label,
    )
    final_topic, decision_reason = resolve_binary_audit_topic(
        classifier,
        base,
        final_topic=final_topic,
        binary_label=audit_binary_label,
        decision_reason=decision_reason,
    )
    topic_binary_label = urban_flag_for_topic_label(final_topic)
    urban_flag = audit_binary_label if audit_binary_label in {"0", "1"} else topic_binary_label
    review_flag, review_reason = classifier._reconcile_final_review_signal(
        base,
        final_topic=final_topic,
        binary_label=audit_binary_label,
        review_flag=review_flag,
        review_reason=review_reason,
    )

    topic_group = topic_group_for_label(final_topic)
    topic_name = topic_name_for_label(final_topic)
    legacy_label, legacy_group, legacy_name = legacy_topic_for_label(final_topic)

    bertopic_hint_label = str(base.get("bertopic_hint_label", "") or "")
    bertopic_conflict_flag = 0
    if final_topic != UNKNOWN_TOPIC_LABEL and bertopic_hint_label and bertopic_hint_label != final_topic:
        bertopic_conflict_flag = 1
    if final_topic == UNKNOWN_TOPIC_LABEL and bertopic_hint_label:
        bertopic_conflict_flag = 1

    if final_topic != UNKNOWN_TOPIC_LABEL:
        topic_family_final = topic_group
    else:
        topic_family_final = UNKNOWN_TOPIC_GROUP
    taxonomy_status = str(base.get("taxonomy_coverage_status", "") or "")
    if final_topic in {OPEN_SET_URBAN_LABEL, OPEN_SET_NONURBAN_LABEL}:
        taxonomy_status = "open_set"
        base["open_set_flag"] = 1
        base["open_set_topic"] = final_topic
        if not base.get("open_set_reason"):
            base["open_set_reason"] = "open_set_topic"
    elif final_topic == UNKNOWN_TOPIC_LABEL:
        taxonomy_status = taxonomy_status or "unknown"
    elif taxonomy_status not in {"hard_negative", "open_set", "binary_resolved"}:
        taxonomy_status = "covered"

    base.update(
        {
            Schema.IS_URBAN_RENEWAL: urban_flag,
            "urban_flag": urban_flag,
            "final_label": urban_flag,
            "urban_parse_reason": decision_source,
            "decision_source": decision_source,
            "decision_reason": decision_reason,
            "confidence": round(float(confidence), 4),
            "review_flag": int(review_flag),
            "review_reason": review_reason,
            "legacy_topic_label": legacy_label,
            "legacy_topic_group": legacy_group,
            "legacy_topic_name": legacy_name,
            "topic_final": final_topic,
            "topic_final_group": topic_group,
            "topic_final_name": topic_name,
            "topic_family_final": topic_family_final,
            "family_predicted_family": base.get("family_predicted_family") or topic_family_final,
            "family_decision_source": base.get("family_decision_source") or decision_source,
            "topic_within_family_label": base.get("topic_within_family_label")
            or (final_topic if final_topic != UNKNOWN_TOPIC_LABEL else ""),
            "topic_label": final_topic,
            "topic_group": topic_group,
            "topic_name": topic_name,
            "bertopic_hint_conflict_flag": bertopic_conflict_flag,
            "binary_topic_consistency_flag": classifier._effective_binary_topic_consistency_flag(
                base,
                binary_label=urban_flag,
                final_topic=final_topic,
            ),
            "taxonomy_coverage_status": taxonomy_status,
        }
    )
    base.update(
        classifier._summarize_decision_explanation(
            base,
            final_topic=final_topic,
            binary_label=urban_flag,
            confidence=float(confidence),
            decision_source=decision_source,
            review_flag=review_flag,
        )
    )
    return base
