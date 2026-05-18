"""Binary scoring helpers for the urban hybrid classifier.

The public classifier remains the facade. These helpers keep binary evidence,
audit, and final-label scoring away from the orchestration code without
changing the scoring contract.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ...runtime.config import Config
from ..taxonomy.core import UNKNOWN_TOPIC_GROUP, UNKNOWN_TOPIC_LABEL, topic_group_for_label, urban_flag_for_topic_label


BINARY_HARD_NEGATIVE_REASONS = {"math_term_misuse", "rural_nonurban"}


def apply_binary_decision(
    classifier: Any,
    base: Dict[str, Any],
    *,
    record: Any,
    route_result: Any,
    final_topic: str,
    decision_source: str,
    decision_reason: str,
    confidence: float,
    review_flag: int,
    review_reason: str,
) -> tuple[str, float, int, str]:
    """Apply the existing deterministic binary scoring policy."""

    if not bool(Config.URBAN_BINARY_DECISION_ENABLED):
        label = urban_flag_for_topic_label(final_topic)
        return label, confidence, review_flag, review_reason

    threshold = float(Config.URBAN_BINARY_DECISION_THRESHOLD)
    base["binary_decision_threshold"] = threshold

    route_reason = str(route_result.reason or "").strip()
    if route_reason in BINARY_HARD_NEGATIVE_REASONS:
        score = 0.02
        binary_label = "0"
        decision_confidence = 0.98
        consistency_flag = classifier._binary_topic_consistency_flag(
            binary_label=binary_label,
            final_topic=final_topic,
        )
        base.update(
            {
                "urban_probability_score": score,
                "binary_decision_source": "binary_hard_negative_override",
                "binary_decision_evidence": f"hard_negative:{route_reason}",
                "binary_topic_consistency_flag": consistency_flag,
                "binary_recall_calibration_flag": 0,
                "binary_recall_calibration_tier": "hard_negative",
                "binary_recall_calibration_reason": route_reason,
            }
        )
        return binary_label, decision_confidence, int(bool(review_flag)), review_reason

    family_probability = classifier._source_probability(base.get("family_probability_urban"), default=0.5)
    topic_binary_probability = classifier._source_probability(base.get("topic_binary_probability"), default=0.5)
    topic_vote_probability = classifier._topic_family_vote_probability(base, final_topic=final_topic)
    anchor_probability, anchor_evidence = classifier._anchor_probability(
        record=record,
        base=base,
        final_topic=final_topic,
    )
    llm_probability = classifier._llm_hint_probability(base)
    risk_adjustment, risk_evidence = classifier._risk_adjustment(base)
    decision_adjustment, decision_adjustment_evidence = classifier._decision_adjustment(
        decision_source=decision_source,
        decision_reason=decision_reason,
    )

    raw_score = (
        0.40 * family_probability
        + 0.25 * topic_binary_probability
        + 0.20 * topic_vote_probability
        + 0.10 * anchor_probability
        + 0.05 * llm_probability
        + risk_adjustment
        + decision_adjustment
    )
    raw_score = round(min(max(raw_score, 0.02), 0.98), 6)
    recall_context = classifier._binary_recall_context(
        record=record,
        base=base,
        final_topic=final_topic,
    )
    score, recall_flag, recall_tier, recall_reason = classifier._apply_binary_recall_calibration(
        base=base,
        raw_score=raw_score,
        context=recall_context,
        final_topic=final_topic,
        decision_source=decision_source,
    )
    score = round(min(max(score, 0.02), 0.98), 6)
    binary_label = "1" if score >= threshold else "0"
    decision_confidence = round(score if binary_label == "1" else 1.0 - score, 6)
    consistency_flag = classifier._binary_topic_consistency_flag(
        binary_label=binary_label,
        final_topic=final_topic,
    )
    evidence = (
        f"family={family_probability:.4f}*0.40;"
        f"topic_binary={topic_binary_probability:.4f}*0.25;"
        f"topic_vote={topic_vote_probability:.4f}*0.20;"
        f"anchor={anchor_probability:.2f}*0.10({anchor_evidence});"
        f"llm_hint={llm_probability:.2f}*0.05;"
        f"risk_adjust={risk_adjustment:+.2f}({risk_evidence});"
        f"decision_adjust={decision_adjustment:+.2f}({decision_adjustment_evidence});"
        f"raw_score={raw_score:.6f};"
        f"recall_calibration={recall_tier}({recall_reason})"
    )
    base.update(
        {
            "urban_probability_score": score,
            "binary_decision_source": "binary_confidence_resolution",
            "binary_decision_evidence": evidence,
            "binary_topic_consistency_flag": consistency_flag,
            "binary_recall_calibration_flag": int(bool(recall_flag)),
            "binary_recall_calibration_tier": recall_tier,
            "binary_recall_calibration_reason": recall_reason,
        }
    )

    review_reasons = [item for item in str(review_reason or "").split(";") if item]
    if decision_confidence < float(Config.URBAN_BINARY_LOW_CONFIDENCE_REVIEW_FLOOR):
        review_reasons.append("binary_low_confidence")
    if consistency_flag:
        review_reasons.append("binary_topic_inconsistency")
    review_flag = int(bool(review_flag) or bool(review_reasons))
    review_reason = ";".join(dict.fromkeys(review_reasons))
    return binary_label, decision_confidence, review_flag, review_reason


def normalize_final_binary_label(
    base: Dict[str, Any],
    *,
    final_topic: str,
    binary_label: Optional[str],
) -> str:
    """Resolve the final binary label from explicit, existing, or topic labels."""

    if binary_label in {"0", "1"}:
        return str(binary_label)
    existing_label = str(base.get("final_label", "") or base.get("urban_flag", "") or "").strip()
    if existing_label.endswith(".0"):
        existing_label = existing_label[:-2]
    if existing_label in {"0", "1"}:
        return existing_label
    topic_label = urban_flag_for_topic_label(final_topic)
    return topic_label if topic_label in {"0", "1"} else ""


def resolve_binary_audit_topic(
    classifier: Any,
    base: Dict[str, Any],
    *,
    final_topic: str,
    binary_label: str,
    decision_reason: str,
) -> tuple[str, str]:
    """Apply the existing binary/topic audit annotations."""

    if not bool(Config.URBAN_BINARY_AUDIT_RESOLUTION_ENABLED):
        return final_topic, decision_reason

    route_reason = str(base.get("metadata_route_reason", "") or "").strip()
    if route_reason in BINARY_HARD_NEGATIVE_REASONS:
        base.update(
            {
                "binary_audit_resolution_flag": 0,
                "binary_audit_resolution_action": "hard_negative_preserved",
                "binary_audit_resolution_reason": route_reason,
                "binary_audit_resolution_evidence": route_reason,
            }
        )
        return final_topic, decision_reason

    topic_group = topic_group_for_label(final_topic)
    if binary_label not in {"0", "1"}:
        base.update(
            {
                "binary_audit_resolution_flag": 0,
                "binary_audit_resolution_action": "missing_binary_label",
                "binary_audit_resolution_reason": f"topic_group={topic_group}",
                "binary_audit_resolution_evidence": "",
            }
        )
        return final_topic, decision_reason

    if binary_label == "1" and topic_group != "urban":
        score = classifier._safe_float(base.get("urban_probability_score"), default=0.0)
        threshold = classifier._safe_float(
            base.get("binary_decision_threshold"),
            default=float(Config.URBAN_BINARY_DECISION_THRESHOLD),
        )
        recall_tier = str(base.get("binary_recall_calibration_tier", "") or "none")
        from_topic = final_topic or UNKNOWN_TOPIC_LABEL
        evidence = (
            f"from={from_topic};score={score:.4f};threshold={threshold:.4f};"
            f"recall={recall_tier};source={base.get('binary_decision_source', '')}"
        )
        base.update(
            {
                "binary_audit_resolution_flag": 1,
                "binary_audit_resolution_action": "positive_binary_conflict_audit",
                "binary_audit_resolution_reason": f"binary_positive_from_{from_topic}",
                "binary_audit_resolution_evidence": evidence,
            }
        )
        decision_reason = f"{decision_reason};binary_audit_positive_conflict:{from_topic}".strip(";")
        return final_topic, decision_reason

    if binary_label == "0" and topic_group != "nonurban":
        score = classifier._safe_float(base.get("urban_probability_score"), default=1.0)
        threshold = classifier._safe_float(
            base.get("binary_decision_threshold"),
            default=float(Config.URBAN_BINARY_DECISION_THRESHOLD),
        )
        from_topic = final_topic or UNKNOWN_TOPIC_LABEL
        evidence = (
            f"from={from_topic};score={score:.4f};threshold={threshold:.4f};"
            f"source={base.get('binary_decision_source', '')}"
        )
        if topic_group == UNKNOWN_TOPIC_GROUP:
            base.update(
                {
                    "binary_audit_resolution_flag": 1,
                    "binary_audit_resolution_action": "negative_binary_conflict_audit",
                    "binary_audit_resolution_reason": f"binary_negative_from_{from_topic}",
                    "binary_audit_resolution_evidence": evidence,
                }
            )
            decision_reason = f"{decision_reason};binary_audit_negative_conflict:{from_topic}".strip(";")
            return final_topic, decision_reason

        base.update(
            {
                "binary_audit_resolution_flag": 1,
                "binary_audit_resolution_action": "negative_binary_conflict_audit",
                "binary_audit_resolution_reason": f"binary_negative_from_{from_topic}",
                "binary_audit_resolution_evidence": evidence,
            }
        )
        decision_reason = f"{decision_reason};binary_audit_negative_conflict:{from_topic}".strip(";")
        return final_topic, decision_reason

    base.update(
        {
            "binary_audit_resolution_flag": 0,
            "binary_audit_resolution_action": "covered",
            "binary_audit_resolution_reason": f"binary_{binary_label}_matches_{topic_group}",
            "binary_audit_resolution_evidence": "",
        }
    )
    return final_topic, decision_reason
