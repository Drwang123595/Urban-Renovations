"""Boundary-guard orchestration for the urban hybrid classifier."""

from __future__ import annotations

from typing import Any, Dict

from ..taxonomy.core import UNKNOWN_TOPIC_LABEL


def apply_boundary_guards(
    classifier: Any,
    base: Dict[str, Any],
    *,
    record: Any,
    route_result: Any,
    topic_prediction: Any,
    bertopic_signal: Any,
    state: Any,
    llm_family_hint: str,
    fusion_final_topic: str,
) -> Any:
    """Apply family, anchor, uncertain-nonurban, review, and open-set guards."""

    state.final_topic, state.decision_source, state.decision_reason, state.confidence = classifier._apply_family_gate(
        base,
        record=record,
        route_result=route_result,
        topic_prediction=topic_prediction,
        bertopic_signal=bertopic_signal,
        llm_family_hint=llm_family_hint,
        candidate_final_topic=state.final_topic,
        decision_source=state.decision_source,
        decision_reason=state.decision_reason,
        confidence=state.confidence,
    )
    (
        state.final_topic,
        state.decision_source,
        state.decision_reason,
        state.confidence,
        state.review_flag,
        state.review_reason,
    ) = classifier._apply_anchor_guard(
        base,
        record=record,
        route_result=route_result,
        topic_prediction=topic_prediction,
        bertopic_signal=bertopic_signal,
        final_topic=state.final_topic,
        decision_source=state.decision_source,
        decision_reason=state.decision_reason,
        confidence=state.confidence,
        review_flag=state.review_flag,
        review_reason=state.review_reason,
    )
    (
        state.final_topic,
        state.decision_source,
        state.decision_reason,
        state.confidence,
        state.review_flag,
        state.review_reason,
    ) = classifier._apply_uncertain_nonurban_guard(
        base,
        record=record,
        route_result=route_result,
        topic_prediction=topic_prediction,
        bertopic_signal=bertopic_signal,
        final_topic=state.final_topic,
        decision_source=state.decision_source,
        decision_reason=state.decision_reason,
        confidence=state.confidence,
        review_flag=state.review_flag,
        review_reason=state.review_reason,
    )
    if state.final_topic != UNKNOWN_TOPIC_LABEL and fusion_final_topic != UNKNOWN_TOPIC_LABEL:
        base["unknown_recovery_path"] = "not_triggered"
        base["unknown_recovery_evidence"] = ""
    if state.final_topic == UNKNOWN_TOPIC_LABEL and str(base.get("unknown_recovery_path", "") or "") in {
        "",
        "not_triggered",
        "pending_review",
    }:
        base["unknown_recovery_path"] = "retained_unknown"
        base["unknown_recovery_evidence"] = str(base.get("unknown_recovery_evidence", "") or state.review_reason or "")
    state.review_flag, state.review_reason = classifier._merge_rule_review_signal(
        base=base,
        final_topic=state.final_topic,
        review_flag=state.review_flag,
        review_reason=state.review_reason,
    )
    (
        state.final_topic,
        state.decision_source,
        state.decision_reason,
        state.confidence,
        state.review_flag,
        state.review_reason,
    ) = classifier._apply_open_set_topic(
        base,
        record=record,
        route_result=route_result,
        final_topic=state.final_topic,
        decision_source=state.decision_source,
        decision_reason=state.decision_reason,
        confidence=state.confidence,
        review_flag=state.review_flag,
        review_reason=state.review_reason,
    )
    return state
