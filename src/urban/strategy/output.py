"""Output helpers and column defaults for stable urban-renewal strategy."""

from __future__ import annotations

from typing import Any

from ...runtime.config import Schema
from ..taxonomy.core import topic_group_for_label, topic_name_for_label
from .evidence import EvidenceBundle, StableDecisionResult


STABLE_STRATEGY_DEFAULTS: dict[str, Any] = {
    "strategy_label": "",
    "strategy_topic": "",
    "strategy_status": "",
    "strategy_reason": "",
    "strategy_confidence": "",
    "positive_evidence": "",
    "negative_evidence": "",
    "llm_semantic_evidence": "",
    "evidence_conflict_type": "",
}

STRATEGY_V3_DEFAULTS: dict[str, Any] = {
    "strategy_v3_label": "",
    "strategy_v3_topic": "",
    "strategy_v3_status": "",
    "strategy_v3_reason": "",
    "strategy_v3_evidence": "",
    "strategy_v3_confidence": "",
}


def apply_decision_to_row(
    row: dict[str, Any],
    evidence: EvidenceBundle,
    decision: StableDecisionResult,
    *,
    mutate_final_fields: bool = False,
) -> dict[str, Any]:
    output = dict(row)
    output.update(
        {
            "strategy_label": decision.final_label,
            "strategy_topic": decision.topic_final,
            "strategy_status": decision.status.value,
            "strategy_reason": decision.reason,
            "strategy_confidence": decision.confidence,
            "positive_evidence": decision.positive_evidence,
            "negative_evidence": decision.negative_evidence,
            "llm_semantic_evidence": decision.llm_semantic_evidence,
            "evidence_conflict_type": decision.evidence_conflict_type,
            # Compatibility aliases for existing workbooks/tests that already
            # know the temporary strategy_v3_* fields.
            "strategy_v3_label": decision.final_label,
            "strategy_v3_topic": decision.topic_final,
            "strategy_v3_status": decision.status.value,
            "strategy_v3_reason": decision.reason,
            "strategy_v3_evidence": decision.positive_evidence,
            "strategy_v3_confidence": decision.confidence,
        }
    )
    if decision.review_flag:
        output["review_flag"] = 1
        prior = str(output.get("review_reason", "") or "").strip()
        reason = decision.review_reason
        output["review_reason"] = ";".join(part for part in [prior, reason] if part)

    if not mutate_final_fields:
        return output

    output[Schema.IS_URBAN_RENEWAL] = decision.final_label
    output["urban_flag"] = decision.urban_flag
    output["final_label"] = decision.final_label
    output["topic_final"] = decision.topic_final
    output["topic_label"] = decision.topic_final
    output["topic_final_group"] = topic_group_for_label(decision.topic_final)
    output["topic_group"] = topic_group_for_label(decision.topic_final)
    output["topic_final_name"] = topic_name_for_label(decision.topic_final)
    output["topic_name"] = topic_name_for_label(decision.topic_final)
    output["decision_source"] = _append_source(output.get("decision_source", ""), "stable_strategy")
    output["binary_decision_source"] = _append_source(output.get("binary_decision_source", ""), "stable_strategy")
    return output


def _append_source(prior: Any, source: str) -> str:
    parts = [part for part in str(prior or "").split("|") if part]
    if source not in parts:
        parts.append(source)
    return "|".join(parts)
