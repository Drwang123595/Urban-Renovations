"""Stable urban-renewal strategy pipeline for rows and data frames."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ...runtime.config import Schema
from ..taxonomy.core import UNKNOWN_TOPIC_GROUP, UNKNOWN_TOPIC_LABEL, topic_group_for_label
from .decision import StableUrbanDecisionEngine, decide_stable_strategy
from .evidence import EvidenceBundle
from .input import build_article_input
from .llm_semantic import LLMSemanticAnalyzer
from .output import apply_decision_to_row
from .rule_evidence import RuleEvidenceExtractor
from .topic_evidence import (
    build_cluster_evidence_from_row,
    build_family_evidence_from_row,
    build_topic_evidence_from_row,
)


class StableUrbanStrategy:
    def __init__(
        self,
        *,
        decision_engine: StableUrbanDecisionEngine | None = None,
        llm_analyzer: Any | None = None,
    ):
        self.rule_extractor = RuleEvidenceExtractor()
        self.decision_engine = decision_engine or StableUrbanDecisionEngine()
        self.llm_analyzer = llm_analyzer

    def classify_row(
        self,
        row: Mapping[str, Any],
        *,
        mutate_final_fields: bool = False,
        session_path: Any = None,
        audit_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        evidence = build_evidence_bundle_from_row(row)
        evidence = self._with_llm_semantic_evidence(
            evidence,
            session_path=session_path,
            audit_metadata=audit_metadata,
        )
        decision = self.decision_engine.decide(evidence)
        return apply_decision_to_row(dict(row), evidence, decision, mutate_final_fields=mutate_final_fields)

    def _with_llm_semantic_evidence(
        self,
        evidence: EvidenceBundle,
        *,
        session_path: Any = None,
        audit_metadata: dict[str, Any] | None = None,
    ) -> EvidenceBundle:
        analyzer = self.llm_analyzer
        if analyzer is None:
            return evidence
        should_call = getattr(analyzer, "should_call", None)
        if callable(should_call) and not should_call(rule=evidence.rule, topic=evidence.topic):
            return evidence
        analyze = getattr(analyzer, "analyze", None)
        if not callable(analyze):
            return evidence
        llm_evidence = analyze(
            evidence.article,
            rule=evidence.rule,
            topic=evidence.topic,
            session_path=session_path,
            audit_metadata=audit_metadata,
        )
        return evidence.with_llm(llm_evidence)


def build_evidence_bundle_from_row(row: Mapping[str, Any]) -> EvidenceBundle:
    article = build_article_input(row)
    topic = build_topic_evidence_from_row(row)
    route_reason = _text(row.get("metadata_route_reason", ""))
    risk_tags = _split_tokens(row.get("stage1_risk_tags", ""))
    rule = RuleEvidenceExtractor().extract_from_article(
        article,
        route_reason=route_reason,
        risk_tags=risk_tags,
        rule_topic_candidate=_text(row.get("topic_rule", "")) or topic.topic_candidate,
        rule_confidence=_safe_float(row.get("topic_rule_score", 0.0)),
        explanation=_text(row.get("metadata_route_reason", "")),
    )
    current_label = _normalize_binary_label(
        row.get("final_label", row.get("urban_flag", row.get(Schema.IS_URBAN_RENEWAL, "")))
    )
    current_topic = _text(row.get("topic_final", row.get("topic_label", ""))) or UNKNOWN_TOPIC_LABEL
    current_group = _text(row.get("topic_final_group", row.get("topic_group", ""))).lower()
    if not current_group:
        current_group = topic_group_for_label(current_topic)
    dynamic = {
        "candidate_label": _normalize_binary_label(row.get("dynamic_binary_candidate_label", "")),
        "candidate_topic": _text(row.get("dynamic_to_fixed_topic_candidate", row.get("dynamic_binary_override_topic", ""))),
        "override_applied": _safe_int(row.get("dynamic_binary_override_applied", 0)),
    }
    return EvidenceBundle(
        article=article,
        rule=rule,
        topic=topic,
        family=build_family_evidence_from_row(row),
        cluster=build_cluster_evidence_from_row(row),
        current_label=current_label,
        current_topic=current_topic,
        current_topic_group=current_group or UNKNOWN_TOPIC_GROUP,
        score=_safe_float(row.get("urban_probability_score", 0.0)),
        threshold=_safe_float(row.get("binary_decision_threshold", 0.45), default=0.45),
        binary_topic_consistency_flag=_safe_int(row.get("binary_topic_consistency_flag", 0)),
        evidence_balance=_text(row.get("evidence_balance", "")),
        decision_source=_text(row.get("decision_source", "")),
        binary_decision_source=_text(row.get("binary_decision_source", "")),
        dynamic=dynamic,
    )


def apply_stable_strategy(
    frame: pd.DataFrame,
    *,
    mutate_final_fields: bool = False,
    llm_analyzer: Any | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    working = frame.copy()
    strategy = StableUrbanStrategy(llm_analyzer=llm_analyzer)
    for idx, row in working.iterrows():
        updated = strategy.classify_row(
            row.to_dict(),
            mutate_final_fields=mutate_final_fields,
        )
        for column, value in updated.items():
            if column not in working.columns:
                working[column] = pd.Series([""] * len(working), index=working.index, dtype=object)
            working.at[idx, column] = value
    return working


def build_llm_semantic_analyzer(llm_strategy: Any, *, enabled: bool = True) -> LLMSemanticAnalyzer:
    return LLMSemanticAnalyzer(llm_strategy, enabled=enabled)


def _normalize_binary_label(value: Any) -> str:
    text = _text(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text if text in {"0", "1"} else ""


def _split_tokens(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").replace(",", ";").split(";") if part.strip()]


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
