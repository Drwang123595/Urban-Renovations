from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..runtime.config import Config, Schema
from .urban_bertopic_service import BERTopicSignal, UrbanBERTopicService
from .urban_family_gate import UrbanFamilyGate
from .urban_metadata import UrbanMetadataRecord, normalize_phrase
from .urban_rule_filter import (
    METADATA_ROUTE_HARD_NEGATIVE,
    MetadataRuleFilter,
    R7_STRONG_RENEWAL_MECHANISMS,
    RISK_BACKGROUND_SUPPORT,
    RISK_EXPLICIT_RENEWAL_OTHER_OBJECT,
    RISK_GENERIC_TECHNICAL,
    RISK_GREENFIELD_EXPANSION,
    RISK_SOCIAL_HISTORY_MEDIA,
)
from .urban_topic_classifier import TopicPrediction, UrbanTopicClassifier
from .urban_topic_taxonomy import (
    COMMON_EXISTING_URBAN_OBJECTS,
    COMMON_RENEWAL_ANCHORS,
    CORE_RENEWAL_ANCHORS,
    OPEN_SET_NONURBAN_LABEL,
    OPEN_SET_URBAN_LABEL,
    UNKNOWN_TOPIC_GROUP,
    UNKNOWN_TOPIC_LABEL,
    UNKNOWN_TOPIC_NAME,
    legacy_topic_for_label,
    topic_group_for_label,
    topic_name_for_label,
    urban_flag_for_topic_label,
)


UNKNOWN_RECOVERY_RULE_TO_LOCAL = {
    ("N3", "U9"): (0.68, 1.0),
    ("N4", "U9"): (0.7, 1.1),
    ("N8", "U9"): (0.68, 1.0),
    ("N3", "U12"): (0.68, 1.0),
    ("N4", "U12"): (0.64, 0.2),
    ("N1", "U12"): (0.62, 0.2),
    ("N2", "U12"): (0.62, 0.2),
    ("N5", "U12"): (0.64, 0.2),
    ("N7", "U12"): (0.62, 0.2),
    ("N9", "U12"): (0.62, 0.2),
    ("N3", "U1"): (0.68, 0.8),
    ("N4", "U1"): (0.60, 0.0),
    ("N8", "U1"): (0.60, 0.0),
    ("N4", "U4"): (0.64, 0.2),
    ("N2", "U5"): (0.62, 0.0),
    ("N9", "U5"): (0.62, 0.0),
    ("N10", "U5"): (0.64, 0.2),
    ("N3", "U10"): (0.7, 1.0),
    ("N4", "U10"): (0.7, 1.0),
}
UNKNOWN_RECOVERY_RULE_TO_TOPIC = {
    ("U12", "N4"): 0.30,
    ("U12", "N1"): 0.30,
    ("U12", "N2"): 0.3,
    ("U12", "N5"): 0.32,
    ("U12", "N7"): 0.32,
    ("U12", "N9"): 0.32,
    ("U1", "N1"): 0.28,
    ("U9", "N3"): 0.65,
    ("U10", "N3"): 0.68,
    ("U10", "N4"): 0.68,
    ("U4", "N1"): 0.28,
    ("U4", "N4"): 0.32,
    ("U5", "N2"): 0.3,
    ("U5", "N9"): 0.3,
    ("U5", "N10"): 0.32,
}
UNKNOWN_RECOVERY_WEAK_URBAN_RULES = {
    "U1": 0.28,
    "U3": 0.28,
    "U11": 0.28,
    "U4": 0.3,
    "U5": 0.3,
    "U9": 0.4,
    "U10": 0.45,
    "U12": 0.32,
    "U15": 0.35,
}
UNKNOWN_RECOVERY_CONSENSUS_SCORE_FLOOR = 3.4
UNKNOWN_RECOVERY_CONSENSUS_MARGIN_FLOOR = 1.0
UNKNOWN_RECOVERY_CONSENSUS_STRONG_SCORE_FLOOR = 4.8
UNKNOWN_FAMILY_GATE_RECOVERY_SCORE_FLOOR = 4.0
UNKNOWN_FAMILY_GATE_RECOVERY_MARGIN_FLOOR = 1.0
UNKNOWN_FAMILY_GATE_RECOVERY_CONFIDENCE_FLOOR = 0.82
OFFLINE_UNKNOWN_RECOVERY_LOCAL_RULES = {
    ("N8", "U9"): (0.84, 2.5, 0.0),
    ("N3", "U9"): (0.89, 3.0, 0.0),
    ("N3", "U12"): (0.85, 4.5, 0.0),
    ("N3", "U1"): (0.67, 1.0, 0.92),
    ("N4", "U1"): (0.67, 1.0, 0.92),
}

ANCHOR_GUARD_HARD_NEGATIVE_REASONS = {"math_term_misuse", "rural_nonurban"}
ANCHOR_GUARD_BOUNDARY_BUCKETS = {
    "nonurban_rule_urban_local",
    "governance_policy_finance_boundary",
    "brownfield_environment_boundary",
}
UNCERTAIN_NONURBAN_HARD_NEGATIVE_REASONS = {"math_term_misuse", "rural_nonurban"}
UNCERTAIN_NONURBAN_PROMOTE_BOUNDARY_BUCKETS = {
    "nonurban_rule_urban_local",
    "governance_policy_finance_boundary",
}
UNCERTAIN_NONURBAN_HIGH_RISK_RULES = {"N1", "N3", "N4", "N5", "N7", "N9", "N10"}
BINARY_HARD_NEGATIVE_REASONS = {"math_term_misuse", "rural_nonurban"}
BINARY_RISK_ADJUSTMENTS = {
    RISK_GENERIC_TECHNICAL: -0.06,
    RISK_BACKGROUND_SUPPORT: -0.08,
    RISK_SOCIAL_HISTORY_MEDIA: -0.06,
    RISK_GREENFIELD_EXPANSION: -0.12,
    RISK_EXPLICIT_RENEWAL_OTHER_OBJECT: 0.03,
}
BINARY_RECALL_URBAN_CONTEXT_TERMS = (
    "urban",
    "city",
    "cities",
    "municipal",
    "metropolitan",
    "neighborhood",
    "neighbourhood",
    "downtown",
    "inner city",
    "inner-city",
    "slum",
    "informal settlement",
    "brownfield",
    "housing estate",
    "public housing",
    "old town",
    "old district",
    "local government",
    "planning",
    "housing",
    "land use",
    "infrastructure",
    "community",
)
OPEN_SET_RENEWAL_ACTION_TERMS = (
    "renewal",
    "regeneration",
    "redevelopment",
    "revitalization",
    "upgrading",
    "rehabilitation",
    "renovation",
    "adaptive reuse",
    "retrofit",
    "retrofitting",
    "requalification",
)
OPEN_SET_EXISTING_BUILT_ENV_TERMS = COMMON_EXISTING_URBAN_OBJECTS + (
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
    "building stock",
    "historic urban fabric",
    "urban fabric",
    "built environment",
    "aging district",
    "ageing district",
    "older district",
    "older neighborhood",
    "older neighbourhood",
)
OPEN_SET_POLICY_INTERVENTION_TERMS = (
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
)


class UrbanHybridClassifier:
    def __init__(
        self,
        llm_strategy,
        *,
        bertopic_service: Optional[UrbanBERTopicService] = None,
        llm_assist_enabled: Optional[bool] = None,
        family_gate: Optional[UrbanFamilyGate] = None,
    ):
        self.llm_strategy = llm_strategy
        self.rule_filter = MetadataRuleFilter()
        self.topic_classifier = UrbanTopicClassifier()
        self.bertopic_service = bertopic_service or UrbanBERTopicService()
        self.family_gate = family_gate or UrbanFamilyGate()
        if llm_assist_enabled is None:
            llm_assist_enabled = Config.URBAN_HYBRID_LLM_ASSIST_ENABLED
        self.llm_assist_enabled = bool(llm_assist_enabled)

    def classify(
        self,
        title: str,
        abstract: str,
        metadata: Optional[Dict[str, Any]] = None,
        session_path: Optional[Path] = None,
        audit_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        row = dict(metadata or {})
        row.update({Schema.TITLE: title, Schema.ABSTRACT: abstract})
        record = UrbanMetadataRecord.from_row(row)
        route_result = self.rule_filter.evaluate(record)
        base = self._build_base_output(record=record, route_result=route_result)

        if route_result.route == METADATA_ROUTE_HARD_NEGATIVE:
            base.update(
                {
                    "topic_family_rule": route_result.topic_rule_group or UNKNOWN_TOPIC_GROUP,
                    "topic_family_local": UNKNOWN_TOPIC_GROUP,
                    "topic_family_final": route_result.topic_rule_group or UNKNOWN_TOPIC_GROUP,
                    "family_decision_source": "stage1_rule",
                    "family_confidence": 0.99,
                    "boundary_bucket": "hard_negative",
                    "family_conflict_pattern": f"{route_result.topic_rule}_vs_Unknown",
                    "taxonomy_coverage_status": "hard_negative",
                    "open_set_reason": "hard_negative",
                    "open_set_evidence": route_result.reason,
                }
            )
            binary_label, binary_confidence, review_flag, review_reason = self._apply_binary_decision(
                base,
                record=record,
                route_result=route_result,
                final_topic=route_result.topic_rule,
                decision_source="stage1_rule",
                decision_reason=route_result.reason,
                confidence=0.99,
                review_flag=0,
                review_reason="",
            )
            return self._build_final_result(
                base,
                final_topic=route_result.topic_rule,
                decision_source="stage1_rule",
                decision_reason=route_result.reason,
                confidence=binary_confidence,
                review_flag=review_flag,
                review_reason=review_reason,
                binary_label=binary_label,
            )

        topic_prediction = self.topic_classifier.predict(record)
        bertopic_signal = self.bertopic_service.predict(record)
        self._attach_local_topic(base, topic_prediction)
        self._attach_bertopic_hint(base, bertopic_signal)

        fusion = self._fuse_rule_and_local(
            route_result=route_result,
            topic_prediction=topic_prediction,
        )
        llm_family_hint = ""
        final_topic = fusion["final_topic"]
        decision_source = fusion["decision_source"]
        decision_reason = fusion["decision_reason"]
        confidence = fusion["confidence"]
        review_flag = 0
        review_reason = ""
        if fusion["final_topic"] == UNKNOWN_TOPIC_LABEL:
            base["unknown_recovery_path"] = "pending_review"
            llm_hint = self._maybe_collect_llm_hint(
                title=title,
                abstract=abstract,
                record=record,
                route_result=route_result,
                topic_prediction=topic_prediction,
                bertopic_signal=bertopic_signal,
                session_path=session_path,
                audit_metadata=audit_metadata,
            )
            base.update(llm_hint)
            llm_family_hint = self._normalize_family_hint_value(base.get("llm_family_hint", ""))
            resolved = self._resolve_unknown_with_hints(
                route_result=route_result,
                topic_prediction=topic_prediction,
                bertopic_signal=bertopic_signal,
                llm_family_hint=base.get("llm_family_hint", ""),
            )
            if resolved["final_topic"] != UNKNOWN_TOPIC_LABEL:
                final_topic = resolved["final_topic"]
                decision_source = resolved["decision_source"]
                decision_reason = resolved["decision_reason"]
                confidence = resolved["confidence"]
                base["unknown_recovery_path"] = str(
                    resolved.get("recovery_path", decision_source or "unknown_hint_resolution")
                )
                base["unknown_recovery_evidence"] = str(resolved.get("recovery_evidence", "") or "")
            else:
                review_reason = fusion["review_reason"]
                decision_source = "unknown_review"
                decision_reason = fusion["decision_reason"]
                confidence = max(route_result.topic_rule_score / 10.0, float(topic_prediction.confidence))
                review_flag = 1
                base["unknown_recovery_path"] = str(resolved.get("recovery_path", "retained_unknown") or "retained_unknown")
                evidence = str(resolved.get("recovery_evidence", "") or "")
                base["unknown_recovery_evidence"] = evidence or review_reason
            if llm_family_hint in {"0", "1"}:
                review_reason = f"{review_reason};llm_family_hint={llm_family_hint}".strip(";")
                if base.get("unknown_recovery_evidence"):
                    base["unknown_recovery_evidence"] = (
                        f"{base['unknown_recovery_evidence']};llm_family_hint={llm_family_hint}"
                    ).strip(";")
                else:
                    base["unknown_recovery_evidence"] = f"llm_family_hint={llm_family_hint}"

        final_topic, decision_source, decision_reason, confidence = self._apply_family_gate(
            base,
            record=record,
            route_result=route_result,
            topic_prediction=topic_prediction,
            bertopic_signal=bertopic_signal,
            llm_family_hint=llm_family_hint,
            candidate_final_topic=final_topic,
            decision_source=decision_source,
            decision_reason=decision_reason,
            confidence=confidence,
        )
        (
            final_topic,
            decision_source,
            decision_reason,
            confidence,
            review_flag,
            review_reason,
        ) = self._apply_anchor_guard(
            base,
            record=record,
            route_result=route_result,
            topic_prediction=topic_prediction,
            bertopic_signal=bertopic_signal,
            final_topic=final_topic,
            decision_source=decision_source,
            decision_reason=decision_reason,
            confidence=confidence,
            review_flag=review_flag,
            review_reason=review_reason,
        )
        (
            final_topic,
            decision_source,
            decision_reason,
            confidence,
            review_flag,
            review_reason,
        ) = self._apply_uncertain_nonurban_guard(
            base,
            record=record,
            route_result=route_result,
            topic_prediction=topic_prediction,
            bertopic_signal=bertopic_signal,
            final_topic=final_topic,
            decision_source=decision_source,
            decision_reason=decision_reason,
            confidence=confidence,
            review_flag=review_flag,
            review_reason=review_reason,
        )
        if final_topic != UNKNOWN_TOPIC_LABEL and fusion["final_topic"] != UNKNOWN_TOPIC_LABEL:
            base["unknown_recovery_path"] = "not_triggered"
            base["unknown_recovery_evidence"] = ""
        if final_topic == UNKNOWN_TOPIC_LABEL and str(base.get("unknown_recovery_path", "") or "") in {
            "",
            "not_triggered",
            "pending_review",
        }:
            base["unknown_recovery_path"] = "retained_unknown"
            base["unknown_recovery_evidence"] = str(base.get("unknown_recovery_evidence", "") or review_reason or "")
        review_flag, review_reason = self._merge_rule_review_signal(
            base=base,
            final_topic=final_topic,
            review_flag=review_flag,
            review_reason=review_reason,
        )
        (
            final_topic,
            decision_source,
            decision_reason,
            confidence,
            review_flag,
            review_reason,
        ) = self._apply_open_set_topic(
            base,
            record=record,
            route_result=route_result,
            final_topic=final_topic,
            decision_source=decision_source,
            decision_reason=decision_reason,
            confidence=confidence,
            review_flag=review_flag,
            review_reason=review_reason,
        )
        binary_label, confidence, review_flag, review_reason = self._apply_binary_decision(
            base,
            record=record,
            route_result=route_result,
            final_topic=final_topic,
            decision_source=decision_source,
            decision_reason=decision_reason,
            confidence=confidence,
            review_flag=review_flag,
            review_reason=review_reason,
        )
        return self._build_final_result(
            base,
            final_topic=final_topic,
            decision_source=decision_source,
            decision_reason=decision_reason,
            confidence=confidence,
            review_flag=review_flag,
            review_reason=review_reason,
            binary_label=binary_label,
        )

    def _build_base_output(
        self,
        *,
        record: UrbanMetadataRecord,
        route_result,
    ) -> Dict[str, Any]:
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

    def _attach_local_topic(self, base: Dict[str, Any], topic_prediction: TopicPrediction) -> None:
        base.update(
            {
                "topic_local_label": topic_prediction.topic_label,
                "topic_local_group": topic_prediction.topic_group,
                "topic_local_name": topic_prediction.topic_name,
                "topic_local_confidence": topic_prediction.confidence,
                "topic_local_margin": topic_prediction.margin,
                "topic_local_top3": "; ".join(topic_prediction.top_candidates),
                "topic_family_local": topic_prediction.topic_group or UNKNOWN_TOPIC_GROUP,
                "topic_confidence": topic_prediction.confidence,
                "topic_margin": topic_prediction.margin,
                "topic_confidence_effective": topic_prediction.confidence,
                "topic_margin_effective": topic_prediction.margin,
                "topic_matches": "; ".join(topic_prediction.matched_terms),
                "topic_binary_score": topic_prediction.binary_score,
                "topic_binary_probability": topic_prediction.binary_probability,
            }
        )

    def _attach_bertopic_hint(self, base: Dict[str, Any], bertopic_signal: BERTopicSignal) -> None:
        cluster_quality = self._bertopic_cluster_quality(bertopic_signal)
        high_purity = int(cluster_quality == "high")
        true_outlier = int(bool(bertopic_signal.available and bertopic_signal.is_outlier))
        base.update(
            {
                "bertopic_status": bertopic_signal.status,
                "bertopic_topic_id": bertopic_signal.topic_id,
                "bertopic_topic_name": bertopic_signal.topic_name,
                "bertopic_probability": bertopic_signal.topic_probability,
                "bertopic_is_outlier": int(bertopic_signal.is_outlier),
                "bertopic_count": bertopic_signal.topic_count,
                "bertopic_pos_rate": bertopic_signal.topic_pos_rate,
                "bertopic_mapped_label": bertopic_signal.mapped_label,
                "bertopic_mapped_group": bertopic_signal.mapped_group,
                "bertopic_mapped_name": bertopic_signal.mapped_name,
                "bertopic_label_purity": bertopic_signal.label_purity,
                "bertopic_mapped_label_share": bertopic_signal.mapped_label_share,
                "bertopic_top_terms": bertopic_signal.top_terms,
                "bertopic_sample_titles": bertopic_signal.sample_titles,
                "bertopic_source_split": bertopic_signal.source_split,
                "bertopic_high_purity": high_purity,
                "bertopic_true_outlier": true_outlier,
                "bertopic_hint_label": bertopic_signal.mapped_label,
                "bertopic_hint_group": bertopic_signal.mapped_group,
                "bertopic_hint_name": bertopic_signal.mapped_name,
                "bertopic_cluster_quality": cluster_quality,
                "bertopic_dynamic_topic_id": bertopic_signal.topic_id,
                "bertopic_dynamic_topic_words": bertopic_signal.top_terms,
                "bertopic_primary_label": bertopic_signal.mapped_label,
                "bertopic_primary_group": bertopic_signal.mapped_group,
                "bertopic_primary_name": bertopic_signal.mapped_name,
                "bertopic_primary_probability": bertopic_signal.topic_probability,
                "bertopic_primary_support": bertopic_signal.topic_count,
                "bertopic_primary_purity": bertopic_signal.label_purity,
                "bertopic_primary_mapped_share": bertopic_signal.mapped_label_share,
                "bertopic_primary_override": 0,
                "bertopic_primary_reason": "deprecated_auxiliary_only" if bertopic_signal.mapped_label else "",
            }
        )

    def _bertopic_cluster_quality(self, bertopic_signal: BERTopicSignal) -> str:
        if not bertopic_signal.available:
            return "unavailable"
        if bertopic_signal.is_outlier:
            return "outlier"
        if (
            bertopic_signal.topic_count >= int(Config.BERTOPIC_PRIMARY_MIN_SUPPORT)
            and bertopic_signal.label_purity >= float(Config.BERTOPIC_PRIMARY_MIN_PURITY)
            and bertopic_signal.mapped_label_share >= float(Config.BERTOPIC_PRIMARY_MIN_MAPPED_SHARE)
        ):
            return "high"
        if bertopic_signal.topic_count >= 10 and bertopic_signal.label_purity >= 0.60:
            return "medium"
        return "low"

    def _rule_confidence(self, route_result) -> float:
        score_signal = min(max(float(route_result.topic_rule_score), 0.0) / 8.0, 1.0)
        margin_signal = min(max(float(route_result.topic_rule_margin), 0.0) / 4.0, 1.0)
        confidence = 0.18 + 0.48 * score_signal + 0.24 * margin_signal
        if route_result.rule_high_confidence:
            confidence += 0.08
        return round(min(max(confidence, 0.2), 0.99), 4)

    def _normalize_family_hint_value(self, value: Any) -> str:
        text = str(value or "").strip()
        if text.endswith(".0"):
            text = text[:-2]
        return text if text in {"0", "1"} else ""

    def _safe_float(self, value: Any, *, default: float = 0.0) -> float:
        if value in ("", None):
            return float(default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _extract_core_anchor_hits(self, *, title: str, abstract: str) -> list[str]:
        normalized_text = normalize_phrase(f"{title or ''} {abstract or ''}").replace("-", " ")
        if not normalized_text:
            return []
        hits: list[str] = []
        for anchor in CORE_RENEWAL_ANCHORS:
            normalized_anchor = normalize_phrase(anchor).replace("-", " ")
            if not normalized_anchor:
                continue
            if normalized_anchor in normalized_text and normalized_anchor not in hits:
                hits.append(normalized_anchor)
        return hits

    def _extract_broad_anchor_hits(self, *, title: str, abstract: str) -> list[str]:
        normalized_text = normalize_phrase(f"{title or ''} {abstract or ''}").replace("-", " ")
        if not normalized_text:
            return []
        hits: list[str] = []
        for anchor in COMMON_RENEWAL_ANCHORS:
            normalized_anchor = normalize_phrase(anchor).replace("-", " ")
            if not normalized_anchor:
                continue
            if normalized_anchor in normalized_text and normalized_anchor not in hits:
                hits.append(normalized_anchor)
        return hits

    def _is_urban_candidate_label(self, label: str) -> bool:
        token = str(label or "").strip()
        return bool(token) and topic_group_for_label(token) == "urban"

    def _collect_anchor_guard_urban_candidates(self, base: Dict[str, Any]) -> list[str]:
        candidates: list[str] = []
        for field in ("topic_within_family_label", "topic_local_label", "bertopic_hint_label"):
            label = str(base.get(field, "") or "").strip()
            if self._is_urban_candidate_label(label) and label not in candidates:
                candidates.append(label)
        return candidates

    def _build_anchor_guard_support_signals(
        self,
        base: Dict[str, Any],
        *,
        route_result,
        topic_prediction: TopicPrediction,
        bertopic_signal: BERTopicSignal,
    ) -> tuple[list[str], list[str]]:
        signals: list[str] = []
        urban_candidates = self._collect_anchor_guard_urban_candidates(base)
        if not urban_candidates:
            within_label, within_score, within_margin = self._select_topic_within_family(
                family="urban",
                candidate_final_topic=UNKNOWN_TOPIC_LABEL,
                route_result=route_result,
                topic_prediction=topic_prediction,
                bertopic_signal=bertopic_signal,
            )
            if (
                within_label != UNKNOWN_TOPIC_LABEL
                and within_score >= float(Config.URBAN_ANCHOR_GUARD_URBAN_WITHIN_SCORE_FLOOR)
                and within_margin >= float(Config.URBAN_ANCHOR_GUARD_URBAN_WITHIN_MARGIN_FLOOR)
            ):
                urban_candidates.append(within_label)
                signals.append(
                    f"urban_within_candidate:{within_label}@{within_score:.2f}/{within_margin:.2f}"
                )
        if urban_candidates:
            signals.append(f"urban_candidate:{urban_candidates[0]}")

        family_probability_urban = self._safe_float(base.get("family_probability_urban"))
        if family_probability_urban >= float(Config.URBAN_ANCHOR_GUARD_FAMILY_PROB_FLOOR):
            signals.append(f"family_probability_urban>={Config.URBAN_ANCHOR_GUARD_FAMILY_PROB_FLOOR:.2f}")

        topic_binary_probability = self._safe_float(base.get("topic_binary_probability"))
        if topic_binary_probability >= float(Config.URBAN_ANCHOR_GUARD_BINARY_PROB_FLOOR):
            signals.append(f"topic_binary_probability>={Config.URBAN_ANCHOR_GUARD_BINARY_PROB_FLOOR:.2f}")

        boundary_bucket = str(base.get("boundary_bucket", "") or "").strip()
        if boundary_bucket in ANCHOR_GUARD_BOUNDARY_BUCKETS:
            signals.append(f"boundary_bucket:{boundary_bucket}")
        return signals, urban_candidates

    def _apply_anchor_guard(
        self,
        base: Dict[str, Any],
        *,
        record: UrbanMetadataRecord,
        route_result,
        topic_prediction: TopicPrediction,
        bertopic_signal: BERTopicSignal,
        final_topic: str,
        decision_source: str,
        decision_reason: str,
        confidence: float,
        review_flag: int,
        review_reason: str,
    ) -> tuple[str, str, str, float, int, str]:
        if not bool(Config.URBAN_ANCHOR_GUARD_ENABLED):
            return final_topic, decision_source, decision_reason, confidence, review_flag, review_reason
        if topic_group_for_label(final_topic) != "nonurban":
            return final_topic, decision_source, decision_reason, confidence, review_flag, review_reason
        if str(route_result.reason or "").strip() in ANCHOR_GUARD_HARD_NEGATIVE_REASONS:
            return final_topic, decision_source, decision_reason, confidence, review_flag, review_reason

        anchor_hits = self._extract_core_anchor_hits(title=record.title, abstract=record.abstract)
        if not anchor_hits:
            return final_topic, decision_source, decision_reason, confidence, review_flag, review_reason

        signals, urban_candidates = self._build_anchor_guard_support_signals(
            base,
            route_result=route_result,
            topic_prediction=topic_prediction,
            bertopic_signal=bertopic_signal,
        )
        base["anchor_guard_flag"] = 1
        base["anchor_guard_hits"] = "; ".join(anchor_hits)
        has_support_signal = bool(signals)
        support_summary = "; ".join(signals) if signals else "support:none"

        if has_support_signal:
            promote_requires_candidate = bool(Config.URBAN_ANCHOR_GUARD_PROMOTE_REQUIRES_URBAN_CANDIDATE)
            promotion_label = urban_candidates[0] if urban_candidates else ""
            if not promotion_label and not promote_requires_candidate:
                fallback_rule = str(base.get("topic_rule", "") or "")
                if self._is_urban_candidate_label(fallback_rule):
                    promotion_label = fallback_rule

            if promotion_label:
                base["anchor_guard_action"] = "promote"
                base["anchor_guard_reason"] = support_summary
                decision_reason = (
                    f"{decision_reason};anchor_guard_core_anchor_promote:{promotion_label};{support_summary}"
                ).strip(";")
                return (
                    promotion_label,
                    "anchor_guard_promotion",
                    decision_reason,
                    round(max(float(confidence), 0.58), 4),
                    review_flag,
                    review_reason,
                )

            base["anchor_guard_action"] = "review"
            base["anchor_guard_reason"] = f"support_without_urban_candidate;{support_summary}"
            decision_reason = (
                f"{decision_reason};anchor_guard_core_anchor_support_without_urban_candidate"
            ).strip(";")
            review_reason = f"{review_reason};anchor_guard_support_without_urban_candidate".strip(";")
            return (
                UNKNOWN_TOPIC_LABEL,
                "anchor_guard_review",
                decision_reason,
                round(float(confidence), 4),
                1,
                review_reason,
            )

        base["anchor_guard_action"] = "review"
        base["anchor_guard_reason"] = "core_anchor_without_support"
        decision_reason = f"{decision_reason};anchor_guard_core_anchor_without_support".strip(";")
        review_reason = f"{review_reason};anchor_guard_core_anchor_without_support".strip(";")
        return (
            UNKNOWN_TOPIC_LABEL,
            "anchor_guard_review",
            decision_reason,
            round(float(confidence), 4),
            1,
            review_reason,
        )

    def _decision_reason_is_uncertain_nonurban_candidate(self, decision_reason: str) -> bool:
        token = str(decision_reason or "").strip()
        return token.startswith("rule_local_agree:") or token.startswith("rule_high_conflict_override_local:")

    def _n8_has_renewal_semantics(self, *, record: UrbanMetadataRecord, base: Dict[str, Any]) -> tuple[bool, str]:
        core_hits = self._extract_core_anchor_hits(title=record.title, abstract=record.abstract)
        if core_hits:
            return True, f"core_anchor:{core_hits[0]}"
        broad_hits = self._extract_broad_anchor_hits(title=record.title, abstract=record.abstract)
        if broad_hits:
            return True, f"broad_anchor:{broad_hits[0]}"
        risk_tags = str(base.get("stage1_risk_tags", "") or "").strip().lower()
        if "explicit_renewal_wording_but_other_object" in risk_tags:
            return True, "risk_tag:explicit_renewal_wording_but_other_object"
        return False, ""

    def _apply_uncertain_nonurban_guard(
        self,
        base: Dict[str, Any],
        *,
        record: UrbanMetadataRecord,
        route_result,
        topic_prediction: TopicPrediction,
        bertopic_signal: BERTopicSignal,
        final_topic: str,
        decision_source: str,
        decision_reason: str,
        confidence: float,
        review_flag: int,
        review_reason: str,
    ) -> tuple[str, str, str, float, int, str]:
        if not bool(Config.URBAN_UNCERTAIN_NONURBAN_GUARD_ENABLED):
            return final_topic, decision_source, decision_reason, confidence, review_flag, review_reason
        if topic_group_for_label(final_topic) != "nonurban":
            return final_topic, decision_source, decision_reason, confidence, review_flag, review_reason
        if str(route_result.reason or "").strip() in UNCERTAIN_NONURBAN_HARD_NEGATIVE_REASONS:
            return final_topic, decision_source, decision_reason, confidence, review_flag, review_reason
        if int(bool(base.get("review_flag_rule", 0))) != 1:
            return final_topic, decision_source, decision_reason, confidence, review_flag, review_reason
        if str(decision_source or "").strip() != "rule_model_fusion":
            return final_topic, decision_source, decision_reason, confidence, review_flag, review_reason
        if not self._decision_reason_is_uncertain_nonurban_candidate(decision_reason):
            return final_topic, decision_source, decision_reason, confidence, review_flag, review_reason

        base["uncertain_nonurban_guard_flag"] = 1
        rule_label = str(base.get("topic_rule", "") or route_result.topic_rule or "").strip()
        boundary_bucket = str(base.get("boundary_bucket", "") or "").strip()

        within_label = str(base.get("topic_within_family_label", "") or "").strip()
        within_score = self._safe_float(base.get("topic_family_within_score"))
        within_margin = self._safe_float(base.get("topic_family_within_margin"))
        within_threshold = (
            self._is_urban_candidate_label(within_label)
            and within_score >= float(Config.URBAN_UNCERTAIN_NONURBAN_PROMOTE_WITHIN_SCORE_FLOOR)
            and within_margin >= float(Config.URBAN_UNCERTAIN_NONURBAN_PROMOTE_WITHIN_MARGIN_FLOOR)
        )

        local_label = str(base.get("topic_local_label", "") or "").strip()
        local_confidence = self._safe_float(base.get("topic_local_confidence"))
        local_margin = self._safe_float(base.get("topic_local_margin"))
        local_threshold = (
            self._is_urban_candidate_label(local_label)
            and local_confidence >= float(Config.URBAN_UNCERTAIN_NONURBAN_PROMOTE_LOCAL_CONF_FLOOR)
            and local_margin >= float(Config.URBAN_UNCERTAIN_NONURBAN_PROMOTE_LOCAL_MARGIN_FLOOR)
        )

        family_probability_urban = self._safe_float(base.get("family_probability_urban"))
        family_bucket_threshold = (
            family_probability_urban >= float(Config.URBAN_UNCERTAIN_NONURBAN_PROMOTE_FAMILY_PROB_FLOOR)
            and boundary_bucket in UNCERTAIN_NONURBAN_PROMOTE_BOUNDARY_BUCKETS
        )

        promote_reasons: list[str] = []
        promotion_label = ""
        if within_threshold:
            promotion_label = within_label
            promote_reasons.append(
                f"within_family:{within_label}@{within_score:.2f}/{within_margin:.2f}"
            )
        if local_threshold:
            if not promotion_label:
                promotion_label = local_label
            promote_reasons.append(
                f"local_urban:{local_label}@{local_confidence:.2f}/{local_margin:.2f}"
            )
        if family_bucket_threshold:
            promote_reasons.append(
                f"family_prob_bucket:{family_probability_urban:.2f}:{boundary_bucket}"
            )
            if not promotion_label:
                fallback_label, fallback_score, fallback_margin = self._select_topic_within_family(
                    family="urban",
                    candidate_final_topic=UNKNOWN_TOPIC_LABEL,
                    route_result=route_result,
                    topic_prediction=topic_prediction,
                    bertopic_signal=bertopic_signal,
                )
                if fallback_label != UNKNOWN_TOPIC_LABEL:
                    promotion_label = fallback_label
                    promote_reasons.append(
                        f"fallback_within:{fallback_label}@{fallback_score:.2f}/{fallback_margin:.2f}"
                    )
        if promote_reasons and promotion_label:
            evidence = "; ".join(promote_reasons)
            base["uncertain_nonurban_guard_action"] = "promote"
            base["uncertain_nonurban_guard_reason"] = evidence
            base["uncertain_nonurban_guard_evidence"] = evidence
            decision_reason = (
                f"{decision_reason};uncertain_nonurban_promote:{promotion_label};{evidence}"
            ).strip(";")
            return (
                promotion_label,
                "uncertain_nonurban_promotion",
                decision_reason,
                round(max(float(confidence), 0.58), 4),
                review_flag,
                review_reason,
            )

        n8_has_renewal_signal, n8_evidence = self._n8_has_renewal_semantics(record=record, base=base)
        high_risk_nonurban = rule_label in UNCERTAIN_NONURBAN_HIGH_RISK_RULES
        if rule_label == "N8" and n8_has_renewal_signal:
            high_risk_nonurban = True
        if high_risk_nonurban:
            reasons = [f"high_risk_rule:{rule_label or 'Unknown'}"]
            if n8_evidence:
                reasons.append(n8_evidence)
            evidence = "; ".join(reasons)
            base["uncertain_nonurban_guard_action"] = "review"
            base["uncertain_nonurban_guard_reason"] = evidence
            base["uncertain_nonurban_guard_evidence"] = evidence
            base["unknown_recovery_path"] = "uncertain_nonurban_guard_review"
            base["unknown_recovery_evidence"] = evidence
            decision_reason = f"{decision_reason};uncertain_nonurban_review:{evidence}".strip(";")
            review_reason = f"{review_reason};uncertain_nonurban_review".strip(";")
            return (
                UNKNOWN_TOPIC_LABEL,
                "uncertain_nonurban_review",
                decision_reason,
                round(float(confidence), 4),
                1,
                review_reason,
            )

        base["uncertain_nonurban_guard_action"] = "keep_0"
        base["uncertain_nonurban_guard_reason"] = f"rule={rule_label or 'Unknown'} not in high_risk_bucket"
        base["uncertain_nonurban_guard_evidence"] = str(base["uncertain_nonurban_guard_reason"])
        return final_topic, decision_source, decision_reason, confidence, review_flag, review_reason

    def _extract_stage1_risk_tags(self, base: Dict[str, Any]) -> set[str]:
        raw = base.get("stage1_risk_tags", "")
        if isinstance(raw, (list, tuple, set)):
            return {str(item).strip() for item in raw if str(item).strip()}
        text = str(raw or "")
        return {item.strip() for item in text.split(";") if item.strip()}

    def _source_probability(self, value: Any, *, default: float = 0.5) -> float:
        if value in ("", None):
            return float(default)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float(default)
        return min(max(number, 0.0), 1.0)

    def _topic_family_vote_probability(self, base: Dict[str, Any], *, final_topic: str) -> float:
        labels = [
            base.get("topic_rule", ""),
            base.get("topic_local_label", ""),
            base.get("topic_within_family_label", ""),
            base.get("bertopic_hint_label", ""),
            final_topic,
        ]
        votes: list[float] = []
        for label in labels:
            group = topic_group_for_label(str(label or "").strip())
            if group == "urban":
                votes.append(1.0)
            elif group == "nonurban":
                votes.append(0.0)
            else:
                votes.append(0.5)
        return round(sum(votes) / len(votes), 6) if votes else 0.5

    def _has_binary_phrase_hit(self, *, title: str, abstract: str, phrases: tuple[str, ...]) -> bool:
        normalized_text = normalize_phrase(f"{title or ''} {abstract or ''}").replace("-", " ")
        if not normalized_text:
            return False
        for phrase in phrases:
            normalized_phrase = normalize_phrase(phrase).replace("-", " ")
            if normalized_phrase and normalized_phrase in normalized_text:
                return True
        return False

    def _binary_phrase_hits(self, *, title: str, abstract: str, phrases: tuple[str, ...]) -> list[str]:
        normalized_text = normalize_phrase(f"{title or ''} {abstract or ''}").replace("-", " ")
        if not normalized_text:
            return []
        hits: list[str] = []
        for phrase in phrases:
            normalized_phrase = normalize_phrase(phrase).replace("-", " ")
            if normalized_phrase and normalized_phrase in normalized_text:
                hits.append(normalized_phrase)
        return hits

    def _has_negated_renewal_context(self, *, title: str, abstract: str) -> bool:
        normalized_text = normalize_phrase(f"{title or ''} {abstract or ''}").replace("-", " ")
        negated_patterns = (
            "without renewal",
            "without any renewal",
            "without urban renewal",
            "without regeneration",
            "without redevelopment",
            "no renewal",
            "no urban renewal",
            "no regeneration",
            "no redevelopment",
            "not renewal",
            "not regeneration",
            "not redevelopment",
            "without intervention",
            "without intervention evidence",
            "without project evidence",
            "without redevelopment intervention",
            "without renewal intervention",
        )
        return any(pattern in normalized_text for pattern in negated_patterns)

    def _anchor_probability(self, *, record: UrbanMetadataRecord, base: Dict[str, Any], final_topic: str) -> tuple[float, str]:
        negated_renewal = self._has_negated_renewal_context(title=record.title, abstract=record.abstract)
        core_hits = self._extract_core_anchor_hits(title=record.title, abstract=record.abstract)
        if core_hits and not negated_renewal:
            return 0.75, f"core_anchor:{core_hits[0]}"

        broad_hits = self._extract_broad_anchor_hits(title=record.title, abstract=record.abstract)
        object_hit = self._has_binary_phrase_hit(
            title=record.title,
            abstract=record.abstract,
            phrases=COMMON_EXISTING_URBAN_OBJECTS,
        )
        mechanism_hit = self._has_binary_phrase_hit(
            title=record.title,
            abstract=record.abstract,
            phrases=R7_STRONG_RENEWAL_MECHANISMS,
        )
        if broad_hits and not negated_renewal:
            return 0.65, f"broad_anchor:{broad_hits[0]}"
        if object_hit:
            return 0.65, "existing_urban_object"
        if mechanism_hit:
            return 0.65, "renewal_mechanism"

        risk_tags = self._extract_stage1_risk_tags(base)
        if RISK_GENERIC_TECHNICAL in risk_tags and str(final_topic or "") == "N8":
            return 0.45, "technical_method_risk"
        return 0.50, "neutral"

    def _binary_recall_context(
        self,
        *,
        record: UrbanMetadataRecord,
        base: Dict[str, Any],
        final_topic: str,
    ) -> Dict[str, Any]:
        negated_renewal = self._has_negated_renewal_context(title=record.title, abstract=record.abstract)
        core_hits = [] if negated_renewal else self._extract_core_anchor_hits(title=record.title, abstract=record.abstract)
        broad_hits = [] if negated_renewal else self._extract_broad_anchor_hits(title=record.title, abstract=record.abstract)
        object_hit = self._has_binary_phrase_hit(
            title=record.title,
            abstract=record.abstract,
            phrases=COMMON_EXISTING_URBAN_OBJECTS,
        )
        mechanism_hit = self._has_binary_phrase_hit(
            title=record.title,
            abstract=record.abstract,
            phrases=R7_STRONG_RENEWAL_MECHANISMS,
        )
        urban_context_hit = self._has_binary_phrase_hit(
            title=record.title,
            abstract=record.abstract,
            phrases=BINARY_RECALL_URBAN_CONTEXT_TERMS,
        )
        topic_labels = {
            "rule": base.get("topic_rule", ""),
            "local": base.get("topic_local_label", ""),
            "within": base.get("topic_within_family_label", ""),
            "bertopic": base.get("bertopic_hint_label", ""),
            "final": final_topic,
        }
        urban_topic_sources = [
            source
            for source, label in topic_labels.items()
            if topic_group_for_label(str(label or "").strip()) == "urban"
        ]
        nonurban_topic_sources = [
            source
            for source, label in topic_labels.items()
            if topic_group_for_label(str(label or "").strip()) == "nonurban"
        ]
        return {
            "negated_renewal": bool(negated_renewal),
            "core_anchor": bool(core_hits),
            "broad_anchor": bool(broad_hits),
            "object_anchor": bool(object_hit),
            "mechanism_anchor": bool(mechanism_hit),
            "urban_context": bool(urban_context_hit),
            "final_topic_urban": topic_group_for_label(str(final_topic or "").strip()) == "urban",
            "any_urban_topic": bool(urban_topic_sources),
            "urban_topic_sources": urban_topic_sources,
            "nonurban_topic_sources": nonurban_topic_sources,
        }

    def _open_set_urban_evidence(
        self,
        *,
        record: UrbanMetadataRecord,
        base: Dict[str, Any],
        final_topic: str,
    ) -> tuple[bool, str, str]:
        if self._has_negated_renewal_context(title=record.title, abstract=record.abstract):
            return False, "negated_renewal_context", "negated renewal wording"

        risk_tags = self._extract_stage1_risk_tags(base)
        high_risk_tags = {
            RISK_GENERIC_TECHNICAL,
            RISK_GREENFIELD_EXPANSION,
            RISK_SOCIAL_HISTORY_MEDIA,
            RISK_BACKGROUND_SUPPORT,
        }
        has_high_risk = bool(risk_tags & high_risk_tags)
        title_core_hits = self._extract_core_anchor_hits(title=record.title, abstract="")
        core_hits = self._extract_core_anchor_hits(title=record.title, abstract=record.abstract)
        if title_core_hits:
            return True, "core_renewal_anchor", "; ".join(title_core_hits[:5])

        broad_hits = self._extract_broad_anchor_hits(title=record.title, abstract=record.abstract)
        action_hits = self._binary_phrase_hits(
            title=record.title,
            abstract=record.abstract,
            phrases=OPEN_SET_RENEWAL_ACTION_TERMS,
        )
        object_hits = self._binary_phrase_hits(
            title=record.title,
            abstract=record.abstract,
            phrases=OPEN_SET_EXISTING_BUILT_ENV_TERMS,
        )
        urban_context_hits = self._binary_phrase_hits(
            title=record.title,
            abstract=record.abstract,
            phrases=BINARY_RECALL_URBAN_CONTEXT_TERMS,
        )
        title_action_hits = self._binary_phrase_hits(
            title=record.title,
            abstract="",
            phrases=OPEN_SET_RENEWAL_ACTION_TERMS,
        )
        title_object_hits = self._binary_phrase_hits(
            title=record.title,
            abstract="",
            phrases=OPEN_SET_EXISTING_BUILT_ENV_TERMS,
        )
        policy_hits = self._binary_phrase_hits(
            title=record.title,
            abstract=record.abstract,
            phrases=OPEN_SET_POLICY_INTERVENTION_TERMS,
        )
        if core_hits and (title_object_hits or title_action_hits or (object_hits and policy_hits)) and not has_high_risk:
            evidence = f"core={','.join(core_hits[:4])};objects={','.join(object_hits[:4])}"
            return True, "core_renewal_anchor", evidence

        if action_hits and object_hits and (title_action_hits or title_object_hits) and not has_high_risk:
            evidence = f"actions={','.join(action_hits[:4])};objects={','.join(object_hits[:4])}"
            return True, "renewal_action_existing_urban_object", evidence

        if policy_hits and action_hits and object_hits and (title_action_hits or title_object_hits) and not has_high_risk:
            evidence = (
                f"policy={','.join(policy_hits[:4])};"
                f"renewal={','.join(broad_hits[:4])};"
                f"objects={','.join(object_hits[:4])}"
            )
            return True, "policy_project_intervention_built_environment", evidence

        llm_hint = self._normalize_family_hint_value(base.get("llm_family_hint", ""))
        family_probability = self._source_probability(base.get("family_probability_urban"), default=0.0)
        family_positive = (
            str(base.get("family_predicted_family", "") or "") == "urban"
            and family_probability >= float(Config.URBAN_OPEN_SET_FAMILY_PROB_FLOOR)
        )
        auxiliary_context = bool(broad_hits or object_hits or action_hits)
        if (
            topic_group_for_label(final_topic) == UNKNOWN_TOPIC_GROUP
            and (llm_hint == "1" or family_positive)
            and auxiliary_context
            and action_hits
            and object_hits
            and (title_action_hits or title_object_hits)
            and not has_high_risk
        ):
            evidence = (
                f"llm_family_hint={llm_hint or 'missing'};"
                f"family_probability={family_probability:.4f};"
                f"context={','.join((broad_hits + object_hits + action_hits)[:5])}"
            )
            return True, "family_or_llm_positive_unmapped", evidence

        return False, "no_open_set_urban_evidence", ""

    def _apply_open_set_topic(
        self,
        base: Dict[str, Any],
        *,
        record: UrbanMetadataRecord,
        route_result,
        final_topic: str,
        decision_source: str,
        decision_reason: str,
        confidence: float,
        review_flag: int,
        review_reason: str,
    ) -> tuple[str, str, str, float, int, str]:
        if not bool(Config.URBAN_OPEN_SET_ENABLED):
            return final_topic, decision_source, decision_reason, confidence, review_flag, review_reason

        route_reason = str(getattr(route_result, "reason", "") or "").strip()
        if route_reason in BINARY_HARD_NEGATIVE_REASONS:
            base.update(
                {
                    "open_set_flag": 0,
                    "open_set_topic": "",
                    "open_set_reason": "hard_negative",
                    "open_set_evidence": route_reason,
                    "taxonomy_coverage_status": "hard_negative",
                }
            )
            return final_topic, decision_source, decision_reason, confidence, review_flag, review_reason

        topic_group = topic_group_for_label(final_topic)
        if topic_group in {"urban", "nonurban"} and final_topic not in {OPEN_SET_URBAN_LABEL, OPEN_SET_NONURBAN_LABEL}:
            base["taxonomy_coverage_status"] = "covered"
            return final_topic, decision_source, decision_reason, confidence, review_flag, review_reason

        has_evidence, reason, evidence = self._open_set_urban_evidence(
            record=record,
            base=base,
            final_topic=final_topic,
        )
        if has_evidence:
            base.update(
                {
                    "open_set_flag": 1,
                    "open_set_topic": OPEN_SET_URBAN_LABEL,
                    "open_set_reason": reason,
                    "open_set_evidence": evidence,
                    "taxonomy_coverage_status": "open_set",
                }
            )
            decision_reason = f"{decision_reason};open_set_urban_other:{reason}".strip(";")
            review_reason = f"{review_reason};open_set_topic".strip(";")
            return (
                OPEN_SET_URBAN_LABEL,
                "open_set_recovery",
                decision_reason,
                max(float(confidence), 0.55),
                1,
                review_reason,
            )

        if final_topic == UNKNOWN_TOPIC_LABEL:
            base["taxonomy_coverage_status"] = "unknown"
        elif final_topic == OPEN_SET_NONURBAN_LABEL:
            base["taxonomy_coverage_status"] = "open_set"
        return final_topic, decision_source, decision_reason, confidence, review_flag, review_reason

    def _binary_recall_blocked_by_risk(
        self,
        *,
        base: Dict[str, Any],
        final_topic: str,
        context: Dict[str, Any],
    ) -> tuple[bool, str]:
        risk_tags = self._extract_stage1_risk_tags(base)
        final_label = str(final_topic or "").strip()
        substantive_anchor = bool(
            context.get("core_anchor")
            or context.get("broad_anchor")
            or context.get("object_anchor")
        )
        if RISK_GENERIC_TECHNICAL in risk_tags and final_label == "N8" and not substantive_anchor:
            return True, "generic_technical_n8_without_substantive_anchor"
        if RISK_GREENFIELD_EXPANSION in risk_tags and not (
            context.get("core_anchor") or context.get("broad_anchor")
        ):
            return True, "greenfield_expansion_without_renewal_anchor"
        return False, "none"

    def _apply_binary_recall_calibration(
        self,
        *,
        base: Dict[str, Any],
        raw_score: float,
        context: Dict[str, Any],
        final_topic: str,
        decision_source: str,
    ) -> tuple[float, int, str, str]:
        if not bool(Config.URBAN_BINARY_RECALL_CALIBRATION_ENABLED):
            return raw_score, 0, "none", "disabled"

        blocked, blocked_reason = self._binary_recall_blocked_by_risk(
            base=base,
            final_topic=final_topic,
            context=context,
        )
        if blocked:
            return raw_score, 0, "blocked", blocked_reason

        candidates: list[tuple[float, str, str]] = []
        if context.get("final_topic_urban"):
            candidates.append(
                (
                    float(Config.URBAN_BINARY_RECALL_FINAL_TOPIC_URBAN_FLOOR),
                    "final_topic_urban_floor",
                    f"final_topic={final_topic}",
                )
            )
        if context.get("core_anchor"):
            candidates.append(
                (
                    float(Config.URBAN_BINARY_RECALL_CORE_ANCHOR_FLOOR),
                    "core_anchor_floor",
                    "core_anchor",
                )
            )
        if context.get("broad_anchor") and (
            context.get("urban_context") or context.get("any_urban_topic") or context.get("object_anchor")
        ):
            candidates.append(
                (
                    float(Config.URBAN_BINARY_RECALL_BROAD_ANCHOR_FLOOR),
                    "broad_anchor_context_floor",
                    "broad_anchor_with_urban_context",
                )
            )
        if context.get("any_urban_topic") and (
            context.get("object_anchor") or context.get("mechanism_anchor") or context.get("urban_context")
        ):
            candidates.append(
                (
                    float(Config.URBAN_BINARY_RECALL_URBAN_EVIDENCE_FLOOR),
                    "urban_evidence_context_floor",
                    "urban_topic_evidence_with_context",
                )
            )
        if (
            raw_score >= float(Config.URBAN_BINARY_RECALL_CONTEXT_SCORE_FLOOR)
            and (context.get("urban_context") or context.get("any_urban_topic") or context.get("object_anchor") or context.get("mechanism_anchor"))
            and not context.get("negated_renewal")
        ):
            candidates.append(
                (
                    float(Config.URBAN_BINARY_RECALL_CONTEXT_POSITIVE_FLOOR),
                    "context_relevance_floor",
                    f"raw_score>={float(Config.URBAN_BINARY_RECALL_CONTEXT_SCORE_FLOOR):.3f}",
                )
            )
        if decision_source in {"anchor_guard_promotion", "uncertain_nonurban_promotion", "family_gate_recovery"}:
            candidates.append((0.62, "guard_promotion_floor", decision_source))

        if not candidates:
            return raw_score, 0, "none", "no_recall_evidence"

        floor, tier, reason = max(candidates, key=lambda item: item[0])
        calibrated = max(raw_score, floor)
        if calibrated <= raw_score:
            return raw_score, 0, "none", "score_already_above_recall_floor"
        sources = ",".join(context.get("urban_topic_sources") or [])
        detail = reason if not sources else f"{reason};urban_sources={sources}"
        return calibrated, 1, tier, detail

    def _llm_hint_probability(self, base: Dict[str, Any]) -> float:
        hint = self._normalize_family_hint_value(base.get("llm_family_hint", ""))
        if hint == "1":
            return 1.0
        if hint == "0":
            return 0.0
        return 0.5

    def _risk_adjustment(self, base: Dict[str, Any]) -> tuple[float, str]:
        risk_tags = self._extract_stage1_risk_tags(base)
        applied: list[str] = []
        total = 0.0
        for tag, adjustment in BINARY_RISK_ADJUSTMENTS.items():
            if tag in risk_tags:
                total += adjustment
                applied.append(f"{tag}:{adjustment:+.2f}")
        return total, ",".join(applied) if applied else "none"

    def _decision_adjustment(self, *, decision_source: str, decision_reason: str) -> tuple[float, str]:
        source = str(decision_source or "").strip()
        reason = str(decision_reason or "").strip()
        if source in {"anchor_guard_promotion", "uncertain_nonurban_promotion", "family_gate_recovery"}:
            return 0.16, source
        if source == "unknown_hint_resolution" and reason.startswith("rule_unknown_local_llm_family_nonurban_weak:"):
            return -0.10, "weak_nonurban_hint"
        return 0.0, "none"

    def _binary_topic_consistency_flag(self, *, binary_label: str, final_topic: str) -> int:
        topic_group = topic_group_for_label(final_topic)
        if topic_group == UNKNOWN_TOPIC_GROUP:
            return 1
        if binary_label == "1" and topic_group != "urban":
            return 1
        if binary_label == "0" and topic_group == "urban":
            return 1
        return 0

    def _apply_binary_decision(
        self,
        base: Dict[str, Any],
        *,
        record: UrbanMetadataRecord,
        route_result,
        final_topic: str,
        decision_source: str,
        decision_reason: str,
        confidence: float,
        review_flag: int,
        review_reason: str,
    ) -> tuple[str, float, int, str]:
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
            consistency_flag = self._binary_topic_consistency_flag(
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

        family_probability = self._source_probability(base.get("family_probability_urban"), default=0.5)
        topic_binary_probability = self._source_probability(base.get("topic_binary_probability"), default=0.5)
        topic_vote_probability = self._topic_family_vote_probability(base, final_topic=final_topic)
        anchor_probability, anchor_evidence = self._anchor_probability(
            record=record,
            base=base,
            final_topic=final_topic,
        )
        llm_probability = self._llm_hint_probability(base)
        risk_adjustment, risk_evidence = self._risk_adjustment(base)
        decision_adjustment, decision_adjustment_evidence = self._decision_adjustment(
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
        recall_context = self._binary_recall_context(
            record=record,
            base=base,
            final_topic=final_topic,
        )
        score, recall_flag, recall_tier, recall_reason = self._apply_binary_recall_calibration(
            base=base,
            raw_score=raw_score,
            context=recall_context,
            final_topic=final_topic,
            decision_source=decision_source,
        )
        score = round(min(max(score, 0.02), 0.98), 6)
        binary_label = "1" if score >= threshold else "0"
        decision_confidence = round(score if binary_label == "1" else 1.0 - score, 6)
        consistency_flag = self._binary_topic_consistency_flag(
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

    def _merge_rule_review_signal(
        self,
        *,
        base: Dict[str, Any],
        final_topic: str,
        review_flag: int,
        review_reason: str,
    ) -> tuple[int, str]:
        current_flag = int(bool(review_flag))
        current_reason = str(review_reason or "")
        if final_topic == UNKNOWN_TOPIC_LABEL:
            if current_flag == 0:
                current_flag = 1
                current_reason = current_reason or str(base.get("review_reason_rule", "") or "unknown_review")
            return current_flag, current_reason

        if current_flag:
            return current_flag, current_reason

        rule_flag = int(bool(base.get("review_flag_rule", 0)))
        if rule_flag:
            return 1, current_reason or str(base.get("review_reason_rule", "") or "")
        return 0, ""

    def _candidate_scores_from_prediction(self, topic_prediction: TopicPrediction) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        raw_scores = getattr(topic_prediction, "scored_topics", None) or []
        for label, score in raw_scores:
            scores[str(label)] = float(score)
        if scores:
            return scores
        for item in getattr(topic_prediction, "top_candidates", []) or []:
            label, _, score = str(item).partition(":")
            try:
                scores[label] = float(score)
            except ValueError:
                continue
        return scores

    def _select_topic_within_family(
        self,
        *,
        family: str,
        candidate_final_topic: str,
        route_result,
        topic_prediction: TopicPrediction,
        bertopic_signal: BERTopicSignal,
    ) -> tuple[str, float, float]:
        if family not in {"urban", "nonurban"}:
            return UNKNOWN_TOPIC_LABEL, 0.0, 0.0

        scores = self._candidate_scores_from_prediction(topic_prediction)
        if candidate_final_topic != UNKNOWN_TOPIC_LABEL and topic_group_for_label(candidate_final_topic) == family:
            scores[candidate_final_topic] = max(
                scores.get(candidate_final_topic, float("-inf")),
                float(topic_prediction.confidence) * 8.0,
            )
        rule_label = route_result.topic_rule or UNKNOWN_TOPIC_LABEL
        if rule_label != UNKNOWN_TOPIC_LABEL and topic_group_for_label(rule_label) == family:
            scores[rule_label] = max(scores.get(rule_label, float("-inf")), float(route_result.topic_rule_score or 0.0))
        bertopic_label = str(bertopic_signal.mapped_label or "")
        if bertopic_label and topic_group_for_label(bertopic_label) == family:
            bonus = 0.0
            if self._bertopic_cluster_quality(bertopic_signal) == "high":
                bonus = 0.8
            elif self._bertopic_cluster_quality(bertopic_signal) == "medium":
                bonus = 0.45
            scores[bertopic_label] = max(scores.get(bertopic_label, 0.0), scores.get(bertopic_label, 0.0) + bonus)

        family_scores = sorted(
            (
                (label, float(score))
                for label, score in scores.items()
                if label != UNKNOWN_TOPIC_LABEL and topic_group_for_label(label) == family
            ),
            key=lambda item: (-item[1], item[0]),
        )
        if not family_scores:
            return UNKNOWN_TOPIC_LABEL, 0.0, 0.0

        top_label, top_score = family_scores[0]
        second_score = family_scores[1][1] if len(family_scores) > 1 else 0.0
        margin = round(max(top_score - second_score, 0.0), 4)
        if top_score < 2.8 or margin < 1.0:
            return UNKNOWN_TOPIC_LABEL, round(top_score, 4), margin
        return top_label, round(top_score, 4), margin

    def _resolve_unknown_with_family_consensus(
        self,
        *,
        route_result,
        topic_prediction: TopicPrediction,
        bertopic_signal: BERTopicSignal,
        llm_hint: str,
        bertopic_urban_support: bool,
        bertopic_nonurban_support: bool,
    ) -> Optional[Dict[str, Any]]:
        preferred_family = {"1": "urban", "0": "nonurban"}.get(llm_hint)
        if preferred_family is None:
            return None

        within_label, within_score, within_margin = self._select_topic_within_family(
            family=preferred_family,
            candidate_final_topic=UNKNOWN_TOPIC_LABEL,
            route_result=route_result,
            topic_prediction=topic_prediction,
            bertopic_signal=bertopic_signal,
        )
        if within_label == UNKNOWN_TOPIC_LABEL:
            return None

        rule_label = route_result.topic_rule or UNKNOWN_TOPIC_LABEL
        local_label = topic_prediction.topic_label or UNKNOWN_TOPIC_LABEL
        rule_group = route_result.topic_rule_group or topic_group_for_label(rule_label)
        local_group = topic_prediction.topic_group or topic_group_for_label(local_label)

        aligned_sources = ["llm"]
        if rule_group == preferred_family:
            aligned_sources.append("rule")
        if local_group == preferred_family:
            aligned_sources.append("local")
        if preferred_family == "urban" and bertopic_urban_support:
            aligned_sources.append("bertopic")
        if preferred_family == "nonurban" and bertopic_nonurban_support:
            aligned_sources.append("bertopic")

        if len(aligned_sources) < 2:
            return None
        has_structural_support = rule_group == preferred_family or (
            preferred_family == "urban" and bertopic_urban_support
        ) or (
            preferred_family == "nonurban" and bertopic_nonurban_support
        )
        if not has_structural_support:
            return None
        if within_margin < UNKNOWN_RECOVERY_CONSENSUS_MARGIN_FLOOR:
            return None

        score_floor = UNKNOWN_RECOVERY_CONSENSUS_SCORE_FLOOR
        if rule_group == preferred_family and within_label == rule_label:
            score_floor = UNKNOWN_RECOVERY_CONSENSUS_STRONG_SCORE_FLOOR
        if within_score < score_floor:
            return None
        if within_label not in {rule_label, local_label} and len(aligned_sources) < 3:
            return None

        confidence = 0.64
        if within_label == rule_label:
            confidence = max(confidence, self._rule_confidence(route_result))
        if within_label == local_label:
            confidence = max(confidence, float(topic_prediction.confidence))

        evidence = (
            f"family={preferred_family};sources={'+'.join(aligned_sources)};"
            f"within={within_label};score={within_score:.2f};margin={within_margin:.2f}"
        )
        return {
            "final_topic": within_label,
            "decision_source": "unknown_hint_resolution",
            "decision_reason": f"llm_family_consensus_within_family:{within_label}",
            "confidence": round(confidence, 4),
            "recovery_path": "unknown_hint_consensus",
            "recovery_evidence": evidence,
        }

    def _has_signal(self, values: Any, needle: str) -> bool:
        token = str(needle or "").strip().lower()
        if not token:
            return False
        for value in values or []:
            if token in str(value or "").strip().lower():
                return True
        return False

    def _offline_unknown_resolution_payload(
        self,
        *,
        final_topic: str,
        confidence: float,
        decision_reason: str,
        recovery_path: str,
        recovery_evidence: str,
    ) -> Dict[str, Any]:
        return {
            "final_topic": final_topic,
            "decision_source": "unknown_hint_resolution",
            "decision_reason": decision_reason,
            "confidence": round(float(confidence), 4),
            "recovery_path": recovery_path,
            "recovery_evidence": recovery_evidence,
        }

    def _resolve_unknown_with_offline_signals(
        self,
        *,
        route_result,
        topic_prediction: TopicPrediction,
        bertopic_signal: BERTopicSignal,
    ) -> Optional[Dict[str, Any]]:
        rule_label = route_result.topic_rule or UNKNOWN_TOPIC_LABEL
        local_label = topic_prediction.topic_label or UNKNOWN_TOPIC_LABEL
        rule_group = route_result.topic_rule_group or topic_group_for_label(rule_label)
        local_group = topic_prediction.topic_group or topic_group_for_label(local_label)
        rule_confidence = self._rule_confidence(route_result)
        local_confidence = float(topic_prediction.confidence)
        local_margin = float(topic_prediction.margin)
        binary_probability = float(topic_prediction.binary_probability)
        risk_tags = route_result.stage1_risk_tags or []
        positive_signals = route_result.matched_positive_signals or route_result.topic_rule_matches or []
        boundary_bucket, conflict_pattern = self.family_gate.describe_conflict(
            rule_label=str(rule_label or ""),
            local_label=str(local_label or ""),
            rule_family=rule_group,
            local_family=local_group,
        )
        bertopic_label = str(bertopic_signal.mapped_label or "")
        bertopic_high_purity = bertopic_label == rule_label and float(bertopic_signal.label_purity or 0.0) >= 0.80

        def build_rule_payload(trigger: str, *, min_confidence: float = 0.66) -> Dict[str, Any]:
            evidence = (
                f"pattern={conflict_pattern};bucket={boundary_bucket};trigger={trigger};"
                f"rule={rule_label};rule_score={float(route_result.topic_rule_score or 0.0):.2f};"
                f"rule_margin={float(route_result.topic_rule_margin or 0.0):.2f};"
                f"binary={binary_probability:.4f}"
            )
            return self._offline_unknown_resolution_payload(
                final_topic=rule_label,
                confidence=max(rule_confidence, min_confidence),
                decision_reason=f"offline_unknown_curated_rule:{trigger}:{rule_label}",
                recovery_path="unknown_offline_curated_rule",
                recovery_evidence=evidence,
            )

        def build_local_payload(trigger: str, *, min_confidence: float = 0.68) -> Dict[str, Any]:
            evidence = (
                f"pattern={conflict_pattern};bucket={boundary_bucket};trigger={trigger};"
                f"local={local_label};local_confidence={local_confidence:.4f};"
                f"local_margin={local_margin:.4f};binary={binary_probability:.4f}"
            )
            return self._offline_unknown_resolution_payload(
                final_topic=local_label,
                confidence=max(local_confidence, min_confidence),
                decision_reason=f"offline_unknown_curated_local:{trigger}:{local_label}",
                recovery_path="unknown_offline_curated_local",
                recovery_evidence=evidence,
            )

        if rule_label != UNKNOWN_TOPIC_LABEL and local_label == UNKNOWN_TOPIC_LABEL:
            if (
                rule_label == "U1"
                and float(route_result.topic_rule_score or 0.0) >= 2.5
                and any(
                    self._has_signal(
                        positive_signals,
                        token,
                    )
                    for token in (
                        "anchor:urban renewal",
                        "anchor:urban regeneration",
                        "anchor:district regeneration",
                        "anchor:redevelopment",
                        "anchor:renewal",
                        "anchor:upgrading",
                    )
                )
            ):
                return build_rule_payload("u1_anchor_bundle")
            if (
                rule_label == "U9"
                and float(route_result.topic_rule_score or 0.0) >= 4.0
                and any(
                    self._has_signal(
                        positive_signals,
                        token,
                    )
                    for token in (
                        "combo:urban regeneration+policy",
                        "combo:urban regeneration+participation",
                        "combo:urban regeneration+governance",
                        "combo:urban renewal+governance",
                    )
                )
            ):
                return build_rule_payload("u9_policy_governance_bundle")
            if (
                rule_label == "U10"
                and float(route_result.topic_rule_score or 0.0) >= 3.5
                and binary_probability >= 0.90
                and any(
                    self._has_signal(
                        positive_signals,
                        token,
                    )
                    for token in (
                        "anchor:urban regeneration",
                        "anchor:urban redevelopment",
                        "anchor:redevelopment",
                    )
                )
            ):
                return build_rule_payload("u10_finance_anchor_bundle")
            if rule_label == "U12":
                has_gentrification_signal = self._has_signal(positive_signals, "gentrification")
                has_social_history_risk = self._has_signal(risk_tags, "social_history_media_risk")
                if (
                    (has_gentrification_signal and not has_social_history_risk)
                    or bertopic_high_purity
                    or float(route_result.topic_rule_margin or 0.0) >= 2.0
                ):
                    return build_rule_payload("u12_gentrification_bundle")
            if (
                rule_label == "U4"
                and (
                    float(route_result.topic_rule_score or 0.0) >= 3.8
                    or (
                        float(route_result.topic_rule_score or 0.0) >= 2.85
                        and self._has_signal(positive_signals, "title:redevelopment")
                    )
                )
            ):
                return build_rule_payload("u4_redevelopment_bundle")
            if (
                rule_label == "U13"
                and float(route_result.topic_rule_score or 0.0) >= 4.5
                and self._has_signal(positive_signals, "anchor:brownfield")
                and self._has_signal(positive_signals, "anchor:revitalization")
            ):
                return build_rule_payload("u13_brownfield_interim_use")

        if rule_group == "nonurban" and local_group == "urban":
            local_override = OFFLINE_UNKNOWN_RECOVERY_LOCAL_RULES.get((rule_label, local_label))
            if (
                local_override is not None
                and local_confidence >= local_override[0]
                and local_margin >= local_override[1]
                and binary_probability >= local_override[2]
            ):
                trigger = f"{rule_label.lower()}_to_{local_label.lower()}_strong_local"
                return build_local_payload(trigger)

        if rule_group == "urban" and local_group == "nonurban":
            if rule_label == "U1" and local_label == "N1" and float(route_result.topic_rule_score or 0.0) >= 2.5:
                return build_rule_payload("u1_over_n1_curated")
            if rule_label == "U12" and local_label == "N1" and float(route_result.topic_rule_score or 0.0) >= 2.0:
                return build_rule_payload("u12_over_n1_curated")
            if (
                rule_label == "U12"
                and local_label in {"N5", "N7", "N9"}
                and float(route_result.topic_rule_score or 0.0) >= 4.5
                and self._has_signal(positive_signals, "gentrification")
                and self._has_signal(positive_signals, "displacement")
            ):
                return build_rule_payload("u12_displacement_cross_group")
            if (
                rule_label == "U4"
                and local_label == "N1"
                and not self._has_signal(risk_tags, "greenfield")
                and (
                    self._has_signal(positive_signals, "redevelopment")
                    or self._has_signal(positive_signals, "revitalization")
                )
            ):
                return build_rule_payload("u4_over_n1_inner_city")
            if (
                rule_label == "U5"
                and local_label in {"N9", "N10"}
                and binary_probability <= 0.03
                and self._has_signal(positive_signals, "anchor:brownfield")
            ):
                return build_rule_payload("u5_brownfield_cross_group")

        return None

    def _allow_family_gate_recovery(
        self,
        *,
        route_result,
        topic_prediction: TopicPrediction,
        bertopic_signal: BERTopicSignal,
        llm_family_hint: str,
        family_decision,
        candidate_final_topic: str,
        within_family_label: str,
        within_family_score: float,
        within_family_margin: float,
    ) -> bool:
        if candidate_final_topic != UNKNOWN_TOPIC_LABEL:
            return False
        if family_decision.final_family not in {"urban", "nonurban"}:
            return False
        if within_family_label == UNKNOWN_TOPIC_LABEL:
            return False
        if within_family_score < UNKNOWN_FAMILY_GATE_RECOVERY_SCORE_FLOOR:
            return False
        if within_family_margin < UNKNOWN_FAMILY_GATE_RECOVERY_MARGIN_FLOOR:
            return False

        llm_hint = self._normalize_family_hint_value(llm_family_hint)
        if llm_hint not in {"0", "1"}:
            return False
        required_confidence = UNKNOWN_FAMILY_GATE_RECOVERY_CONFIDENCE_FLOOR
        if family_decision.confidence < required_confidence:
            return False
        rule_label = route_result.topic_rule or UNKNOWN_TOPIC_LABEL
        rule_family = route_result.topic_rule_group or topic_group_for_label(rule_label)
        local_label = topic_prediction.topic_label or UNKNOWN_TOPIC_LABEL
        local_family = topic_prediction.topic_group or topic_group_for_label(local_label)
        bertopic_family = str(bertopic_signal.mapped_group or "")

        expected_family = "urban" if llm_hint == "1" else "nonurban"
        if family_decision.final_family != expected_family:
            return False
        has_structural_support = rule_family == expected_family or bertopic_family == expected_family
        if not has_structural_support:
            return False
        aligned_support = 1  # llm
        if rule_family == expected_family:
            aligned_support += 1
        if local_family == expected_family:
            aligned_support += 1
        if bertopic_family == expected_family:
            aligned_support += 1
        return aligned_support >= 2

    def _apply_family_gate(
        self,
        base: Dict[str, Any],
        *,
        record: UrbanMetadataRecord,
        route_result,
        topic_prediction: TopicPrediction,
        bertopic_signal: BERTopicSignal,
        llm_family_hint: str,
        candidate_final_topic: str,
        decision_source: str,
        decision_reason: str,
        confidence: float,
    ) -> tuple[str, str, str, float]:
        if not self.family_gate.enabled:
            rule_family = route_result.topic_rule_group or topic_group_for_label(route_result.topic_rule)
            local_family = topic_prediction.topic_group or topic_group_for_label(topic_prediction.topic_label)
            boundary_bucket, conflict_pattern = self.family_gate.describe_conflict(
                rule_label=str(route_result.topic_rule or ""),
                local_label=str(topic_prediction.topic_label or ""),
                rule_family=rule_family,
                local_family=local_family,
            )
            final_family = (
                topic_group_for_label(candidate_final_topic)
                if candidate_final_topic != UNKNOWN_TOPIC_LABEL
                else UNKNOWN_TOPIC_GROUP
            )
            base.update(
                {
                    "topic_family_rule": rule_family,
                    "topic_family_local": local_family,
                    "topic_family_final": final_family,
                    "family_decision_source": "family_gate_disabled",
                    "family_confidence": 0.0,
                    "topic_within_family_label": candidate_final_topic if candidate_final_topic != UNKNOWN_TOPIC_LABEL else "",
                    "boundary_bucket": boundary_bucket,
                    "family_conflict_pattern": conflict_pattern,
                    "family_probability_urban": "",
                    "topic_family_within_score": 0.0,
                    "topic_family_within_margin": 0.0,
                }
            )
            return candidate_final_topic, decision_source, decision_reason, confidence

        family_decision = self.family_gate.predict(
            record=record,
            route_result=route_result,
            topic_prediction=topic_prediction,
            bertopic_signal=bertopic_signal,
            llm_family_hint=llm_family_hint,
        )
        within_family_label, within_family_score, within_family_margin = self._select_topic_within_family(
            family=family_decision.final_family,
            candidate_final_topic=candidate_final_topic,
            route_result=route_result,
            topic_prediction=topic_prediction,
            bertopic_signal=bertopic_signal,
        )
        base.update(
            {
                "topic_family_rule": family_decision.rule_family,
                "topic_family_local": family_decision.local_family,
                "topic_family_final": family_decision.final_family,
                "family_predicted_family": family_decision.final_family,
                "family_decision_source": family_decision.decision_source,
                "family_confidence": family_decision.confidence,
                "topic_within_family_label": within_family_label if within_family_label != UNKNOWN_TOPIC_LABEL else "",
                "boundary_bucket": family_decision.boundary_bucket,
                "family_conflict_pattern": family_decision.family_conflict_pattern,
                "family_probability_urban": family_decision.probability_urban,
                "topic_family_within_score": within_family_score,
                "topic_family_within_margin": within_family_margin,
            }
        )

        final_topic = candidate_final_topic
        final_source = decision_source
        final_reason = decision_reason
        final_confidence = confidence

        if self._allow_family_gate_recovery(
            route_result=route_result,
            topic_prediction=topic_prediction,
            bertopic_signal=bertopic_signal,
            llm_family_hint=llm_family_hint,
            family_decision=family_decision,
            candidate_final_topic=candidate_final_topic,
            within_family_label=within_family_label,
            within_family_score=within_family_score,
            within_family_margin=within_family_margin,
        ):
            final_topic = within_family_label
            final_source = "family_gate_recovery"
            final_reason = f"{decision_reason};family_gate:{family_decision.final_family}:{within_family_label}".strip(";")
            final_confidence = round(max(confidence, 0.58 + family_decision.confidence * 0.25), 4)
            base["unknown_recovery_path"] = "family_gate_recovery"
            base["unknown_recovery_evidence"] = (
                f"family={family_decision.final_family};within={within_family_label};"
                f"family_confidence={family_decision.confidence:.4f};"
                f"within_score={within_family_score:.2f};within_margin={within_family_margin:.2f}"
            )

        if final_topic != UNKNOWN_TOPIC_LABEL:
            base["topic_family_final"] = topic_group_for_label(final_topic)
        elif str(base.get("unknown_recovery_path", "") or "") in {"", "pending_review"}:
            base["unknown_recovery_path"] = "retained_unknown"

        return final_topic, final_source, final_reason, final_confidence

    def _fuse_rule_and_local(
        self,
        *,
        route_result,
        topic_prediction: TopicPrediction,
    ) -> Dict[str, Any]:
        rule_label = route_result.topic_rule or UNKNOWN_TOPIC_LABEL
        local_label = topic_prediction.topic_label or UNKNOWN_TOPIC_LABEL
        rule_confidence = self._rule_confidence(route_result)
        local_confidence = float(topic_prediction.confidence)
        same_label = rule_label == local_label and rule_label != UNKNOWN_TOPIC_LABEL
        same_group = (
            rule_label != UNKNOWN_TOPIC_LABEL
            and local_label != UNKNOWN_TOPIC_LABEL
            and route_result.topic_rule_group == topic_prediction.topic_group
        )

        if same_label:
            return {
                "final_topic": local_label,
                "decision_source": "rule_model_fusion",
                "decision_reason": f"rule_local_agree:{local_label}",
                "confidence": round(max(rule_confidence, local_confidence), 4),
                "review_reason": "",
            }

        if rule_label == UNKNOWN_TOPIC_LABEL and local_label == UNKNOWN_TOPIC_LABEL:
            return {
                "final_topic": UNKNOWN_TOPIC_LABEL,
                "decision_source": "unknown_review",
                "decision_reason": "rule_unknown_local_unknown",
                "confidence": round(max(rule_confidence, local_confidence), 4),
                "review_reason": "rule_unknown_local_unknown",
            }

        if route_result.rule_high_confidence and rule_label != UNKNOWN_TOPIC_LABEL:
            if local_label == UNKNOWN_TOPIC_LABEL:
                return {
                    "final_topic": rule_label,
                    "decision_source": "rule_model_fusion",
                    "decision_reason": f"rule_high_confidence_local_unknown:{rule_label}",
                    "confidence": rule_confidence,
                    "review_reason": "",
                }
            if same_group:
                if rule_confidence >= local_confidence + 0.05:
                    return {
                        "final_topic": rule_label,
                        "decision_source": "rule_model_fusion",
                        "decision_reason": f"rule_high_confidence_same_group:{rule_label}",
                        "confidence": rule_confidence,
                        "review_reason": "",
                    }
                return {
                    "final_topic": local_label,
                    "decision_source": "stage2_classifier",
                    "decision_reason": f"local_stronger_same_group:{local_label}",
                    "confidence": local_confidence,
                    "review_reason": "",
                }
            if rule_confidence >= local_confidence + 0.18:
                return {
                    "final_topic": rule_label,
                    "decision_source": "rule_model_fusion",
                    "decision_reason": f"rule_high_conflict_override_local:{rule_label}",
                    "confidence": rule_confidence,
                    "review_reason": "",
                }
            if local_confidence >= rule_confidence + 0.18 and float(topic_prediction.margin) >= 1.5:
                return {
                    "final_topic": local_label,
                    "decision_source": "stage2_classifier",
                    "decision_reason": f"local_stronger_cross_group:{local_label}",
                    "confidence": local_confidence,
                    "review_reason": "",
                }
            return {
                "final_topic": UNKNOWN_TOPIC_LABEL,
                "decision_source": "unknown_review",
                "decision_reason": "rule_local_cross_group_conflict",
                "confidence": round(max(rule_confidence, local_confidence), 4),
                "review_reason": "rule_local_cross_group_conflict",
            }

        if local_label == UNKNOWN_TOPIC_LABEL:
            if rule_confidence >= 0.62 and rule_label != UNKNOWN_TOPIC_LABEL:
                return {
                    "final_topic": rule_label,
                    "decision_source": "rule_model_fusion",
                    "decision_reason": f"rule_preferred_local_unknown:{rule_label}",
                    "confidence": rule_confidence,
                    "review_reason": "",
                }
            return {
                "final_topic": UNKNOWN_TOPIC_LABEL,
                "decision_source": "unknown_review",
                "decision_reason": "local_unknown_rule_weak",
                "confidence": round(max(rule_confidence, local_confidence), 4),
                "review_reason": "local_unknown_rule_weak",
            }

        if same_group and rule_label != UNKNOWN_TOPIC_LABEL:
            if local_confidence >= 0.45:
                return {
                    "final_topic": local_label,
                    "decision_source": "stage2_classifier",
                    "decision_reason": f"local_same_group:{local_label}",
                    "confidence": local_confidence,
                    "review_reason": "",
                }
            return {
                "final_topic": rule_label,
                "decision_source": "rule_model_fusion",
                "decision_reason": f"rule_same_group_backup:{rule_label}",
                "confidence": rule_confidence,
                "review_reason": "",
            }

        if rule_label == UNKNOWN_TOPIC_LABEL:
            return {
                "final_topic": local_label,
                "decision_source": "stage2_classifier",
                "decision_reason": f"rule_unknown_local_direct:{local_label}",
                "confidence": local_confidence,
                "review_reason": "",
            }

        if local_confidence >= 0.72 and float(topic_prediction.margin) >= 1.6:
            return {
                "final_topic": local_label,
                "decision_source": "stage2_classifier",
                "decision_reason": f"local_cross_group_strong:{local_label}",
                "confidence": local_confidence,
                "review_reason": "",
            }
        if rule_confidence >= 0.72 and float(route_result.topic_rule_margin) >= 2.5:
            return {
                "final_topic": rule_label,
                "decision_source": "rule_model_fusion",
                "decision_reason": f"rule_cross_group_strong:{rule_label}",
                "confidence": rule_confidence,
                "review_reason": "",
            }
        return {
            "final_topic": UNKNOWN_TOPIC_LABEL,
            "decision_source": "unknown_review",
            "decision_reason": "rule_local_cross_group_both_weak",
            "confidence": round(max(rule_confidence, local_confidence), 4),
            "review_reason": "rule_local_cross_group_both_weak",
        }

    def _resolve_unknown_with_hints(
        self,
        *,
        route_result,
        topic_prediction: TopicPrediction,
        bertopic_signal: BERTopicSignal,
        llm_family_hint: Any,
    ) -> Dict[str, Any]:
        rule_label = route_result.topic_rule or UNKNOWN_TOPIC_LABEL
        local_label = topic_prediction.topic_label or UNKNOWN_TOPIC_LABEL
        rule_group = route_result.topic_rule_group or topic_group_for_label(rule_label)
        local_group = topic_prediction.topic_group or topic_group_for_label(local_label)
        rule_confidence = self._rule_confidence(route_result)
        local_confidence = float(topic_prediction.confidence)
        local_margin = float(topic_prediction.margin)
        llm_hint = self._normalize_family_hint_value(llm_family_hint)
        bertopic_group = str(bertopic_signal.mapped_group or "")
        bertopic_quality = self._bertopic_cluster_quality(bertopic_signal)
        bertopic_urban_support = bertopic_group == "urban" and bertopic_quality in {"high", "medium"}
        bertopic_nonurban_support = bertopic_group == "nonurban" and bertopic_quality in {"high", "medium"}

        offline_resolved = self._resolve_unknown_with_offline_signals(
            route_result=route_result,
            topic_prediction=topic_prediction,
            bertopic_signal=bertopic_signal,
        )
        if offline_resolved is not None:
            return offline_resolved

        if rule_label != UNKNOWN_TOPIC_LABEL and local_label == UNKNOWN_TOPIC_LABEL:
            weak_nonurban_floor = 0.24
            if rule_group == "nonurban" and llm_hint == "0" and rule_confidence >= weak_nonurban_floor:
                return {
                    "final_topic": rule_label,
                    "decision_source": "unknown_hint_resolution",
                    "decision_reason": f"rule_unknown_local_llm_family_nonurban_weak:{rule_label}",
                    "confidence": round(max(rule_confidence, 0.60), 4),
                }
            weak_urban_floor = UNKNOWN_RECOVERY_WEAK_URBAN_RULES.get(rule_label)
            if weak_urban_floor is not None and llm_hint == "1" and rule_confidence >= weak_urban_floor:
                return {
                    "final_topic": rule_label,
                    "decision_source": "unknown_hint_resolution",
                    "decision_reason": f"rule_unknown_local_llm_family_curated:{rule_label}",
                    "confidence": round(max(rule_confidence, 0.60), 4),
                }
            if rule_group == "urban" and llm_hint == "1" and rule_confidence >= 0.35:
                return {
                    "final_topic": rule_label,
                    "decision_source": "unknown_hint_resolution",
                    "decision_reason": f"rule_unknown_local_llm_family:{rule_label}",
                    "confidence": round(max(rule_confidence, 0.62), 4),
                }
            if rule_group == "nonurban" and llm_hint == "0" and rule_confidence >= 0.35:
                return {
                    "final_topic": rule_label,
                    "decision_source": "unknown_hint_resolution",
                    "decision_reason": f"rule_unknown_local_llm_family:{rule_label}",
                    "confidence": round(max(rule_confidence, 0.62), 4),
                }
            if rule_group == "urban" and bertopic_urban_support and rule_confidence >= 0.50:
                return {
                    "final_topic": rule_label,
                    "decision_source": "unknown_hint_resolution",
                    "decision_reason": f"rule_unknown_local_bertopic_support:{rule_label}",
                    "confidence": round(max(rule_confidence, 0.64), 4),
                }
            if rule_group == "nonurban" and bertopic_nonurban_support and rule_confidence >= 0.50:
                return {
                    "final_topic": rule_label,
                    "decision_source": "unknown_hint_resolution",
                    "decision_reason": f"rule_unknown_local_bertopic_support:{rule_label}",
                    "confidence": round(max(rule_confidence, 0.64), 4),
                }

        if rule_group == "urban" and local_group == "nonurban":
            if (
                llm_hint == "1"
                and bertopic_urban_support
                and rule_confidence >= 0.35
                and local_confidence <= 0.60
                and local_margin <= 1.0
            ):
                return {
                    "final_topic": rule_label,
                    "decision_source": "unknown_hint_resolution",
                    "decision_reason": f"cross_group_dual_urban_support:{rule_label}",
                    "confidence": round(max(rule_confidence, 0.64), 4),
                }
            curated_threshold = UNKNOWN_RECOVERY_RULE_TO_TOPIC.get((rule_label, local_label))
            if llm_hint == "1" and curated_threshold is not None and rule_confidence >= curated_threshold:
                return {
                    "final_topic": rule_label,
                    "decision_source": "unknown_hint_resolution",
                    "decision_reason": f"cross_group_curated_rule_urban:{rule_label}",
                    "confidence": round(max(rule_confidence, 0.62), 4),
                }

        if rule_group == "nonurban" and local_group == "urban":
            local_override = UNKNOWN_RECOVERY_RULE_TO_LOCAL.get((rule_label, local_label))
            if (
                llm_hint == "1"
                and local_override is not None
                and local_confidence >= local_override[0]
                and local_margin >= local_override[1]
            ):
                return {
                    "final_topic": local_label,
                    "decision_source": "unknown_hint_resolution",
                    "decision_reason": f"cross_group_curated_local_urban:{local_label}",
                    "confidence": round(max(local_confidence, 0.68), 4),
                }
            if (
                llm_hint == "0"
                and bertopic_nonurban_support
                and rule_confidence >= 0.35
                and local_confidence <= 0.60
                and local_margin <= 1.0
            ):
                return {
                    "final_topic": rule_label,
                    "decision_source": "unknown_hint_resolution",
                    "decision_reason": f"cross_group_dual_nonurban_support:{rule_label}",
                    "confidence": round(max(rule_confidence, 0.64), 4),
                }

        if rule_label == UNKNOWN_TOPIC_LABEL and local_label != UNKNOWN_TOPIC_LABEL:
            if local_group == "urban" and llm_hint == "1" and bertopic_urban_support and local_confidence >= 0.62:
                return {
                    "final_topic": local_label,
                    "decision_source": "unknown_hint_resolution",
                    "decision_reason": f"local_unknown_rule_llm_bertopic_support:{local_label}",
                    "confidence": round(max(local_confidence, 0.64), 4),
                    "recovery_path": "unknown_hint_bertopic",
                    "recovery_evidence": f"family=urban;sources=llm+local+bertopic;label={local_label}",
                }
            if local_group == "nonurban" and llm_hint == "0" and bertopic_nonurban_support and local_confidence >= 0.62:
                return {
                    "final_topic": local_label,
                    "decision_source": "unknown_hint_resolution",
                    "decision_reason": f"local_unknown_rule_llm_bertopic_support:{local_label}",
                    "confidence": round(max(local_confidence, 0.64), 4),
                    "recovery_path": "unknown_hint_bertopic",
                    "recovery_evidence": f"family=nonurban;sources=llm+local+bertopic;label={local_label}",
                }

        consensus = self._resolve_unknown_with_family_consensus(
            route_result=route_result,
            topic_prediction=topic_prediction,
            bertopic_signal=bertopic_signal,
            llm_hint=llm_hint,
            bertopic_urban_support=bertopic_urban_support,
            bertopic_nonurban_support=bertopic_nonurban_support,
        )
        if consensus is not None:
            return consensus

        return {
            "final_topic": UNKNOWN_TOPIC_LABEL,
            "decision_source": "unknown_review",
            "decision_reason": "",
            "confidence": round(max(rule_confidence, local_confidence), 4),
            "recovery_path": "retained_unknown",
            "recovery_evidence": "",
        }

    def _maybe_collect_llm_hint(
        self,
        *,
        title: str,
        abstract: str,
        record: UrbanMetadataRecord,
        route_result,
        topic_prediction: TopicPrediction,
        bertopic_signal: BERTopicSignal,
        session_path: Optional[Path],
        audit_metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not self.llm_assist_enabled:
            return {
                "llm_attempted": 0,
                "llm_used": 0,
                "llm_failure_reason": "",
                "llm_family_hint": "",
                "llm_family_hint_reason": "",
            }
        if not Config.URBAN_HYBRID_ONLINE_LLM_HINTS_ENABLED:
            return {
                "llm_attempted": 0,
                "llm_used": 0,
                "llm_failure_reason": "online_llm_hints_disabled",
                "llm_family_hint": "",
                "llm_family_hint_reason": "",
            }

        try:
            llm_result = self.llm_strategy.process(
                title,
                abstract,
                session_path=session_path,
                metadata=record.to_output_dict(),
                audit_metadata=audit_metadata,
                auxiliary_context={
                    "topic_rule": route_result.topic_rule,
                    "topic_rule_score": route_result.topic_rule_score,
                    "topic_rule_margin": route_result.topic_rule_margin,
                    "topic_rule_top3": route_result.topic_rule_top3,
                    "topic_local_label": topic_prediction.topic_label,
                    "topic_local_confidence": topic_prediction.confidence,
                    "topic_local_margin": topic_prediction.margin,
                    "topic_local_top3": topic_prediction.top_candidates,
                    "bertopic_hint_label": bertopic_signal.mapped_label,
                    "bertopic_hint_group": bertopic_signal.mapped_group,
                    "bertopic_probability": bertopic_signal.topic_probability,
                    "bertopic_label_purity": bertopic_signal.label_purity,
                    "bertopic_mapped_label_share": bertopic_signal.mapped_label_share,
                },
            )
        except Exception as error:
            return {
                "llm_attempted": 1,
                "llm_used": 0,
                "llm_failure_reason": f"exception:{type(error).__name__}",
                "llm_family_hint": "",
                "llm_family_hint_reason": "",
            }

        llm_label, llm_failure_reason = self._normalize_llm_result(llm_result)
        if llm_failure_reason:
            return {
                "llm_attempted": 1,
                "llm_used": 0,
                "llm_failure_reason": llm_failure_reason,
                "llm_family_hint": "",
                "llm_family_hint_reason": "",
            }
        return {
            "llm_attempted": 1,
            "llm_used": 0,
            "llm_failure_reason": "",
            "llm_family_hint": llm_label,
            "llm_family_hint_reason": str(llm_result.get("urban_parse_reason", "llm_family_hint") or "llm_family_hint"),
        }

    def _normalize_llm_result(self, llm_result: Any) -> tuple[str, str]:
        if not isinstance(llm_result, dict) or not llm_result:
            return "0", "empty_response"
        parse_reason = str(llm_result.get("urban_parse_reason", "") or "")
        if parse_reason == "empty_response":
            return "0", "empty_response"

        raw_label = llm_result.get(Schema.IS_URBAN_RENEWAL, llm_result.get("final_label"))
        if raw_label in (None, ""):
            return "0", "missing_label"

        llm_label = self._normalize_family_hint_value(raw_label)
        if llm_label not in {"0", "1"}:
            return "0", "invalid_label"
        return llm_label, ""

    def _normalize_final_binary_label(
        self,
        base: Dict[str, Any],
        *,
        final_topic: str,
        binary_label: Optional[str],
    ) -> str:
        if binary_label in {"0", "1"}:
            return str(binary_label)
        existing_label = str(base.get("final_label", "") or base.get("urban_flag", "") or "").strip()
        if existing_label.endswith(".0"):
            existing_label = existing_label[:-2]
        if existing_label in {"0", "1"}:
            return existing_label
        topic_label = urban_flag_for_topic_label(final_topic)
        return topic_label if topic_label in {"0", "1"} else "0"

    def _resolve_binary_audit_topic(
        self,
        base: Dict[str, Any],
        *,
        final_topic: str,
        binary_label: str,
        decision_reason: str,
    ) -> tuple[str, str]:
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
        if binary_label == "1" and topic_group != "urban":
            score = self._safe_float(base.get("urban_probability_score"), default=0.0)
            threshold = self._safe_float(
                base.get("binary_decision_threshold"),
                default=float(Config.URBAN_BINARY_DECISION_THRESHOLD),
            )
            recall_tier = str(base.get("binary_recall_calibration_tier", "") or "none")
            from_topic = final_topic or UNKNOWN_TOPIC_LABEL
            evidence = (
                f"from={from_topic};score={score:.4f};threshold={threshold:.4f};"
                f"recall={recall_tier};source={base.get('binary_decision_source', '')}"
            )
            strong_open_set = (
                str(base.get("open_set_topic", "") or "") == OPEN_SET_URBAN_LABEL
                or str(base.get("anchor_guard_action", "") or "") == "promote"
                or str(base.get("uncertain_nonurban_guard_action", "") or "") == "promote"
                or recall_tier == "guard_promotion_floor"
            )
            if not strong_open_set:
                base.update(
                    {
                        "binary_audit_resolution_flag": 1,
                        "binary_audit_resolution_action": "positive_binary_resolved",
                        "binary_audit_resolution_reason": f"binary_positive_from_{from_topic}",
                        "binary_audit_resolution_evidence": evidence,
                        "taxonomy_coverage_status": "binary_resolved",
                    }
                )
                decision_reason = f"{decision_reason};binary_audit_positive_resolved:{from_topic}".strip(";")
                return final_topic, decision_reason

            base.update(
                {
                    "binary_audit_resolution_flag": 1,
                    "binary_audit_resolution_action": "positive_open_set_topic",
                    "binary_audit_resolution_reason": f"binary_positive_from_{from_topic}",
                    "binary_audit_resolution_evidence": evidence,
                    "open_set_flag": 1,
                    "open_set_topic": OPEN_SET_URBAN_LABEL,
                    "open_set_reason": str(base.get("open_set_reason", "") or "binary_positive_resolution"),
                    "open_set_evidence": str(base.get("open_set_evidence", "") or evidence),
                    "taxonomy_coverage_status": "open_set",
                }
            )
            decision_reason = f"{decision_reason};binary_audit_positive_open_set:{from_topic}".strip(";")
            return OPEN_SET_URBAN_LABEL, decision_reason

        if binary_label == "0" and topic_group != "nonurban":
            score = self._safe_float(base.get("urban_probability_score"), default=1.0)
            threshold = self._safe_float(
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
                        "binary_audit_resolution_action": "negative_binary_resolved",
                        "binary_audit_resolution_reason": f"binary_negative_from_{from_topic}",
                        "binary_audit_resolution_evidence": evidence,
                        "taxonomy_coverage_status": "binary_resolved",
                    }
                )
                decision_reason = f"{decision_reason};binary_audit_negative_resolved:{from_topic}".strip(";")
                return final_topic, decision_reason

            base.update(
                {
                    "binary_audit_resolution_flag": 1,
                    "binary_audit_resolution_action": "negative_binary_resolved",
                    "binary_audit_resolution_reason": f"binary_negative_from_{from_topic}",
                    "binary_audit_resolution_evidence": evidence,
                    "taxonomy_coverage_status": "binary_resolved",
                }
            )
            decision_reason = f"{decision_reason};binary_audit_negative_resolved:{from_topic}".strip(";")
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

    def _effective_binary_topic_consistency_flag(
        self,
        base: Dict[str, Any],
        *,
        binary_label: str,
        final_topic: str,
    ) -> int:
        if str(base.get("taxonomy_coverage_status", "") or "") == "binary_resolved":
            return 0
        return self._binary_topic_consistency_flag(
            binary_label=binary_label,
            final_topic=final_topic,
        )

    def _reconcile_final_review_signal(
        self,
        base: Dict[str, Any],
        *,
        final_topic: str,
        binary_label: str,
        review_flag: int,
        review_reason: str,
    ) -> tuple[int, str]:
        base["review_flag_raw"] = int(bool(review_flag))
        base["review_reason_raw"] = str(review_reason or "")

        route_reason = str(base.get("metadata_route_reason", "") or "").strip()
        if route_reason in BINARY_HARD_NEGATIVE_REASONS:
            return 0, ""

        score = self._safe_float(base.get("urban_probability_score"), default=0.5)
        threshold = self._safe_float(
            base.get("binary_decision_threshold"),
            default=float(Config.URBAN_BINARY_DECISION_THRESHOLD),
        )
        consistency_flag = self._effective_binary_topic_consistency_flag(
            base,
            binary_label=binary_label,
            final_topic=final_topic,
        )
        base["binary_topic_consistency_flag"] = consistency_flag

        reasons: list[str] = []
        if abs(score - threshold) <= float(Config.URBAN_BINARY_REVIEW_MARGIN):
            reasons.append("binary_near_threshold")
        if consistency_flag:
            reasons.append("binary_topic_inconsistency")

        recall_tier = str(base.get("binary_recall_calibration_tier", "") or "")
        if recall_tier == "blocked" and binary_label == "1":
            reasons.append("risk_blocked_positive")

        old_reasons = [item for item in str(review_reason or "").split(";") if item]
        unresolved_manual_reasons = {
            "anchor_guard_core_anchor_without_support",
            "anchor_guard_support_without_urban_candidate",
        }
        if consistency_flag:
            for item in old_reasons:
                if item in unresolved_manual_reasons:
                    reasons.append(item)

        reasons = list(dict.fromkeys(item for item in reasons if item))
        return int(bool(reasons)), ";".join(reasons)

    def _summarize_decision_explanation(
        self,
        base: Dict[str, Any],
        *,
        final_topic: str,
        binary_label: str,
        confidence: float,
        decision_source: str,
        review_flag: int,
    ) -> Dict[str, str]:
        score = self._safe_float(base.get("urban_probability_score"), default=(float(confidence) if binary_label == "1" else 1.0 - float(confidence)))
        threshold = self._safe_float(
            base.get("binary_decision_threshold"),
            default=float(Config.URBAN_BINARY_DECISION_THRESHOLD),
        )
        topic_group = topic_group_for_label(final_topic)
        taxonomy_status = str(base.get("taxonomy_coverage_status", "") or "unknown")
        positive: list[str] = []
        negative: list[str] = []

        if topic_group == "urban":
            positive.append(f"topic_final={final_topic}")
        elif topic_group == "nonurban":
            negative.append(f"topic_final={final_topic}")
        else:
            negative.append("topic_final=Unknown")

        open_set_topic = str(base.get("open_set_topic", "") or "")
        if open_set_topic == OPEN_SET_URBAN_LABEL:
            reason = str(base.get("open_set_reason", "") or "open_set")
            evidence = str(base.get("open_set_evidence", "") or "")
            positive.append(f"open_set={reason}" + (f"({evidence})" if evidence else ""))
        elif open_set_topic == OPEN_SET_NONURBAN_LABEL:
            negative.append("open_set=nonurban_other")

        anchor_action = str(base.get("anchor_guard_action", "") or "")
        if anchor_action == "promote":
            positive.append(f"anchor_guard=promote({base.get('anchor_guard_hits', '')})")
        elif anchor_action == "review":
            positive.append("anchor_guard=core_anchor_review")

        uncertain_action = str(base.get("uncertain_nonurban_guard_action", "") or "")
        if uncertain_action == "promote":
            positive.append(f"uncertain_nonurban_guard=promote({base.get('uncertain_nonurban_guard_reason', '')})")
        elif uncertain_action == "review":
            negative.append(f"uncertain_nonurban_guard=review({base.get('uncertain_nonurban_guard_reason', '')})")
        elif uncertain_action == "keep_0":
            negative.append(f"uncertain_nonurban_guard=keep_0({base.get('uncertain_nonurban_guard_reason', '')})")

        family_probability = self._safe_float(base.get("family_probability_urban"), default=0.5)
        if family_probability >= 0.60:
            positive.append(f"family_probability={family_probability:.4f}")
        elif family_probability <= 0.35:
            negative.append(f"family_probability={family_probability:.4f}")

        topic_binary_probability = self._safe_float(base.get("topic_binary_probability"), default=0.5)
        if topic_binary_probability >= 0.60:
            positive.append(f"topic_binary_probability={topic_binary_probability:.4f}")
        elif topic_binary_probability <= 0.35:
            negative.append(f"topic_binary_probability={topic_binary_probability:.4f}")

        llm_hint = self._normalize_family_hint_value(base.get("llm_family_hint", ""))
        if llm_hint == "1":
            positive.append("llm_family_hint=1")
        elif llm_hint == "0":
            negative.append("llm_family_hint=0")

        risk_tags = sorted(self._extract_stage1_risk_tags(base))
        for tag in risk_tags:
            if tag == RISK_EXPLICIT_RENEWAL_OTHER_OBJECT:
                positive.append(f"risk_tag={tag}")
            else:
                negative.append(f"risk_tag={tag}")

        recall_tier = str(base.get("binary_recall_calibration_tier", "") or "none")
        if int(bool(base.get("binary_recall_calibration_flag", 0))):
            positive.append(f"recall_calibration={recall_tier}")
        elif recall_tier == "blocked":
            negative.append(f"recall_calibration_blocked={base.get('binary_recall_calibration_reason', '')}")

        hard_negative = taxonomy_status == "hard_negative" or str(base.get("metadata_route_reason", "") or "") in BINARY_HARD_NEGATIVE_REASONS
        consistency_flag = int(bool(base.get("binary_topic_consistency_flag", 0)))
        if hard_negative:
            balance = "hard_negative"
        elif consistency_flag:
            balance = "conflict_positive" if binary_label == "1" else "conflict_negative"
        elif binary_label == "1" and confidence >= 0.75:
            balance = "strong_positive"
        elif binary_label == "1" and confidence >= 0.60:
            balance = "positive"
        elif binary_label == "1":
            balance = "low_confidence_positive"
        elif binary_label == "0" and confidence >= 0.75:
            balance = "strong_negative"
        elif binary_label == "0" and confidence >= 0.60:
            balance = "negative"
        else:
            balance = "low_confidence_negative"

        comparator = ">=" if score >= threshold else "<"
        explanation = (
            f"final={binary_label}; score={score:.4f}{comparator}threshold={threshold:.4f}; "
            f"confidence={float(confidence):.4f}; topic={final_topic}/{topic_group}; "
            f"coverage={taxonomy_status}; source={decision_source}; review={int(bool(review_flag))}"
        )
        stack_parts = [
            f"route={base.get('metadata_route', '')}:{base.get('metadata_route_reason', '')}",
            f"rule={base.get('topic_rule', '')}",
            f"local={base.get('topic_local_label', '')}",
            f"family={base.get('family_decision_source', '')}:{base.get('family_predicted_family', '')}",
            f"anchor={anchor_action or 'none'}",
            f"uncertain_nonurban={uncertain_action or 'none'}",
            f"open_set={open_set_topic or 'none'}:{base.get('open_set_reason', '')}",
            f"binary_audit={base.get('binary_audit_resolution_action', 'none')}",
            f"binary={base.get('binary_decision_source', '')}:{recall_tier}",
        ]
        return {
            "decision_explanation": explanation,
            "primary_positive_evidence": "; ".join(dict.fromkeys(item for item in positive if item)) or "none",
            "primary_negative_evidence": "; ".join(dict.fromkeys(item for item in negative if item)) or "none",
            "evidence_balance": balance,
            "decision_rule_stack": " > ".join(stack_parts),
        }

    def _build_final_result(
        self,
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
        urban_flag = self._normalize_final_binary_label(
            base,
            final_topic=final_topic,
            binary_label=binary_label,
        )
        final_topic, decision_reason = self._resolve_binary_audit_topic(
            base,
            final_topic=final_topic,
            binary_label=urban_flag,
            decision_reason=decision_reason,
        )
        review_flag, review_reason = self._reconcile_final_review_signal(
            base,
            final_topic=final_topic,
            binary_label=urban_flag,
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

        topic_family_final = str(base.get("topic_family_final", "") or "")
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
                "topic_within_family_label": base.get("topic_within_family_label") or (final_topic if final_topic != UNKNOWN_TOPIC_LABEL else ""),
                "topic_label": final_topic,
                "topic_group": topic_group,
                "topic_name": topic_name,
                "bertopic_hint_conflict_flag": bertopic_conflict_flag,
                "binary_topic_consistency_flag": self._effective_binary_topic_consistency_flag(
                    base,
                    binary_label=urban_flag,
                    final_topic=final_topic,
                ),
                "taxonomy_coverage_status": taxonomy_status,
            }
        )
        base.update(
            self._summarize_decision_explanation(
                base,
                final_topic=final_topic,
                binary_label=urban_flag,
                confidence=float(confidence),
                decision_source=decision_source,
                review_flag=review_flag,
            )
        )
        return base
