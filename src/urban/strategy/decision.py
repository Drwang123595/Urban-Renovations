"""Stable final decision logic for urban-renewal extraction."""

from __future__ import annotations

from ..taxonomy.core import OPEN_SET_URBAN_LABEL, UNKNOWN_TOPIC_GROUP, UNKNOWN_TOPIC_LABEL
from .evidence import EvidenceBundle, FourStepEvidence, StableDecisionResult, StableDecisionStatus
from .four_step import build_four_step_evidence


HIGH_RISK_NONURBAN_TOPICS = {"N1", "N3", "N4", "N5", "N7", "N9", "N10"}


class StableUrbanDecisionEngine:
    """Convert normalized evidence into one stable decision and explanation."""

    def decide(self, evidence: EvidenceBundle) -> StableDecisionResult:
        four_step = build_four_step_evidence(evidence)
        if evidence.rule.hard_exclusion_reason:
            return self._result(
                "0",
                evidence.rule.rule_topic_candidate or evidence.current_topic,
                StableDecisionStatus.EXCLUDED_NEGATIVE,
                f"hard_exclusion:{evidence.rule.hard_exclusion_reason}",
                evidence,
                four_step,
                confidence=max(0.99, 1.0 - evidence.score),
            )

        if self._llm_confirms_core_urban_renewal(evidence):
            topic = evidence.llm.suggested_topic or self._best_positive_topic(evidence)
            review_needed = evidence.current_topic_group in {UNKNOWN_TOPIC_GROUP, "nonurban"}
            return self._result(
                "1",
                topic,
                StableDecisionStatus.LLM_SUPPORTED_POSITIVE,
                evidence.llm.reason or "llm_supported_core_urban_renewal",
                evidence,
                four_step,
                confidence=max(evidence.llm.confidence, evidence.score, evidence.topic.confidence),
                review_flag=int(review_needed),
                review_reason="llm_supported_boundary_case" if review_needed else "",
                conflict_type="llm_supported_boundary_case" if review_needed else "",
            )

        if self._llm_rejects_boundary(evidence):
            return self._result(
                "0",
                evidence.current_topic,
                StableDecisionStatus.LLM_REJECTED_BOUNDARY,
                evidence.llm.reason or "llm_rejected_boundary",
                evidence,
                four_step,
                confidence=max(1.0 - evidence.score, evidence.llm.confidence, 0.5),
                review_flag=1,
                review_reason="llm_rejected_boundary",
                conflict_type="llm_rejected_boundary",
            )

        if self._dynamic_candidate_requires_review(evidence):
            if self._dynamic_candidate_has_core_positive_support(evidence, four_step):
                return self._result(
                    "1",
                    self._dynamic_candidate_topic(evidence) or self._best_positive_topic(evidence),
                    StableDecisionStatus.ACCEPTED_POSITIVE,
                    "dynamic_positive_candidate_with_core_evidence",
                    evidence,
                    four_step,
                    confidence=max(evidence.score, evidence.topic.confidence, 0.65),
                )
            return self._result(
                evidence.current_label or "0",
                evidence.current_topic,
                StableDecisionStatus.DYNAMIC_CANDIDATE_REVIEW,
                "dynamic_candidate_review",
                evidence,
                four_step,
                confidence=max(evidence.score, 0.5),
                review_flag=1,
                review_reason="dynamic_candidate_review",
                conflict_type="dynamic_candidate",
            )

        if self._has_background_or_method_only_subject(four_step):
            return self._result(
                "0",
                evidence.current_topic,
                StableDecisionStatus.CONFLICT_REVIEW,
                "background_or_method_only",
                evidence,
                four_step,
                confidence=max(1.0 - evidence.score, 0.5),
                review_flag=1,
                review_reason="background_or_method_only",
                conflict_type="background_or_method_only",
            )

        if self._has_strong_positive_rule(evidence, four_step):
            return self._result(
                "1",
                self._best_positive_topic(evidence),
                StableDecisionStatus.ACCEPTED_POSITIVE,
                "core_object_action_main_subject",
                evidence,
                four_step,
                confidence=max(evidence.score, evidence.topic.confidence, 0.75),
            )

        if self._has_consistent_positive_support(evidence):
            return self._result(
                "1",
                self._best_positive_topic(evidence),
                StableDecisionStatus.ACCEPTED_POSITIVE,
                "topic_family_consistent_positive",
                evidence,
                four_step,
                confidence=max(evidence.score, evidence.topic.confidence, evidence.family.family_probability_urban, 0.65),
            )

        if self._has_strong_multisource_positive_support(evidence, four_step):
            return self._result(
                "1",
                self._best_positive_topic(evidence),
                StableDecisionStatus.ACCEPTED_POSITIVE,
                "strong_multisource_positive_support",
                evidence,
                four_step,
                confidence=max(evidence.score, evidence.topic.confidence, evidence.family.family_probability_urban, 0.75),
            )

        if self._has_positive_conflict(evidence):
            return self._result(
                "0",
                evidence.current_topic,
                StableDecisionStatus.CONFLICT_REVIEW,
                "conflicting_positive_without_core_evidence_rejected",
                evidence,
                four_step,
                confidence=max(1.0 - evidence.score, 0.5),
                review_flag=1,
                review_reason="strategy_conflict_review",
                conflict_type=self._conflict_type(evidence),
            )

        if evidence.current_label == "1":
            if evidence.current_topic_group == UNKNOWN_TOPIC_GROUP:
                return self._result(
                    "0",
                    evidence.current_topic,
                    StableDecisionStatus.ACCEPTED_NEGATIVE,
                    "unknown_topic_positive_without_core_evidence_rejected",
                    evidence,
                    four_step,
                    confidence=max(1.0 - evidence.score, 0.5),
                )
            return self._result(
                "1",
                self._best_positive_topic(evidence),
                StableDecisionStatus.ACCEPTED_POSITIVE,
                "current_positive_without_strategy_conflict",
                evidence,
                four_step,
                confidence=max(evidence.score, evidence.topic.confidence, 0.5),
            )

        return self._result(
            "0",
            evidence.current_topic,
            StableDecisionStatus.ACCEPTED_NEGATIVE,
            "no_sufficient_urban_renewal_evidence",
            evidence,
            four_step,
            confidence=max(1.0 - evidence.score, 0.5),
        )

    def _has_strong_positive_rule(self, evidence: EvidenceBundle, four_step: FourStepEvidence) -> bool:
        if evidence.rule.hard_exclusion_reason or self._has_hard_risk(evidence):
            return False
        return bool(
            evidence.rule.has_renewal_action
            and evidence.rule.has_existing_urban_object
            and self._has_main_subject_support(four_step)
        )

    def _llm_confirms_core_urban_renewal(self, evidence: EvidenceBundle) -> bool:
        llm = evidence.llm
        return bool(
            llm.used
            and llm.label_hint == "1"
            and llm.confidence >= 0.75
            and llm.object_is_existing_urban is True
            and llm.renewal_action_present is True
            and llm.action_is_main_subject is True
            and llm.is_background_only is False
            and not llm.exclusion_risk
        )

    def _llm_rejects_boundary(self, evidence: EvidenceBundle) -> bool:
        llm = evidence.llm
        return bool(
            llm.used
            and llm.label_hint == "0"
            and llm.confidence >= 0.75
            and (
                llm.is_background_only is True
                or bool(llm.exclusion_risk)
                or llm.object_is_existing_urban is False
                or llm.renewal_action_present is False
            )
        )

    def _has_consistent_positive_support(self, evidence: EvidenceBundle) -> bool:
        if self._has_hard_risk(evidence):
            return False
        return bool(
            evidence.current_label == "1"
            and evidence.topic.topic_group == "urban"
            and evidence.family.family_probability_urban >= 0.60
        )

    def _has_strong_multisource_positive_support(self, evidence: EvidenceBundle, four_step: FourStepEvidence) -> bool:
        if self._has_exclusionary_risk(evidence) or evidence.rule.hard_exclusion_reason:
            return False
        if self._has_background_or_method_only_subject(four_step):
            return False
        if evidence.current_label != "1" or evidence.topic.topic_group != "urban":
            return False
        if not evidence.rule.has_renewal_action:
            return False
        return bool(evidence.family.family_probability_urban >= 0.75 and evidence.score >= evidence.threshold)

    def _has_positive_conflict(self, evidence: EvidenceBundle) -> bool:
        if evidence.current_label != "1":
            return False
        if evidence.binary_topic_consistency_flag:
            return True
        if evidence.current_topic_group == "nonurban":
            return True
        if evidence.current_topic in HIGH_RISK_NONURBAN_TOPICS:
            return True
        if evidence.evidence_balance == "conflict_positive":
            return True
        if self._has_hard_risk(evidence):
            return True
        if evidence.score and abs(evidence.score - evidence.threshold) <= 0.03:
            return True
        return False

    def _dynamic_candidate_requires_review(self, evidence: EvidenceBundle) -> bool:
        candidate_label = str(evidence.dynamic.get("candidate_label", "") or "")
        override_applied = int(bool(evidence.dynamic.get("override_applied", 0)))
        return bool(candidate_label in {"0", "1"} and not override_applied)

    def _dynamic_candidate_has_core_positive_support(self, evidence: EvidenceBundle, four_step: FourStepEvidence) -> bool:
        candidate_label = str(evidence.dynamic.get("candidate_label", "") or "")
        if candidate_label != "1":
            return False
        if self._has_hard_risk(evidence) or evidence.rule.hard_exclusion_reason:
            return False
        return bool(
            evidence.rule.has_renewal_action
            and evidence.rule.has_existing_urban_object
            and self._has_main_subject_support(four_step)
        )

    def _has_main_subject_support(self, four_step: FourStepEvidence) -> bool:
        return four_step.main_subject_evidence == "core_object_and_action_in_title_or_abstract"

    def _has_background_or_method_only_subject(self, four_step: FourStepEvidence) -> bool:
        return four_step.main_subject_evidence == "background_or_method_only"

    def _dynamic_candidate_topic(self, evidence: EvidenceBundle) -> str:
        topic = str(evidence.dynamic.get("candidate_topic", "") or "").strip()
        return topic if topic else ""

    def _has_hard_risk(self, evidence: EvidenceBundle) -> bool:
        risks = set(evidence.rule.risk_hits)
        return bool({"rural_risk", "greenfield_risk", "method_only_risk"} & risks)

    def _has_exclusionary_risk(self, evidence: EvidenceBundle) -> bool:
        risks = set(evidence.rule.risk_hits)
        return bool({"rural_risk", "greenfield_risk"} & risks)

    def _best_positive_topic(self, evidence: EvidenceBundle) -> str:
        if evidence.topic.topic_group == "urban":
            return evidence.topic.topic_candidate
        if evidence.current_topic_group == "urban":
            return evidence.current_topic
        return OPEN_SET_URBAN_LABEL

    def _conflict_type(self, evidence: EvidenceBundle) -> str:
        parts: list[str] = []
        if evidence.binary_topic_consistency_flag:
            parts.append("binary_topic_inconsistency")
        if evidence.current_topic_group == "nonurban":
            parts.append("nonurban_topic_positive")
        if evidence.current_topic in HIGH_RISK_NONURBAN_TOPICS:
            parts.append(f"high_risk_topic_{evidence.current_topic}")
        for risk in evidence.rule.risk_hits:
            parts.append(risk)
        if evidence.evidence_balance == "conflict_positive":
            parts.append("conflict_positive")
        return "|".join(dict.fromkeys(parts)) or "strategy_conflict"

    def _result(
        self,
        label: str,
        topic: str,
        status: StableDecisionStatus,
        reason: str,
        evidence: EvidenceBundle,
        four_step: FourStepEvidence,
        *,
        confidence: float,
        review_flag: int = 0,
        review_reason: str = "",
        conflict_type: str = "",
    ) -> StableDecisionResult:
        normalized_label = "1" if str(label) == "1" else "0"
        return StableDecisionResult(
            final_label=normalized_label,
            urban_flag=normalized_label,
            topic_final=topic or UNKNOWN_TOPIC_LABEL,
            confidence=round(min(max(float(confidence), 0.0), 1.0), 6),
            status=status,
            reason=reason,
            positive_evidence=four_step.positive_evidence,
            negative_evidence=four_step.negative_evidence,
            review_flag=int(bool(review_flag)),
            review_reason=review_reason,
            llm_semantic_evidence=four_step.llm_semantic_evidence,
            evidence_conflict_type=conflict_type,
            core_object_evidence=four_step.core_object_evidence,
            renewal_action_evidence=four_step.renewal_action_evidence,
            main_subject_evidence=four_step.main_subject_evidence,
            risk_evidence=four_step.risk_evidence,
            auxiliary_evidence=four_step.auxiliary_evidence,
        )


def decide_stable_strategy(evidence: EvidenceBundle) -> StableDecisionResult:
    return StableUrbanDecisionEngine().decide(evidence)
