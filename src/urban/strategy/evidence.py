"""Stable evidence contracts for urban-renewal strategy decisions."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class StableDecisionStatus(str, Enum):
    ACCEPTED_POSITIVE = "accepted_positive"
    ACCEPTED_NEGATIVE = "accepted_negative"
    EXCLUDED_NEGATIVE = "excluded_negative"
    HARD_NEGATIVE_PROTECTED = "excluded_negative"
    CONFLICT_REVIEW = "conflict_review"
    CONFLICT_POSITIVE_REVIEW = "conflict_review"
    UNKNOWN_REVIEW = "unknown_review"
    UNKNOWN_POSITIVE_REVIEW = "unknown_review"
    LLM_SUPPORTED_POSITIVE = "llm_supported_positive"
    LLM_REJECTED_BOUNDARY = "llm_rejected_boundary"
    DYNAMIC_CANDIDATE_REVIEW = "dynamic_candidate_review"


class EvidenceKind(str, Enum):
    RENEWAL_ACTION = "renewal_action"
    EXISTING_URBAN_OBJECT = "existing_urban_object"
    POLICY_PROJECT_INTERVENTION = "policy_project_intervention"
    FIXED_URBAN_TOPIC = "fixed_urban_topic"
    FAMILY_SUPPORT = "family_support"
    METHOD_ONLY_RISK = "method_only_risk"
    RURAL_RISK = "rural_risk"
    GREENFIELD_RISK = "greenfield_risk"
    GENERIC_CONTEXT_RISK = "generic_context_risk"
    HARD_EXCLUSION = "hard_exclusion"
    DYNAMIC_CANDIDATE = "dynamic_candidate"
    LLM_SEMANTIC = "llm_semantic"


@dataclass(frozen=True)
class ArticleEvidenceInput:
    title: str
    abstract: str
    author_keywords: str = ""
    keywords_plus: str = ""
    keywords: str = ""
    wos_categories: str = ""
    research_areas: str = ""
    normalized_text: str = ""


@dataclass(frozen=True)
class RuleEvidence:
    renewal_action_hits: tuple[str, ...] = ()
    existing_urban_object_hits: tuple[str, ...] = ()
    policy_project_hits: tuple[str, ...] = ()
    risk_hits: tuple[str, ...] = ()
    hard_exclusion_reason: str = ""
    rule_topic_candidate: str = "Unknown"
    rule_confidence: float = 0.0
    explanation: str = ""

    @property
    def has_renewal_action(self) -> bool:
        return bool(self.renewal_action_hits)

    @property
    def has_existing_urban_object(self) -> bool:
        return bool(self.existing_urban_object_hits)

    @property
    def has_policy_project(self) -> bool:
        return bool(self.policy_project_hits)


@dataclass(frozen=True)
class TopicEvidence:
    topic_candidate: str = "Unknown"
    topic_group: str = "unknown"
    confidence: float = 0.0
    margin: float = 0.0
    top3: tuple[str, ...] = ()
    evidence: str = ""
    conflict_flag: int = 0


@dataclass(frozen=True)
class FamilyConsistencyEvidence:
    rule_group: str = "unknown"
    model_group: str = "unknown"
    consistency_status: str = "unknown"
    conflict_pattern: str = ""
    boundary_bucket: str = ""
    family_probability_urban: float = 0.0


@dataclass(frozen=True)
class ClusterEvidence:
    cluster_id: str = ""
    cluster_label_hint: str = ""
    cluster_positive_rate: float = 0.0
    cluster_topic_words: str = ""
    support: str = ""
    conflict: str = ""


@dataclass(frozen=True)
class LLMSemanticEvidence:
    attempted: bool = False
    used: bool = False
    research_object: str = ""
    object_is_existing_urban: bool | None = None
    existing_urban_object: str = ""
    renewal_action_present: bool | None = None
    renewal_action: str = ""
    action_is_main_subject: bool | None = None
    policy_or_governance_context: bool | None = None
    is_background_only: bool | None = None
    exclusion_risk: str = ""
    suggested_topic: str = ""
    label_hint: str = ""
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class EvidenceBundle:
    article: ArticleEvidenceInput
    rule: RuleEvidence
    topic: TopicEvidence
    family: FamilyConsistencyEvidence = field(default_factory=FamilyConsistencyEvidence)
    cluster: ClusterEvidence = field(default_factory=ClusterEvidence)
    llm: LLMSemanticEvidence = field(default_factory=LLMSemanticEvidence)
    dynamic: dict[str, Any] = field(default_factory=dict)
    current_label: str = ""
    current_topic: str = "Unknown"
    current_topic_group: str = "unknown"
    score: float = 0.0
    threshold: float = 0.45
    binary_topic_consistency_flag: int = 0
    evidence_balance: str = ""
    decision_source: str = ""
    binary_decision_source: str = ""

    def with_llm(self, llm: LLMSemanticEvidence) -> "EvidenceBundle":
        return replace(self, llm=llm)


@dataclass(frozen=True)
class StableDecisionResult:
    final_label: str
    urban_flag: str
    topic_final: str
    confidence: float
    status: StableDecisionStatus
    reason: str
    positive_evidence: str
    negative_evidence: str
    review_flag: int = 0
    review_reason: str = ""
    llm_semantic_evidence: str = ""
    evidence_conflict_type: str = ""


@dataclass(frozen=True)
class DecisionCandidate:
    label: str
    topic: str
    source: str
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class DecisionResult:
    """Compatibility decision result for legacy strategy_v3 imports."""

    label: str
    topic: str
    status: StableDecisionStatus
    reason: str
    evidence: str = ""
    confidence: float = 0.0
    candidates: tuple[DecisionCandidate, ...] = field(default_factory=tuple)


StrategyDecisionStatus = StableDecisionStatus
