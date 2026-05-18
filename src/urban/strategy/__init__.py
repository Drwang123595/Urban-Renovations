"""Stable urban-renewal strategy contracts and decision helpers."""

from .contracts import (
    ArticleEvidenceInput,
    ClusterEvidence,
    DecisionCandidate,
    DecisionResult,
    EvidenceBundle,
    EvidenceKind,
    FamilyConsistencyEvidence,
    LLMSemanticEvidence,
    RuleEvidence,
    StableDecisionResult,
    StableDecisionStatus,
    StrategyDecisionStatus,
    TopicEvidence,
)
from .decision import StableUrbanDecisionEngine, decide_stable_strategy
from .llm_semantic import LLMSemanticAnalyzer
from .output import STABLE_STRATEGY_DEFAULTS, STRATEGY_V3_DEFAULTS
from .pipeline import StableUrbanStrategy, apply_stable_strategy, build_evidence_bundle_from_row
from .v3 import apply_strategy_v3_shadow, decide_strategy_v3

__all__ = [
    "ArticleEvidenceInput",
    "ClusterEvidence",
    "DecisionCandidate",
    "DecisionResult",
    "EvidenceBundle",
    "EvidenceKind",
    "FamilyConsistencyEvidence",
    "LLMSemanticAnalyzer",
    "LLMSemanticEvidence",
    "RuleEvidence",
    "STABLE_STRATEGY_DEFAULTS",
    "StableDecisionResult",
    "StableDecisionStatus",
    "StableUrbanDecisionEngine",
    "StableUrbanStrategy",
    "StrategyDecisionStatus",
    "STRATEGY_V3_DEFAULTS",
    "TopicEvidence",
    "apply_stable_strategy",
    "apply_strategy_v3_shadow",
    "build_evidence_bundle_from_row",
    "decide_stable_strategy",
    "decide_strategy_v3",
]
