"""Compatibility exports for the stable urban-renewal strategy contracts."""

from __future__ import annotations

from .evidence import (
    ArticleEvidenceInput,
    ClusterEvidence,
    DecisionCandidate,
    DecisionResult,
    EvidenceBundle,
    EvidenceKind,
    FamilyConsistencyEvidence,
    FourStepEvidence,
    LLMSemanticEvidence,
    RuleEvidence,
    StableDecisionResult,
    StableDecisionStatus,
    StrategyDecisionStatus,
    TopicEvidence,
)

UrbanEvidenceBundle = EvidenceBundle

__all__ = [
    "ArticleEvidenceInput",
    "ClusterEvidence",
    "DecisionCandidate",
    "DecisionResult",
    "EvidenceBundle",
    "EvidenceKind",
    "FamilyConsistencyEvidence",
    "FourStepEvidence",
    "LLMSemanticEvidence",
    "RuleEvidence",
    "StableDecisionResult",
    "StableDecisionStatus",
    "StrategyDecisionStatus",
    "TopicEvidence",
    "UrbanEvidenceBundle",
]
