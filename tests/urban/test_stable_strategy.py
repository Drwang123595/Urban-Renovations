import pandas as pd

from src.runtime.config import Schema
from src.urban.hybrid.classifier import UrbanHybridClassifier
from src.urban.pipeline.io import build_urban_output_row
from src.urban.pipeline.postprocess import postprocess_urban_predictions
from src.urban.strategy import (
    ArticleEvidenceInput,
    LLMSemanticAnalyzer,
    LLMSemanticEvidence,
    RuleEvidence,
    StableDecisionStatus,
    StableUrbanStrategy,
    TopicEvidence,
    apply_stable_strategy,
    build_evidence_bundle_from_row,
    decide_stable_strategy,
)


class _NoCallLLMStrategy:
    def process(self, *args, **kwargs):
        raise AssertionError("Stable strategy overlay should not call the LLM when LLM assist is disabled")


class _BoundarySemanticAnalyzer:
    def __init__(self):
        self.called = False

    def should_call(self, *, rule, topic):
        return True

    def analyze(self, article, *, rule, topic, session_path=None, audit_metadata=None):
        self.called = True
        return LLMSemanticEvidence(
            attempted=True,
            used=True,
            research_object="property-led regeneration of an existing inner-city district",
            object_is_existing_urban=True,
            existing_urban_object="existing inner-city district",
            renewal_action_present=True,
            renewal_action="regeneration",
            action_is_main_subject=True,
            is_background_only=False,
            suggested_topic="U9",
            label_hint="1",
            confidence=0.9,
            reason="LLM confirms object-action-main-subject relation",
        )


class _LegacyJSONLLMStrategy:
    def __init__(self):
        self.calls = 0

    def process(self, title, abstract, session_path=None):
        self.calls += 1
        return {
            "label_hint": "1",
            "confidence": 0.91,
            "object_is_existing_urban": True,
            "renewal_action_present": True,
            "action_is_main_subject": True,
            "is_background_only": False,
            "suggested_topic": "U9",
            "reason": "legacy strategy still returns structured JSON-compatible data",
        }


def test_stable_strategy_protects_hard_exclusion_without_mutating_current_output():
    frame = pd.DataFrame(
        [
            {
                Schema.TITLE: "Urban renewal moves in dimer models",
                Schema.ABSTRACT: "This paper studies bipartite graph tiling methods and combinatorics.",
                Schema.IS_URBAN_RENEWAL: "1",
                "final_label": "1",
                "urban_flag": "1",
                "topic_final": "U1",
                "topic_final_group": "urban",
                "metadata_route_reason": "math_term_misuse",
                "binary_decision_source": "binary_hard_negative_override",
            }
        ]
    )

    stable = apply_stable_strategy(frame, mutate_final_fields=False)

    assert stable.loc[0, "final_label"] == "1"
    assert stable.loc[0, "strategy_label"] == "0"
    assert stable.loc[0, "strategy_status"] == StableDecisionStatus.EXCLUDED_NEGATIVE.value
    assert "math_term_misuse" in stable.loc[0, "strategy_reason"]


def test_stable_strategy_rejects_generic_governance_without_object_action_relation():
    row = pd.Series(
        {
            Schema.TITLE: "City governance and policy networks",
            Schema.ABSTRACT: "This paper studies city policy discourse and institutional narratives.",
            "final_label": "1",
            "urban_flag": "1",
            "topic_final": "Unknown",
            "topic_final_group": "unknown",
            "urban_probability_score": 0.47,
            "binary_decision_threshold": 0.45,
            "decision_source": "unknown_review",
            "binary_topic_consistency_flag": 1,
        }
    )

    decision = decide_stable_strategy(build_evidence_bundle_from_row(row))

    assert decision.final_label == "1"
    assert decision.status == StableDecisionStatus.UNKNOWN_REVIEW
    assert "unknown_topic" in decision.reason


def test_stable_strategy_accepts_strong_positive_with_action_and_existing_object():
    row = pd.Series(
        {
            Schema.TITLE: "Urban renewal of old industrial districts",
            Schema.ABSTRACT: "The study evaluates brownfield redevelopment and adaptive reuse of existing buildings.",
            "final_label": "1",
            "urban_flag": "1",
            "topic_final": "U5",
            "topic_final_group": "urban",
            "family_probability_urban": 0.82,
            "urban_probability_score": 0.78,
            "binary_decision_threshold": 0.45,
        }
    )

    decision = decide_stable_strategy(build_evidence_bundle_from_row(row))

    assert decision.final_label == "1"
    assert decision.status == StableDecisionStatus.ACCEPTED_POSITIVE
    assert "renewal_action_and_existing_urban_object" in decision.reason
    assert "renewal_action" in decision.positive_evidence
    assert "existing_urban_object" in decision.positive_evidence


def test_stable_strategy_uses_structured_llm_evidence_for_boundary_positive():
    row = pd.Series(
        {
            Schema.TITLE: "Property-led regeneration and local policy networks",
            Schema.ABSTRACT: "The article studies regeneration governance for an existing inner-city district.",
            "final_label": "0",
            "urban_flag": "0",
            "topic_final": "N3",
            "topic_final_group": "nonurban",
            "urban_probability_score": 0.49,
            "binary_decision_threshold": 0.5,
        }
    )
    evidence = build_evidence_bundle_from_row(row)
    evidence = evidence.with_llm(
        LLMSemanticEvidence(
            attempted=True,
            used=True,
            research_object="property-led regeneration of an existing inner-city district",
            object_is_existing_urban=True,
            existing_urban_object="existing inner-city district",
            renewal_action_present=True,
            renewal_action="regeneration",
            action_is_main_subject=True,
            is_background_only=False,
            suggested_topic="U9",
            label_hint="1",
            confidence=0.88,
            reason="regeneration governance is the main subject",
        )
    )

    decision = decide_stable_strategy(evidence)

    assert decision.final_label == "1"
    assert decision.topic_final == "U9"
    assert decision.status == StableDecisionStatus.LLM_SUPPORTED_POSITIVE
    assert decision.review_flag == 1
    assert "llm_supported_boundary_case" in decision.review_reason


def test_stable_strategy_keeps_dynamic_candidate_as_review_without_final_mutation():
    frame = pd.DataFrame(
        [
            {
                Schema.TITLE: "Urban regeneration of old districts",
                Schema.ABSTRACT: "This paper studies urban regeneration and redevelopment of old districts.",
                Schema.IS_URBAN_RENEWAL: "0",
                "final_label": "0",
                "urban_flag": "0",
                "topic_final": "N4",
                "topic_final_group": "nonurban",
                "dynamic_binary_candidate_label": "1",
                "dynamic_to_fixed_topic_candidate": "U1",
                "dynamic_binary_override_applied": 0,
            }
        ]
    )

    stable = apply_stable_strategy(frame, mutate_final_fields=False)

    assert stable.loc[0, "final_label"] == "0"
    assert stable.loc[0, "strategy_label"] == "0"
    assert stable.loc[0, "strategy_status"] == StableDecisionStatus.DYNAMIC_CANDIDATE_REVIEW.value
    assert stable.loc[0, "review_flag"] == 1


def test_urban_output_row_contract_includes_stable_strategy_defaults_and_compat_aliases():
    row = build_urban_output_row(
        "Generic city research",
        "This paper studies city policy discourse.",
        {"final_label": "0", "urban_flag": "0", Schema.IS_URBAN_RENEWAL: "0"},
    )

    assert row["strategy_label"] == ""
    assert row["strategy_status"] == ""
    assert row["strategy_reason"] == ""
    assert row["positive_evidence"] == ""
    assert row["negative_evidence"] == ""
    assert row["strategy_v3_label"] == ""
    assert row["strategy_v3_status"] == ""


def test_postprocess_appends_stable_strategy_without_mutating_current_output_by_default():
    frame = pd.DataFrame(
        [
            {
                Schema.TITLE: "City governance and policy networks",
                Schema.ABSTRACT: "This paper studies city policy discourse and institutional narratives.",
                Schema.IS_URBAN_RENEWAL: "1",
                "final_label": "1",
                "urban_flag": "1",
                "topic_final": "Unknown",
                "topic_final_group": "unknown",
                "urban_probability_score": 0.47,
                "binary_decision_threshold": 0.45,
                "binary_topic_consistency_flag": 1,
            }
        ]
    )

    processed = postprocess_urban_predictions(
        frame,
        run_context={"urban_binary_policy_v2_enabled": False},
    )

    assert processed.loc[0, "final_label"] == "1"
    assert processed.loc[0, "strategy_label"] == "1"
    assert processed.loc[0, "strategy_status"] == StableDecisionStatus.UNKNOWN_REVIEW.value


def test_hybrid_classifier_appends_stable_strategy_explanation_without_mutating_final_fields():
    classifier = UrbanHybridClassifier(
        _NoCallLLMStrategy(),
        llm_assist_enabled=False,
    )

    result = classifier.classify(
        "Urban renewal of old industrial districts",
        "The study evaluates brownfield redevelopment and adaptive reuse of existing buildings.",
    )

    assert result["final_label"] in {"0", "1"}
    assert result["urban_flag"] == result["final_label"]
    assert result[Schema.IS_URBAN_RENEWAL] == result["final_label"]
    assert result["strategy_label"] == "1"
    assert result["strategy_status"] == StableDecisionStatus.ACCEPTED_POSITIVE.value
    assert "renewal_action" in result["positive_evidence"]


def test_stable_strategy_pipeline_can_use_llm_semantic_evidence_on_boundary_samples():
    analyzer = _BoundarySemanticAnalyzer()
    strategy = StableUrbanStrategy(llm_analyzer=analyzer)

    result = strategy.classify_row(
        {
            Schema.TITLE: "Property-led regeneration and local policy networks",
            Schema.ABSTRACT: "The article studies regeneration governance for an existing inner-city district.",
            "final_label": "0",
            "urban_flag": "0",
            "topic_final": "N3",
            "topic_final_group": "nonurban",
            "urban_probability_score": 0.49,
            "binary_decision_threshold": 0.5,
        },
        mutate_final_fields=False,
    )

    assert analyzer.called is True
    assert result["final_label"] == "0"
    assert result["strategy_label"] == "1"
    assert result["strategy_topic"] == "U9"
    assert result["strategy_status"] == StableDecisionStatus.LLM_SUPPORTED_POSITIVE.value
    assert "LLM confirms" in result["llm_semantic_evidence"]


def test_llm_semantic_analyzer_supports_legacy_strategy_process_signature():
    strategy = _LegacyJSONLLMStrategy()
    analyzer = LLMSemanticAnalyzer(strategy, enabled=True)

    evidence = analyzer.analyze(
        ArticleEvidenceInput(
            title="Property-led regeneration",
            abstract="Regeneration governance for an existing inner-city district.",
            normalized_text="property led regeneration governance existing inner city district",
        ),
        rule=RuleEvidence(
            renewal_action_hits=("regeneration",),
            policy_project_hits=("governance",),
            rule_topic_candidate="N3",
        ),
        topic=TopicEvidence(topic_candidate="N3", topic_group="nonurban", conflict_flag=1),
    )

    assert strategy.calls == 1
    assert evidence.used is True
    assert evidence.label_hint == "1"
    assert evidence.suggested_topic == "U9"
