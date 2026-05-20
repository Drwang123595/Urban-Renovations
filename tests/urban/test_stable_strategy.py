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
from src.urban.pipeline import postprocess as postprocess_module
from src.urban.topic_model.bertopic_service import BERTopicSignal
from src.urban.topic_model.local_classifier import TopicPrediction


def _prediction(
    topic_label: str,
    topic_group: str,
    *,
    confidence: float = 0.55,
    margin: float = 0.2,
) -> TopicPrediction:
    return TopicPrediction(
        topic_label=topic_label,
        topic_group=topic_group,
        topic_name="topic",
        confidence=confidence,
        matched_terms=[],
        binary_score=0.0,
        binary_probability=0.5,
        margin=margin,
        top_candidates=[],
    )


class _BoundaryNonurbanClassifier:
    def predict(self, _record):
        return _prediction("N3", "nonurban", confidence=0.52, margin=0.15)


class _NullBERTopicService:
    def predict(self, _record):
        return BERTopicSignal(
            available=False,
            status="disabled",
            mapped_label="",
            mapped_group="",
            mapped_name="",
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


class _StructuredSemanticOnlyStrategy:
    supports_structured_urban_semantic_evidence = True

    def __init__(self):
        self.tasks = []

    def process(self, title, abstract, session_path=None, **kwargs):
        task = str((kwargs.get("auxiliary_context") or {}).get("task", "") or "legacy_family_hint")
        self.tasks.append(task)
        return {
            "label_hint": "1",
            "confidence": 0.91,
            "object_is_existing_urban": True,
            "renewal_action_present": True,
            "action_is_main_subject": True,
            "is_background_only": False,
            "suggested_topic": "U9",
            "reason": "structured semantic evidence confirms the boundary case",
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

    assert decision.final_label == "0"
    assert decision.status == StableDecisionStatus.CONFLICT_REVIEW
    assert "without_core_evidence_rejected" in decision.reason


def test_stable_strategy_converts_current_positive_generic_governance_to_negative_when_mutating():
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

    stable = apply_stable_strategy(frame, mutate_final_fields=True)

    assert stable.loc[0, "final_label"] == "0"
    assert stable.loc[0, "urban_flag"] == "0"
    assert stable.loc[0, Schema.IS_URBAN_RENEWAL] == "0"
    assert stable.loc[0, "strategy_status"] == StableDecisionStatus.CONFLICT_REVIEW.value
    assert "without_core_evidence_rejected" in stable.loc[0, "strategy_reason"]


def test_stable_strategy_preserves_binary_decision_source_when_mutating_final_fields():
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
                "binary_decision_source": "binary_confidence_resolution",
            }
        ]
    )

    stable = apply_stable_strategy(frame, mutate_final_fields=True)

    assert stable.loc[0, "decision_source"] == "stable_strategy"
    assert stable.loc[0, "binary_decision_source"] == "binary_confidence_resolution"


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
    assert decision.reason == "core_object_action_main_subject"
    assert "renewal_action" in decision.positive_evidence
    assert "existing_urban_object" in decision.positive_evidence


def test_stable_strategy_outputs_four_step_evidence_fields_when_mutating():
    frame = pd.DataFrame(
        [
            {
                Schema.TITLE: "Adaptive reuse of heritage buildings in old districts",
                Schema.ABSTRACT: (
                    "The study evaluates adaptive reuse and rehabilitation of existing "
                    "heritage buildings in old urban districts."
                ),
                Schema.IS_URBAN_RENEWAL: "0",
                "final_label": "0",
                "urban_flag": "0",
                "topic_final": "U5",
                "topic_final_group": "urban",
                "family_probability_urban": 0.82,
                "urban_probability_score": 0.78,
                "binary_decision_threshold": 0.45,
            }
        ]
    )

    stable = apply_stable_strategy(frame, mutate_final_fields=True)

    assert stable.loc[0, "core_object_evidence"]
    assert stable.loc[0, "renewal_action_evidence"]
    assert stable.loc[0, "main_subject_evidence"] == "core_object_and_action_in_title_or_abstract"
    assert stable.loc[0, "risk_evidence"] == "none"
    assert "topic=U5" in stable.loc[0, "auxiliary_evidence"]
    assert stable.loc[0, "strategy_reason"] == "core_object_action_main_subject"


def test_stable_strategy_rejects_method_background_even_with_object_and_action_words():
    row = pd.Series(
        {
            Schema.TITLE: "Machine learning framework for urban renewal datasets",
            Schema.ABSTRACT: (
                "This method paper develops a graph neural network and uses old district "
                "redevelopment records only as a benchmark dataset."
            ),
            Schema.IS_URBAN_RENEWAL: "1",
            "final_label": "1",
            "urban_flag": "1",
            "topic_final": "U9",
            "topic_final_group": "urban",
            "family_probability_urban": 0.82,
            "urban_probability_score": 0.78,
            "binary_decision_threshold": 0.45,
        }
    )

    decision = decide_stable_strategy(build_evidence_bundle_from_row(row))

    assert decision.final_label == "0"
    assert decision.status == StableDecisionStatus.CONFLICT_REVIEW
    assert "background_or_method_only" in decision.reason
    assert "main_subject=background_or_method_only" in decision.negative_evidence


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
    assert stable.loc[0, "strategy_label"] == "1"
    assert stable.loc[0, "strategy_status"] == StableDecisionStatus.ACCEPTED_POSITIVE.value
    assert "dynamic_positive_candidate_with_core_evidence" in stable.loc[0, "strategy_reason"]


def test_stable_strategy_accepts_dynamic_positive_candidate_with_core_anchor_when_mutating():
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

    stable = apply_stable_strategy(frame, mutate_final_fields=True)

    assert stable.loc[0, "final_label"] == "1"
    assert stable.loc[0, "urban_flag"] == "1"
    assert stable.loc[0, "topic_final"] == "U1"
    assert stable.loc[0, "strategy_status"] == StableDecisionStatus.ACCEPTED_POSITIVE.value
    assert "dynamic_positive_candidate_with_core_evidence" in stable.loc[0, "strategy_reason"]


def test_stable_strategy_accepts_method_mixed_case_with_strong_urban_support():
    row = pd.Series(
        {
            Schema.TITLE: "Urban renewal governance evaluation model",
            Schema.ABSTRACT: "This paper evaluates urban renewal governance using a hybrid quantitative model.",
            Schema.IS_URBAN_RENEWAL: "1",
            "final_label": "1",
            "urban_flag": "1",
            "topic_final": "U9",
            "topic_final_group": "urban",
            "topic_rule": "N8",
            "topic_rule_group": "nonurban",
            "topic_local_label": "U9",
            "topic_local_group": "urban",
            "family_probability_urban": 0.86,
            "urban_probability_score": 0.9265,
            "binary_decision_threshold": 0.45,
            "stage1_risk_tags": "explicit_renewal_wording_but_other_object",
            "binary_topic_consistency_flag": 0,
        }
    )

    decision = decide_stable_strategy(build_evidence_bundle_from_row(row))

    assert decision.final_label == "1"
    assert decision.topic_final == "U9"
    assert decision.status == StableDecisionStatus.ACCEPTED_POSITIVE
    assert "strong_multisource_positive" in decision.reason


def test_urban_output_row_contract_includes_stable_strategy_defaults_and_compat_aliases():
    row = build_urban_output_row(
        "Generic city research",
        "This paper studies city policy discourse.",
        {"final_label": "0", "urban_flag": "0", Schema.IS_URBAN_RENEWAL: "0"},
    )

    assert row["strategy_label"] == ""
    assert row["strategy_status"] == ""
    assert row["strategy_reason"] == ""
    assert row["core_object_evidence"] == ""
    assert row["renewal_action_evidence"] == ""
    assert row["main_subject_evidence"] == ""
    assert row["risk_evidence"] == ""
    assert row["auxiliary_evidence"] == ""
    assert row["positive_evidence"] == ""
    assert row["negative_evidence"] == ""
    assert row["strategy_v3_label"] == ""
    assert row["strategy_v3_status"] == ""


def test_postprocess_stable_strategy_is_final_mutating_layer_by_default():
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

    assert processed.loc[0, "final_label"] == "0"
    assert processed.loc[0, "urban_flag"] == "0"
    assert processed.loc[0, "strategy_label"] == "0"
    assert processed.loc[0, "strategy_status"] == StableDecisionStatus.CONFLICT_REVIEW.value


def test_postprocess_binary_policy_records_conflict_without_final_label_mutation(monkeypatch):
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
            }
        ]
    )

    original_apply = postprocess_module.UrbanBinaryPolicyV2.apply

    def wrapped_apply(self, incoming):
        result = original_apply(self, incoming)
        assert result.loc[0, "final_label"] == "1"
        return result

    monkeypatch.setattr(postprocess_module.UrbanBinaryPolicyV2, "apply", wrapped_apply)

    processed = postprocess_urban_predictions(frame, run_context={})

    assert processed.loc[0, "binary_policy_action"] in {"conflict_review", "accept_positive"}
    assert processed.loc[0, "final_label"] == "0"
    assert processed.loc[0, "strategy_label"] == "0"


def test_postprocess_dynamic_binary_records_candidate_without_direct_mutation():
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
                "dynamic_topic_id": "DUR_0001",
                "dynamic_topic_confidence": 0.95,
                "dynamic_topic_size": 50,
                "dynamic_mapping_status": "mapped",
                "dynamic_to_fixed_topic_candidate": "U1",
                "dynamic_binary_candidate_label": "1",
                "dynamic_binary_candidate_action": "promote",
                "review_flag": 1,
                "uncertain_nonurban_guard_action": "review",
            }
        ]
    )

    processed = postprocess_urban_predictions(
        frame,
        run_context={
            "dynamic_binary_refinement_enabled": True,
            "dynamic_binary_refinement_unknown_only": False,
            "dynamic_binary_refinement_allow_flip": True,
            "dynamic_topics_min_topic_size": 1,
            "dynamic_topics_max_topics": 1,
            "dynamic_binary_refinement_min_topic_size": 1,
            "dynamic_topics_keyword_fallback_only": True,
        },
    )

    assert processed.loc[0, "dynamic_binary_override_applied"] == 0
    assert processed.loc[0, "dynamic_binary_override_source"] == "dynamic_topic_refiner_flip_review"
    assert processed.loc[0, "final_label"] == "1"
    assert processed.loc[0, "strategy_reason"] == "dynamic_positive_candidate_with_core_evidence"


def test_postprocess_strict_stable_release_raises_when_binary_policy_fails(monkeypatch):
    frame = pd.DataFrame(
        [
            {
                Schema.TITLE: "Urban renewal of old districts",
                Schema.ABSTRACT: "The paper studies redevelopment.",
                "final_label": "1",
                "urban_flag": "1",
                "topic_final": "U1",
            }
        ]
    )

    def fail_policy(self, _frame):
        raise RuntimeError("policy boom")

    monkeypatch.setattr(postprocess_module.UrbanBinaryPolicyV2, "apply", fail_policy)

    try:
        postprocess_urban_predictions(frame, run_context={"experiment_track": "stable_release"})
    except RuntimeError as exc:
        assert "Urban binary policy failed" in str(exc)
    else:
        raise AssertionError("stable_release postprocess must fail closed when binary policy fails")


def test_postprocess_research_matrix_records_warning_when_binary_policy_fails(monkeypatch):
    frame = pd.DataFrame(
        [
            {
                Schema.TITLE: "Urban renewal of old districts",
                Schema.ABSTRACT: "The paper studies redevelopment.",
                "final_label": "1",
                "urban_flag": "1",
                "topic_final": "U1",
            }
        ]
    )

    def fail_policy(self, _frame):
        raise RuntimeError("policy boom")

    monkeypatch.setattr(postprocess_module.UrbanBinaryPolicyV2, "apply", fail_policy)
    context = {"experiment_track": "research_matrix"}

    processed = postprocess_urban_predictions(frame, run_context=context)

    assert processed.loc[0, "final_label"] == "1"
    failed_layers = [
        item for item in context["urban_postprocess_layers"] if item["status"] == "failed"
    ]
    assert failed_layers[-1]["layer"] == "binary_policy_v2"
    assert "policy boom" in failed_layers[-1]["error"]


def test_postprocess_strict_stable_release_raises_when_enabled_dynamic_topic_fails(monkeypatch):
    frame = pd.DataFrame([{Schema.TITLE: "Urban renewal", Schema.ABSTRACT: "Redevelopment."}])

    class FailingDiscovery:
        def __init__(self, _config):
            pass

        def enrich(self, *_args, **_kwargs):
            raise RuntimeError("dynamic topic boom")

    monkeypatch.setattr(postprocess_module, "DynamicTopicDiscovery", FailingDiscovery)

    try:
        postprocess_urban_predictions(
            frame,
            run_context={"experiment_track": "stable_release", "dynamic_topics_enabled": True},
        )
    except RuntimeError as exc:
        assert "Dynamic topic enrichment failed" in str(exc)
    else:
        raise AssertionError("stable_release postprocess must fail closed for enabled dynamic topics")


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


def test_hybrid_classifier_uses_stable_strategy_llm_for_boundary_final_decision():
    from src.urban.hybrid import classifier as classifier_module

    original_classifier = classifier_module.UrbanTopicClassifier
    classifier_module.UrbanTopicClassifier = lambda: _BoundaryNonurbanClassifier()
    classifier = UrbanHybridClassifier(
        _LegacyJSONLLMStrategy(),
        bertopic_service=_NullBERTopicService(),
        llm_assist_enabled=True,
    )

    try:
        result = classifier.classify(
            "Property-led regeneration and local policy networks",
            "The article studies regeneration policy and institutional networks.",
        )
    finally:
        classifier_module.UrbanTopicClassifier = original_classifier

    assert result["decision_source"].endswith("stable_strategy")
    assert result["strategy_status"] == StableDecisionStatus.LLM_SUPPORTED_POSITIVE.value
    assert result["final_label"] == "1"
    assert result["llm_attempted"] == 1
    assert result["llm_used"] == 1
    assert "object_is_existing_urban=True" in result["llm_semantic_evidence"]


def test_hybrid_classifier_calls_llm_only_from_stable_strategy_boundary_layer():
    from src.urban.hybrid import classifier as classifier_module

    strategy = _LegacyJSONLLMStrategy()
    original_classifier = classifier_module.UrbanTopicClassifier
    classifier_module.UrbanTopicClassifier = lambda: _BoundaryNonurbanClassifier()
    classifier = UrbanHybridClassifier(
        strategy,
        bertopic_service=_NullBERTopicService(),
        llm_assist_enabled=True,
    )

    try:
        result = classifier.classify(
            "Property-led regeneration and local policy networks",
            "The article studies regeneration policy and institutional networks.",
        )
    finally:
        classifier_module.UrbanTopicClassifier = original_classifier

    assert strategy.calls == 1
    assert result["strategy_status"] == StableDecisionStatus.LLM_SUPPORTED_POSITIVE.value
    assert result["llm_family_hint"] == ""


def test_hybrid_classifier_skips_legacy_family_hint_when_strategy_supports_structured_semantics():
    from src.urban.hybrid import classifier as classifier_module

    strategy = _StructuredSemanticOnlyStrategy()
    original_classifier = classifier_module.UrbanTopicClassifier
    classifier_module.UrbanTopicClassifier = lambda: _BoundaryNonurbanClassifier()
    classifier = UrbanHybridClassifier(
        strategy,
        bertopic_service=_NullBERTopicService(),
        llm_assist_enabled=True,
    )

    try:
        result = classifier.classify(
            "Property-led regeneration and local policy networks",
            "The article studies regeneration policy and institutional networks.",
        )
    finally:
        classifier_module.UrbanTopicClassifier = original_classifier

    assert strategy.tasks == ["urban_renewal_semantic_evidence"]
    assert result["llm_family_hint"] == ""
    assert result["strategy_status"] == StableDecisionStatus.LLM_SUPPORTED_POSITIVE.value
    assert result["final_label"] == "1"


def test_hybrid_classifier_can_defer_stable_strategy_for_batch_postprocess():
    from src.urban.hybrid import classifier as classifier_module

    strategy = _StructuredSemanticOnlyStrategy()
    original_classifier = classifier_module.UrbanTopicClassifier
    classifier_module.UrbanTopicClassifier = lambda: _BoundaryNonurbanClassifier()
    classifier = UrbanHybridClassifier(
        strategy,
        bertopic_service=_NullBERTopicService(),
        llm_assist_enabled=True,
    )

    try:
        result = classifier.classify(
            "Property-led regeneration and local policy networks",
            "The article studies regeneration policy and institutional networks.",
            finalize_strategy=False,
        )
    finally:
        classifier_module.UrbanTopicClassifier = original_classifier

    assert strategy.tasks == []
    assert result.get("strategy_status", "") == ""
    assert not str(result.get("decision_source", "")).endswith("stable_strategy")


def test_batch_deferred_strategy_calls_structured_llm_once_in_postprocess():
    from src.urban.hybrid import classifier as classifier_module

    strategy = _StructuredSemanticOnlyStrategy()
    original_classifier = classifier_module.UrbanTopicClassifier
    classifier_module.UrbanTopicClassifier = lambda: _BoundaryNonurbanClassifier()
    classifier = UrbanHybridClassifier(
        strategy,
        bertopic_service=_NullBERTopicService(),
        llm_assist_enabled=True,
    )

    try:
        row = classifier.classify(
            "Property-led regeneration and local policy networks",
            "The article studies regeneration policy and institutional networks.",
            finalize_strategy=False,
        )
    finally:
        classifier_module.UrbanTopicClassifier = original_classifier

    frame = pd.DataFrame([{Schema.TITLE: "Property-led regeneration and local policy networks", Schema.ABSTRACT: "The article studies regeneration policy and institutional networks.", **row}])
    processed = postprocess_urban_predictions(
        frame,
        run_context={
            "urban_stable_strategy_llm_strategy": strategy,
            "urban_stable_strategy_llm_enabled": True,
        },
    )

    assert strategy.tasks == ["urban_renewal_semantic_evidence"]
    assert processed.loc[0, "strategy_status"] == StableDecisionStatus.LLM_SUPPORTED_POSITIVE.value
    assert processed.loc[0, "final_label"] == "1"


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


def test_stable_strategy_pipeline_uses_llm_to_mutate_uncertain_negative_boundary_sample():
    analyzer = _BoundarySemanticAnalyzer()
    strategy = StableUrbanStrategy(llm_analyzer=analyzer)

    result = strategy.classify_row(
        {
            Schema.TITLE: "Property-led regeneration and local policy networks",
            Schema.ABSTRACT: "The article studies regeneration governance for an existing inner-city district.",
            Schema.IS_URBAN_RENEWAL: "0",
            "final_label": "0",
            "urban_flag": "0",
            "topic_final": "N3",
            "topic_final_group": "nonurban",
            "urban_probability_score": 0.49,
            "binary_decision_threshold": 0.5,
        },
        mutate_final_fields=True,
    )

    assert analyzer.called is True
    assert result["final_label"] == "1"
    assert result["urban_flag"] == "1"
    assert result["topic_final"] == "U9"
    assert result["decision_source"].endswith("stable_strategy")
    assert result["strategy_status"] == StableDecisionStatus.LLM_SUPPORTED_POSITIVE.value
    assert result["llm_attempted"] == 1
    assert result["llm_used"] == 1


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


def test_llm_semantic_analyzer_does_not_use_positive_label_without_required_semantic_triplet():
    class WeakPositiveLLMStrategy:
        def process(self, *args, **kwargs):
            return {
                "label_hint": "1",
                "confidence": 0.95,
                "object_is_existing_urban": True,
                "renewal_action_present": True,
                "action_is_main_subject": False,
                "is_background_only": False,
                "suggested_topic": "U9",
                "reason": "mentions regeneration but not as the main subject",
            }

    analyzer = LLMSemanticAnalyzer(WeakPositiveLLMStrategy(), enabled=True)

    evidence = analyzer.analyze(
        ArticleEvidenceInput(
            title="Governance discourse around regeneration",
            abstract="Regeneration is mentioned as policy context.",
            normalized_text="governance discourse regeneration policy context",
        ),
        rule=RuleEvidence(renewal_action_hits=("regeneration",), rule_topic_candidate="N3"),
        topic=TopicEvidence(topic_candidate="N3", topic_group="nonurban", conflict_flag=1),
    )

    assert evidence.attempted is True
    assert evidence.label_hint == "1"
    assert evidence.used is False
    assert "requires_existing_object_action_main_subject" in evidence.reason
