from src.runtime.config import Schema
from src.urban.core.metadata import UrbanMetadataRecord
from src.urban.hybrid.binary_scoring import apply_binary_decision
from src.urban.hybrid.classifier import UrbanHybridClassifier
from src.urban.hybrid.result_contract import build_final_result
from src.urban.rules.metadata_filter import METADATA_ROUTE_UNCERTAIN, MetadataRouteResult


class _NoCallLLMStrategy:
    def process(self, *args, **kwargs):
        raise AssertionError("LLM review should not be called")


class _NullBERTopicService:
    def signal_for_record(self, *_args, **_kwargs):
        return None


def _uncertain_route() -> MetadataRouteResult:
    return MetadataRouteResult(route=METADATA_ROUTE_UNCERTAIN, reason="uncertain_pass")


def test_result_contract_helper_preserves_unknown_positive_binary_label():
    classifier = UrbanHybridClassifier(_NoCallLLMStrategy(), bertopic_service=_NullBERTopicService())
    base = {
        "urban_probability_score": 0.72,
        "binary_decision_threshold": 0.45,
        "binary_decision_source": "binary_confidence_resolution",
        "binary_decision_evidence": "score_above_threshold",
        "taxonomy_coverage_status": "unknown",
        "topic_rule": "",
        "topic_local_label": "Unknown",
        "family_probability_urban": 0.8,
        "topic_binary_probability": 0.7,
        "llm_family_hint": "",
        "stage1_risk_tags": "",
    }

    result = build_final_result(
        classifier,
        base,
        final_topic="Unknown",
        decision_source="unknown_review",
        decision_reason="binary_positive_topic_unknown",
        confidence=0.72,
        review_flag=1,
        review_reason="binary_topic_inconsistency",
        binary_label="1",
    )

    assert result["final_label"] == "1"
    assert result["urban_flag"] == "1"
    assert result[Schema.IS_URBAN_RENEWAL] == "1"
    assert result["topic_final"] == "Unknown"
    assert result["binary_audit_resolution_action"] == "positive_binary_conflict_audit"
    assert "topic=Unknown/unknown" in result["decision_explanation"]


def test_binary_scoring_helper_applies_context_relevance_floor():
    classifier = UrbanHybridClassifier(_NoCallLLMStrategy(), bertopic_service=_NullBERTopicService())
    base = {
        "family_probability_urban": 0.02,
        "topic_binary_probability": 0.0,
        "topic_rule": "N3",
        "topic_local_label": "N3",
        "topic_within_family_label": "",
        "bertopic_hint_label": "",
        "llm_family_hint": "",
        "stage1_risk_tags": "",
    }

    label, confidence, review_flag, review_reason = apply_binary_decision(
        classifier,
        base,
        record=UrbanMetadataRecord(
            title="Urban governance in metropolitan housing policy",
            abstract="This paper studies policy coordination, planning and community infrastructure in cities.",
        ),
        route_result=_uncertain_route(),
        final_topic="Unknown",
        decision_source="unknown_review",
        decision_reason="rule_unknown_local_unknown",
        confidence=0.2,
        review_flag=1,
        review_reason="unknown_review",
    )

    assert label == "1"
    assert confidence >= 0.45
    assert review_flag == 1
    assert "binary_low_confidence" in review_reason
    assert base["binary_recall_calibration_flag"] == 1
    assert base["binary_recall_calibration_tier"] == "context_relevance_floor"
