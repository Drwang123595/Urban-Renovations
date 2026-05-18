import pandas as pd

from src.runtime.config import Schema
from src.urban.hybrid.binary_policy_v2 import UrbanBinaryPolicyV2


class _FakeLLMClient:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def chat_completion(self, messages, temperature=0.0, max_retries=2):
        self.calls += 1
        return self.response


def _frame(row):
    defaults = {
        Schema.TITLE: "Generic city research",
        Schema.ABSTRACT: "This article studies urban policy and governance.",
        Schema.IS_URBAN_RENEWAL: "1",
        "urban_flag": "1",
        "final_label": "1",
        "topic_final": "Unknown",
        "topic_final_group": "unknown",
        "urban_probability_score": 0.62,
        "binary_decision_threshold": 0.45,
        "binary_topic_consistency_flag": 1,
        "binary_decision_source": "binary_confidence_resolution",
        "decision_source": "binary_confidence_resolution",
        "decision_explanation": "",
        "binary_decision_evidence": "",
        "evidence_balance": "conflict_positive",
        "metadata_route_reason": "",
        "stage1_risk_tags": "",
        "dynamic_binary_candidate_label": "",
        "llm_used": 0,
        "llm_attempted": 0,
    }
    defaults.update(row)
    return pd.DataFrame([defaults])


def test_policy_v2_hard_negative_cannot_be_lifted_to_positive():
    frame = _frame(
        {
            "metadata_route_reason": "rural_nonurban",
            Schema.TITLE: "Rural regeneration and property-led renewal",
            Schema.ABSTRACT: "This paper studies rural renewal and countryside development.",
        }
    )

    result = UrbanBinaryPolicyV2().apply(frame)

    assert result.loc[0, "final_label"] == "0"
    assert result.loc[0, "urban_flag"] == "0"
    assert result.loc[0, Schema.IS_URBAN_RENEWAL] == "0"
    assert result.loc[0, "binary_policy_action"] == "protected_negative"
    assert result.loc[0, "llm_adjudication_required"] == 0


def test_policy_v2_keeps_generic_unknown_positive_for_conflict_review():
    frame = _frame(
        {
            Schema.TITLE: "Urban governance and sustainability transitions",
            Schema.ABSTRACT: "This article studies city governance and sustainability policy without redevelopment projects.",
        }
    )

    result = UrbanBinaryPolicyV2().apply(frame)

    assert result.loc[0, "final_label"] == "1"
    assert result.loc[0, "binary_policy_action"] == "conflict_review"
    assert result.loc[0, "llm_adjudication_required"] == 1
    assert "binary_positive_unknown_topic" in result.loc[0, "binary_policy_conflict_type"]


def test_policy_v2_preserves_conflict_positive_when_core_renewal_and_existing_object_are_present():
    frame = _frame(
        {
            Schema.TITLE: "Brownfield redevelopment and community regeneration in an old industrial district",
            Schema.ABSTRACT: "The study evaluates urban regeneration, brownfield redevelopment, and adaptive reuse of an existing industrial site.",
            "topic_final": "N3",
            "topic_final_group": "nonurban",
        }
    )

    result = UrbanBinaryPolicyV2().apply(frame)

    assert result.loc[0, "final_label"] == "1"
    assert result.loc[0, "binary_policy_action"] == "accept_positive"
    assert result.loc[0, "llm_adjudication_required"] == 0


def test_policy_v2_uses_dynamic_negative_candidate_as_conflict_evidence():
    frame = _frame(
        {
            "topic_final": "N8",
            "topic_final_group": "nonurban",
            "dynamic_binary_candidate_label": "0",
            Schema.TITLE: "Machine learning model for general urban policy text mining",
            Schema.ABSTRACT: "This paper proposes an algorithmic framework and uses urban renewal only as an application phrase.",
        }
    )

    result = UrbanBinaryPolicyV2().apply(frame)

    assert result.loc[0, "final_label"] == "1"
    assert result.loc[0, "binary_policy_action"] == "conflict_review"
    assert "dynamic_topic_negative_candidate" in result.loc[0, "binary_policy_conflict_type"]


def test_policy_v2_llm_adjudication_overrides_only_high_confidence_conflict_rows():
    frame = _frame(
        {
            Schema.TITLE: "State-led gentrification in an old neighborhood",
            Schema.ABSTRACT: "The article studies displacement during urban regeneration and redevelopment of an existing neighborhood.",
            "topic_final": "Unknown",
            "topic_final_group": "unknown",
        }
    )
    client = _FakeLLMClient('{"label":"1","confidence":0.91,"reason":"existing neighborhood regeneration"}')

    result = UrbanBinaryPolicyV2(llm_client=client, llm_enabled=True).apply(frame)

    assert client.calls == 0
    assert result.loc[0, "final_label"] == "1"
    assert result.loc[0, "llm_used"] == 0


def test_policy_v2_llm_low_confidence_does_not_override_rule_result():
    frame = _frame({})
    client = _FakeLLMClient('{"label":"1","confidence":0.40,"reason":"generic urban context"}')

    result = UrbanBinaryPolicyV2(llm_client=client, llm_enabled=True).apply(frame)

    assert client.calls == 1
    assert result.loc[0, "final_label"] == "1"
    assert result.loc[0, "llm_attempted"] == 1
    assert result.loc[0, "llm_used"] == 0


def test_policy_v2_llm_high_confidence_can_restore_difficult_positive():
    frame = _frame(
        {
            Schema.TITLE: "Renewal governance and resident relocation",
            Schema.ABSTRACT: "The paper studies regeneration governance and resident relocation in planning discourse.",
        }
    )
    client = _FakeLLMClient('{"label":"1","confidence":0.88,"reason":"regeneration of existing urban area"}')

    result = UrbanBinaryPolicyV2(llm_client=client, llm_enabled=True).apply(frame)

    assert client.calls == 1
    assert result.loc[0, "final_label"] == "1"
    assert result.loc[0, "llm_attempted"] == 1
    assert result.loc[0, "llm_used"] == 1


def test_policy_v2_llm_high_confidence_can_reject_difficult_false_positive():
    frame = _frame(
        {
            Schema.TITLE: "Generic digital participation in cities",
            Schema.ABSTRACT: "The paper studies an algorithmic framework for municipal participation platforms without redevelopment or upgrading.",
        }
    )
    client = _FakeLLMClient('{"label":"0","confidence":0.90,"reason":"generic urban governance only"}')

    result = UrbanBinaryPolicyV2(llm_client=client, llm_enabled=True).apply(frame)

    assert client.calls == 1
    assert result.loc[0, "final_label"] == "0"
    assert result.loc[0, "llm_attempted"] == 1
    assert result.loc[0, "llm_used"] == 1
