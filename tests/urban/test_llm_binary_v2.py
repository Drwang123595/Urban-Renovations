import pandas as pd

from src.runtime.config import Schema
from src.urban.hybrid.binary_finalizer import LlmBinaryFinalizer
from src.urban.pipeline.contracts import URBAN_RESULT_COLUMNS, apply_urban_output_defaults
from src.urban.hybrid.llm_adjudicator import LlmAdjudicator
from src.urban.hybrid.llm_triage import LlmTriagePolicy
from src.urban.pipeline.postprocess import postprocess_urban_predictions


class _FakeLLMClient:
    def __init__(self, response: str):
        self.response = response
        self.calls = 0
        self.messages = []

    def chat_completion(self, messages, temperature=0.0, max_retries=2):
        self.calls += 1
        self.messages.append(messages)
        return self.response


def _row(**overrides):
    base = {
        Schema.TITLE: "Generic urban research",
        Schema.ABSTRACT: "This article studies municipal policy.",
        Schema.IS_URBAN_RENEWAL: "0",
        "final_label": "0",
        "urban_flag": "0",
        "urban_probability_score": 0.2,
        "binary_decision_threshold": 0.45,
        "binary_decision_source": "binary_confidence_resolution",
        "decision_source": "binary_confidence_resolution",
        "decision_reason": "baseline",
        "metadata_route_reason": "",
        "stage1_risk_tags": "",
        "topic_final": "N3",
        "topic_final_group": "nonurban",
        "topic_rule": "N3",
        "topic_rule_group": "nonurban",
        "topic_local_label": "N3",
        "topic_local_group": "nonurban",
        "family_predicted_family": "nonurban",
        "family_probability_urban": 0.1,
        "review_flag": 0,
        "review_reason": "",
        "unknown_recovery_path": "",
        "boundary_bucket": "",
        "binary_policy_action": "accept_negative",
        "binary_policy_conflict_type": "",
        "llm_used": 0,
        "llm_attempted": 0,
    }
    base.update(overrides)
    return base


def test_triage_triggers_near_threshold_and_skips_high_confidence_positive():
    policy = LlmTriagePolicy()

    near = pd.Series(_row(urban_probability_score=0.50, final_label="1", urban_flag="1"))
    assert policy.evaluate(near).should_call is True
    assert "near_threshold" in policy.evaluate(near).reasons

    confident = pd.Series(
        _row(
            final_label="1",
            urban_flag="1",
            urban_probability_score=0.86,
            topic_final="U5",
            topic_final_group="urban",
            topic_rule="U5",
            topic_rule_group="urban",
            topic_local_label="U5",
            topic_local_group="urban",
            family_predicted_family="urban",
            family_probability_urban=0.96,
        )
    )
    decision = policy.evaluate(confident)
    assert decision.should_call is False
    assert decision.action == "skip_high_confidence_positive"


def test_triage_never_calls_llm_for_hard_negative():
    policy = LlmTriagePolicy()

    rural = pd.Series(
        _row(
            metadata_route_reason="rural_nonurban",
            final_label="0",
            urban_probability_score=0.81,
            topic_final="N9",
            topic_final_group="nonurban",
        )
    )

    decision = policy.evaluate(rural)
    assert decision.should_call is False
    assert decision.action == "protected_hard_negative"


def test_adjudicator_parses_structured_json_and_marks_valid_positive():
    client = _FakeLLMClient(
        """
        {
          "label": "1",
          "confidence": 0.91,
          "decision_type": "core_renewal",
          "object_is_existing_urban": true,
          "renewal_action_present": true,
          "action_is_main_subject": true,
          "background_only": false,
          "exclusion_risk": "none",
          "evidence": ["old neighborhood redevelopment"],
          "reason": "main subject is redevelopment of an existing urban area"
        }
        """
    )
    result = LlmAdjudicator(client).adjudicate(pd.Series(_row()))

    assert client.calls == 1
    assert result.status == "valid"
    assert result.label == "1"
    assert result.confidence == 0.91
    assert result.used is False
    assert "old neighborhood redevelopment" in result.evidence


def test_adjudicator_prompt_contains_strict_json_schema_example():
    client = _FakeLLMClient(
        '{"label":"0","confidence":0.91,"decision_type":"background_only",'
        '"object_is_existing_urban":false,"renewal_action_present":false,'
        '"action_is_main_subject":false,"background_only":true,'
        '"exclusion_risk":"background","evidence":["municipal policy"],'
        '"reason":"background only"}'
    )

    LlmAdjudicator(client).adjudicate(pd.Series(_row()))
    system_prompt = client.messages[0][0]["content"]

    assert "JSON object" in system_prompt
    assert "json" in system_prompt.lower()
    assert '"label": "0 or 1"' in system_prompt
    assert '"confidence": 0.0' in system_prompt
    assert '"decision_type": "core_renewal"' in system_prompt
    assert "Return no Markdown" in system_prompt


def test_adjudicator_invalid_json_is_safe_fallback():
    result = LlmAdjudicator(_FakeLLMClient("not json")).adjudicate(pd.Series(_row()))

    assert result.status == "invalid_json"
    assert result.label == ""
    assert result.confidence == 0.0


def test_binary_finalizer_uses_high_confidence_llm_to_correct_false_positive():
    frame = pd.DataFrame(
        [
            _row(
                final_label="1",
                urban_flag="1",
                urban_probability_score=0.56,
                topic_final="Unknown",
                topic_final_group="unknown",
                review_flag=1,
                binary_policy_action="conflict_review",
                **{
                    Schema.TITLE: "Generic digital participation in cities",
                    Schema.ABSTRACT: "The paper studies municipal participation platforms without redevelopment or upgrading.",
                },
            )
        ]
    )
    client = _FakeLLMClient(
        '{"label":"0","confidence":0.90,"decision_type":"method_only",'
        '"object_is_existing_urban":false,"renewal_action_present":false,'
        '"action_is_main_subject":false,"background_only":false,'
        '"exclusion_risk":"method_only","evidence":["municipal participation platforms"],'
        '"reason":"generic platform study without renewal object"}'
    )

    result = LlmBinaryFinalizer(llm_client=client, llm_enabled=True).apply(frame)

    assert client.calls == 1
    assert result.loc[0, "final_label"] == "0"
    assert result.loc[0, "urban_flag"] == "0"
    assert result.loc[0, Schema.IS_URBAN_RENEWAL] == "0"
    assert result.loc[0, "binary_final_source"] == "llm_binary_v2"
    assert result.loc[0, "llm_adjudication_used"] == 1


def test_binary_finalizer_does_not_override_with_mid_confidence_llm():
    frame = pd.DataFrame(
        [
            _row(
                final_label="0",
                urban_flag="0",
                urban_probability_score=0.44,
                review_flag=1,
            )
        ]
    )
    client = _FakeLLMClient(
        '{"label":"1","confidence":0.72,"decision_type":"boundary_positive",'
        '"object_is_existing_urban":true,"renewal_action_present":true,'
        '"action_is_main_subject":true,"background_only":false,'
        '"exclusion_risk":"none","evidence":["redevelopment"],'
        '"reason":"possible redevelopment context"}'
    )

    result = LlmBinaryFinalizer(llm_client=client, llm_enabled=True).apply(frame)

    assert client.calls == 1
    assert result.loc[0, "final_label"] == "0"
    assert result.loc[0, "llm_adjudication_used"] == 0
    assert result.loc[0, "llm_adjudication_status"] == "informative_only"


def test_postprocess_llm_binary_v2_runs_only_when_workflow_enabled():
    frame = pd.DataFrame(
        [
            _row(
                final_label="1",
                urban_flag="1",
                urban_probability_score=0.51,
                topic_final="Unknown",
                topic_final_group="unknown",
                review_flag=1,
            )
        ]
    )
    client = _FakeLLMClient(
        '{"label":"0","confidence":0.90,"decision_type":"background_only",'
        '"object_is_existing_urban":false,"renewal_action_present":false,'
        '"action_is_main_subject":false,"background_only":true,'
        '"exclusion_risk":"background_only","evidence":["background phrase"],'
        '"reason":"urban renewal appears only as background"}'
    )

    stable = postprocess_urban_predictions(
        frame,
        run_context={"urban_binary_workflow_version": "stable_v1"},
        llm_client=client,
        hybrid_llm_assist_enabled=True,
        urban_method="three_stage_hybrid",
    )
    assert client.calls == 0
    assert "binary_final_source" not in stable.columns

    candidate = postprocess_urban_predictions(
        frame,
        run_context={
            "urban_binary_workflow_version": "llm_binary_v2",
            "experiment_track": "research_matrix",
            "hybrid_llm_assist_enabled": True,
        },
        llm_client=client,
        hybrid_llm_assist_enabled=True,
        urban_method="three_stage_hybrid",
    )
    assert client.calls == 1
    assert candidate.loc[0, "final_label"] == "0"
    assert candidate.loc[0, "binary_final_source"] == "llm_binary_v2"


def test_output_contract_includes_llm_binary_v2_fields():
    row = {}
    apply_urban_output_defaults(row)

    assert "binary_final_source" in URBAN_RESULT_COLUMNS
    assert row["binary_final_source"] == ""
    assert row["llm_adjudication_attempted"] == 0
    assert row["llm_adjudication_prompt_version"] == ""
