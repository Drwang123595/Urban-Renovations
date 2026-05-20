import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.runtime.config import Config, Schema
from src.runtime.llm_client import LLMQuotaExceededError
from src.runtime.project_paths import run_paths
from src.tasks.merged_output import (
    REVIEW_BINARY_EVIDENCE_COLUMN,
    REVIEW_BINARY_POLICY_ACTION_COLUMN,
    REVIEW_DERIVED_COLUMNS,
    REVIEW_DECISION_EXPLANATION_COLUMN,
    REVIEW_DYNAMIC_BINARY_ACTION_COLUMN,
    REVIEW_DYNAMIC_BINARY_CONFIDENCE_COLUMN,
    REVIEW_DYNAMIC_BINARY_LABEL_COLUMN,
    REVIEW_DYNAMIC_BINARY_PRIORITY_COLUMN,
    REVIEW_DYNAMIC_BINARY_REASON_COLUMN,
    REVIEW_DYNAMIC_FIXED_CANDIDATE_COLUMN,
    REVIEW_DYNAMIC_MAPPING_STATUS_COLUMN,
    REVIEW_DYNAMIC_TOPIC_CONFIDENCE_COLUMN,
    REVIEW_DYNAMIC_TOPIC_ID_COLUMN,
    REVIEW_DYNAMIC_TOPIC_KEYWORDS_COLUMN,
    REVIEW_DYNAMIC_TOPIC_NAME_COLUMN,
    REVIEW_DYNAMIC_TOPIC_SIZE_COLUMN,
    REVIEW_DYNAMIC_TOPIC_SOURCE_POOL_COLUMN,
    REVIEW_EVIDENCE_BALANCE_COLUMN,
    REVIEW_INPUT_COLUMNS,
    REVIEW_LLM_USED_COLUMN,
    REVIEW_NEGATIVE_EVIDENCE_COLUMN,
    REVIEW_PREDICT_SPATIAL_COLUMN,
    REVIEW_PREDICT_SPATIAL_DESC_COLUMN,
    REVIEW_PREDICT_SPATIAL_LEVEL_COLUMN,
    REVIEW_POSITIVE_EVIDENCE_COLUMN,
    REVIEW_REASONING_COLUMN,
    REVIEW_RULE_STACK_COLUMN,
    REVIEW_SPATIAL_CONFIDENCE_COLUMN,
    REVIEW_SPATIAL_AREA_EVIDENCE_COLUMN,
    REVIEW_SPATIAL_VALIDATION_REASON_COLUMN,
    REVIEW_SPATIAL_VALIDATION_STATUS_COLUMN,
    REVIEW_TAXONOMY_COVERAGE_COLUMN,
    REVIEW_TOPIC_FINAL_NAME_EN_COLUMN,
    REVIEW_TOPIC_FINAL_NAME_ZH_COLUMN,
    REVIEW_URBAN_CONFIDENCE_COLUMN,
    REVIEW_UNKNOWN_RECOVERY_EVIDENCE_COLUMN,
    REVIEW_UNKNOWN_RECOVERY_PATH_COLUMN,
    build_review_ready_merged_frame,
    load_task_input_frame,
)
from src.tasks.router import TaskRouter, UrbanMethod
from src.urban.pipeline.postprocess import postprocess_urban_predictions
from src.urban.taxonomy.core import topic_name_for_label, topic_name_zh_for_label


SPATIAL_STUDY = "空间研究"
DISTRICT_LEVEL = "城区"
OLD_CITY_AREA = "旧城片区"


def test_spatial_session_path_contains_run_id(tmp_path):
    router = TaskRouter.__new__(TaskRouter)
    router.config = type("Cfg", (), {"SESSIONS_DIR": tmp_path})()
    path = router._get_spatial_session_path("demo_task", 3, "run_001")
    assert str(path).endswith(str(Path("demo_task") / "run_001" / "spatial_3" / "session.json"))


def test_explicit_output_parent_is_created(tmp_path):
    router = TaskRouter.__new__(TaskRouter)
    output_path = tmp_path / "nested" / "reports" / "result.xlsx"

    returned = TaskRouter._ensure_output_parent(router, output_path)

    assert returned == output_path
    assert output_path.parent.is_dir()


def test_run_both_executes_strict_serial(monkeypatch, tmp_path):
    router = TaskRouter.__new__(TaskRouter)
    order = []

    def fake_run_urban(self, input_file=None, output_file=None, limit=None, run_id=None, run_context=None):
        order.append("urban")
        assert output_file is None
        return tmp_path / "urban.xlsx"

    def fake_run_spatial(self, input_file=None, output_file=None, limit=None, run_id=None, run_context=None):
        order.append("spatial")
        assert output_file is None
        return tmp_path / "spatial.xlsx"

    def fake_merge(self, urban_path, spatial_path, timestamp, output_file=None):
        order.append("merge")
        return tmp_path / "merged.xlsx"

    monkeypatch.setattr(TaskRouter, "run_urban_renewal", fake_run_urban)
    monkeypatch.setattr(TaskRouter, "run_spatial", fake_run_spatial)
    monkeypatch.setattr(TaskRouter, "_merge_results", fake_merge)

    result = TaskRouter.run_both(router, input_file="input.xlsx", output_file=None, limit=5)
    assert order == ["urban", "spatial", "merge"]
    assert result["urban_renewal"].name == "urban.xlsx"
    assert result["spatial"].name == "spatial.xlsx"
    assert result["merged"].name == "merged.xlsx"


def test_task_router_default_prediction_dirs_are_partitioned_by_task(tmp_path):
    router = TaskRouter.__new__(TaskRouter)
    router.config = type("Cfg", (), {"DATA_DIR": tmp_path / "Data"})()
    run_context = {
        "dataset_id": "demo_dataset",
        "experiment_track": "research_matrix",
    }

    urban_dir = TaskRouter._default_prediction_dir(
        router,
        "demo_dataset",
        "run_001",
        run_context,
        task_type="urban_renewal",
    )
    spatial_dir = TaskRouter._default_prediction_dir(
        router,
        "demo_dataset",
        "run_001",
        run_context,
        task_type="spatial",
    )
    merged_dir = TaskRouter._default_prediction_dir(
        router,
        "demo_dataset",
        "run_001",
        run_context,
        task_type="merged",
    )
    layout = run_paths("demo_dataset", "research_matrix", "run_001", project_root=tmp_path)

    assert urban_dir == layout.urban_prediction_dir
    assert spatial_dir == layout.spatial_prediction_dir
    assert merged_dir == layout.merged_prediction_dir


def test_task_router_default_prediction_names_include_dataset_task_and_run_tag(tmp_path):
    router = TaskRouter.__new__(TaskRouter)
    router.config = type("Cfg", (), {"DATA_DIR": tmp_path / "Data"})()
    router.urban_method = UrbanMethod.THREE_STAGE_HYBRID
    router.urban_shot_mode = "few"
    router.spatial_shot_mode = "zero"
    run_context = {
        "dataset_id": "Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407",
        "experiment_track": "research_matrix",
        "hybrid_llm_assist_enabled": True,
    }

    urban_name = TaskRouter._default_prediction_file(
        router,
        "input-stem",
        "20260520_150000",
        run_context,
        task_type="urban_renewal",
    ).name
    spatial_name = TaskRouter._default_prediction_file(
        router,
        "input-stem",
        "20260520_150000",
        run_context,
        task_type="spatial",
    ).name
    merged_name = TaskRouter._default_prediction_file(
        router,
        "input-stem",
        "20260520_150000",
        run_context,
        task_type="merged",
    ).name

    assert (
        urban_name
        == "urban_renovation_v2_0_20260407__urban_renewal__three_stage_hybrid_few_llm_on__20260520_150000.xlsx"
    )
    assert spatial_name == "urban_renovation_v2_0_20260407__spatial__zero__20260520_150000.xlsx"
    assert merged_name == "urban_renovation_v2_0_20260407__merged__urban_renewal_spatial__20260520_150000.xlsx"


def test_prepare_frame_for_run_honors_canonical_title_order():
    router = TaskRouter.__new__(TaskRouter)
    frame = TaskRouter._prepare_frame_for_run(
        router,
        pd.DataFrame(
            {
                Schema.TITLE: ["b title", "A title", "c title"],
                Schema.ABSTRACT: ["b", "a", "c"],
            }
        ),
        run_context={"order_id": "canonical_title_order"},
    )
    assert frame[Schema.TITLE].tolist() == ["A title", "b title", "c title"]


def test_prepare_frame_for_run_honors_order_seed():
    router = TaskRouter.__new__(TaskRouter)
    frame = TaskRouter._prepare_frame_for_run(
        router,
        pd.DataFrame(
            {
                Schema.TITLE: ["A", "B", "C", "D"],
                Schema.ABSTRACT: ["a", "b", "c", "d"],
            }
        ),
        run_context={"order_id": "shuffle_seed_20260415_a", "order_seed": 7},
    )
    assert frame[Schema.TITLE].tolist() == ["C", "B", "A", "D"]


def test_run_urban_method_dispatches_to_pure_llm(tmp_path):
    router = TaskRouter.__new__(TaskRouter)
    router.urban_method = UrbanMethod.PURE_LLM_API

    called = {}

    def fake_llm(title, abstract, record, session_path, audit_metadata=None):
        called["method"] = "pure_llm"
        called["session_path"] = session_path
        return {Schema.IS_URBAN_RENEWAL: "1"}

    router._run_urban_pure_llm = fake_llm
    router._run_urban_local_classifier = lambda record: {Schema.IS_URBAN_RENEWAL: "0"}
    router.urban_hybrid_classifier = type(
        "Hybrid",
        (),
        {"classify": lambda self, title, abstract, metadata=None, session_path=None: {Schema.IS_URBAN_RENEWAL: "0"}},
    )()

    result = TaskRouter._run_urban_method(
        router,
        "Urban renewal policy",
        "Studies redevelopment and financing.",
        {Schema.KEYWORDS_PLUS: "redevelopment"},
        tmp_path / "session.json",
        run_context={"session_policy": "per_paper_isolated"},
    )
    assert called["method"] == "pure_llm"
    assert called["session_path"] == tmp_path / "session.json"
    assert result[Schema.IS_URBAN_RENEWAL] == "1"


def test_run_urban_renewal_resumes_completed_rows_after_quota(tmp_path):
    input_path = tmp_path / "input.xlsx"
    output_path = tmp_path / "out.xlsx"
    pd.DataFrame(
        [
            {Schema.TITLE: "A", Schema.ABSTRACT: "a"},
            {Schema.TITLE: "B", Schema.ABSTRACT: "b"},
            {Schema.TITLE: "C", Schema.ABSTRACT: "c"},
        ]
    ).to_excel(input_path, index=False, engine="openpyxl")

    router = TaskRouter.__new__(TaskRouter)
    router.config = type("Cfg", (), {"SESSIONS_DIR": tmp_path / "sessions", "DATA_DIR": tmp_path / "Data"})()
    router.urban_method = UrbanMethod.PURE_LLM_API
    router.urban_shot_mode = "zero"
    router.hybrid_llm_assist_enabled = False
    router._postprocess_urban_prediction_frame = lambda frame, run_context=None: frame

    first_seen = []

    def quota_on_second(title, abstract, metadata, session_path, audit_metadata=None, run_context=None):
        first_seen.append(title)
        if title == "B":
            raise LLMQuotaExceededError("daily limit", status_code=429, code="USAGE_LIMIT_EXCEEDED")
        return {Schema.IS_URBAN_RENEWAL: "1", "final_label": "1", "urban_flag": "1"}

    router._run_urban_method = quota_on_second

    with pytest.raises(LLMQuotaExceededError):
        TaskRouter.run_urban_renewal(
            router,
            input_file=str(input_path),
            output_file=str(output_path),
            run_context={"urban_flow_audit_enabled": False},
        )

    assert first_seen == ["A", "B"]
    assert (tmp_path / "out.xlsx.checkpoint.jsonl").exists()

    second_seen = []

    def complete_remaining(title, abstract, metadata, session_path, audit_metadata=None, run_context=None):
        second_seen.append(title)
        return {Schema.IS_URBAN_RENEWAL: "0", "final_label": "0", "urban_flag": "0"}

    router._run_urban_method = complete_remaining
    result_path = TaskRouter.run_urban_renewal(
        router,
        input_file=str(input_path),
        output_file=str(output_path),
        run_context={"urban_flow_audit_enabled": False},
    )

    assert result_path == output_path
    assert second_seen == ["B", "C"]
    output = pd.read_excel(output_path, engine="openpyxl")
    assert output[Schema.TITLE].tolist() == ["A", "B", "C"]
    assert output["final_label"].tolist() == [1, 0, 0]


def test_run_spatial_resumes_completed_rows_after_quota_and_preserves_order(tmp_path):
    input_path = tmp_path / "input.xlsx"
    output_path = tmp_path / "spatial.xlsx"
    pd.DataFrame(
        [
            {Schema.TITLE: "Duplicate", Schema.ABSTRACT: "first"},
            {Schema.TITLE: "Duplicate", Schema.ABSTRACT: "second"},
            {Schema.TITLE: "C", Schema.ABSTRACT: "third"},
        ]
    ).to_excel(input_path, index=False, engine="openpyxl")

    router = TaskRouter.__new__(TaskRouter)
    router.config = type(
        "Cfg",
        (),
        {"SESSIONS_DIR": tmp_path / "sessions", "DATA_DIR": tmp_path / "Data", "MAX_WORKERS": 1},
    )()
    router.spatial_shot_mode = "zero"

    first_seen = []

    class FirstSpatialStrategy:
        def process(self, title, abstract, session_path=None, audit_metadata=None):
            first_seen.append((title, abstract))
            if abstract == "second":
                raise LLMQuotaExceededError("daily limit", status_code=429, code="USAGE_LIMIT_EXCEEDED")
            return {Schema.IS_SPATIAL: "1", Schema.SPATIAL_LEVEL: "7. Single-city / Municipal Scale", Schema.SPATIAL_DESC: "First City"}

    router.spatial_strategy = FirstSpatialStrategy()
    with pytest.raises(LLMQuotaExceededError):
        TaskRouter.run_spatial(
            router,
            input_file=str(input_path),
            output_file=str(output_path),
            run_context={"order_id": "input_order"},
        )

    assert first_seen == [("Duplicate", "first"), ("Duplicate", "second")]
    assert output_path.with_name("spatial.xlsx.checkpoint.jsonl").exists()

    second_seen = []

    class SecondSpatialStrategy:
        def process(self, title, abstract, session_path=None, audit_metadata=None):
            second_seen.append((title, abstract))
            return {Schema.IS_SPATIAL: "1", Schema.SPATIAL_LEVEL: "8. District / County Scale", Schema.SPATIAL_DESC: abstract}

    router.spatial_strategy = SecondSpatialStrategy()
    TaskRouter.run_spatial(
        router,
        input_file=str(input_path),
        output_file=str(output_path),
        run_context={"order_id": "input_order"},
    )

    assert second_seen == [("Duplicate", "second"), ("C", "third")]
    output = pd.read_excel(output_path, engine="openpyxl")
    assert output[Schema.TITLE].tolist() == ["Duplicate", "Duplicate", "C"]
    assert output[Schema.SPATIAL_DESC].tolist() == ["First City", "second", "third"]


def test_postprocess_llm_binary_v2_resumes_after_quota(tmp_path):
    checkpoint_path = tmp_path / "postprocess.checkpoint.jsonl"
    frame = pd.DataFrame(
        [
            {
                Schema.TITLE: "A",
                Schema.ABSTRACT: "Urban regeneration of an old district.",
                "final_label": "1",
                "urban_flag": "1",
                Schema.IS_URBAN_RENEWAL: "1",
                "urban_probability_score": 0.46,
                "binary_decision_threshold": 0.45,
                "topic_final_group": "nonurban",
                "topic_final": "N3",
            },
            {
                Schema.TITLE: "B",
                Schema.ABSTRACT: "Urban renewal policy in a historic quarter.",
                "final_label": "1",
                "urban_flag": "1",
                Schema.IS_URBAN_RENEWAL: "1",
                "urban_probability_score": 0.47,
                "binary_decision_threshold": 0.45,
                "topic_final_group": "nonurban",
                "topic_final": "N3",
            },
        ]
    )
    context = {
        "urban_binary_workflow_version": "llm_binary_v2",
        "hybrid_llm_assist_enabled": True,
        "urban_stable_strategy_enabled": False,
        "resume_checkpoint": str(checkpoint_path),
        "resume_run_id": "run-1",
        "resume_input_fingerprint": "fp-1",
    }

    class FirstClient:
        def __init__(self):
            self.calls = []

        def chat_completion(self, messages, temperature=0.0, max_retries=2):
            text = messages[-1]["content"]
            self.calls.append(text)
            if "[TITLE] B" in text:
                raise LLMQuotaExceededError("daily limit", status_code=429, code="USAGE_LIMIT_EXCEEDED")
            return '{"label":"1","confidence":0.91,"decision_type":"core_renewal","object_is_existing_urban":true,"renewal_action_present":true,"action_is_main_subject":true,"background_only":false,"exclusion_risk":"none","evidence":["district"],"reason":"core renewal"}'

    first_client = FirstClient()
    with pytest.raises(LLMQuotaExceededError):
        postprocess_urban_predictions(
            frame,
            run_context=context,
            llm_client=first_client,
            hybrid_llm_assist_enabled=True,
            urban_method=UrbanMethod.THREE_STAGE_HYBRID,
        )

    assert len(first_client.calls) == 2
    assert checkpoint_path.exists()

    class SecondClient:
        def __init__(self):
            self.calls = []

        def chat_completion(self, messages, temperature=0.0, max_retries=2):
            text = messages[-1]["content"]
            self.calls.append(text)
            return '{"label":"0","confidence":0.93,"decision_type":"background_only","object_is_existing_urban":false,"renewal_action_present":false,"action_is_main_subject":false,"background_only":true,"exclusion_risk":"background","evidence":["policy"],"reason":"background only"}'

    second_client = SecondClient()
    result = postprocess_urban_predictions(
        frame,
        run_context=context,
        llm_client=second_client,
        hybrid_llm_assist_enabled=True,
        urban_method=UrbanMethod.THREE_STAGE_HYBRID,
    )

    assert len(second_client.calls) == 1
    assert "[TITLE] B" in second_client.calls[0]
    assert result.loc[0, "llm_adjudication_reason"] == "core renewal"
    assert result.loc[1, "llm_adjudication_reason"] == "background only"


def test_run_urban_method_uses_shared_context_for_cross_paper_long_context(tmp_path):
    router = TaskRouter.__new__(TaskRouter)
    router.urban_method = UrbanMethod.PURE_LLM_API
    called = {}

    def fake_llm(title, abstract, record, session_path, audit_metadata=None):
        called["session_path"] = session_path
        return {Schema.IS_URBAN_RENEWAL: "1"}

    router._run_urban_pure_llm = fake_llm
    router._run_urban_local_classifier = lambda record: {Schema.IS_URBAN_RENEWAL: "0"}
    router.urban_hybrid_classifier = type(
        "Hybrid",
        (),
        {"classify": lambda self, title, abstract, metadata=None, session_path=None: {Schema.IS_URBAN_RENEWAL: "0"}},
    )()

    TaskRouter._run_urban_method(
        router,
        "Urban renewal policy",
        "Studies redevelopment and financing.",
        {Schema.KEYWORDS_PLUS: "redevelopment"},
        tmp_path / "session.json",
        run_context={"session_policy": "cross_paper_long_context"},
    )
    assert called["session_path"] is None


def test_run_urban_method_dispatches_to_local_classifier(tmp_path):
    router = TaskRouter.__new__(TaskRouter)
    router.urban_method = UrbanMethod.LOCAL_TOPIC_CLASSIFIER
    router._run_urban_pure_llm = lambda title, abstract, record, session_path: {Schema.IS_URBAN_RENEWAL: "0"}

    called = {}

    def fake_local(record):
        called["method"] = "local_classifier"
        return {Schema.IS_URBAN_RENEWAL: "1"}

    router._run_urban_local_classifier = fake_local
    router.urban_hybrid_classifier = type(
        "Hybrid",
        (),
        {"classify": lambda self, title, abstract, metadata=None, session_path=None: {Schema.IS_URBAN_RENEWAL: "0"}},
    )()

    result = TaskRouter._run_urban_method(
        router,
        "Urban renewal policy",
        "Studies redevelopment and financing.",
        {Schema.KEYWORDS_PLUS: "redevelopment"},
        tmp_path / "session.json",
    )
    assert called["method"] == "local_classifier"
    assert result[Schema.IS_URBAN_RENEWAL] == "1"


def test_build_urban_output_row_resolves_blank_unknown_label_to_binary_default():
    router = TaskRouter.__new__(TaskRouter)
    row = TaskRouter._build_urban_output_row(
        router,
        "Urban renewal policy",
        "Studies redevelopment and financing.",
        {
            Schema.IS_URBAN_RENEWAL: "",
            "final_label": "",
            "urban_flag": "",
            "decision_source": "unknown_review",
            "review_flag": 1,
            "review_reason": "rule_local_cross_group_conflict",
            "bertopic_hint_label": "N3",
        },
    )
    assert row[Schema.IS_URBAN_RENEWAL] == "0"
    assert row["final_label"] == "0"
    assert row["urban_flag"] == "0"
    assert row["decision_source"] == "unknown_review"
    assert row["review_flag"] == 1
    assert row["bertopic_hint_label"] == "N3"
    assert row["dynamic_topic_id"] == ""
    assert row["dynamic_mapping_status"] == ""
    assert row["dynamic_binary_candidate_label"] == ""
    assert row["dynamic_binary_candidate_action"] == ""


def test_load_task_input_frame_backfills_publication_year_from_train(tmp_path):
    task_dir = tmp_path / "demo_task"
    labels_dir = task_dir / "labels"
    train_dir = tmp_path / "train"
    labels_dir.mkdir(parents=True)
    train_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                Schema.TITLE: "Urban renewal policy",
                Schema.KEYWORDS_PLUS: "urban policy",
                Schema.ABSTRACT: "Studies redevelopment and financing.",
                Schema.WOS_CATEGORIES: "Urban Studies",
                Schema.RESEARCH_AREAS: "Geography",
                Schema.IS_URBAN_RENEWAL: "1",
            }
        ]
    ).to_excel(labels_dir / "demo_task.xlsx", index=False, engine="openpyxl")

    pd.DataFrame(
        [
            {
                Schema.TITLE: "Urban renewal policy",
                Schema.ABSTRACT: "Studies redevelopment and financing.",
                "Publication Year": 2022,
            }
        ]
    ).to_excel(train_dir / "Urban Renovation V2.0.xlsx", index=False, engine="openpyxl")

    loaded = load_task_input_frame(task_dir)

    assert loaded is not None
    assert loaded.columns.tolist()[0:2] == [Schema.TITLE, "Publication Year"]
    assert loaded.at[0, "Publication Year"] == 2022


def test_build_review_ready_merged_frame_preserves_source_input_columns():
    input_df = pd.DataFrame(
        [
            {
                Schema.TITLE: "Urban renewal policy",
                "Publication Year": 2024,
                Schema.KEYWORDS_PLUS: "urban policy",
                Schema.ABSTRACT: "Studies redevelopment and financing.",
                Schema.WOS_CATEGORIES: "Urban Studies",
                Schema.RESEARCH_AREAS: "Geography",
                Schema.IS_URBAN_RENEWAL: "1",
                "Unnamed: 6": "",
                "theme_gold": "U10",
                "theme_gold_source": "manual",
                "review_status": "reviewed",
            }
        ]
    )
    merged = pd.DataFrame(
        [
            {
                Schema.TITLE: "Urban renewal policy",
                Schema.ABSTRACT: "Studies redevelopment and financing.",
                Schema.IS_URBAN_RENEWAL: "1",
                f"{Schema.IS_SPATIAL}_spatial": SPATIAL_STUDY,
                f"{Schema.SPATIAL_LEVEL}_spatial": DISTRICT_LEVEL,
                f"{Schema.SPATIAL_DESC}_spatial": OLD_CITY_AREA,
                "final_label": "1",
                "topic_final": "U10",
                "topic_final_name": "redevelopment finance",
                "confidence": 0.93,
                "Reasoning": "mentions redevelopment finance",
                "Confidence": "High",
                "review_flag": 0,
                "review_reason": "",
                "taxonomy_coverage_status": "covered",
                "binary_policy_action": "accept_positive",
                "llm_used": 0,
                "decision_explanation": "final=1; score=0.9300>=threshold=0.4500",
                "primary_positive_evidence": "topic_final=U10",
                "primary_negative_evidence": "none",
                "evidence_balance": "strong_positive",
                "decision_rule_stack": "route=pass > rule=U10 > binary=confidence",
                "binary_decision_evidence": "raw_score=0.9300",
                "unknown_recovery_path": "not_triggered",
                "unknown_recovery_evidence": "",
                "dynamic_topic_id": "DUR_0001",
                "dynamic_topic_name_zh": "\u68d5\u5730\u518d\u5f00\u53d1",
                "dynamic_topic_keywords": "brownfield; redevelopment",
                "dynamic_topic_size": 12,
                "dynamic_topic_confidence": 0.88,
                "dynamic_topic_source_pool": "unknown_pool",
                "dynamic_to_fixed_topic_candidate": "U2",
                "dynamic_mapping_status": "mapped_to_fixed",
                "dynamic_binary_candidate_label": "1",
                "dynamic_binary_candidate_confidence": 0.88,
                "dynamic_binary_candidate_action": "supports_current_label",
                "dynamic_binary_candidate_reason": "dynamic_topic=DUR_0001",
                "dynamic_binary_review_priority": "low",
            }
        ]
    )

    review = build_review_ready_merged_frame(merged, input_df=input_df)

    assert review.columns.tolist() == REVIEW_INPUT_COLUMNS + REVIEW_DERIVED_COLUMNS
    assert "Unnamed: 6" not in review.columns
    assert "theme_gold" not in review.columns
    assert "theme_gold_source" not in review.columns
    assert "review_status" not in review.columns
    assert review.at[0, Schema.IS_URBAN_RENEWAL] == "1"
    assert review.at[0, "Publication Year"] == 2024
    assert review.at[0, REVIEW_PREDICT_SPATIAL_COLUMN] == SPATIAL_STUDY
    assert review.at[0, REVIEW_PREDICT_SPATIAL_LEVEL_COLUMN] == DISTRICT_LEVEL
    assert review.at[0, REVIEW_PREDICT_SPATIAL_DESC_COLUMN] == OLD_CITY_AREA
    assert review.at[0, REVIEW_TOPIC_FINAL_NAME_EN_COLUMN] == topic_name_for_label("U10")
    assert review.at[0, REVIEW_TOPIC_FINAL_NAME_ZH_COLUMN] == topic_name_zh_for_label("U10")
    assert review.at[0, REVIEW_TAXONOMY_COVERAGE_COLUMN] == "covered"
    assert review.at[0, REVIEW_URBAN_CONFIDENCE_COLUMN] == 0.93
    assert review.at[0, REVIEW_SPATIAL_CONFIDENCE_COLUMN] == "High"
    assert review.at[0, REVIEW_DECISION_EXPLANATION_COLUMN].startswith("final=1")
    assert review.at[0, REVIEW_POSITIVE_EVIDENCE_COLUMN] == "topic_final=U10"
    assert review.at[0, REVIEW_NEGATIVE_EVIDENCE_COLUMN] == "none"
    assert review.at[0, REVIEW_EVIDENCE_BALANCE_COLUMN] == "strong_positive"
    assert "rule=U10" in review.at[0, REVIEW_RULE_STACK_COLUMN]
    assert review.at[0, REVIEW_BINARY_EVIDENCE_COLUMN] == "raw_score=0.9300"
    assert review.at[0, REVIEW_UNKNOWN_RECOVERY_PATH_COLUMN] == "not_triggered"
    assert review.at[0, REVIEW_UNKNOWN_RECOVERY_EVIDENCE_COLUMN] == ""
    assert review.at[0, REVIEW_DYNAMIC_TOPIC_ID_COLUMN] == "DUR_0001"
    assert review.at[0, REVIEW_DYNAMIC_TOPIC_NAME_COLUMN] == "\u68d5\u5730\u518d\u5f00\u53d1"
    assert review.at[0, REVIEW_DYNAMIC_TOPIC_KEYWORDS_COLUMN] == "brownfield; redevelopment"
    assert review.at[0, REVIEW_DYNAMIC_TOPIC_SIZE_COLUMN] == 12
    assert review.at[0, REVIEW_DYNAMIC_TOPIC_CONFIDENCE_COLUMN] == 0.88
    assert review.at[0, REVIEW_DYNAMIC_TOPIC_SOURCE_POOL_COLUMN] == "unknown_pool"
    assert review.at[0, REVIEW_DYNAMIC_FIXED_CANDIDATE_COLUMN] == "U2"
    assert review.at[0, REVIEW_DYNAMIC_MAPPING_STATUS_COLUMN] == "mapped_to_fixed"
    assert review.at[0, REVIEW_DYNAMIC_BINARY_LABEL_COLUMN] == "1"
    assert review.at[0, REVIEW_DYNAMIC_BINARY_CONFIDENCE_COLUMN] == 0.88
    assert review.at[0, REVIEW_DYNAMIC_BINARY_ACTION_COLUMN] == "supports_current_label"
    assert review.at[0, REVIEW_DYNAMIC_BINARY_REASON_COLUMN] == "dynamic_topic=DUR_0001"
    assert review.at[0, REVIEW_DYNAMIC_BINARY_PRIORITY_COLUMN] == "low"
    assert review.at[0, REVIEW_BINARY_POLICY_ACTION_COLUMN] == "accept_positive"
    assert review.at[0, REVIEW_LLM_USED_COLUMN] == 0


def test_topic_name_zh_for_label_covers_nonurban_and_unknown():
    assert topic_name_for_label("N7") == "transport mobility and accessibility"
    assert topic_name_zh_for_label("N7") == "交通、流动性与可达性"
    assert topic_name_zh_for_label("Unknown") == "未知主题"


def test_merge_results_writes_source_input_plus_review_columns(tmp_path):
    task_dir = tmp_path / "demo_task"
    output_dir = task_dir / "output"
    labels_dir = task_dir / "labels"
    output_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    urban_path = output_dir / "urban.xlsx"
    spatial_path = output_dir / "spatial.xlsx"
    merged_path = output_dir / "merged.xlsx"

    input_df = pd.DataFrame(
        [
            {
                Schema.TITLE: "Urban renewal policy",
                "Publication Year": 2024,
                Schema.KEYWORDS_PLUS: "urban policy",
                Schema.ABSTRACT: "Studies redevelopment and financing.",
                Schema.WOS_CATEGORIES: "Urban Studies",
                Schema.RESEARCH_AREAS: "Geography",
                Schema.IS_URBAN_RENEWAL: "1",
                "Unnamed: 6": "",
                "theme_gold": "U10",
                "theme_gold_source": "manual",
                "review_status": "reviewed",
            }
        ]
    )
    input_df.to_excel(labels_dir / "demo_task.xlsx", index=False, engine="openpyxl")

    pd.DataFrame(
        [
            {
                Schema.TITLE: "Urban renewal policy",
                Schema.ABSTRACT: "Studies redevelopment and financing.",
                Schema.IS_URBAN_RENEWAL: "1",
                "final_label": "1",
                "topic_final": "U10",
                "confidence": 0.93,
                "review_flag": 0,
                "review_reason": "",
            }
        ]
    ).to_excel(urban_path, index=False, engine="openpyxl")

    pd.DataFrame(
        [
            {
                Schema.TITLE: "Urban renewal policy",
                Schema.IS_SPATIAL: SPATIAL_STUDY,
                Schema.SPATIAL_LEVEL: DISTRICT_LEVEL,
                Schema.SPATIAL_DESC: OLD_CITY_AREA,
                "Reasoning": "mentions district scale",
                "Confidence": "High",
                Schema.SPATIAL_VALIDATION_STATUS: "accepted",
                Schema.SPATIAL_VALIDATION_REASON: "explicit_area_evidence",
                Schema.SPATIAL_AREA_EVIDENCE: OLD_CITY_AREA,
            }
        ]
    ).to_excel(spatial_path, index=False, engine="openpyxl")

    router = TaskRouter.__new__(TaskRouter)
    result = TaskRouter._merge_results(
        router,
        urban_path=urban_path,
        spatial_path=spatial_path,
        timestamp="20260416_000000",
        output_file=str(merged_path),
    )

    assert result == merged_path

    merged = pd.read_excel(merged_path, engine="openpyxl")
    assert merged.columns.tolist() == REVIEW_INPUT_COLUMNS + REVIEW_DERIVED_COLUMNS
    assert set(pd.ExcelFile(merged_path, engine="openpyxl").sheet_names) == {
        "Review View",
        "Metric Dictionary",
        "Raw Predictions",
    }
    assert merged.at[0, "Publication Year"] == 2024
    assert "theme_gold" not in merged.columns
    assert "theme_gold_source" not in merged.columns
    assert "review_status" not in merged.columns
    assert merged.at[0, REVIEW_PREDICT_SPATIAL_COLUMN] == SPATIAL_STUDY
    assert merged.at[0, REVIEW_TOPIC_FINAL_NAME_EN_COLUMN] == topic_name_for_label("U10")
    assert merged.at[0, REVIEW_TOPIC_FINAL_NAME_ZH_COLUMN] == topic_name_zh_for_label("U10")
    assert merged.at[0, REVIEW_REASONING_COLUMN] == "mentions district scale"
    assert merged.at[0, REVIEW_SPATIAL_VALIDATION_STATUS_COLUMN] == "accepted"
    assert merged.at[0, REVIEW_SPATIAL_VALIDATION_REASON_COLUMN] == "explicit_area_evidence"
    assert merged.at[0, REVIEW_SPATIAL_AREA_EVIDENCE_COLUMN] == OLD_CITY_AREA


def test_merge_results_uses_row_order_for_duplicate_titles(tmp_path):
    output_dir = tmp_path / "predictions"
    output_dir.mkdir(parents=True)
    urban_path = output_dir / "urban.xlsx"
    spatial_path = output_dir / "spatial.xlsx"
    merged_path = output_dir / "merged.xlsx"

    pd.DataFrame(
        [
            {
                Schema.TITLE: "Duplicate title",
                Schema.ABSTRACT: "First abstract about redevelopment finance.",
                Schema.IS_URBAN_RENEWAL: "1",
                "final_label": "1",
                "topic_final": "U10",
            },
            {
                Schema.TITLE: "Duplicate title",
                Schema.ABSTRACT: "Second abstract about heritage conservation.",
                Schema.IS_URBAN_RENEWAL: "1",
                "final_label": "1",
                "topic_final": "U8",
            },
        ]
    ).to_excel(urban_path, index=False, engine="openpyxl")

    pd.DataFrame(
        [
            {
                Schema.TITLE: "Duplicate title",
                Schema.IS_SPATIAL: "1",
                Schema.SPATIAL_LEVEL: "7. Single-city / Municipal Scale",
                Schema.SPATIAL_DESC: "First City",
                "Reasoning": "first row spatial reasoning",
                "Confidence": "High",
            },
            {
                Schema.TITLE: "Duplicate title",
                Schema.IS_SPATIAL: "1",
                Schema.SPATIAL_LEVEL: "8. District / County Scale",
                Schema.SPATIAL_DESC: "Second District",
                "Reasoning": "second row spatial reasoning",
                "Confidence": "Medium",
            },
        ]
    ).to_excel(spatial_path, index=False, engine="openpyxl")

    router = TaskRouter.__new__(TaskRouter)
    result = TaskRouter._merge_results(
        router,
        urban_path=urban_path,
        spatial_path=spatial_path,
        timestamp="20260420_000000",
        output_file=str(merged_path),
    )

    assert result == merged_path
    merged = pd.read_excel(merged_path, engine="openpyxl")
    assert len(merged) == 2
    assert merged[REVIEW_PREDICT_SPATIAL_DESC_COLUMN].tolist() == ["First City", "Second District"]
    assert merged[REVIEW_REASONING_COLUMN].tolist() == [
        "first row spatial reasoning",
        "second row spatial reasoning",
    ]


def test_merge_results_creates_nested_explicit_output_parent(tmp_path):
    output_dir = tmp_path / "predictions"
    output_dir.mkdir(parents=True)
    urban_path = output_dir / "urban.xlsx"
    spatial_path = output_dir / "spatial.xlsx"
    merged_path = tmp_path / "missing" / "nested" / "merged.xlsx"

    pd.DataFrame(
        [
            {
                Schema.TITLE: "A",
                Schema.ABSTRACT: "Urban renewal abstract",
                Schema.IS_URBAN_RENEWAL: "1",
                "final_label": "1",
                "topic_final": "U1",
            }
        ]
    ).to_excel(urban_path, index=False, engine="openpyxl")

    pd.DataFrame(
        [
            {
                Schema.TITLE: "A",
                Schema.IS_SPATIAL: "1",
                Schema.SPATIAL_LEVEL: "7. Single-city / Municipal Scale",
                Schema.SPATIAL_DESC: "A City",
            }
        ]
    ).to_excel(spatial_path, index=False, engine="openpyxl")

    router = TaskRouter.__new__(TaskRouter)
    result = TaskRouter._merge_results(
        router,
        urban_path=urban_path,
        spatial_path=spatial_path,
        timestamp="20260420_000000",
        output_file=str(merged_path),
    )

    assert result == merged_path
    assert merged_path.exists()


def test_merge_results_auto_names_canonical_merged_output_with_dataset_slug(tmp_path):
    pred_root = (
        tmp_path
        / "Data"
        / "output"
        / "Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407"
        / "runs"
        / "research_matrix"
        / "run_001"
        / "predictions"
    )
    urban_dir = pred_root / "urban_renewal"
    spatial_dir = pred_root / "spatial"
    urban_dir.mkdir(parents=True)
    spatial_dir.mkdir(parents=True)
    urban_path = urban_dir / "urban.xlsx"
    spatial_path = spatial_dir / "spatial.xlsx"

    pd.DataFrame(
        [
            {
                Schema.TITLE: "A",
                Schema.ABSTRACT: "Urban renewal abstract",
                Schema.IS_URBAN_RENEWAL: "1",
            }
        ]
    ).to_excel(urban_path, index=False, engine="openpyxl")
    pd.DataFrame(
        [
            {
                Schema.TITLE: "A",
                Schema.IS_SPATIAL: "1",
                Schema.SPATIAL_LEVEL: "7. Single-city / Municipal Scale",
                Schema.SPATIAL_DESC: "A City",
            }
        ]
    ).to_excel(spatial_path, index=False, engine="openpyxl")

    router = TaskRouter.__new__(TaskRouter)
    result = TaskRouter._merge_results(
        router,
        urban_path=urban_path,
        spatial_path=spatial_path,
        timestamp="20260520_160000",
    )

    assert result == pred_root / "merged" / (
        "urban_renovation_v2_0_20260407__merged__urban_renewal_spatial__20260520_160000.xlsx"
    )
    assert result.exists()
