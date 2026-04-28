import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Schema
from src.merged_output import (
    REVIEW_BINARY_EVIDENCE_COLUMN,
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
    REVIEW_NEGATIVE_EVIDENCE_COLUMN,
    REVIEW_POSITIVE_EVIDENCE_COLUMN,
    REVIEW_RULE_STACK_COLUMN,
    REVIEW_UNKNOWN_RECOVERY_EVIDENCE_COLUMN,
    REVIEW_UNKNOWN_RECOVERY_PATH_COLUMN,
    build_review_ready_merged_frame,
    load_task_input_frame,
)
from src.task_router import TaskRouter, UrbanMethod
from src.urban_topic_taxonomy import topic_name_for_label, topic_name_zh_for_label


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
    assert review.at[0, REVIEW_DERIVED_COLUMNS[1]] == SPATIAL_STUDY
    assert review.at[0, REVIEW_DERIVED_COLUMNS[2]] == DISTRICT_LEVEL
    assert review.at[0, REVIEW_DERIVED_COLUMNS[3]] == OLD_CITY_AREA
    assert review.at[0, "topic_final_name_en"] == topic_name_for_label("U10")
    assert review.at[0, "topic_final_name_zh"] == topic_name_zh_for_label("U10")
    assert review.at[0, REVIEW_DERIVED_COLUMNS[7]] == 0.93
    assert review.at[0, REVIEW_DERIVED_COLUMNS[9]] == "High"
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
    assert merged.at[0, "Publication Year"] == 2024
    assert "theme_gold" not in merged.columns
    assert "theme_gold_source" not in merged.columns
    assert "review_status" not in merged.columns
    assert merged.at[0, REVIEW_DERIVED_COLUMNS[1]] == SPATIAL_STUDY
    assert merged.at[0, "topic_final_name_en"] == topic_name_for_label("U10")
    assert merged.at[0, "topic_final_name_zh"] == topic_name_zh_for_label("U10")
    assert merged.at[0, REVIEW_DERIVED_COLUMNS[8]] == "mentions district scale"
    assert merged.at[0, Schema.SPATIAL_VALIDATION_STATUS] == "accepted"
    assert merged.at[0, Schema.SPATIAL_VALIDATION_REASON] == "explicit_area_evidence"
    assert merged.at[0, Schema.SPATIAL_AREA_EVIDENCE] == OLD_CITY_AREA


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
    assert merged[REVIEW_DERIVED_COLUMNS[3]].tolist() == ["First City", "Second District"]
    assert merged[REVIEW_DERIVED_COLUMNS[8]].tolist() == [
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
