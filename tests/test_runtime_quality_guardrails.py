import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.evaluate import (
    build_group_summaries,
    build_long_context_stability,
    build_prediction_guardrails,
    build_urban_error_analysis,
    collect_pred_files,
    evaluate_one_file,
    flatten_diagnostics,
    resolve_truth_files,
    resolve_truth_for_prediction,
)
from src.config import Schema
from src.evaluation_core import align_truth_pred, evaluate_merged, summarize_chunked_binary_metrics
from src.task_router import TaskRouter


class _DummyPromptGen:
    def get_step_prompt(self, step, title, abstract, include_context=True):
        return f"{step}:{title}:{abstract}"


class _DummyClient:
    def __init__(self, response):
        self.response = response

    def chat_completion(self, _messages):
        return self.response


class _DummyMemory:
    def __init__(self):
        self.messages = []
        self.saved = 0

    def add_user_message(self, text):
        self.messages.append({"role": "user", "content": text})

    def add_assistant_message(self, text):
        self.messages.append({"role": "assistant", "content": text})

    def get_messages(self):
        return self.messages

    def save(self):
        self.saved += 1


def test_collect_pred_files_default_urban_scope(tmp_path):
    (tmp_path / "urban_renewal_zero_1.xlsx").touch()
    (tmp_path / "spatial_zero_1.xlsx").touch()
    (tmp_path / "merged_1.xlsx").touch()
    (tmp_path / "Eval_urban_renewal_zero_1.xlsx").touch()
    (tmp_path / "~$urban_renewal_zero_1.xlsx").touch()
    files = collect_pred_files(pred_dir_arg=str(tmp_path), pred_scope="urban_renewal")
    names = [file.name for file in files]
    assert "urban_renewal_zero_1.xlsx" in names
    assert "merged_1.xlsx" not in names
    assert "spatial_zero_1.xlsx" not in names
    assert "Eval_urban_renewal_zero_1.xlsx" not in names
    assert "~$urban_renewal_zero_1.xlsx" not in names


def test_evaluation_core_accepts_manual_truth_alias():
    truth_df = pd.DataFrame(
        {
            "Article Title": ["A", "B"],
            "是否属于城市更新研究(人工)": [1, 0],
        }
    )
    pred_df = pd.DataFrame(
        {
            "Article Title": ["A", "B"],
            "是否属于城市更新研究": [1, 0],
        }
    )
    aligned = align_truth_pred(truth_df, pred_df, strict=True)
    metrics_df, _ = evaluate_merged(aligned.merged, source_name="demo")
    urban = metrics_df[metrics_df["Metric"] == "Urban Renewal"].iloc[0]
    assert int(urban["Correct"]) == 2
    assert int(urban["Total"]) == 2


def test_resolve_truth_for_prediction_prefers_contains_match(tmp_path):
    truth_files = [tmp_path / "test1.xlsx", tmp_path / "test2.xlsx"]
    for file_path in truth_files:
        file_path.touch()
    truth_file, mode = resolve_truth_for_prediction(
        tmp_path / "test2_urban_renewal_few.xlsx",
        truth_files,
        experiment_track="legacy_archive",
    )
    assert truth_file.name == "test2.xlsx"
    assert mode == "contains_match"


def test_resolve_truth_for_prediction_fallback_first(tmp_path):
    truth_files = [tmp_path / "alpha.xlsx", tmp_path / "beta.xlsx"]
    for file_path in truth_files:
        file_path.touch()
    truth_file, mode = resolve_truth_for_prediction(
        tmp_path / "urban_renewal_few_20260330.xlsx",
        truth_files,
        experiment_track="legacy_archive",
    )
    assert truth_file.name == "alpha.xlsx"
    assert mode == "fallback_first_truth"


def test_resolve_truth_for_prediction_rejects_heuristics_in_stable_release(tmp_path):
    truth_files = [tmp_path / "alpha.xlsx", tmp_path / "beta.xlsx"]
    for file_path in truth_files:
        file_path.touch()
    try:
        resolve_truth_for_prediction(
            tmp_path / "urban_renewal_few_20260330.xlsx",
            truth_files,
            experiment_track="stable_release",
        )
        assert False, "expected strict truth binding failure"
    except ValueError as exc:
        assert "forbids heuristic truth matching" in str(exc)


def test_resolve_truth_files_requires_explicit_truth_when_multiple_labels_exist(tmp_path):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "a.xlsx").touch()
    (labels_dir / "b.xlsx").touch()
    try:
        resolve_truth_files(labels_dir, experiment_track="research_matrix")
        assert False, "expected explicit truth requirement"
    except ValueError as exc:
        assert "requires an explicit --truth" in str(exc)


def test_flatten_diagnostics_default_minimal():
    diagnostics = {
        "truth_unmatched": pd.DataFrame({"_key": ["a"], "x": ["long text"]}),
    }
    rows = flatten_diagnostics({"truth_count": 1.0}, diagnostics, verbose=False)
    assert len(rows[rows["Type"] == "truth_unmatched"]) == 0
    section = rows[(rows["Type"] == "section") & (rows["Key"] == "truth_unmatched")].iloc[0]
    assert section["Value"] == 1


def test_parse_single_output_prefers_explicit_answer():
    router = TaskRouter.__new__(TaskRouter)
    label, reason = TaskRouter._parse_single_output(router, "Step 1: 1\n最终答案: 0")
    assert label == "0"
    assert reason == "explicit_answer_pattern"


def test_process_urban_renewal_persists_session():
    router = TaskRouter.__new__(TaskRouter)
    router.urban_prompt_gen = _DummyPromptGen()
    router.urban_client = _DummyClient("最终答案: 1")
    memory = _DummyMemory()
    result = TaskRouter._process_urban_renewal(router, "t", "a", memory)
    assert result["是否属于城市更新研究"] == "1"
    assert memory.saved == 1


def test_build_output_rows_do_not_leak_input_labels():
    router = TaskRouter.__new__(TaskRouter)
    urban_row = TaskRouter._build_urban_output_row(
        router,
        "title",
        "abstract",
        {"是否属于城市更新研究": "1", "urban_parse_reason": "single_digit_line"},
    )
    assert list(urban_row.keys())[:5] == [
        "Article Title",
        "Abstract",
        "是否属于城市更新研究",
        "urban_flag",
        "urban_parse_reason",
    ]
    for column in [
        "decision_explanation",
        "primary_positive_evidence",
        "primary_negative_evidence",
        "evidence_balance",
        "decision_rule_stack",
        "binary_decision_evidence",
        "urban_probability_score",
        "binary_decision_source",
        "taxonomy_coverage_status",
        "unknown_recovery_path",
        "unknown_recovery_evidence",
    ]:
        assert column in urban_row

    spatial_row = TaskRouter._build_spatial_output_row(
        router,
        "title",
        "abstract",
        {
            "空间研究/非空间研究": "1",
            "空间等级": "7",
            "具体空间描述": "beijing",
            "Reasoning": "case study",
            "Confidence": "High",
        },
    )
    assert list(spatial_row.keys()) == ["Article Title", "Abstract", "空间研究/非空间研究", "空间等级", "具体空间描述", "Reasoning", "Confidence"]


def test_build_urban_output_row_keeps_anchor_guard_audit_fields():
    router = TaskRouter.__new__(TaskRouter)
    urban_row = TaskRouter._build_urban_output_row(
        router,
        "title",
        "abstract",
        {
            "鏄惁灞炰簬鍩庡競鏇存柊鐮旂┒": "1",
            "urban_parse_reason": "anchor_guard_promotion",
            "anchor_guard_flag": 1,
            "anchor_guard_action": "promote",
            "anchor_guard_reason": "urban_within_candidate:U9@5.20/1.30",
            "anchor_guard_hits": "urban redevelopment",
        },
    )
    assert urban_row["anchor_guard_flag"] == 1
    assert urban_row["anchor_guard_action"] == "promote"
    assert "urban_within_candidate" in urban_row["anchor_guard_reason"]
    assert urban_row["anchor_guard_hits"] == "urban redevelopment"


def test_build_urban_output_row_keeps_uncertain_nonurban_guard_audit_fields():
    router = TaskRouter.__new__(TaskRouter)
    urban_row = TaskRouter._build_urban_output_row(
        router,
        "title",
        "abstract",
        {
            "urban_parse_reason": "uncertain_nonurban_review",
            "urban_flag": "",
            "final_label": "",
            "uncertain_nonurban_guard_flag": 1,
            "uncertain_nonurban_guard_action": "review",
            "uncertain_nonurban_guard_reason": "high_risk_rule:N3",
            "uncertain_nonurban_guard_evidence": "high_risk_rule:N3; broad_anchor:urban redevelopment",
            "urban_probability_score": 0.43,
            "binary_decision_threshold": 0.45,
            "binary_decision_source": "binary_confidence_resolution",
            "binary_decision_evidence": "family=0.40*0.40",
            "binary_topic_consistency_flag": 1,
            "binary_recall_calibration_flag": 1,
            "binary_recall_calibration_tier": "context_relevance_floor",
            "binary_recall_calibration_reason": "raw_score>=0.07",
            "open_set_flag": 1,
            "open_set_topic": "Urban_Renewal_Other",
            "open_set_reason": "core_renewal_anchor",
            "open_set_evidence": "urban renewal",
            "taxonomy_coverage_status": "open_set",
            "decision_explanation": "final=1; score=0.6200>=threshold=0.4500",
            "primary_positive_evidence": "open_set=core_renewal_anchor(urban renewal)",
            "primary_negative_evidence": "none",
            "evidence_balance": "positive",
            "decision_rule_stack": "route=uncertain > open_set=Urban_Renewal_Other",
        },
    )
    assert urban_row["uncertain_nonurban_guard_flag"] == 1
    assert urban_row["uncertain_nonurban_guard_action"] == "review"
    assert "high_risk_rule" in urban_row["uncertain_nonurban_guard_reason"]
    assert "broad_anchor" in urban_row["uncertain_nonurban_guard_evidence"]
    assert urban_row["urban_probability_score"] == 0.43
    assert urban_row["binary_decision_source"] == "binary_confidence_resolution"
    assert urban_row["binary_topic_consistency_flag"] == 1
    assert urban_row["binary_recall_calibration_flag"] == 1
    assert urban_row["binary_recall_calibration_tier"] == "context_relevance_floor"
    assert urban_row["open_set_topic"] == "Urban_Renewal_Other"
    assert urban_row["taxonomy_coverage_status"] == "open_set"
    assert urban_row["decision_explanation"].startswith("final=1")
    assert "open_set=core_renewal_anchor" in urban_row["primary_positive_evidence"]
    assert urban_row["evidence_balance"] == "positive"
    assert "open_set=Urban_Renewal_Other" in urban_row["decision_rule_stack"]


def test_chunk_metrics_capture_chunk_level_recall_drop():
    truth_df = pd.DataFrame(
        {
            "Article Title": ["A", "B", "C", "D"],
            "是否属于城市更新研究": [1, 1, 1, 1],
            "空间研究/非空间研究": [1, 1, 1, 1],
            "空间等级": ["7", "7", "7", "7"],
            "具体空间描述": ["x", "x", "x", "x"],
        }
    )
    pred_df = pd.DataFrame(
        {
            "Article Title": ["A", "B", "C", "D"],
            "是否属于城市更新研究": [1, 1, 0, 0],
            "空间研究/非空间研究": [1, 1, 1, 1],
            "空间等级": ["7", "7", "7", "7"],
            "具体空间描述": ["x", "x", "x", "x"],
        }
    )
    aligned = align_truth_pred(truth_df, pred_df, strict=True)
    chunk_df = summarize_chunked_binary_metrics(aligned.merged, source_name="demo", chunk_size=2)
    urban_chunk_2 = chunk_df[(chunk_df["Metric"] == "Urban Renewal") & (chunk_df["Chunk"] == 2)].iloc[0]
    assert float(urban_chunk_2["Recall"]) == 0.0
    assert float(urban_chunk_2["Predicted Positive Rate"]) == 0.0


def test_build_prediction_guardrails_flags_near_zero_positive_rate():
    pred_df = pd.DataFrame(
        {
            "Article Title": ["A", "B", "C", "D"],
            "是否属于城市更新研究": [0, 0, 0, 0],
            "urban_parse_reason": ["single_digit_line", "single_digit_line", "fallback_first_digit", "fallback_first_digit"],
        }
    )
    guardrail_df = build_prediction_guardrails(pred_df, source_name="demo", chunk_size=2)
    first_chunk = guardrail_df.iloc[0]
    second_chunk = guardrail_df.iloc[1]
    assert "predicted_positive_rate_near_zero" in first_chunk["Warnings"]
    assert "high_parse_fallback_rate" in second_chunk["Warnings"]


def test_evaluate_one_file_writes_guardrail_sheets(tmp_path):
    truth_df = pd.DataFrame(
        {
            "Article Title": ["A"],
            "Abstract": ["abs"],
            "是否属于城市更新研究": [1],
            "空间研究/非空间研究": [1],
            "空间等级": ["7"],
            "具体空间描述": ["beijing"],
        }
    )
    pred_df = pd.DataFrame(
        {
            "Article Title": ["A"],
            "Abstract": ["abs"],
            "是否属于城市更新研究": [1],
            "空间研究/非空间研究": [1],
            "空间等级": ["7"],
            "具体空间描述": ["beijing"],
        }
    )
    pred_path = tmp_path / "urban_renewal_demo.xlsx"
    pred_df.to_excel(pred_path, index=False, engine="openpyxl")
    result = evaluate_one_file(
        truth_df=truth_df,
        pred_file=pred_path,
        report_dir=tmp_path,
        strict=True,
        coverage_threshold=0.8,
        spatial_desc_threshold=0.6,
        chunk_size=100,
        verbose_diagnostics=False,
    )
    report_path = result[-1]
    workbook = pd.ExcelFile(report_path, engine="openpyxl")
    assert workbook.sheet_names == [
        "Detail Comparison",
        "Quality Metrics",
        "Theme Metrics",
        "Theme Confusion",
        "U-N Family Metrics",
        "Unknown Rate",
        "Decision Source Metrics",
        "Topic Distribution",
        "Boundary Bucket Metrics",
        "Unknown Conflict Analysis",
        "Explainability Quality",
        "Evidence Balance Metrics",
        "Dynamic Topic Quality",
        "Dynamic Topic Distribution",
        "Dynamic Fixed Crosswalk",
        "Dynamic Topic Candidates",
        "Dynamic Binary Recommendations",
        "Bootstrap CI",
        "Chunk Metrics",
        "Guardrails",
        "Urban Error Analysis",
    ]


def test_build_urban_error_analysis_extracts_fn_fp_and_categories():
    detail_df = pd.DataFrame(
        {
            "Article Title": [
                "Impact of Urban Renewal on Surface Temperature",
                "The hotel and the city",
            ],
            "Abstract": [
                "This study uses remote sensing to evaluate temperature changes caused by urban renewal in Guangzhou.",
                "This article discusses hotels in twentieth-century urbanism and mentions urban renewal strategies.",
            ],
            f"{Schema.IS_URBAN_RENEWAL}_truth": [1, 0],
            f"{Schema.IS_URBAN_RENEWAL}_pred": [0, 1],
        }
    )
    pred_df = pd.DataFrame(
        {
            "Article Title": [
                "Impact of Urban Renewal on Surface Temperature",
                "The hotel and the city",
            ],
            "urban_parse_reason": ["single_digit_line", "single_digit_line"],
        }
    )
    error_df = build_urban_error_analysis(detail_df, pred_df, "demo")
    assert list(error_df["Error Type"]) == ["FN", "FP"]
    assert bool(error_df.iloc[0]["Contains Explicit Renewal Anchor"]) is True
    assert error_df.iloc[0]["Error Category"] == "method_design_evaluation_under_renewal"
    assert error_df.iloc[1]["Error Category"] == "explicit_renewal_wording_but_other_object"


def test_build_long_context_stability_groups_runs_by_order_insensitive_signature():
    merged_metrics = pd.DataFrame(
        [
            {"File": "run_a", "Metric": "Urban Renewal", "Accuracy": 85.0, "Precision": 0.95, "Recall": 0.91, "F1": 0.93},
            {"File": "run_b", "Metric": "Urban Renewal", "Accuracy": 86.0, "Precision": 0.96, "Recall": 0.92, "F1": 0.94},
            {"File": "run_c", "Metric": "Urban Renewal", "Accuracy": 84.7, "Precision": 0.949, "Recall": 0.915, "F1": 0.931},
        ]
    )
    merged_unknown_rates = pd.DataFrame(
        [
            {"File": "run_a", "Predicted Unknown Count": 40},
            {"File": "run_b", "Predicted Unknown Count": 42},
            {"File": "run_c", "Predicted Unknown Count": 41},
        ]
    )
    run_metadata_df = pd.DataFrame(
        [
            {
                "prediction_file": "run_a.xlsx",
                "prediction_stem": "run_a",
                "session_policy": "cross_paper_long_context",
                "order_id": "canonical_title_order",
                "long_context_group_signature": "sig-1",
            },
            {
                "prediction_file": "run_b.xlsx",
                "prediction_stem": "run_b",
                "session_policy": "cross_paper_long_context",
                "order_id": "shuffle_seed_20260415_a",
                "long_context_group_signature": "sig-1",
            },
            {
                "prediction_file": "run_c.xlsx",
                "prediction_stem": "run_c",
                "session_policy": "cross_paper_long_context",
                "order_id": "shuffle_seed_20260415_b",
                "long_context_group_signature": "sig-1",
            },
        ]
    )
    stability_df = build_long_context_stability(merged_metrics, merged_unknown_rates, run_metadata_df)
    row = stability_df.iloc[0]
    assert row["Run Count"] == 3
    assert row["Long Context Group Signature"] == "sig-1"
    assert row["Unknown Range"] == 2.0
    assert bool(row["Order Sensitive"]) is False


def test_group_summary_excludes_single_long_context_runs_from_main_table():
    merged_metrics = pd.DataFrame(
        [
            {"File": "run_a", "Metric": "Urban Renewal", "Accuracy": 85.0, "Correct": 85, "Total": 100, "TP": 50, "TN": 35, "FP": 5, "FN": 10, "Precision": 0.91, "Recall": 0.83, "F1": 0.87},
            {"File": "run_b", "Metric": "Urban Renewal", "Accuracy": 86.0, "Correct": 86, "Total": 100, "TP": 51, "TN": 35, "FP": 4, "FN": 10, "Precision": 0.927, "Recall": 0.836, "F1": 0.879},
            {"File": "stable_run", "Metric": "Urban Renewal", "Accuracy": 90.0, "Correct": 90, "Total": 100, "TP": 60, "TN": 30, "FP": 3, "FN": 7, "Precision": 0.952, "Recall": 0.896, "F1": 0.923},
        ]
    )
    comparability_df = pd.DataFrame(
        [
            {"prediction_file": "run_a.xlsx", "comparability_signature": "sig-long-a"},
            {"prediction_file": "run_b.xlsx", "comparability_signature": "sig-long-b"},
            {"prediction_file": "stable_run.xlsx", "comparability_signature": "sig-stable"},
        ]
    )
    run_metadata_df = pd.DataFrame(
        [
            {"prediction_file": "run_a.xlsx", "session_policy": "cross_paper_long_context"},
            {"prediction_file": "run_b.xlsx", "session_policy": "cross_paper_long_context"},
            {"prediction_file": "stable_run.xlsx", "session_policy": "per_paper_isolated"},
        ]
    )
    group_summary_df = build_group_summaries(merged_metrics, comparability_df, run_metadata_df=run_metadata_df)
    assert group_summary_df["Files"].tolist() == ["stable_run.xlsx"]
