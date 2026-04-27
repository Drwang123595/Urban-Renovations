import tomllib
from pathlib import Path

import pandas as pd

from scripts.pipeline.run_stable_release import (
    DEFAULT_DATASET_ID,
    DEFAULT_TAG,
    StableThresholds,
    build_classification_command,
    build_paths,
    validate_gates,
)
from src.runtime.project_paths import dataset_paths, run_paths


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STABLE_RESULT_DIR = (
    PROJECT_ROOT
    / "Data"
    / "Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407"
    / "runs"
    / "stable_release"
    / DEFAULT_TAG
    / "reports"
)
STABLE_OUTPUT_DIR = (
    PROJECT_ROOT
    / "Data"
    / "Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407"
    / "runs"
    / "stable_release"
    / DEFAULT_TAG
    / "predictions"
)
STABLE_OUTPUT_FILE = STABLE_OUTPUT_DIR / f"urban_renewal_three_stage_hybrid_few_llm_on_{DEFAULT_TAG}.xlsx"
STABLE_SUMMARY_FILE = STABLE_RESULT_DIR / "Eval_Summary.xlsx"
STABLE_UNKNOWN_REVIEW_FILE = (
    PROJECT_ROOT
    / "Data"
    / "Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407"
    / "runs"
    / "stable_release"
    / DEFAULT_TAG
    / "reviews"
    / f"Unknown_Review_hybrid_llm_on_{DEFAULT_TAG}.xlsx"
)
STABLE_FILE_STEM = f"urban_renewal_three_stage_hybrid_few_llm_on_{DEFAULT_TAG}"


def _dependency_names(requirements):
    names = set()
    for requirement in requirements:
        normalized = requirement.lower()
        for separator in [">=", "==", "<=", "~=", ">", "<"]:
            normalized = normalized.split(separator, 1)[0]
        names.add(normalized.strip())
    return names


def test_stable_pipeline_entry_locks_paths_and_command():
    paths = build_paths()
    assert paths.dataset_id == DEFAULT_DATASET_ID
    assert paths.tag == DEFAULT_TAG
    assert paths.labels_file.name == f"{DEFAULT_DATASET_ID}.xlsx"
    assert paths.prediction_file == STABLE_OUTPUT_FILE
    assert paths.eval_summary_file == STABLE_SUMMARY_FILE
    assert paths.unknown_review_file == STABLE_UNKNOWN_REVIEW_FILE

    command = build_classification_command(Path("python"), paths)
    command_text = " ".join(command)
    assert "scripts\\pipeline\\main_py313.py" in command_text or "scripts/pipeline/main_py313.py" in command_text
    assert "--experiment-track stable_release" in command_text
    assert "--urban-method three_stage_hybrid" in command_text
    assert "--hybrid-llm-assist on" in command_text
    assert "--order-id canonical_title_order" in command_text
    assert "--max-samples-per-window 50" in command_text


def test_project_paths_define_canonical_input_and_run_layout():
    dataset = dataset_paths(DEFAULT_DATASET_ID)
    assert dataset.labels_dir.name == "labels"
    assert dataset.labels_dir.parent.name == "input"
    assert dataset.runs_dir.name == "runs"

    run = run_paths(DEFAULT_DATASET_ID, "stable_release", DEFAULT_TAG)
    assert run.prediction_dir.name == "predictions"
    assert run.report_dir.name == "reports"
    assert run.review_dir.name == "reviews"
    assert run.log_dir.name == "logs"
    assert run.run_dir == dataset.runs_dir / "stable_release" / DEFAULT_TAG


def test_stable_pipeline_gate_thresholds_match_locked_release():
    metrics = {
        "model_name": "deepseek-v4-flash",
        "rows": 1000,
        "total": 1000,
        "accuracy": 92.2,
        "precision": 0.9599,
        "recall": 0.94335,
        "f1": 0.951553,
        "fp": 32,
        "fn": 46,
        "predicted_unknown_count": 38,
        "predicted_unknown_rate": 0.038,
        "unknown_hint_resolution_accuracy": 94.8980,
        "llm_used_sum": 0,
        "explanation_coverage": 1.0,
        "rule_stack_coverage": 1.0,
        "binary_evidence_coverage": 1.0,
    }
    assert validate_gates(metrics, StableThresholds(), expected_rows=1000) == []

    regressed = dict(metrics)
    regressed["predicted_unknown_count"] = 39
    assert validate_gates(regressed, StableThresholds(), expected_rows=1000)

    explainability_regressed = dict(metrics)
    explainability_regressed["binary_evidence_coverage"] = 0.99
    assert "binary decision evidence coverage below stable threshold" in validate_gates(
        explainability_regressed,
        StableThresholds(),
        expected_rows=1000,
    )


def test_readme_locks_current_stable_release():
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    assert "Current project contract as of 2026-04-27" in readme
    assert "Data/<dataset_id>/runs/<track>/<tag>/..." in readme
    assert "scripts/pipeline/run_stable_release.py" in readme
    assert "three_stage_hybrid --hybrid-llm-assist on" in readme
    assert f"runs/stable_release/{DEFAULT_TAG}" in readme
    assert "deepseek-v4-flash" in readme
    assert "Accuracy = 92.2" in readme
    assert "Precision = 0.959900" in readme
    assert "F1 = 0.951553" in readme
    assert "FP = 32" in readme
    assert "FN = 46" in readme
    assert "Predicted Unknown Count = 38" in readme
    assert "unknown_hint_resolution Accuracy = 94.8980" in readme


def test_report_dependencies_are_declared_in_recommended_install_targets():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    required = {"pymupdf", "matplotlib", "seaborn", "reportlab", "pillow"}

    assert required.issubset(_dependency_names(extras["report"]))
    assert required.issubset(_dependency_names(extras["dev"]))


def test_stable_release_doc_records_thresholds_and_artifacts():
    doc_text = (PROJECT_ROOT / "doc" / "evaluation_baseline_20260408.md").read_text(encoding="utf-8")
    assert "stable release as of 2026-04-27" in doc_text
    assert f"runs/stable_release/{DEFAULT_TAG}" in doc_text
    assert "Data/<dataset_id>/input/labels/<dataset_id>.xlsx" in doc_text
    assert "deepseek-v4-flash" in doc_text
    assert "| `three_stage_hybrid + LLM on` | 92.2% | 0.959900 | 0.943350 | 0.951553 | 766 | 156 | 32 | 46 | 38 |" in doc_text
    assert "Precision >= 0.956" in doc_text
    assert "Recall >= 0.940" in doc_text
    assert "unknown_hint_resolution" in doc_text


def test_stable_release_artifacts_exist():
    assert STABLE_OUTPUT_FILE.exists()
    assert STABLE_SUMMARY_FILE.exists()
    assert STABLE_UNKNOWN_REVIEW_FILE.exists()


def test_stable_release_metrics_meet_locked_thresholds():
    metrics_df = pd.read_excel(STABLE_SUMMARY_FILE, sheet_name="All Metrics", engine="openpyxl")
    urban = metrics_df[
        (metrics_df["File"] == STABLE_FILE_STEM) & (metrics_df["Metric"] == "Urban Renewal")
    ].iloc[0]
    assert float(urban["Accuracy"]) >= 88.0
    assert float(urban["Precision"]) >= 0.956
    assert float(urban["Recall"]) >= 0.940
    assert float(urban["F1"]) >= 0.948
    assert int(urban["FP"]) <= 34
    assert int(urban["FN"]) <= 48


def test_stable_release_decision_source_and_unknown_contract():
    unknown_df = pd.read_excel(STABLE_SUMMARY_FILE, sheet_name="Unknown Rate", engine="openpyxl")
    unknown_row = unknown_df[unknown_df["File"] == STABLE_FILE_STEM].iloc[0]
    assert int(unknown_row["Predicted Unknown Count"]) <= 38
    assert float(unknown_row["Predicted Unknown Rate"]) <= 0.038

    decision_df = pd.read_excel(STABLE_SUMMARY_FILE, sheet_name="Decision Source Metrics", engine="openpyxl")
    stable_decisions = decision_df[decision_df["File"] == STABLE_FILE_STEM]
    assert {"rule_model_fusion", "stage1_rule", "unknown_hint_resolution"}.issubset(
        set(stable_decisions["Decision Source"])
    )
    recovered = stable_decisions[stable_decisions["Decision Source"] == "unknown_hint_resolution"].iloc[0]
    assert float(recovered["Accuracy"]) >= 92.0


def test_stable_release_output_contract_and_llm_usage():
    output_df = pd.read_excel(STABLE_OUTPUT_FILE, engine="openpyxl")
    expected_columns = {
        "topic_final",
        "urban_flag",
        "final_label",
        "review_flag",
        "review_reason",
        "unknown_recovery_path",
        "unknown_recovery_evidence",
        "decision_source",
        "llm_used",
        "llm_attempted",
        "bertopic_hint_label",
        "bertopic_hint_group",
        "bertopic_hint_name",
        "family_predicted_family",
    }
    assert expected_columns.issubset(output_df.columns)
    assert int(pd.to_numeric(output_df["llm_used"], errors="coerce").fillna(0).sum()) == 0
    assert int(pd.to_numeric(output_df["llm_attempted"], errors="coerce").fillna(0).sum()) > 0
    comparable = output_df[output_df["urban_flag"].fillna("").astype(str) != ""].copy()
    assert (
        comparable["urban_flag"].fillna("").astype(str)
        == comparable["final_label"].fillna("").astype(str)
    ).all()


def test_stable_release_theme_metrics_remain_empty_without_theme_gold():
    theme_metrics = pd.read_excel(STABLE_SUMMARY_FILE, sheet_name="Theme Metrics", engine="openpyxl")
    assert list(theme_metrics.columns) == [
        "File",
        "Theme",
        "Theme Name",
        "Theme Group",
        "Accuracy",
        "Correct",
        "Total",
        "Truth Support",
        "Pred Support",
        "TP",
        "FP",
        "FN",
        "Precision",
        "Recall",
        "F1",
    ]
    assert theme_metrics.empty
