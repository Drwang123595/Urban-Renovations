from argparse import Namespace
from pathlib import Path

import pandas as pd

from scripts.pipeline.run_stable_release import DEFAULT_DATASET_ID, DEFAULT_TAG
from scripts.reporting.generate_stage_report import generate_report, resolve_report_inputs


def _args(tmp_path: Path, **overrides):
    values = {
        "dataset_id": DEFAULT_DATASET_ID,
        "tag": DEFAULT_TAG,
        "pred": None,
        "report_dir": None,
        "eval_summary": None,
        "run_summary": None,
        "unknown_review": None,
        "output_dir": tmp_path,
        "tables": None,
        "pdf": None,
        "no_pdf": True,
    }
    values.update(overrides)
    return Namespace(**values)


def test_stage_report_resolves_current_stable_paths(tmp_path):
    inputs = resolve_report_inputs(_args(tmp_path))

    assert inputs.dataset_id == DEFAULT_DATASET_ID
    assert inputs.tag == DEFAULT_TAG
    assert inputs.prediction_file.name == f"urban_renewal_three_stage_hybrid_few_llm_on_{DEFAULT_TAG}.xlsx"
    assert inputs.eval_summary_file.name == "Eval_Summary.xlsx"
    assert inputs.run_summary_file.name == "Stable_Run_Summary.json"


def test_stage_report_exports_tables_without_pdf_dependencies(tmp_path):
    outputs = generate_report(_args(tmp_path))

    table_path = outputs["tables"]
    assert table_path.exists()
    assert "pdf" not in outputs

    sheets = pd.ExcelFile(table_path, engine="openpyxl").sheet_names
    assert "Explainability Quality" in sheets
    assert "Evidence Balance Metrics" in sheets

    stable_metrics = pd.read_excel(table_path, sheet_name="Stable Metrics", engine="openpyxl")
    values = dict(zip(stable_metrics["Field"], stable_metrics["Value"]))
    assert values["tag"] == DEFAULT_TAG
    assert int(values["rows"]) == 1000
    assert int(values["llm_used_sum"]) == 0
    assert "explanation_coverage" in values
    assert "binary_evidence_coverage" in values
