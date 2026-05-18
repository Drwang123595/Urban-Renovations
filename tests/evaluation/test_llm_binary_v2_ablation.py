from pathlib import Path

import pandas as pd

from scripts.analysis.urban.evaluate_llm_binary_v2_ablation import run_ablation
from src.runtime.config import Schema


def test_llm_binary_v2_ablation_outputs_summary_and_flips(tmp_path: Path):
    truth = pd.DataFrame(
        {
            Schema.TITLE: ["A", "B", "C"],
            Schema.ABSTRACT: ["a", "b", "c"],
            Schema.IS_URBAN_RENEWAL: [1, 0, 1],
        }
    )
    baseline = pd.DataFrame(
        {
            Schema.TITLE: ["A", "B", "C"],
            Schema.ABSTRACT: ["a", "b", "c"],
            Schema.IS_URBAN_RENEWAL: [1, 1, 0],
            "final_label": [1, 1, 0],
            "boundary_bucket": ["same", "method_boundary", "same"],
        }
    )
    candidate = pd.DataFrame(
        {
            Schema.TITLE: ["A", "B", "C"],
            Schema.ABSTRACT: ["a", "b", "c"],
            Schema.IS_URBAN_RENEWAL: [1, 0, 1],
            "final_label": [1, 0, 1],
            "binary_final_source": ["deterministic_binary", "llm_binary_v2", "llm_binary_v2"],
            "llm_adjudication_attempted": [0, 1, 1],
            "llm_adjudication_used": [0, 1, 1],
            "boundary_bucket": ["same", "method_boundary", "same"],
        }
    )

    truth_path = tmp_path / "truth.xlsx"
    baseline_path = tmp_path / "baseline.xlsx"
    candidate_path = tmp_path / "candidate.xlsx"
    output_dir = tmp_path / "out"
    truth.to_excel(truth_path, index=False, engine="openpyxl")
    baseline.to_excel(baseline_path, index=False, engine="openpyxl")
    candidate.to_excel(candidate_path, index=False, engine="openpyxl")

    result = run_ablation(
        truth_workbook=truth_path,
        baseline_workbook=baseline_path,
        candidate_workbook=candidate_path,
        output_dir=output_dir,
    )

    assert result["rows"] == 3
    assert result["summary"].set_index("Group").loc["candidate_llm_binary_v2", "F1"] == 1.0
    assert result["flip_count"] == 2
    assert (output_dir / "Llm_Binary_V2_Ablation_Summary.xlsx").exists()
    assert (output_dir / "llm_binary_v2_ablation_manifest.json").exists()
