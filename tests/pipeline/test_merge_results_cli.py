import pandas as pd

from scripts.data.merge_results import merge_results
from src.runtime.config import Config, Schema


def test_merge_results_creates_managed_report_dir_and_applies_strategy_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(Config, "DATA_OUTPUT_DIR", tmp_path / "output", raising=False)
    output_dir = tmp_path / "output" / "demo" / "legacy_output"
    output_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                Schema.TITLE: "A",
                Schema.ABSTRACT: "A abstract",
                Schema.IS_URBAN_RENEWAL: "1",
            }
        ]
    ).to_excel(output_dir / "single_zero_20260420.xlsx", index=False, engine="openpyxl")
    pd.DataFrame(
        [
            {
                Schema.TITLE: "A",
                Schema.ABSTRACT: "A abstract",
                Schema.IS_SPATIAL: "1",
            }
        ]
    ).to_excel(output_dir / "spatial_zero_20260420.xlsx", index=False, engine="openpyxl")

    merged_path = merge_results("demo", ["single"])

    assert merged_path is not None
    assert merged_path.parent == tmp_path / "output" / "demo" / "legacy_result"
    assert merged_path.exists()

    merged = pd.read_excel(merged_path, engine="openpyxl")
    assert any("SINGLE_ZERO_20260420" in column for column in merged.columns)
    assert not any("SPATIAL_ZERO_20260420" in column for column in merged.columns)


def test_merge_results_supports_canonical_run_dir(tmp_path):
    run_dir = tmp_path / "Data" / "output" / "Demo Dataset 20260520" / "runs" / "stable_release" / "tag"
    pred_dir = run_dir / "predictions" / "urban_renewal"
    pred_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                Schema.TITLE: "A",
                Schema.ABSTRACT: "A abstract",
                Schema.IS_URBAN_RENEWAL: "1",
            }
        ]
    ).to_excel(pred_dir / "urban_renewal_three_stage_hybrid.xlsx", index=False, engine="openpyxl")

    merged_path = merge_results(run_dir=run_dir)

    assert merged_path is not None
    assert merged_path.parent == run_dir / "reports"
    assert merged_path.name == "comparison__demo_dataset_20260520__urban_renewal__tag.xlsx"
    assert merged_path.exists()


def test_merge_results_can_select_spatial_canonical_task_dir(tmp_path):
    run_dir = tmp_path / "runs" / "research_matrix" / "tag"
    urban_dir = run_dir / "predictions" / "urban_renewal"
    spatial_dir = run_dir / "predictions" / "spatial"
    urban_dir.mkdir(parents=True)
    spatial_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {
                Schema.TITLE: "A",
                Schema.ABSTRACT: "A abstract",
                Schema.IS_URBAN_RENEWAL: "1",
            }
        ]
    ).to_excel(urban_dir / "urban_renewal_zero.xlsx", index=False, engine="openpyxl")
    pd.DataFrame(
        [
            {
                Schema.TITLE: "A",
                Schema.ABSTRACT: "A abstract",
                Schema.IS_SPATIAL: "1",
            }
        ]
    ).to_excel(spatial_dir / "spatial_zero.xlsx", index=False, engine="openpyxl")

    merged_path = merge_results(run_dir=run_dir, prediction_task="spatial")

    assert merged_path is not None
    merged = pd.read_excel(merged_path, engine="openpyxl")
    assert any("SPATIAL_ZERO" in column for column in merged.columns)
    assert not any("URBAN_RENEWAL_ZERO" in column for column in merged.columns)
