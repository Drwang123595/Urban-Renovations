import pandas as pd

from scripts.data.merge_results import merge_results
from src.config import Config, Schema


def test_legacy_merge_results_creates_result_dir_and_applies_strategy_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "DATA_DIR", tmp_path)
    output_dir = tmp_path / "demo" / "output"
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
    assert merged_path.parent == tmp_path / "demo" / "Result"
    assert merged_path.exists()

    merged = pd.read_excel(merged_path, engine="openpyxl")
    assert any("SINGLE_ZERO_20260420" in column for column in merged.columns)
    assert not any("SPATIAL_ZERO_20260420" in column for column in merged.columns)


def test_merge_results_supports_canonical_run_dir(tmp_path):
    run_dir = tmp_path / "runs" / "stable_release" / "tag"
    pred_dir = run_dir / "predictions"
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
    assert merged_path.exists()
