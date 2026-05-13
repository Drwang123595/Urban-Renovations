import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.analysis.evaluate_spatial_gpt_vs_pipeline import load_and_align
from src.runtime.config import Schema


def _write_labels(path: Path) -> None:
    pd.DataFrame(
        {
            "row_index": [0, 1],
            Schema.TITLE: ["A spatial paper", "B non-spatial paper"],
            Schema.ABSTRACT: ["case in A", "theory"],
            "gpt_is_spatial": [1, 0],
            "gpt_spatial_scale_level": ["7. Single-city / Municipal Scale", None],
            "gpt_specific_study_area": ["A", None],
            "label_status": ["valid", "valid"],
        }
    ).to_excel(path, index=False, engine="openpyxl")


def test_spatial_eval_auto_aligns_by_title_when_pipeline_is_sorted(tmp_path: Path):
    labels_path = tmp_path / "labels.xlsx"
    pipeline_path = tmp_path / "pipeline.xlsx"
    _write_labels(labels_path)
    pd.DataFrame(
        {
            Schema.TITLE: ["B non-spatial paper", "A spatial paper"],
            Schema.ABSTRACT: ["theory", "case in A"],
            Schema.IS_SPATIAL: [0, 1],
            Schema.SPATIAL_LEVEL: ["Not mentioned", "7. Single-city / Municipal Scale"],
            Schema.SPATIAL_DESC: ["Not mentioned", "A"],
        }
    ).to_excel(pipeline_path, index=False, engine="openpyxl")

    aligned = load_and_align(labels_path, pipeline_path)

    assert aligned.attrs["alignment_mode"] == "title"
    assert aligned[Schema.TITLE + "_gpt_label_file"].tolist() == [
        "A spatial paper",
        "B non-spatial paper",
    ]
    assert aligned[Schema.IS_SPATIAL].tolist() == [1, 0]
    assert aligned["pipeline_row_index"].tolist() == [1, 0]


def test_spatial_eval_auto_keeps_row_index_when_titles_match(tmp_path: Path):
    labels_path = tmp_path / "labels.xlsx"
    pipeline_path = tmp_path / "pipeline.xlsx"
    _write_labels(labels_path)
    pd.DataFrame(
        {
            Schema.TITLE: ["A spatial paper", "B non-spatial paper"],
            Schema.ABSTRACT: ["case in A", "theory"],
            Schema.IS_SPATIAL: [1, 0],
            Schema.SPATIAL_LEVEL: ["7. Single-city / Municipal Scale", "Not mentioned"],
            Schema.SPATIAL_DESC: ["A", "Not mentioned"],
        }
    ).to_excel(pipeline_path, index=False, engine="openpyxl")

    aligned = load_and_align(labels_path, pipeline_path)

    assert aligned.attrs["alignment_mode"] == "row_index"
    assert aligned[Schema.IS_SPATIAL].tolist() == [1, 0]
