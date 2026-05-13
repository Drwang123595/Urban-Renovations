import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.analysis.evaluate_spatial_gpt_vs_pipeline import prepare_eval_frame
from scripts.analysis.spatial_blind_label_common import (
    area_match_type,
    binary_metrics,
    parse_batch_json,
    parse_label_json,
    validate_label,
)
from src.config import Schema


def test_parse_label_json_accepts_standard_object():
    parsed = parse_label_json(
        '{"Is_Spatial_Research": true, '
        '"Spatial_Scale_Level": "7. Single-city / Municipal Scale", '
        '"Specific_Study_Area": "New York City"}'
    )
    assert parsed["gpt_is_spatial"] == 1
    assert parsed["gpt_spatial_scale_level"] == "7. Single-city / Municipal Scale"
    assert parsed["gpt_specific_study_area"] == "New York City"
    assert validate_label(parsed) == ("valid", "")


def test_parse_batch_json_accepts_array_with_string_boolean():
    parsed = parse_batch_json(
        """
        [
          {"row_index": 0, "source_row_0based": 10, "gpt_is_spatial": "false",
           "gpt_spatial_scale_level": "7. Single-city / Municipal Scale",
           "gpt_specific_study_area": "Paris"}
        ]
        """
    )
    assert parsed == [
        {
            "row_index": 0,
            "source_row_0based": 10,
            "gpt_is_spatial": 0,
            "gpt_spatial_scale_level": None,
            "gpt_specific_study_area": None,
        }
    ]


def test_validate_label_rejects_missing_scale_and_placeholder_area():
    assert validate_label(
        {
            "gpt_is_spatial": 1,
            "gpt_spatial_scale_level": None,
            "gpt_specific_study_area": "Shenzhen",
        }
    ) == ("invalid", "gpt_label_invalid_missing_scale")
    assert validate_label(
        {
            "gpt_is_spatial": 1,
            "gpt_spatial_scale_level": "7. Single-city / Municipal Scale",
            "gpt_specific_study_area": "the study area in a city",
        }
    ) == ("invalid", "gpt_placeholder_or_missing_area")


def test_binary_metrics_fixed_confusion_matrix():
    metrics = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["f1"] == 0.5


def test_area_match_handles_accents_hyphens_parentheses_and_containment():
    assert area_match_type("Poznan", "Poznań") == "exact"
    assert area_match_type("the 22@ district", "22@ Innovation District (Barcelona)") in {
        "containment",
        "token_overlap",
    }
    assert area_match_type("Sham Shui Po", "Sham Shui Po in Hong Kong") == "containment"


def test_prepare_eval_frame_only_counts_scale_for_mutual_positive():
    df = pd.DataFrame(
        {
            "row_index": [0, 1],
            "label_status": ["valid", "valid"],
            "gpt_is_spatial": [1, 1],
            "gpt_spatial_scale_level": [
                "7. Single-city / Municipal Scale",
                "3. National / Single-country Scale",
            ],
            "gpt_specific_study_area": ["New York City", "China"],
            Schema.IS_SPATIAL: [1, 0],
            Schema.SPATIAL_LEVEL: ["7. Single-city / Municipal Scale", "Not mentioned"],
            Schema.SPATIAL_DESC: ["New York City", "Not mentioned"],
        }
    )
    eval_df = prepare_eval_frame(df)
    assert eval_df.loc[0, "binary_bucket"] == "TP"
    assert eval_df.loc[1, "binary_bucket"] == "FN"
