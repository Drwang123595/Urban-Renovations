import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.analysis.spatial import label_with_openai
from scripts.analysis.spatial.evaluate_gpt_vs_pipeline import prepare_eval_frame
from scripts.analysis.spatial.blind_label_common import (
    area_match_type,
    binary_metrics,
    parse_batch_json,
    parse_label_json,
    validate_label,
)
from src.runtime.config import Schema
from src.runtime.llm_client import LLMQuotaExceededError


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


def test_openai_label_checkpoint_skips_valid_and_stops_on_quota(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "labels.jsonl"
    label_with_openai.append_checkpoint(
        checkpoint_path,
        {
            "row_index": 0,
            "source_row_0based": 0,
            Schema.TITLE: "A",
            Schema.ABSTRACT: "already labeled",
            "gpt_is_spatial": 1,
            "gpt_spatial_scale_level": "7. Single-city / Municipal Scale",
            "gpt_specific_study_area": "Paris",
            "label_status": "valid",
            "label_error": "",
            "label_model": "gpt-test",
            "label_attempts": 1,
            "raw_response": "{}",
        },
    )
    df = pd.DataFrame(
        [
            {Schema.TITLE: "A", Schema.ABSTRACT: "already labeled"},
            {Schema.TITLE: "B", Schema.ABSTRACT: "needs API"},
        ]
    )
    calls = []

    def fake_call(_client, _model, title, abstract, retry_reason=""):
        calls.append((title, abstract, retry_reason))
        raise LLMQuotaExceededError("daily limit", status_code=429, code="USAGE_LIMIT_EXCEEDED")

    monkeypatch.setattr(label_with_openai, "call_openai_label", fake_call)

    with pytest.raises(LLMQuotaExceededError):
        label_with_openai.label_dataframe(
            df,
            client=object(),
            model="gpt-test",
            checkpoint_path=checkpoint_path,
        )

    checkpoint = label_with_openai.load_checkpoint(checkpoint_path)
    assert calls == [("B", "needs API", "")]
    assert checkpoint[0]["label_status"] == "valid"
    assert checkpoint[1]["label_status"] == "quota_exhausted"
    assert checkpoint[1]["label_status"] != "valid"


def test_openai_label_detects_429_too_many_requests_as_quota():
    class FakeOpenAIError(Exception):
        status_code = 429
        body = {"message": "Too Many Requests"}

    with pytest.raises(LLMQuotaExceededError):
        label_with_openai.raise_if_quota_exceeded(FakeOpenAIError("429 Too Many Requests"))
