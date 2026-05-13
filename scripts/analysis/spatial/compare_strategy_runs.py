import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.analysis.spatial.evaluate_gpt_vs_pipeline import (  # noqa: E402
    build_summary,
    load_and_align,
    prepare_eval_frame,
)
from scripts.analysis.spatial.blind_label_common import area_match_type, normalize_scale  # noqa: E402
from src.runtime.config import Schema  # noqa: E402


DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "Data" / "spatial_sample_2000_20260428" / "analysis"
DEFAULT_LABELS = DEFAULT_ANALYSIS_DIR / "codex_gpt_blind_labels_2000_20260428.xlsx"
DEFAULT_OLD_PIPELINE = DEFAULT_ANALYSIS_DIR / "spatial_zero_2000_final_v2_20260428.xlsx"
DEFAULT_NEW_PIPELINE = DEFAULT_ANALYSIS_DIR / "spatial_strategy_v2_2000_20260512.xlsx"
DEFAULT_OUTPUT_JSON = DEFAULT_ANALYSIS_DIR / "spatial_strategy_v2_2000_comparison_20260512.json"
DEFAULT_OUTPUT_XLSX = DEFAULT_ANALYSIS_DIR / "spatial_strategy_v2_2000_comparison_20260512.xlsx"


def _transition_frame(labels_path: Path, old_path: Path, new_path: Path) -> pd.DataFrame:
    old_eval = prepare_eval_frame(load_and_align(labels_path, old_path, align_on="title"))
    new_eval = prepare_eval_frame(load_and_align(labels_path, new_path, align_on="title"))
    old_eval = old_eval.copy()
    new_eval = new_eval.copy()
    old_eval["_title_key"] = old_eval[Schema.TITLE + "_gpt_label_file"].astype(str).str.strip().str.lower()
    new_eval["_title_key"] = new_eval[Schema.TITLE + "_gpt_label_file"].astype(str).str.strip().str.lower()

    keep_new = [
        "_title_key",
        Schema.IS_SPATIAL,
        Schema.SPATIAL_LEVEL,
        Schema.SPATIAL_DESC,
        "Reasoning",
        "Confidence",
        Schema.SPATIAL_VALIDATION_STATUS,
        Schema.SPATIAL_VALIDATION_REASON,
        Schema.SPATIAL_AREA_EVIDENCE,
        "pipeline_is_spatial_norm",
        "pipeline_scale_norm",
        "binary_bucket",
    ]
    merged = old_eval.merge(
        new_eval[keep_new],
        on="_title_key",
        how="inner",
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    if len(merged) != len(old_eval):
        raise RuntimeError(f"Comparison row mismatch: old={len(old_eval)} merged={len(merged)}")

    merged["old_to_new"] = merged["binary_bucket_old"] + "->" + merged["binary_bucket_new"]
    merged["change_type"] = merged.apply(_change_type, axis=1)
    merged["new_area_match"] = merged.apply(
        lambda row: area_match_type(row.get("gpt_specific_study_area"), row.get(f"{Schema.SPATIAL_DESC}_new")),
        axis=1,
    )
    merged["old_scale_norm"] = merged["pipeline_scale_norm_old"].map(normalize_scale)
    merged["new_scale_norm"] = merged["pipeline_scale_norm_new"].map(normalize_scale)
    return merged


def _change_type(row: pd.Series) -> str:
    old_correct = row["gpt_is_spatial_norm"] == row["pipeline_is_spatial_norm_old"]
    new_correct = row["gpt_is_spatial_norm"] == row["pipeline_is_spatial_norm_new"]
    if old_correct and new_correct:
        return "still_correct"
    if old_correct and not new_correct:
        return "regressed"
    if not old_correct and new_correct:
        return "improved"
    return "still_wrong"


def _build_summary(old_summary: Dict[str, Any], new_summary: Dict[str, Any], detail: pd.DataFrame) -> Dict[str, Any]:
    old_metrics = old_summary["binary_metrics"]
    new_metrics = new_summary["binary_metrics"]
    metric_delta = {
        key: new_metrics.get(key, 0) - old_metrics.get(key, 0)
        for key in new_metrics
        if isinstance(new_metrics.get(key), (int, float)) and isinstance(old_metrics.get(key), (int, float))
    }
    return {
        "alignment_mode": "title",
        "old_binary_metrics": old_metrics,
        "new_binary_metrics": new_metrics,
        "binary_metric_delta": metric_delta,
        "transition_counts": detail["old_to_new"].value_counts().to_dict(),
        "change_type_counts": detail["change_type"].value_counts().to_dict(),
        "validation_reason_by_transition": pd.crosstab(
            detail["old_to_new"],
            detail.get(f"{Schema.SPATIAL_VALIDATION_REASON}_new", pd.Series(dtype=str)),
        ).to_dict(),
        "old_scale": {
            "scale_both_positive_n": old_summary["scale_both_positive_n"],
            "scale_exact_match_n": old_summary["scale_exact_match_n"],
            "scale_exact_match_accuracy": old_summary["scale_exact_match_accuracy"],
        },
        "new_scale": {
            "scale_both_positive_n": new_summary["scale_both_positive_n"],
            "scale_exact_match_n": new_summary["scale_exact_match_n"],
            "scale_exact_match_accuracy": new_summary["scale_exact_match_accuracy"],
        },
        "new_tp_area_match_counts": detail.loc[
            detail["binary_bucket_new"].eq("TP"),
            "new_area_match",
        ].value_counts().to_dict(),
    }


def _write_outputs(detail: pd.DataFrame, summary: Dict[str, Any], output_json: Path, output_xlsx: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        pd.DataFrame([{"key": key, "value": json.dumps(value, ensure_ascii=False)} for key, value in summary.items()]).to_excel(
            writer,
            sheet_name="Overview",
            index=False,
        )
        pd.crosstab(detail["binary_bucket_old"], detail["binary_bucket_new"]).to_excel(writer, sheet_name="Transition")
        pd.crosstab(detail["old_to_new"], detail[f"{Schema.SPATIAL_VALIDATION_REASON}_new"]).to_excel(
            writer,
            sheet_name="Reason_by_Transition",
        )
        for sheet_name, mask in {
            "TP_to_FN": detail["old_to_new"].eq("TP->FN"),
            "FN_to_TP": detail["old_to_new"].eq("FN->TP"),
            "FP_to_TN": detail["old_to_new"].eq("FP->TN"),
            "TN_to_FP": detail["old_to_new"].eq("TN->FP"),
            "Detail_All": pd.Series(True, index=detail.index),
        }.items():
            detail.loc[mask].to_excel(writer, sheet_name=sheet_name, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare old and new spatial strategy runs with title alignment.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--old-pipeline", type=Path, default=DEFAULT_OLD_PIPELINE)
    parser.add_argument("--new-pipeline", type=Path, default=DEFAULT_NEW_PIPELINE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-xlsx", type=Path, default=DEFAULT_OUTPUT_XLSX)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    old_eval = prepare_eval_frame(load_and_align(args.labels, args.old_pipeline, align_on="title"))
    new_eval = prepare_eval_frame(load_and_align(args.labels, args.new_pipeline, align_on="title"))
    old_summary = build_summary(old_eval)
    new_summary = build_summary(new_eval)
    detail = _transition_frame(args.labels, args.old_pipeline, args.new_pipeline)
    summary = _build_summary(old_summary, new_summary, detail)
    _write_outputs(detail, summary, args.output_json, args.output_xlsx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
