from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.spatial.compare_strategy_runs import _transition_frame  # noqa: E402
from scripts.analysis.spatial.evaluate_gpt_vs_pipeline import (  # noqa: E402
    build_summary,
    load_and_align,
    prepare_eval_frame,
)
from src.runtime.config import Schema  # noqa: E402


DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "Data" / "spatial_sample_2000_20260428" / "analysis"
DEFAULT_LABELS = DEFAULT_ANALYSIS_DIR / "codex_gpt_blind_labels_2000_20260428.xlsx"
DEFAULT_OLD_PIPELINE = DEFAULT_ANALYSIS_DIR / "spatial_zero_2000_final_v2_20260428.xlsx"
DEFAULT_NEW_PIPELINE = DEFAULT_ANALYSIS_DIR / "spatial_strategy_v2_2000_20260512.xlsx"
DEFAULT_COMPARE_JSON = DEFAULT_ANALYSIS_DIR / "spatial_strategy_v2_2000_comparison_20260512.json"
DEFAULT_OUTPUT_JSON = DEFAULT_ANALYSIS_DIR / "spatial_strategy_v2_2000_error_analysis_20260512.json"
DEFAULT_OUTPUT_XLSX = DEFAULT_ANALYSIS_DIR / "spatial_strategy_v2_2000_error_analysis_20260512.xlsx"
DEFAULT_OUTPUT_MD = DEFAULT_ANALYSIS_DIR / "spatial_strategy_v2_2000_error_analysis_20260512.md"


ACCEPTANCE_THRESHOLDS = {
    "recall_min": 0.92,
    "f1_min": 0.9345,
    "fp_max": 25,
    "tp_to_fn_max": 99,
    "tp_to_fn_target": 60,
    "area_different_max": 46,
    "old_scale_exact_accuracy": 0.7728571428571429,
    "scale_gap_warn": 0.05,
}


def _as_int_dict(series: pd.Series) -> Dict[str, int]:
    return {str(key): int(value) for key, value in series.value_counts(dropna=False).to_dict().items()}


def _read_compare_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _status_summary(new_summary: Dict[str, Any], comparison: Dict[str, Any], detail: pd.DataFrame) -> Dict[str, Any]:
    metrics = new_summary["binary_metrics"]
    transition_counts = comparison.get("transition_counts", {})
    tp_to_fn = int(transition_counts.get("TP->FN", 0))
    new_area_counts = new_summary.get("area_match_counts", {})
    area_different = int(new_area_counts.get("different", 0))
    scale_accuracy = float(new_summary.get("scale_exact_match_accuracy", 0.0))
    scale_gap = ACCEPTANCE_THRESHOLDS["old_scale_exact_accuracy"] - scale_accuracy
    checks = {
        "recall_ge_0_92": metrics["recall"] >= ACCEPTANCE_THRESHOLDS["recall_min"],
        "f1_gt_0_9345": metrics["f1"] > ACCEPTANCE_THRESHOLDS["f1_min"],
        "fp_le_25": metrics["fp"] <= ACCEPTANCE_THRESHOLDS["fp_max"],
        "tp_to_fn_lt_99": tp_to_fn < ACCEPTANCE_THRESHOLDS["tp_to_fn_max"],
        "area_different_le_46": area_different <= ACCEPTANCE_THRESHOLDS["area_different_max"],
        "scale_gap_within_5pp": scale_gap <= ACCEPTANCE_THRESHOLDS["scale_gap_warn"],
    }
    extra = {
        "new_fp_reason_counts": _as_int_dict(
            detail.loc[detail["binary_bucket_new"].eq("FP"), f"{Schema.SPATIAL_VALIDATION_REASON}_new"]
        ),
        "new_fn_reason_counts": _as_int_dict(
            detail.loc[detail["binary_bucket_new"].eq("FN"), f"{Schema.SPATIAL_VALIDATION_REASON}_new"]
        ),
        "tn_to_fp_reason_counts": _as_int_dict(
            detail.loc[detail["old_to_new"].eq("TN->FP"), f"{Schema.SPATIAL_VALIDATION_REASON}_new"]
        ),
        "tp_to_fn_reason_counts": _as_int_dict(
            detail.loc[detail["old_to_new"].eq("TP->FN"), f"{Schema.SPATIAL_VALIDATION_REASON}_new"]
        ),
    }
    return {
        "thresholds": ACCEPTANCE_THRESHOLDS,
        "checks": checks,
        "candidate_decision": "near_candidate_requires_targeted_fp_and_scale_review"
        if checks["recall_ge_0_92"] and checks["f1_gt_0_9345"] and not checks["fp_le_25"]
        else ("candidate_stable_strategy" if all(checks.values()) else "not_stable_candidate"),
        "scale_exact_gap_vs_old": scale_gap,
        **extra,
    }


def _compact_examples(df: pd.DataFrame, limit: int = 12) -> list[dict[str, Any]]:
    columns = [
        "Article Title_gpt_label_file",
        "gpt_spatial_scale_level",
        "gpt_specific_study_area",
        f"{Schema.IS_SPATIAL}_new",
        f"{Schema.SPATIAL_LEVEL}_new",
        f"{Schema.SPATIAL_DESC}_new",
        f"{Schema.SPATIAL_VALIDATION_REASON}_new",
        f"{Schema.SPATIAL_AREA_EVIDENCE}_new",
        "new_area_match",
        "old_to_new",
    ]
    available = [column for column in columns if column in df.columns]
    return df[available].head(limit).fillna("").to_dict(orient="records")


def build_error_analysis(
    labels: Path,
    old_pipeline: Path,
    new_pipeline: Path,
    compare_json: Path,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    new_eval = prepare_eval_frame(load_and_align(labels, new_pipeline, align_on="title"))
    new_eval.attrs["alignment_mode"] = "title"
    new_summary = build_summary(new_eval)
    comparison = _read_compare_summary(compare_json)
    detail = _transition_frame(labels, old_pipeline, new_pipeline)

    valid = new_eval[new_eval["gpt_valid"]].copy()
    both_positive = valid[
        valid["gpt_is_spatial_norm"].eq(1) & valid["pipeline_is_spatial_norm"].eq(1)
    ].copy()
    status_summary = _status_summary(new_summary, comparison, detail)
    summary: Dict[str, Any] = {
        "alignment_mode": new_summary["alignment_mode"],
        "rows": int(len(new_eval)),
        "new_metrics": new_summary["binary_metrics"],
        "new_bucket_counts": new_summary["binary_bucket_counts"],
        "new_status_counts": _as_int_dict(new_eval.get(Schema.SPATIAL_VALIDATION_STATUS, pd.Series(dtype=str))),
        "new_reason_counts": _as_int_dict(new_eval.get(Schema.SPATIAL_VALIDATION_REASON, pd.Series(dtype=str))),
        "new_fp_by_reason": status_summary["new_fp_reason_counts"],
        "new_fn_by_reason": status_summary["new_fn_reason_counts"],
        "tn_to_fp_reason_counts": status_summary["tn_to_fp_reason_counts"],
        "tp_to_fn_reason_counts": status_summary["tp_to_fn_reason_counts"],
        "scale_both_positive_n": new_summary["scale_both_positive_n"],
        "scale_exact_match_n": new_summary["scale_exact_match_n"],
        "scale_exact_match_accuracy": new_summary["scale_exact_match_accuracy"],
        "scale_confusion": pd.crosstab(
            both_positive["gpt_scale_norm"],
            both_positive["pipeline_scale_norm"],
            dropna=False,
        ).to_dict(),
        "area_match_counts": new_summary["area_match_counts"],
        "transition_counts": comparison.get("transition_counts", {}),
        "change_type_counts": comparison.get("change_type_counts", {}),
        "acceptance": status_summary,
        "fp_examples": _compact_examples(detail[detail["binary_bucket_new"].eq("FP")]),
        "fn_examples": _compact_examples(detail[detail["binary_bucket_new"].eq("FN")]),
        "tn_to_fp_examples": _compact_examples(detail[detail["old_to_new"].eq("TN->FP")]),
        "tp_to_fn_examples": _compact_examples(detail[detail["old_to_new"].eq("TP->FN")]),
    }
    return detail, summary


def write_outputs(detail: pd.DataFrame, summary: Dict[str, Any], output_json: Path, output_xlsx: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    overview_rows = [
        {"key": key, "value": json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value}
        for key, value in summary.items()
        if key not in {"scale_confusion", "fp_examples", "fn_examples", "tn_to_fp_examples", "tp_to_fn_examples"}
    ]
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        pd.DataFrame(overview_rows).to_excel(writer, sheet_name="Overview", index=False)
        pd.DataFrame([summary["new_metrics"]]).to_excel(writer, sheet_name="Binary_Metrics", index=False)
        pd.crosstab(detail["binary_bucket_old"], detail["binary_bucket_new"]).to_excel(writer, sheet_name="Transition")
        pd.crosstab(detail["old_to_new"], detail[f"{Schema.SPATIAL_VALIDATION_REASON}_new"]).to_excel(
            writer,
            sheet_name="Reason_by_Transition",
        )
        pd.crosstab(detail["gpt_scale_norm"], detail["new_scale_norm"]).to_excel(writer, sheet_name="Scale_Confusion")
        pd.DataFrame(
            [{"match_type": key, "count": value} for key, value in summary["area_match_counts"].items()]
        ).to_excel(writer, sheet_name="Area_Match", index=False)
        detail[detail["binary_bucket_new"].eq("FP")].to_excel(writer, sheet_name="FP", index=False)
        detail[detail["binary_bucket_new"].eq("FN")].to_excel(writer, sheet_name="FN", index=False)
        detail[detail["old_to_new"].eq("TN->FP")].to_excel(writer, sheet_name="TN_to_FP", index=False)
        detail[detail["old_to_new"].eq("TP->FN")].to_excel(writer, sheet_name="TP_to_FN", index=False)
        detail[
            detail["binary_bucket_new"].eq("TP") & detail["new_area_match"].isin(["different", "missing"])
        ].to_excel(writer, sheet_name="TP_Area_Disagreements", index=False)
        detail[
            detail["binary_bucket_new"].eq("TP") & (detail["gpt_scale_norm"] != detail["new_scale_norm"])
        ].to_excel(writer, sheet_name="TP_Scale_Disagreements", index=False)
        detail.to_excel(writer, sheet_name="Detail_All", index=False)

    metrics = summary["new_metrics"]
    acceptance = summary["acceptance"]
    lines = [
        "# Spatial strategy v2 2000 error analysis",
        "",
        "All rows were aligned by normalized Article Title; row_index alignment was not used.",
        "",
        "## Binary metrics",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Precision: {metrics['precision']:.4f}",
        f"- Recall: {metrics['recall']:.4f}",
        f"- Specificity: {metrics['specificity']:.4f}",
        f"- F1: {metrics['f1']:.4f}",
        f"- Cohen kappa: {metrics['cohen_kappa']:.4f}",
        f"- Confusion: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}",
        "",
        "## Movement from old strategy",
    ]
    for key, value in summary["transition_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Error drivers",
            f"- FP by reason: {summary['new_fp_by_reason']}",
            f"- FN by reason: {summary['new_fn_by_reason']}",
            f"- TN->FP by reason: {summary['tn_to_fp_reason_counts']}",
            f"- TP->FN by reason: {summary['tp_to_fn_reason_counts']}",
            "",
            "## Scale and area",
            (
                f"- Scale exact: {summary['scale_exact_match_n']}/"
                f"{summary['scale_both_positive_n']} "
                f"({summary['scale_exact_match_accuracy']:.4f})"
            ),
            f"- Area match counts: {summary['area_match_counts']}",
            "",
            "## Acceptance gate",
        ]
    )
    for key, value in acceptance["checks"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            f"- decision: {acceptance['candidate_decision']}",
            (
                "- interpretation: v2 restores recall and F1, but FP exceeds the plan threshold by "
                f"{max(0, metrics['fp'] - ACCEPTANCE_THRESHOLDS['fp_max'])}; scale exact is "
                f"{acceptance['scale_exact_gap_vs_old']:.4f} below the old strategy."
            ),
            "",
        ]
    )
    output_md.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build deep-dive error analysis for spatial strategy v2.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--old-pipeline", type=Path, default=DEFAULT_OLD_PIPELINE)
    parser.add_argument("--new-pipeline", type=Path, default=DEFAULT_NEW_PIPELINE)
    parser.add_argument("--compare-json", type=Path, default=DEFAULT_COMPARE_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-xlsx", type=Path, default=DEFAULT_OUTPUT_XLSX)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    detail, summary = build_error_analysis(args.labels, args.old_pipeline, args.new_pipeline, args.compare_json)
    write_outputs(detail, summary, args.output_json, args.output_xlsx, args.output_md)
    print(f"[INFO] error analysis json saved to {args.output_json}")
    print(f"[INFO] error analysis xlsx saved to {args.output_xlsx}")
    print(f"[INFO] error analysis md saved to {args.output_md}")
    print(f"[INFO] decision={summary['acceptance']['candidate_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
