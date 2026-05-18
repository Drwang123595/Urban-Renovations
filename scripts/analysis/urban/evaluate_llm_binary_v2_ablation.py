from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.core import align_truth_pred, normalize_binary_value
from src.runtime.config import Schema


OUTPUT_WORKBOOK = "Llm_Binary_V2_Ablation_Summary.xlsx"
OUTPUT_MANIFEST = "llm_binary_v2_ablation_manifest.json"


def _metrics(truth: pd.Series, pred: pd.Series) -> dict[str, Any]:
    truth_norm = truth.apply(normalize_binary_value)
    pred_norm = pred.apply(normalize_binary_value)
    valid = truth_norm.isin([0, 1]) & pred_norm.isin([0, 1])
    truth_norm = truth_norm[valid]
    pred_norm = pred_norm[valid]
    tp = int(((truth_norm == 1) & (pred_norm == 1)).sum())
    tn = int(((truth_norm == 0) & (pred_norm == 0)).sum())
    fp = int(((truth_norm == 0) & (pred_norm == 1)).sum())
    fn = int(((truth_norm == 1) & (pred_norm == 0)).sum())
    total = int(len(truth_norm))
    correct = int(tp + tn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "Total": total,
        "Correct": correct,
        "Accuracy": round(correct / total * 100.0, 6) if total else 0.0,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Precision": round(precision, 6),
        "Recall": round(recall, 6),
        "F1": round(f1, 6),
    }


def _prediction_column(frame: pd.DataFrame, suffix: str) -> str:
    candidates = [
        f"{Schema.IS_URBAN_RENEWAL}_{suffix}",
        f"final_label_{suffix}",
        f"urban_flag_{suffix}",
    ]
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"Missing prediction label column for suffix={suffix}")


def _align_pair(truth_df: pd.DataFrame, pred_df: pd.DataFrame, *, suffix: str) -> pd.DataFrame:
    aligned = align_truth_pred(truth_df, pred_df, strict=True).merged.copy()
    truth_col = f"{Schema.IS_URBAN_RENEWAL}_truth"
    pred_col = _prediction_column(aligned, "pred")
    title_col = f"{Schema.TITLE}_truth" if f"{Schema.TITLE}_truth" in aligned.columns else Schema.TITLE
    abstract_col = f"{Schema.ABSTRACT}_truth" if f"{Schema.ABSTRACT}_truth" in aligned.columns else Schema.ABSTRACT
    output = pd.DataFrame(
        {
            "Article Title": aligned[title_col],
            "Abstract": aligned[abstract_col] if abstract_col in aligned.columns else "",
            "truth": aligned[truth_col],
            f"{suffix}_pred": aligned[pred_col],
        }
    )
    passthrough = [
        "binary_final_source",
        "binary_final_reason",
        "llm_adjudication_attempted",
        "llm_adjudication_used",
        "llm_adjudication_status",
        "llm_adjudication_label",
        "llm_adjudication_confidence",
        "llm_adjudication_decision_type",
        "llm_adjudication_reason",
        "boundary_bucket",
        "binary_policy_conflict_type",
        "topic_final_group",
    ]
    for column in passthrough:
        pred_column = f"{column}_pred"
        source_column = pred_column if pred_column in aligned.columns else column
        if source_column in aligned.columns:
            output[f"{suffix}_{column}"] = aligned[source_column]
    return output


def _build_boundary_coverage(predictions: pd.DataFrame) -> pd.DataFrame:
    boundary_col = "candidate_boundary_bucket"
    if boundary_col not in predictions.columns:
        return pd.DataFrame(columns=["Boundary Bucket", "Total", "LLM Attempted", "LLM Used"])
    attempted = _numeric_series(predictions, "candidate_llm_adjudication_attempted")
    used = _numeric_series(predictions, "candidate_llm_adjudication_used")
    working = predictions[[boundary_col]].copy()
    working["attempted"] = attempted.astype(int)
    working["used"] = used.astype(int)
    rows = []
    for bucket, group in working.groupby(boundary_col, dropna=False):
        rows.append(
            {
                "Boundary Bucket": str(bucket or ""),
                "Total": int(len(group)),
                "LLM Attempted": int(group["attempted"].sum()),
                "LLM Used": int(group["used"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([0] * len(frame), index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0)


def run_ablation(
    *,
    truth_workbook: Path,
    baseline_workbook: Path,
    candidate_workbook: Path,
    output_dir: Path,
) -> dict[str, Any]:
    truth_df = pd.read_excel(truth_workbook, engine="openpyxl")
    baseline_df = pd.read_excel(baseline_workbook, engine="openpyxl")
    candidate_df = pd.read_excel(candidate_workbook, engine="openpyxl")

    baseline = _align_pair(truth_df, baseline_df, suffix="baseline")
    candidate = _align_pair(truth_df, candidate_df, suffix="candidate")
    predictions = baseline.merge(
        candidate.drop(columns=["truth", "Abstract"], errors="ignore"),
        on="Article Title",
        how="inner",
    )
    predictions["truth_norm"] = predictions["truth"].apply(normalize_binary_value)
    predictions["baseline_norm"] = predictions["baseline_pred"].apply(normalize_binary_value)
    predictions["candidate_norm"] = predictions["candidate_pred"].apply(normalize_binary_value)
    predictions["flip_type"] = ""
    predictions.loc[
        predictions["baseline_norm"] != predictions["candidate_norm"],
        "flip_type",
    ] = predictions.apply(
        lambda row: (
            "improved"
            if row["candidate_norm"] == row["truth_norm"] and row["baseline_norm"] != row["truth_norm"]
            else "regressed"
            if row["baseline_norm"] == row["truth_norm"] and row["candidate_norm"] != row["truth_norm"]
            else "changed_same_correctness"
        ),
        axis=1,
    )

    summary = pd.DataFrame(
        [
            {"Group": "baseline_stable_v1", **_metrics(predictions["truth"], predictions["baseline_pred"])},
            {"Group": "candidate_llm_binary_v2", **_metrics(predictions["truth"], predictions["candidate_pred"])},
        ]
    )
    flips = predictions[predictions["baseline_norm"] != predictions["candidate_norm"]].copy()
    llm_calls = predictions[_numeric_series(predictions, "candidate_llm_adjudication_attempted").astype(int) == 1].copy()
    boundary = _build_boundary_coverage(predictions)

    output_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = output_dir / OUTPUT_WORKBOOK
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        predictions.to_excel(writer, sheet_name="Predictions", index=False)
        flips.to_excel(writer, sheet_name="Flips", index=False)
        llm_calls.to_excel(writer, sheet_name="LLM Calls", index=False)
        boundary.to_excel(writer, sheet_name="Boundary Coverage", index=False)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "truth_workbook": str(Path(truth_workbook).resolve()),
        "baseline_workbook": str(Path(baseline_workbook).resolve()),
        "candidate_workbook": str(Path(candidate_workbook).resolve()),
        "output_workbook": str(workbook_path.resolve()),
        "rows": int(len(predictions)),
        "flip_count": int(len(flips)),
        "llm_attempted_count": int(len(llm_calls)),
        "groups_evaluated": ["baseline_stable_v1", "candidate_llm_binary_v2"],
    }
    manifest_path = output_dir / OUTPUT_MANIFEST
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {
        **manifest,
        "summary": summary,
        "predictions": predictions,
        "flips": flips,
        "llm_calls": llm_calls,
        "boundary": boundary,
        "manifest_path": manifest_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate stable_v1 vs llm_binary_v2 urban binary predictions.")
    parser.add_argument("--truth-workbook", required=True)
    parser.add_argument("--baseline-workbook", required=True)
    parser.add_argument("--candidate-workbook", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_ablation(
        truth_workbook=Path(args.truth_workbook),
        baseline_workbook=Path(args.baseline_workbook),
        candidate_workbook=Path(args.candidate_workbook),
        output_dir=Path(args.output_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
