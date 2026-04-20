import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import Config, Schema
from src.evaluation_core import (
    align_truth_pred,
    evaluate_merged,
    summarize_bootstrap_ci,
    summarize_boundary_bucket_metrics,
    normalize_binary_value,
    summarize_mcnemar,
    summarize_decision_source_metrics,
    summarize_chunked_binary_metrics,
    summarize_metrics,
    summarize_theme_confusion,
    summarize_theme_family_metrics,
    summarize_theme_metrics,
    summarize_topic_final_distribution,
    summarize_unknown_conflict_analysis,
    summarize_unknown_rate,
    validate_accuracy_bounds,
)
from src.prompt_manifest import (
    build_comparability_signature,
    build_long_context_group_signature,
    compare_manifests,
    load_prompt_manifest,
    manifest_path_for_output,
)
from src.urban_training_contract import allowed_training_workbooks, assert_training_source_contract


GUARDRAIL_OUTPUT_COLUMNS = [
    "File",
    "Chunk",
    "Chunk Start",
    "Chunk End",
    "Total",
    "Predicted Positive Rate",
    "Empty Response Rate",
    "Parse Fallback Rate",
    "Warnings",
]

FALLBACK_PARSE_REASONS = {
    "fallback_first_digit",
    "fallback_boolean_yes",
    "no_label_detected",
    "empty_text",
    "missing_parse_reason",
}

URBAN_ERROR_OUTPUT_COLUMNS = [
    "File",
    "Error Type",
    "Article Title",
    "Truth Label",
    "Pred Label",
    "Error Category",
    "Contains Explicit Renewal Anchor",
    "Matched Anchor Terms",
    "Urban Parse Reason",
]

URBAN_EXPLICIT_RENEWAL_PATTERNS = {
    "urban_renewal": r"\burban renewal\b",
    "urban_regeneration": r"\burban regeneration\b|\bregeneration\b",
    "redevelopment": r"\bredevelopment\b|\bredevelop\b",
    "revitalization": r"\brevitali[sz]ation\b|\brevitali[sz]e\b",
    "retrofit": r"\bretrofit(?:ting)?\b",
    "adaptive_reuse": r"\badaptive reuse\b",
    "densification": r"\bdensification\b",
    "renewal_mission_law": r"\brenewal mission\b|\brenewal law\b",
    "renewal_project_area": r"\brenewal project\b|\brenewal area\b|\bregenerated area\b|\bold community renewal\b|\burban village renewal\b",
}

URBAN_ERROR_CATEGORY_RULES = [
    (
        "method_design_evaluation_under_renewal",
        [
            r"\bmodel\b|\bmodelling\b|\bmodeling\b|\bframework\b|\bremote sensing\b|\bdeep learning\b|\bmachine learning\b|\balgorithm\b|\bgis\b|\bsatellite\b|\bassessment\b|\bevaluation\b|\boptimi[sz]ation\b|\bdesign\b",
        ],
    ),
    (
        "governance_program_law_under_renewal",
        [
            r"\bpolicy\b|\bgovernance\b|\bmission\b|\blaw\b|\bimplementation\b|\breit\b|\bprogram(?:me)?\b|\bparticipation\b|\bcompensation\b|\brelocation\b|\bbudget\b",
        ],
    ),
    (
        "heritage_adaptive_reuse_public_realm",
        [
            r"\bheritage\b|\bconservation\b|\bhistoric\b|\badaptive reuse\b|\bpublic realm\b|\bstreet\b|\bretrofitting\b|\bretrofit\b",
        ],
    ),
    (
        "health_social_effects_under_regeneration",
        [
            r"\bhealth\b|\bphysical activity\b|\bgentrification\b|\bdisplacement\b|\brehousing\b|\bcommunity\b|\bwell-being\b|\bsocial\b",
        ],
    ),
    (
        "contamination_risk_history_with_renewal_area_wording",
        [
            r"\bpollution\b|\bcontaminat\w+\b|\bsoil\b|\brisk\b|\barchaeolog\w+\b|\bhistor\w+\b|\bmosque\b|\bpops\b|\bocps\b",
        ],
    ),
]

STRICT_TRUTH_TRACKS = {"stable_release", "research_matrix"}
LONG_CONTEXT_ACCURACY_DELTA_THRESHOLD = 1.5
LONG_CONTEXT_F1_DELTA_THRESHOLD = 0.015


def list_tasks():
    tasks = []
    if not Config.DATA_DIR.exists():
        return tasks
    for path in Config.DATA_DIR.iterdir():
        if path.is_dir() and path.name != "train":
            tasks.append(path)
    return tasks


def select_from_list(items, prompt="Select item:"):
    if not items:
        print("No items found.")
        return None
    print(f"\n{prompt}")
    for index, item in enumerate(items):
        print(f"{index + 1}: {item.name}")
    while True:
        choice = input("\nEnter number: ").strip()
        if not choice:
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                return items[idx]
            print("Invalid number.")
        except ValueError:
            print("Please enter a number.")


def resolve_truth_files(
    labels_dir: Path,
    truth_arg: str = None,
    experiment_track: str = "stable_release",
):
    if truth_arg:
        truth_file = Path(truth_arg)
        if not truth_file.exists():
            raise FileNotFoundError(f"Ground-truth file does not exist: {truth_file}")
        return [truth_file]

    truth_files = sorted(labels_dir.glob("*.xlsx"))
    if not truth_files:
        raise FileNotFoundError(f"No ground-truth files found: {labels_dir}")
    if experiment_track in STRICT_TRUTH_TRACKS and len(truth_files) != 1:
        raise ValueError(
            f"{experiment_track} requires an explicit --truth when labels dir has {len(truth_files)} workbooks: {labels_dir}"
        )
    return truth_files


def _tokenize_for_match(name: str):
    return {token for token in re.split(r"[^a-z0-9]+", name.lower()) if token}


def resolve_truth_for_prediction(
    pred_file: Path,
    truth_files,
    experiment_track: str = "stable_release",
):
    if len(truth_files) == 1:
        return truth_files[0], "single_truth"
    if experiment_track in STRICT_TRUTH_TRACKS:
        raise ValueError(
            f"{experiment_track} forbids heuristic truth matching for {pred_file.name}. Pass --truth explicitly."
        )

    pred_stem = pred_file.stem.lower()
    exact = [file_path for file_path in truth_files if file_path.stem.lower() == pred_stem]
    if exact:
        return exact[0], "exact_stem_match"

    contains = [
        file_path
        for file_path in truth_files
        if file_path.stem.lower() in pred_stem or pred_stem in file_path.stem.lower()
    ]
    if contains:
        return contains[0], "contains_match"

    pred_tokens = _tokenize_for_match(pred_stem)
    best_file = None
    best_score = 0
    for file_path in truth_files:
        score = len(pred_tokens & _tokenize_for_match(file_path.stem))
        if score > best_score:
            best_file = file_path
            best_score = score
    if best_file and best_score > 0:
        return best_file, "token_overlap_match"

    return truth_files[0], "fallback_first_truth"


def _scope_match(file_name: str, pred_scope: str) -> bool:
    lower_name = file_name.lower()
    if lower_name.startswith("eval_") or lower_name.startswith("~$"):
        return False
    if pred_scope == "all":
        return True
    if pred_scope == "urban_renewal":
        return lower_name.startswith("urban_renewal_")
    if pred_scope == "spatial":
        return lower_name.startswith("spatial_")
    if pred_scope == "merged":
        return lower_name.startswith("merged_")
    return True


def collect_pred_files(
    pred_arg: str = None,
    pred_dir_arg: str = None,
    default_output_dir: Path = None,
    pred_scope: str = "urban_renewal",
):
    if pred_arg:
        pred_file = Path(pred_arg)
        if not pred_file.exists():
            raise FileNotFoundError(f"Prediction file does not exist: {pred_file}")
        return [pred_file]

    pred_dir = Path(pred_dir_arg) if pred_dir_arg else default_output_dir
    if pred_dir is None or not pred_dir.exists():
        raise FileNotFoundError("Prediction directory does not exist. Please pass --pred-dir.")

    pred_files = sorted(
        file_path
        for file_path in pred_dir.glob("*.xlsx")
        if _scope_match(file_path.name, pred_scope)
    )
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {pred_dir} (scope={pred_scope})")
    return pred_files


def evaluate_one_file(
    truth_df: pd.DataFrame,
    pred_file: Path,
    report_dir: Path,
    strict: bool,
    coverage_threshold: float,
    spatial_desc_threshold: float,
    chunk_size: int,
    verbose_diagnostics: bool = False,
):
    df_pred = pd.read_excel(pred_file, engine="openpyxl")
    alignment = align_truth_pred(
        truth_df=truth_df,
        pred_df=df_pred,
        strict=strict,
        coverage_threshold=coverage_threshold,
    )
    metrics_df, detail_df = evaluate_merged(
        alignment.merged,
        source_name=pred_file.stem,
        spatial_desc_threshold=spatial_desc_threshold,
    )
    chunk_metrics_df = summarize_chunked_binary_metrics(
        alignment.merged,
        source_name=pred_file.stem,
        chunk_size=chunk_size,
    )
    theme_metrics_df = summarize_theme_metrics(
        alignment.merged,
        source_name=pred_file.stem,
    )
    theme_confusion_df = summarize_theme_confusion(
        alignment.merged,
        source_name=pred_file.stem,
    )
    theme_family_df = summarize_theme_family_metrics(
        alignment.merged,
        source_name=pred_file.stem,
    )
    unknown_rate_df = summarize_unknown_rate(
        alignment.merged,
        source_name=pred_file.stem,
    )
    decision_source_df = summarize_decision_source_metrics(
        alignment.merged,
        source_name=pred_file.stem,
    )
    topic_distribution_df = summarize_topic_final_distribution(
        alignment.merged,
        source_name=pred_file.stem,
    )
    boundary_bucket_df = summarize_boundary_bucket_metrics(
        alignment.merged,
        source_name=pred_file.stem,
    )
    unknown_conflict_df = summarize_unknown_conflict_analysis(
        alignment.merged,
        source_name=pred_file.stem,
    )
    bootstrap_ci_df = summarize_bootstrap_ci(
        alignment.merged,
        source_name=pred_file.stem,
    )
    guardrail_df = build_prediction_guardrails(
        df_pred,
        source_name=pred_file.stem,
        chunk_size=chunk_size,
    )
    urban_error_df = build_urban_error_analysis(
        detail_df=detail_df,
        pred_df=df_pred,
        source_name=pred_file.stem,
    )
    report_path = report_dir / f"Eval_{pred_file.name}"
    with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
        detail_df.to_excel(writer, sheet_name="Detail Comparison", index=False)
        metrics_df.to_excel(writer, sheet_name="Quality Metrics", index=False)
        theme_metrics_df.to_excel(writer, sheet_name="Theme Metrics", index=False)
        theme_confusion_df.to_excel(writer, sheet_name="Theme Confusion", index=False)
        theme_family_df.to_excel(writer, sheet_name="U-N Family Metrics", index=False)
        unknown_rate_df.to_excel(writer, sheet_name="Unknown Rate", index=False)
        decision_source_df.to_excel(writer, sheet_name="Decision Source Metrics", index=False)
        topic_distribution_df.to_excel(writer, sheet_name="Topic Distribution", index=False)
        boundary_bucket_df.to_excel(writer, sheet_name="Boundary Bucket Metrics", index=False)
        unknown_conflict_df.to_excel(writer, sheet_name="Unknown Conflict Analysis", index=False)
        bootstrap_ci_df.to_excel(writer, sheet_name="Bootstrap CI", index=False)
        chunk_metrics_df.to_excel(writer, sheet_name="Chunk Metrics", index=False)
        guardrail_df.to_excel(writer, sheet_name="Guardrails", index=False)
        urban_error_df.to_excel(writer, sheet_name="Urban Error Analysis", index=False)
    return (
        alignment.merged,
        metrics_df,
        chunk_metrics_df,
        guardrail_df,
        urban_error_df,
        theme_metrics_df,
        theme_confusion_df,
        theme_family_df,
        unknown_rate_df,
        decision_source_df,
        topic_distribution_df,
        boundary_bucket_df,
        unknown_conflict_df,
        bootstrap_ci_df,
        report_path,
    )


def flatten_diagnostics(alignment_summary: dict, diagnostics: dict, verbose: bool = False):
    rows = [
        {"Type": "summary", "Key": key, "Value": value}
        for key, value in alignment_summary.items()
    ]

    for diag_name, diag_df in diagnostics.items():
        rows.append({"Type": "section", "Key": diag_name, "Value": len(diag_df)})
        if len(diag_df) == 0 or not verbose:
            continue
        preview_cols = [
            col
            for col in ["_key", "Article Title", "Article Title_truth", "Article Title_pred"]
            if col in diag_df.columns
        ]
        if not preview_cols:
            preview_cols = diag_df.columns[:4].tolist()
        for _, row in diag_df[preview_cols].head(50).iterrows():
            rows.append(
                {
                    "Type": diag_name,
                    "Key": str(row.get("_key", "")),
                    "Value": " | ".join(str(row[col]) for col in preview_cols if col != "_key"),
                }
            )
    return pd.DataFrame(rows)


def build_prediction_guardrails(
    pred_df: pd.DataFrame,
    source_name: str,
    chunk_size: int = 100,
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if pred_df.empty:
        return pd.DataFrame(columns=GUARDRAIL_OUTPUT_COLUMNS)

    working = pred_df.reset_index(drop=True).copy()
    if Schema.IS_URBAN_RENEWAL in working.columns:
        predicted = working[Schema.IS_URBAN_RENEWAL].apply(normalize_binary_value)
    else:
        predicted = pd.Series([-1] * len(working))

    has_parse_reason = "urban_parse_reason" in working.columns
    if has_parse_reason:
        parse_reason = working["urban_parse_reason"].fillna("missing_parse_reason").astype(str)
    else:
        parse_reason = pd.Series([""] * len(working))

    rows = []
    for chunk_index, start in enumerate(range(0, len(working), chunk_size), start=1):
        end = min(start + chunk_size, len(working))
        pred_chunk = predicted.iloc[start:end]
        parse_chunk = parse_reason.iloc[start:end]
        total = len(pred_chunk)
        predicted_positive_rate = (
            float((pred_chunk == 1).mean()) if total and Schema.IS_URBAN_RENEWAL in working.columns else float("nan")
        )
        empty_response_rate = float((parse_chunk == "empty_response").mean()) if total and has_parse_reason else 0.0
        parse_fallback_rate = (
            float(parse_chunk.isin(FALLBACK_PARSE_REASONS).mean()) if total and has_parse_reason else 0.0
        )

        warnings = []
        if total and Schema.IS_URBAN_RENEWAL in working.columns and predicted_positive_rate <= 0.01:
            warnings.append("predicted_positive_rate_near_zero")
        if empty_response_rate > 0:
            warnings.append("empty_response_detected")
        if parse_fallback_rate >= 0.2:
            warnings.append("high_parse_fallback_rate")

        rows.append(
            {
                "File": source_name,
                "Chunk": chunk_index,
                "Chunk Start": start + 1,
                "Chunk End": end,
                "Total": total,
                "Predicted Positive Rate": round(predicted_positive_rate, 6) if not pd.isna(predicted_positive_rate) else pd.NA,
                "Empty Response Rate": round(empty_response_rate, 6),
                "Parse Fallback Rate": round(parse_fallback_rate, 6),
                "Warnings": ";".join(warnings),
            }
        )

    return pd.DataFrame(rows).reindex(columns=GUARDRAIL_OUTPUT_COLUMNS)


def _find_matched_anchor_terms(text: str) -> list[str]:
    matched = []
    for label, pattern in URBAN_EXPLICIT_RENEWAL_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            matched.append(label)
    return matched


def _classify_urban_error(text: str) -> str:
    for category, patterns in URBAN_ERROR_CATEGORY_RULES:
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
            return category
    if _find_matched_anchor_terms(text):
        return "explicit_renewal_wording_but_other_object"
    return "other"


def build_urban_error_analysis(
    detail_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    truth_col = f"{Schema.IS_URBAN_RENEWAL}_truth"
    pred_col = f"{Schema.IS_URBAN_RENEWAL}_pred"
    if truth_col not in detail_df.columns or pred_col not in detail_df.columns:
        return pd.DataFrame(columns=URBAN_ERROR_OUTPUT_COLUMNS)

    parse_reason_map = {}
    if Schema.TITLE in pred_df.columns and "urban_parse_reason" in pred_df.columns:
        parse_reason_map = (
            pred_df[[Schema.TITLE, "urban_parse_reason"]]
            .drop_duplicates(subset=[Schema.TITLE], keep="first")
            .set_index(Schema.TITLE)["urban_parse_reason"]
            .to_dict()
        )

    rows = []
    for _, row in detail_df.iterrows():
        truth_value = normalize_binary_value(row.get(truth_col))
        pred_value = normalize_binary_value(row.get(pred_col))
        if truth_value == pred_value or truth_value not in {0, 1} or pred_value not in {0, 1}:
            continue

        title = str(row.get(Schema.TITLE, "") or "")
        abstract = str(row.get(Schema.ABSTRACT, "") or "")
        combined_text = f"{title}\n{abstract}"
        matched_anchor_terms = _find_matched_anchor_terms(combined_text)
        rows.append(
            {
                "File": source_name,
                "Error Type": "FN" if truth_value == 1 and pred_value == 0 else "FP",
                "Article Title": title,
                "Truth Label": truth_value,
                "Pred Label": pred_value,
                "Error Category": _classify_urban_error(combined_text),
                "Contains Explicit Renewal Anchor": bool(matched_anchor_terms),
                "Matched Anchor Terms": ";".join(matched_anchor_terms),
                "Urban Parse Reason": parse_reason_map.get(title, ""),
            }
        )

    return pd.DataFrame(rows).reindex(columns=URBAN_ERROR_OUTPUT_COLUMNS)


def build_protocol_df(args, pred_files, truth_files, report_dir: Path) -> pd.DataFrame:
    rows = [
        {"Field": "experiment_track", "Value": args.experiment_track},
        {"Field": "pred_scope", "Value": args.pred_scope},
        {"Field": "truth_binding_policy", "Value": "strict_explicit_or_unique" if args.experiment_track in STRICT_TRUTH_TRACKS else "legacy_heuristic_allowed"},
        {"Field": "accuracy_scale", "Value": "0-100"},
        {"Field": "precision_recall_f1_scale", "Value": "0-1"},
        {"Field": "long_context_accuracy_delta_threshold", "Value": LONG_CONTEXT_ACCURACY_DELTA_THRESHOLD},
        {"Field": "long_context_f1_delta_threshold", "Value": LONG_CONTEXT_F1_DELTA_THRESHOLD},
        {
            "Field": "required_summary_sheets",
            "Value": "All Metrics;Run Metadata;Protocol;Comparability;Long Context Stability;Decision Source Metrics;Unknown Rate;Boundary Bucket Metrics;Unknown Conflict Analysis;Bootstrap CI;McNemar",
        },
        {
            "Field": "training_sources",
            "Value": ";".join(str(path.resolve()) for path in allowed_training_workbooks(Config.TRAIN_DIR)),
        },
        {"Field": "prediction_file_count", "Value": len(pred_files)},
        {"Field": "truth_file_count", "Value": len(truth_files)},
        {"Field": "report_dir", "Value": str(report_dir.resolve())},
    ]
    return pd.DataFrame(rows)


def build_long_context_stability(
    merged_metrics: pd.DataFrame,
    merged_unknown_rates: pd.DataFrame,
    run_metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "Long Context Group Signature",
        "Run Count",
        "Orders",
        "Mean Accuracy",
        "Mean Precision",
        "Mean Recall",
        "Mean F1",
        "Max Delta Accuracy",
        "Max Delta F1",
        "Unknown Min",
        "Unknown Max",
        "Unknown Range",
        "Order Sensitive",
        "Files",
    ]
    if merged_metrics.empty or run_metadata_df.empty:
        return pd.DataFrame(columns=columns)

    metadata = run_metadata_df.copy()
    if "session_policy" not in metadata.columns:
        return pd.DataFrame(columns=columns)

    metadata = metadata[
        metadata["session_policy"].fillna("") == "cross_paper_long_context"
    ].copy()
    if metadata.empty:
        return pd.DataFrame(columns=columns)
    metadata["prediction_stem"] = metadata["prediction_file"].apply(lambda value: Path(str(value)).stem)

    urban_metrics = merged_metrics[merged_metrics["Metric"] == "Urban Renewal"].copy()
    if urban_metrics.empty:
        return pd.DataFrame(columns=columns)

    merged = urban_metrics.merge(
        metadata[["prediction_stem", "prediction_file", "long_context_group_signature", "order_id"]],
        left_on="File",
        right_on="prediction_stem",
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)

    unknown_df = merged_unknown_rates.copy() if not merged_unknown_rates.empty else pd.DataFrame()
    if not unknown_df.empty:
        unknown_df = unknown_df.merge(
            metadata[["prediction_stem", "prediction_file", "long_context_group_signature"]],
            left_on="File",
            right_on="prediction_stem",
            how="inner",
        )

    rows = []
    for signature, frame in merged.groupby("long_context_group_signature"):
        accuracy_values = pd.to_numeric(frame["Accuracy"], errors="coerce").dropna()
        f1_values = pd.to_numeric(frame["F1"], errors="coerce").dropna()
        precision_values = pd.to_numeric(frame["Precision"], errors="coerce").dropna()
        recall_values = pd.to_numeric(frame["Recall"], errors="coerce").dropna()
        unknown_values = pd.Series(dtype=float)
        if not unknown_df.empty:
            unknown_values = pd.to_numeric(
                unknown_df[unknown_df["long_context_group_signature"] == signature]["Predicted Unknown Count"],
                errors="coerce",
            ).dropna()
        max_delta_accuracy = float(accuracy_values.max() - accuracy_values.min()) if not accuracy_values.empty else 0.0
        max_delta_f1 = float(f1_values.max() - f1_values.min()) if not f1_values.empty else 0.0
        rows.append(
            {
                "Long Context Group Signature": signature,
                "Run Count": int(len(frame)),
                "Orders": ", ".join(sorted({str(value) for value in frame["order_id"].dropna().tolist()})),
                "Mean Accuracy": float(accuracy_values.mean()) if not accuracy_values.empty else pd.NA,
                "Mean Precision": float(precision_values.mean()) if not precision_values.empty else pd.NA,
                "Mean Recall": float(recall_values.mean()) if not recall_values.empty else pd.NA,
                "Mean F1": float(f1_values.mean()) if not f1_values.empty else pd.NA,
                "Max Delta Accuracy": max_delta_accuracy,
                "Max Delta F1": max_delta_f1,
                "Unknown Min": float(unknown_values.min()) if not unknown_values.empty else pd.NA,
                "Unknown Max": float(unknown_values.max()) if not unknown_values.empty else pd.NA,
                "Unknown Range": float(unknown_values.max() - unknown_values.min()) if not unknown_values.empty else pd.NA,
                "Order Sensitive": bool(
                    max_delta_accuracy > LONG_CONTEXT_ACCURACY_DELTA_THRESHOLD
                    or max_delta_f1 > LONG_CONTEXT_F1_DELTA_THRESHOLD
                ),
                "Files": ", ".join(sorted(frame["File"].astype(str).tolist())),
            }
        )

    return pd.DataFrame(rows, columns=columns)


def build_group_summaries(
    merged_metrics: pd.DataFrame,
    comparability_df: pd.DataFrame,
    run_metadata_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if merged_metrics.empty or comparability_df.empty:
        return pd.DataFrame()

    excluded_files = set()
    if run_metadata_df is not None and not run_metadata_df.empty and "session_policy" in run_metadata_df.columns:
        excluded_files = {
            str(value)
            for value in run_metadata_df[
                run_metadata_df["session_policy"].fillna("") == "cross_paper_long_context"
            ]["prediction_file"].tolist()
        }

    grouped_frames = []
    for signature, files in comparability_df.groupby("comparability_signature"):
        file_names = set(files["prediction_file"].tolist()) - excluded_files
        if not file_names:
            continue
        subset = merged_metrics[merged_metrics["File"].isin({Path(name).stem for name in file_names})].copy()
        if subset.empty:
            continue
        summary = summarize_metrics(subset)
        summary["Comparability Signature"] = signature
        summary["Files"] = ", ".join(sorted(file_names))
        grouped_frames.append(summary)
    if not grouped_frames:
        return pd.DataFrame()
    return pd.concat(grouped_frames, ignore_index=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Offline evaluator for prediction files against ground truth")
    parser.add_argument("--task", type=str, default=None, help="Task folder under Data/, e.g. test1")
    parser.add_argument(
        "--experiment-track",
        type=str,
        default="stable_release",
        choices=list(Config.EXPERIMENT_TRACKS),
        help="Experiment governance track controlling truth binding strictness",
    )
    parser.add_argument("--truth", type=str, default=None, help="Ground-truth xlsx path")
    parser.add_argument("--pred", type=str, default=None, help="Single prediction xlsx path")
    parser.add_argument("--pred-dir", type=str, default=None, help="Prediction directory path")
    parser.add_argument(
        "--pred-scope",
        type=str,
        default="urban_renewal",
        choices=["urban_renewal", "spatial", "merged", "all"],
        help="Prediction file scope filter, default urban_renewal",
    )
    parser.add_argument("--report-dir", type=str, default=None, help="Output report directory path")
    parser.add_argument("--strict-truth-match", action="store_true", help="Require high-confidence truth matching")
    parser.add_argument("--strict", action="store_true", help="Fail on duplicate keys or low coverage")
    parser.add_argument(
        "--strict-comparable",
        action="store_true",
        help="Fail if files are not comparable by dataset/runtime/prompt manifest",
    )
    parser.add_argument("--coverage-threshold", type=float, default=0.8, help="Coverage threshold for strict mode")
    parser.add_argument("--spatial-desc-threshold", type=float, default=0.6, help="Jaccard threshold for spatial description")
    parser.add_argument("--chunk-size", type=int, default=100, help="Chunk size for chunk-level metrics and guardrails")
    return parser.parse_args()


def evaluate():
    Config.load_env()
    args = parse_args()
    assert_training_source_contract(allowed_training_workbooks(Config.TRAIN_DIR))

    task_dir = None
    if args.task:
        task_dir = Config.DATA_DIR / args.task
        if not task_dir.exists():
            raise FileNotFoundError(f"Task directory does not exist: {task_dir}")
    elif not args.truth and not args.pred and not args.pred_dir:
        tasks = list_tasks()
        selected = select_from_list(tasks, prompt="Available Tasks:")
        if not selected:
            print("No task selected.")
            return
        task_dir = selected

    if task_dir:
        labels_dir = task_dir / "labels"
        default_output_dir = task_dir / "output"
        default_report_dir = task_dir / "Result"
    else:
        labels_dir = Path(".")
        default_output_dir = None
        default_report_dir = Path("Result")

    pred_files = collect_pred_files(args.pred, args.pred_dir, default_output_dir, args.pred_scope)
    truth_files = resolve_truth_files(labels_dir, args.truth, experiment_track=args.experiment_track)

    report_dir = Path(args.report_dir) if args.report_dir else default_report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    if len(truth_files) == 1:
        print(f"Ground truth: {truth_files[0]}")
    else:
        print(f"Ground truth candidates: {len(truth_files)} (auto-match enabled)")
    print(f"Prediction files: {len(pred_files)}")
    print(f"Prediction scope: {args.pred_scope}")
    print(f"Report dir: {report_dir}")

    all_metrics = []
    all_chunk_metrics = []
    all_guardrails = []
    all_urban_errors = []
    all_theme_metrics = []
    all_theme_confusion = []
    all_theme_family = []
    all_unknown_rates = []
    all_decision_source_metrics = []
    all_topic_distributions = []
    all_boundary_bucket_metrics = []
    all_unknown_conflict_metrics = []
    all_bootstrap_ci = []
    aligned_frames = {}
    truth_cache = {}
    truth_match_rows = []
    comparability_rows = []
    run_metadata_rows = []
    strict_modes = {"single_truth", "exact_stem_match", "contains_match"}

    baseline_manifest = None
    baseline_truth_file = None

    for pred_file in pred_files:
        truth_file, match_mode = resolve_truth_for_prediction(
            pred_file,
            truth_files,
            experiment_track=args.experiment_track,
        )
        if args.strict_truth_match and match_mode not in strict_modes:
            raise ValueError(
                f"Truth matching is not strict enough: pred={pred_file.name}, truth={truth_file.name}, mode={match_mode}. "
                f"Use --truth or disable --strict-truth-match."
            )
        if truth_file not in truth_cache:
            truth_cache[truth_file] = pd.read_excel(truth_file, engine="openpyxl")
        truth_df = truth_cache[truth_file]

        print(f"Evaluating: {pred_file.name}")
        print(f"Matched truth: {truth_file.name} ({match_mode})")

        (
            aligned_merged_df,
            metrics_df,
            chunk_metrics_df,
            guardrail_df,
            urban_error_df,
            theme_metrics_df,
            theme_confusion_df,
            theme_family_df,
            unknown_rate_df,
            decision_source_df,
            topic_distribution_df,
            boundary_bucket_df,
            unknown_conflict_df,
            bootstrap_ci_df,
            report_path,
        ) = evaluate_one_file(
            truth_df=truth_df,
            pred_file=pred_file,
            report_dir=report_dir,
            strict=args.strict,
            coverage_threshold=args.coverage_threshold,
            spatial_desc_threshold=args.spatial_desc_threshold,
            chunk_size=args.chunk_size,
        )
        all_metrics.append(metrics_df)
        all_chunk_metrics.append(chunk_metrics_df)
        all_guardrails.append(guardrail_df)
        all_urban_errors.append(urban_error_df)
        all_theme_metrics.append(theme_metrics_df)
        all_theme_confusion.append(theme_confusion_df)
        all_theme_family.append(theme_family_df)
        all_unknown_rates.append(unknown_rate_df)
        all_decision_source_metrics.append(decision_source_df)
        all_topic_distributions.append(topic_distribution_df)
        all_boundary_bucket_metrics.append(boundary_bucket_df)
        all_unknown_conflict_metrics.append(unknown_conflict_df)
        all_bootstrap_ci.append(bootstrap_ci_df)
        aligned_frames[pred_file.stem] = aligned_merged_df
        truth_match_rows.append(
            {
                "prediction_file": pred_file.name,
                "truth_file": truth_file.name,
                "match_mode": match_mode,
            }
        )

        manifest = load_prompt_manifest(pred_file)
        manifest_file = manifest_path_for_output(pred_file)
        signature = build_comparability_signature(manifest, truth_file)
        long_context_signature = build_long_context_group_signature(manifest, truth_file)
        if baseline_manifest is None:
            baseline_manifest = manifest
            baseline_truth_file = truth_file
            comparable = True
            mismatches = []
        else:
            mismatches = compare_manifests(
                baseline_manifest,
                manifest,
                baseline_truth_file,
                truth_file,
            )
            comparable = len(mismatches) == 0

        comparability_rows.append(
            {
                "prediction_file": pred_file.name,
                "truth_file": truth_file.name,
                "match_mode": match_mode,
                "manifest_found": manifest is not None,
                "manifest_path": str(manifest_file.resolve()),
                "comparability_signature": signature,
                "comparable_with_first": comparable,
                "comparability_issues": ";".join(mismatches),
            }
        )
        runtime = (manifest or {}).get("runtime") or {}
        experiment = (manifest or {}).get("experiment") or {}
        run_metadata_rows.append(
            {
                "prediction_file": pred_file.name,
                "prediction_stem": pred_file.stem,
                "truth_file": str(truth_file.resolve()),
                "truth_match_mode": match_mode,
                "manifest_found": manifest is not None,
                "manifest_path": str(manifest_file.resolve()),
                "entrypoint": runtime.get("entrypoint", ""),
                "runtime_python": runtime.get("python_version", ""),
                "task_mode": manifest.get("task_mode") if manifest else "",
                "experiment_track": experiment.get("experiment_track", args.experiment_track),
                "dataset_id": experiment.get("dataset_id", ""),
                "session_policy": experiment.get("session_policy", ""),
                "order_id": experiment.get("order_id", ""),
                "order_seed": experiment.get("order_seed", ""),
                "max_samples_per_window": experiment.get("max_samples_per_window", ""),
                "pred_scope": experiment.get("pred_scope", args.pred_scope),
                "urban_method": experiment.get("urban_method", ""),
                "hybrid_llm_assist_enabled": experiment.get("hybrid_llm_assist_enabled", ""),
                "comparability_signature": signature,
                "long_context_group_signature": long_context_signature,
            }
        )

        warned_rows = guardrail_df[guardrail_df["Warnings"].astype(str) != ""]
        for _, warned_row in warned_rows.iterrows():
            print(
                "[WARN] Guardrail triggered | "
                f"file={warned_row['File']} | chunk={warned_row['Chunk']} | warnings={warned_row['Warnings']}"
            )
        if not comparable:
            print(
                f"[WARN] Comparability mismatch | file={pred_file.name} | issues={';'.join(mismatches) or 'unknown'}"
            )
        if manifest is None:
            print(f"[WARN] Missing prompt manifest for {pred_file.name}")
        print(f"Saved report: {report_path.name}")

    comparability_df = pd.DataFrame(comparability_rows)
    non_comparable_df = comparability_df[comparability_df["comparable_with_first"] == False]
    if args.strict_comparable and not non_comparable_df.empty:
        raise ValueError(
            "Comparability check failed. Incompatible files: "
            + ", ".join(non_comparable_df["prediction_file"].tolist())
        )

    merged_metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    merged_chunk_metrics = pd.concat(all_chunk_metrics, ignore_index=True) if all_chunk_metrics else pd.DataFrame()
    merged_guardrails = pd.concat(all_guardrails, ignore_index=True) if all_guardrails else pd.DataFrame()
    merged_urban_errors = pd.concat(all_urban_errors, ignore_index=True) if all_urban_errors else pd.DataFrame()
    merged_theme_metrics = pd.concat(all_theme_metrics, ignore_index=True) if all_theme_metrics else pd.DataFrame()
    merged_theme_confusion = pd.concat(all_theme_confusion, ignore_index=True) if all_theme_confusion else pd.DataFrame()
    merged_theme_family = pd.concat(all_theme_family, ignore_index=True) if all_theme_family else pd.DataFrame()
    merged_unknown_rates = pd.concat(all_unknown_rates, ignore_index=True) if all_unknown_rates else pd.DataFrame()
    merged_decision_source_metrics = (
        pd.concat(all_decision_source_metrics, ignore_index=True) if all_decision_source_metrics else pd.DataFrame()
    )
    merged_topic_distributions = (
        pd.concat(all_topic_distributions, ignore_index=True) if all_topic_distributions else pd.DataFrame()
    )
    merged_boundary_bucket_metrics = (
        pd.concat(all_boundary_bucket_metrics, ignore_index=True) if all_boundary_bucket_metrics else pd.DataFrame()
    )
    merged_unknown_conflicts = (
        pd.concat(all_unknown_conflict_metrics, ignore_index=True) if all_unknown_conflict_metrics else pd.DataFrame()
    )
    merged_bootstrap_ci = (
        pd.concat(all_bootstrap_ci, ignore_index=True) if all_bootstrap_ci else pd.DataFrame()
    )
    run_metadata_df = pd.DataFrame(run_metadata_rows)

    all_comparable = comparability_df.empty or comparability_df["comparable_with_first"].all()
    summary_df = summarize_metrics(merged_metrics) if all_comparable else pd.DataFrame()
    if not merged_metrics.empty:
        validate_accuracy_bounds(merged_metrics, context="merged_metrics")
    if not summary_df.empty:
        validate_accuracy_bounds(summary_df, context="global_summary")
    group_summary_df = build_group_summaries(merged_metrics, comparability_df, run_metadata_df=run_metadata_df)
    if not group_summary_df.empty:
        validate_accuracy_bounds(group_summary_df, context="group_summary")
    protocol_df = build_protocol_df(args, pred_files, truth_files, report_dir)
    long_context_stability_df = build_long_context_stability(
        merged_metrics,
        merged_unknown_rates,
        run_metadata_df,
    )
    mcnemar_df = summarize_mcnemar(aligned_frames)

    if not merged_metrics.empty:
        summary_path = report_dir / "Eval_Summary.xlsx"
        with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
            merged_metrics.to_excel(writer, sheet_name="All Metrics", index=False)
            run_metadata_df.to_excel(writer, sheet_name="Run Metadata", index=False)
            protocol_df.to_excel(writer, sheet_name="Protocol", index=False)
            if not summary_df.empty:
                summary_df.to_excel(writer, sheet_name="Global Summary", index=False)
            if not group_summary_df.empty:
                group_summary_df.to_excel(writer, sheet_name="Group Summary", index=False)
            long_context_stability_df.to_excel(writer, sheet_name="Long Context Stability", index=False)
            merged_theme_metrics.to_excel(writer, sheet_name="Theme Metrics", index=False)
            merged_theme_confusion.to_excel(writer, sheet_name="Theme Confusion", index=False)
            merged_theme_family.to_excel(writer, sheet_name="U-N Family Metrics", index=False)
            merged_unknown_rates.to_excel(writer, sheet_name="Unknown Rate", index=False)
            merged_decision_source_metrics.to_excel(writer, sheet_name="Decision Source Metrics", index=False)
            merged_topic_distributions.to_excel(writer, sheet_name="Topic Distribution", index=False)
            merged_boundary_bucket_metrics.to_excel(writer, sheet_name="Boundary Bucket Metrics", index=False)
            merged_unknown_conflicts.to_excel(writer, sheet_name="Unknown Conflict Analysis", index=False)
            merged_bootstrap_ci.to_excel(writer, sheet_name="Bootstrap CI", index=False)
            mcnemar_df.to_excel(writer, sheet_name="McNemar", index=False)
            merged_chunk_metrics.to_excel(writer, sheet_name="Chunk Metrics", index=False)
            merged_guardrails.to_excel(writer, sheet_name="Guardrails", index=False)
            merged_urban_errors.to_excel(writer, sheet_name="Urban Error Analysis", index=False)
            pd.DataFrame(truth_match_rows).to_excel(writer, sheet_name="Truth Match", index=False)
            comparability_df.to_excel(writer, sheet_name="Comparability", index=False)
        print(f"Saved summary: {summary_path.name}")

    print("Evaluation complete.")


if __name__ == "__main__":
    evaluate()
