import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prompting.generator import PromptGenerator
from src.runtime.config import Config, Schema
from src.runtime.llm_client import DeepSeekClient
from src.strategies.stepwise_long import StepwiseLongContextStrategy
from src.urban.urban_hybrid_classifier import UrbanHybridClassifier
from src.urban.urban_metadata import UrbanMetadataRecord
from src.urban.urban_topic_classifier import UrbanTopicClassifier


THREAD_STATE = threading.local()


def _coerce_binary_prediction(value):
    if value is None or value is pd.NA:
        return pd.NA
    if isinstance(value, float) and pd.isna(value):
        return pd.NA
    if isinstance(value, str) and not value.strip():
        return pd.NA
    text = str(value).strip()
    if text in {"0", "1"}:
        return int(text)
    try:
        numeric = int(float(text))
    except (TypeError, ValueError):
        return pd.NA
    return numeric if numeric in {0, 1} else pd.NA


def _coerce_binary_series(series: pd.Series) -> pd.Series:
    return series.apply(_coerce_binary_prediction).astype("Int64")


def compute_binary_metrics(truth: pd.Series, pred: pd.Series) -> Dict[str, float]:
    truth_s = _coerce_binary_series(truth)
    pred_s = _coerce_binary_series(pred)
    decided_mask = truth_s.notna() & pred_s.notna()
    truth_eval = truth_s[decided_mask].astype(int)
    pred_eval = pred_s[decided_mask].astype(int)

    tp = int(((truth_eval == 1) & (pred_eval == 1)).sum())
    tn = int(((truth_eval == 0) & (pred_eval == 0)).sum())
    fp = int(((truth_eval == 0) & (pred_eval == 1)).sum())
    fn = int(((truth_eval == 1) & (pred_eval == 0)).sum())
    total = int(len(truth_s))
    decided = int(decided_mask.sum())
    unknown = total - decided
    accuracy = (tp + tn) / decided if decided else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    balanced_accuracy = (recall + specificity) / 2 if decided else 0.0
    return {
        "total": total,
        "decided": decided,
        "unknown": unknown,
        "coverage": round(decided / total, 6) if total else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "specificity": round(specificity, 6),
        "f1": round(f1, 6),
        "balanced_accuracy": round(balanced_accuracy, 6),
        "truth_positive_rate": round(float(truth_eval.mean()), 6) if decided else 0.0,
        "pred_positive_rate": round(float(pred_eval.mean()), 6) if decided else 0.0,
    }


def detect_truth_column(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested
    candidates = [col for col in df.columns if str(col).endswith("local_v2")]
    if candidates:
        return candidates[0]
    raise KeyError(f"Truth column not found: {requested}")


def get_thread_strategy(shot_mode: str) -> StepwiseLongContextStrategy:
    strategy = getattr(THREAD_STATE, "strategy", None)
    if strategy is None:
        client = DeepSeekClient()
        prompt_gen = PromptGenerator(shot_mode=shot_mode, default_theme="urban_renewal")
        strategy = StepwiseLongContextStrategy(client, prompt_gen)
        THREAD_STATE.strategy = strategy
    return strategy


def get_thread_hybrid_classifier(shot_mode: str) -> UrbanHybridClassifier:
    classifier = getattr(THREAD_STATE, "hybrid_classifier", None)
    if classifier is None:
        classifier = UrbanHybridClassifier(get_thread_strategy(shot_mode))
        THREAD_STATE.hybrid_classifier = classifier
    return classifier


def run_single_llm_prediction(
    *,
    row_id: int,
    row: pd.Series,
    shot_mode: str,
    session_root: Path,
) -> Dict[str, object]:
    strategy = get_thread_strategy(shot_mode)
    title = str(row.get(Schema.TITLE, "") or "")
    abstract = str(row.get(Schema.ABSTRACT, "") or "")
    metadata = {
        Schema.AUTHOR_KEYWORDS: row.get(Schema.AUTHOR_KEYWORDS, ""),
        Schema.KEYWORDS_PLUS: row.get(Schema.KEYWORDS_PLUS, ""),
        Schema.KEYWORDS: row.get(Schema.KEYWORDS, ""),
        Schema.WOS_CATEGORIES: row.get(Schema.WOS_CATEGORIES, ""),
        Schema.RESEARCH_AREAS: row.get(Schema.RESEARCH_AREAS, ""),
    }
    session_path = session_root / f"pure_llm_row_{row_id:04d}.json"
    started = time.perf_counter()
    result = strategy.process(
        title,
        abstract,
        session_path=session_path,
        metadata=metadata,
        auxiliary_context=None,
    )
    elapsed = time.perf_counter() - started
    prediction = _coerce_binary_prediction(result.get(Schema.IS_URBAN_RENEWAL, "0"))
    return {
        "_row_id": row_id,
        Schema.TITLE: title,
        "LLM_Prediction": prediction,
        "LLM_Parse_Reason": str(result.get("urban_parse_reason", "")),
        "LLM_Runtime_Sec": round(elapsed, 4),
    }


def run_single_hybrid_prediction(
    *,
    row_id: int,
    row: pd.Series,
    shot_mode: str,
    session_root: Path,
) -> Dict[str, object]:
    classifier = get_thread_hybrid_classifier(shot_mode)
    title = str(row.get(Schema.TITLE, "") or "")
    abstract = str(row.get(Schema.ABSTRACT, "") or "")
    metadata = {
        Schema.AUTHOR_KEYWORDS: row.get(Schema.AUTHOR_KEYWORDS, ""),
        Schema.KEYWORDS_PLUS: row.get(Schema.KEYWORDS_PLUS, ""),
        Schema.KEYWORDS: row.get(Schema.KEYWORDS, ""),
        Schema.WOS_CATEGORIES: row.get(Schema.WOS_CATEGORIES, ""),
        Schema.RESEARCH_AREAS: row.get(Schema.RESEARCH_AREAS, ""),
    }
    session_path = session_root / f"hybrid_row_{row_id:04d}.json"
    started = time.perf_counter()
    result = classifier.classify(
        title,
        abstract,
        metadata=metadata,
        session_path=session_path,
    )
    elapsed = time.perf_counter() - started
    prediction = _coerce_binary_prediction(
        result.get("final_label", result.get(Schema.IS_URBAN_RENEWAL, "0"))
    )
    return {
        "_row_id": row_id,
        Schema.TITLE: title,
        "Hybrid_Prediction": prediction,
        "Hybrid_Runtime_Sec": round(elapsed, 4),
        "Hybrid_Decision_Source": str(result.get("decision_source", "")),
        "Hybrid_Decision_Reason": str(result.get("decision_reason", "")),
        "Hybrid_Metadata_Route": str(result.get("metadata_route", "")),
        "Hybrid_Metadata_Candidate_Buckets": str(result.get("metadata_candidate_topic_buckets", "")),
        "Hybrid_Topic_Label": str(result.get("topic_label", "")),
        "Hybrid_Topic_Group": str(result.get("topic_group", "")),
        "Hybrid_Topic_Name": str(result.get("topic_name", "")),
        "Hybrid_Topic_Confidence": result.get("topic_confidence"),
        "Hybrid_Topic_Margin": result.get("topic_margin"),
        "Hybrid_Topic_Confidence_Effective": result.get("topic_confidence_effective"),
        "Hybrid_Topic_Margin_Effective": result.get("topic_margin_effective"),
        "Hybrid_Topic_Binary_Probability": result.get("topic_binary_probability"),
        "Hybrid_BERTopic_Status": str(result.get("bertopic_status", "")),
        "Hybrid_BERTopic_Topic_ID": result.get("bertopic_topic_id"),
        "Hybrid_BERTopic_Topic_Name": str(result.get("bertopic_topic_name", "")),
        "Hybrid_BERTopic_Probability": result.get("bertopic_probability"),
        "Hybrid_BERTopic_Is_Outlier": result.get("bertopic_is_outlier"),
        "Hybrid_BERTopic_Count": result.get("bertopic_count"),
        "Hybrid_BERTopic_Pos_Rate": result.get("bertopic_pos_rate"),
        "Hybrid_BERTopic_Mapped_Label": str(result.get("bertopic_mapped_label", "")),
        "Hybrid_BERTopic_Mapped_Group": str(result.get("bertopic_mapped_group", "")),
        "Hybrid_BERTopic_Label_Purity": result.get("bertopic_label_purity"),
        "Hybrid_BERTopic_Mapped_Share": result.get("bertopic_mapped_label_share"),
        "Hybrid_BERTopic_High_Purity": result.get("bertopic_high_purity"),
        "Hybrid_BERTopic_True_Outlier": result.get("bertopic_true_outlier"),
        "Hybrid_BERTopic_Prior_Mode": str(result.get("bertopic_prior_mode", "")),
        "Hybrid_BERTopic_Confidence_Delta": result.get("bertopic_confidence_delta"),
        "Hybrid_BERTopic_Margin_Delta": result.get("bertopic_margin_delta"),
        "Hybrid_BERTopic_Primary_Label": str(result.get("bertopic_primary_label", "")),
        "Hybrid_BERTopic_Primary_Group": str(result.get("bertopic_primary_group", "")),
        "Hybrid_BERTopic_Primary_Name": str(result.get("bertopic_primary_name", "")),
        "Hybrid_BERTopic_Primary_Probability": result.get("bertopic_primary_probability"),
        "Hybrid_BERTopic_Primary_Support": result.get("bertopic_primary_support"),
        "Hybrid_BERTopic_Primary_Purity": result.get("bertopic_primary_purity"),
        "Hybrid_BERTopic_Primary_Mapped_Share": result.get("bertopic_primary_mapped_share"),
        "Hybrid_BERTopic_Primary_Override": result.get("bertopic_primary_override"),
        "Hybrid_BERTopic_Primary_Reason": str(result.get("bertopic_primary_reason", "")),
        "Hybrid_LLM_Attempted": int(result.get("llm_attempted", result.get("llm_used", 0)) or 0),
        "Hybrid_LLM_Used": int(result.get("llm_used", 0) or 0),
        "Hybrid_LLM_Failure_Reason": str(result.get("llm_failure_reason", "")),
    }


def run_parallel_predictions(
    df: pd.DataFrame,
    *,
    runner,
    shot_mode: str,
    max_workers: int,
    session_root: Path,
    progress_label: str,
) -> pd.DataFrame:
    session_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                runner,
                row_id=int(row["_row_id"]),
                row=row,
                shot_mode=shot_mode,
                session_root=session_root,
            ): int(row["_row_id"])
            for _, row in df.iterrows()
        }
        for idx, future in enumerate(as_completed(future_map), start=1):
            row_id = future_map[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                title = df.loc[df["_row_id"] == row_id, Schema.TITLE].iloc[0]
                fallback = {
                    "_row_id": row_id,
                    Schema.TITLE: title,
                    f"{progress_label}_Prediction": pd.NA,
                    f"{progress_label}_Runtime_Sec": None,
                }
                if progress_label == "LLM":
                    fallback["LLM_Parse_Reason"] = f"error:{type(exc).__name__}"
                else:
                    fallback["Hybrid_Decision_Source"] = "error"
                    fallback["Hybrid_Decision_Reason"] = f"error:{type(exc).__name__}"
                    fallback["Hybrid_Metadata_Route"] = ""
                    fallback["Hybrid_Metadata_Candidate_Buckets"] = ""
                    fallback["Hybrid_Topic_Label"] = ""
                    fallback["Hybrid_Topic_Group"] = ""
                    fallback["Hybrid_Topic_Name"] = ""
                    fallback["Hybrid_Topic_Confidence"] = None
                    fallback["Hybrid_Topic_Margin"] = None
                    fallback["Hybrid_Topic_Confidence_Effective"] = None
                    fallback["Hybrid_Topic_Margin_Effective"] = None
                    fallback["Hybrid_Topic_Binary_Probability"] = None
                    fallback["Hybrid_BERTopic_Status"] = ""
                    fallback["Hybrid_BERTopic_Topic_ID"] = None
                    fallback["Hybrid_BERTopic_Topic_Name"] = ""
                    fallback["Hybrid_BERTopic_Probability"] = None
                    fallback["Hybrid_BERTopic_Is_Outlier"] = None
                    fallback["Hybrid_BERTopic_Count"] = None
                    fallback["Hybrid_BERTopic_Pos_Rate"] = None
                    fallback["Hybrid_BERTopic_Mapped_Label"] = ""
                    fallback["Hybrid_BERTopic_Mapped_Group"] = ""
                    fallback["Hybrid_BERTopic_Label_Purity"] = None
                    fallback["Hybrid_BERTopic_Mapped_Share"] = None
                    fallback["Hybrid_BERTopic_High_Purity"] = None
                    fallback["Hybrid_BERTopic_True_Outlier"] = None
                    fallback["Hybrid_BERTopic_Prior_Mode"] = ""
                    fallback["Hybrid_BERTopic_Confidence_Delta"] = None
                    fallback["Hybrid_BERTopic_Margin_Delta"] = None
                    fallback["Hybrid_BERTopic_Primary_Label"] = ""
                    fallback["Hybrid_BERTopic_Primary_Group"] = ""
                    fallback["Hybrid_BERTopic_Primary_Name"] = ""
                    fallback["Hybrid_BERTopic_Primary_Probability"] = None
                    fallback["Hybrid_BERTopic_Primary_Support"] = None
                    fallback["Hybrid_BERTopic_Primary_Purity"] = None
                    fallback["Hybrid_BERTopic_Primary_Mapped_Share"] = None
                    fallback["Hybrid_BERTopic_Primary_Override"] = None
                    fallback["Hybrid_BERTopic_Primary_Reason"] = ""
                    fallback["Hybrid_LLM_Attempted"] = 0
                    fallback["Hybrid_LLM_Used"] = 0
                    fallback["Hybrid_LLM_Failure_Reason"] = ""
                rows.append(fallback)

            if idx % 50 == 0 or idx == len(future_map):
                print(f"[{progress_label}] completed {idx}/{len(future_map)}")

    elapsed = time.perf_counter() - started
    print(f"[{progress_label}] total runtime: {elapsed:.2f}s")
    return pd.DataFrame(rows).sort_values("_row_id").reset_index(drop=True)


def run_classifier_predictions(df: pd.DataFrame) -> pd.DataFrame:
    classifier = UrbanTopicClassifier()
    rows: List[Dict[str, object]] = []
    started = time.perf_counter()
    for _, row in df.iterrows():
        record = UrbanMetadataRecord.from_row(row.to_dict())
        pred = classifier.predict(record)
        rows.append(
            {
                "_row_id": int(row["_row_id"]),
                Schema.TITLE: row[Schema.TITLE],
                "Classifier_Prediction": 1 if pred.topic_group == "urban" else 0,
                "Classifier_Topic_Label": pred.topic_label,
                "Classifier_Topic_Group": pred.topic_group,
                "Classifier_Topic_Name": pred.topic_name,
                "Classifier_Confidence": pred.confidence,
                "Classifier_Margin": pred.margin,
                "Classifier_Binary_Probability": pred.binary_probability,
                "Classifier_Matches": "; ".join(pred.matched_terms),
            }
        )
    elapsed = time.perf_counter() - started
    print(f"[CLASSIFIER] total runtime: {elapsed:.2f}s")
    return pd.DataFrame(rows).sort_values("_row_id").reset_index(drop=True)


def load_reused_llm_predictions(
    *,
    df: pd.DataFrame,
    pure_llm_result_path: Path,
) -> pd.DataFrame:
    reused = pd.read_excel(pure_llm_result_path, engine="openpyxl").copy()
    if "LLM_Prediction" not in reused.columns:
        raise KeyError(f"LLM_Prediction column not found in {pure_llm_result_path}")

    reused = reused.reset_index(drop=True)
    reused["_row_id"] = range(1, len(reused) + 1)
    base = df[["_row_id", Schema.TITLE, Schema.ABSTRACT]].copy()

    if len(reused) == len(base):
        title_matches = (
            reused.get(Schema.TITLE, pd.Series([""] * len(reused))).fillna("").astype(str).tolist()
            == base[Schema.TITLE].fillna("").astype(str).tolist()
        )
        if title_matches:
            keep_cols = ["_row_id", Schema.TITLE, "LLM_Prediction"]
            if "LLM_Parse_Reason" in reused.columns:
                keep_cols.append("LLM_Parse_Reason")
            if "LLM_Runtime_Sec" in reused.columns:
                keep_cols.append("LLM_Runtime_Sec")
            return reused[keep_cols].copy()

    left = base.copy()
    right = reused.copy()
    left["_merge_key"] = (
        left[Schema.TITLE].fillna("").astype(str).str.strip().str.lower()
        + " || "
        + left[Schema.ABSTRACT].fillna("").astype(str).str.strip().str.lower()
    )
    right["_merge_key"] = (
        right[Schema.TITLE].fillna("").astype(str).str.strip().str.lower()
        + " || "
        + right[Schema.ABSTRACT].fillna("").astype(str).str.strip().str.lower()
    )
    keep_cols = ["_merge_key", "LLM_Prediction"]
    if "LLM_Parse_Reason" in right.columns:
        keep_cols.append("LLM_Parse_Reason")
    if "LLM_Runtime_Sec" in right.columns:
        keep_cols.append("LLM_Runtime_Sec")
    merged = left.merge(right[keep_cols], on="_merge_key", how="left")
    if merged["LLM_Prediction"].isna().any():
        missing = int(merged["LLM_Prediction"].isna().sum())
        raise RuntimeError(
            f"Failed to align {missing} rows from reused pure LLM result: {pure_llm_result_path}"
        )
    output_cols = ["_row_id", Schema.TITLE, "LLM_Prediction"]
    if "LLM_Parse_Reason" in merged.columns:
        output_cols.append("LLM_Parse_Reason")
    if "LLM_Runtime_Sec" in merged.columns:
        output_cols.append("LLM_Runtime_Sec")
    return merged[output_cols].copy()


def build_metrics_summary(
    *,
    truth: pd.Series,
    llm_pred: pd.Series,
    classifier_pred: pd.Series,
    hybrid_pred: pd.Series,
    llm_runtime_sec: float,
    classifier_runtime_sec: float,
    hybrid_runtime_sec: float,
    llm_runtime_samples: pd.Series,
    hybrid_runtime_samples: pd.Series,
    hybrid_llm_attempted: pd.Series,
) -> pd.DataFrame:
    rows = []
    model_inputs = [
        (
            "pure_llm_api",
            llm_pred,
            llm_runtime_sec,
            float(llm_runtime_samples.dropna().mean()) if len(llm_runtime_samples.dropna()) else None,
            int(len(truth)),
            1.0,
        ),
        (
            "local_topic_classifier",
            classifier_pred,
            classifier_runtime_sec,
            classifier_runtime_sec / len(truth) if len(truth) else None,
            0,
            0.0,
        ),
        (
            "three_stage_hybrid",
            hybrid_pred,
            hybrid_runtime_sec,
            float(hybrid_runtime_samples.dropna().mean()) if len(hybrid_runtime_samples.dropna()) else None,
            int(hybrid_llm_attempted.sum()),
            round(float(hybrid_llm_attempted.mean()), 6) if len(hybrid_llm_attempted) else 0.0,
        ),
    ]
    for model_name, pred, runtime_total, runtime_avg, llm_calls, llm_call_rate in model_inputs:
        row = {"model": model_name}
        row.update(compute_binary_metrics(truth, pred))
        row["runtime_total_sec"] = round(runtime_total, 4)
        row["runtime_avg_sec_per_item"] = round(runtime_avg, 4) if runtime_avg is not None else None
        row["llm_calls"] = llm_calls
        row["llm_call_rate"] = llm_call_rate
        rows.append(row)
    return pd.DataFrame(rows)


def build_prediction_comparison(df: pd.DataFrame, truth_col: str) -> pd.DataFrame:
    comparison = df.copy()
    truth_s = _coerce_binary_series(comparison[truth_col])
    for prefix, pred_col in [
        ("LLM", "LLM_Prediction"),
        ("Classifier", "Classifier_Prediction"),
        ("Hybrid", "Hybrid_Prediction"),
    ]:
        pred_s = _coerce_binary_series(comparison[pred_col])
        comparison[f"{prefix}_Correct"] = ((truth_s == pred_s) & pred_s.notna()).astype(int)
    return comparison


def build_hybrid_error_breakdown(comparison: pd.DataFrame) -> pd.DataFrame:
    hybrid_correct = comparison["Hybrid_Correct"] == 1
    classifier_wrong = comparison["Classifier_Correct"] == 0
    llm_used = comparison["Hybrid_LLM_Used"] == 1
    llm_attempted = comparison.get("Hybrid_LLM_Attempted", comparison["Hybrid_LLM_Used"]) == 1
    source = comparison["Hybrid_Decision_Source"].fillna("")
    return pd.DataFrame(
        [
            {"metric": "stage1_reject_error", "value": int(((source == "stage1_rule") & (~hybrid_correct)).sum())},
            {"metric": "stage2_only_error", "value": int(((source == "stage2_classifier") & (~hybrid_correct)).sum())},
            {"metric": "llm_corrected_cases", "value": int((llm_used & hybrid_correct & classifier_wrong).sum())},
            {"metric": "residual_hard_cases", "value": int((llm_attempted & (~hybrid_correct)).sum())},
            {"metric": "hybrid_llm_calls", "value": int(llm_attempted.sum())},
            {"metric": "hybrid_llm_call_rate", "value": round(float(llm_attempted.mean()), 6) if len(comparison) else 0.0},
        ]
    )


def build_hybrid_decision_sources(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(comparison)
    llm_attempted = comparison.get("Hybrid_LLM_Attempted", comparison["Hybrid_LLM_Used"])
    for decision_source, group in comparison.groupby("Hybrid_Decision_Source", dropna=False):
        rows.append(
            {
                "decision_source": decision_source if decision_source == decision_source else "",
                "count": int(len(group)),
                "share": round(len(group) / total, 6) if total else 0.0,
                "accuracy": round(float(group["Hybrid_Correct"].mean()), 6) if len(group) else 0.0,
                "llm_attempted": int(llm_attempted.loc[group.index].sum()),
                "llm_used": int(group["Hybrid_LLM_Used"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["count", "decision_source"], ascending=[False, True])


def build_model_disagreements(comparison: pd.DataFrame) -> pd.DataFrame:
    llm_pred = _coerce_binary_series(comparison["LLM_Prediction"]).fillna(-1)
    classifier_pred = _coerce_binary_series(comparison["Classifier_Prediction"]).fillna(-1)
    hybrid_pred = _coerce_binary_series(comparison["Hybrid_Prediction"]).fillna(-1)
    return comparison[
        (
            llm_pred != classifier_pred
        )
        | (
            llm_pred != hybrid_pred
        )
        | (
            classifier_pred != hybrid_pred
        )
    ].copy()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark pure LLM API, local topic classifier, and three-stage hybrid against the labeled standard."
    )
    parser.add_argument(
        "--input",
        default=r"C:\Users\26409\Desktop\Urban Renovation\output\spreadsheet\Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407.xlsx",
        help="Standard labeled workbook path",
    )
    parser.add_argument(
        "--truth-column",
        default="是否属于城市更新研究_local_v2",
        help="Truth label column in the standard workbook",
    )
    parser.add_argument(
        "--shot-mode",
        default="few",
        help="Urban renewal prompt strategy for pure API / hybrid benchmark",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Concurrent worker count for LLM-backed paths",
    )
    parser.add_argument(
        "--tag",
        default="20260407",
        help="Output file tag",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for smoke tests",
    )
    parser.add_argument(
        "--outdir",
        default=r"C:\Users\26409\Desktop\Urban Renovation\output\spreadsheet",
        help="Output directory",
    )
    parser.add_argument(
        "--reuse-pure-llm",
        default=None,
        help="Reuse an existing pure LLM workbook instead of rerunning pure LLM API",
    )
    args = parser.parse_args()

    Config.load_env()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    session_root = Path(r"C:\Users\26409\Desktop\Urban Renovation\tmp\benchmark_sessions") / args.tag

    df = pd.read_excel(input_path, engine="openpyxl")
    truth_col = detect_truth_column(df, args.truth_column)
    if args.limit:
        df = df.head(args.limit).copy()
    df = df.copy()
    df["_row_id"] = range(1, len(df) + 1)

    llm_input_columns = [
        "_row_id",
        Schema.TITLE,
        Schema.ABSTRACT,
        Schema.AUTHOR_KEYWORDS,
        Schema.KEYWORDS_PLUS,
        Schema.KEYWORDS,
        Schema.WOS_CATEGORIES,
        Schema.RESEARCH_AREAS,
    ]
    for column in llm_input_columns:
        if column not in df.columns:
            df[column] = ""

    reused_pure_llm_path = Path(args.reuse_pure_llm) if args.reuse_pure_llm else None
    if reused_pure_llm_path:
        llm_started = time.perf_counter()
        llm_df = load_reused_llm_predictions(
            df=df[llm_input_columns],
            pure_llm_result_path=reused_pure_llm_path,
        )
        if "LLM_Runtime_Sec" in llm_df.columns and len(llm_df["LLM_Runtime_Sec"].dropna()):
            llm_runtime_total = float(llm_df["LLM_Runtime_Sec"].dropna().sum())
        else:
            llm_runtime_total = time.perf_counter() - llm_started
        print(f"[LLM] reused existing predictions from: {reused_pure_llm_path}")
    else:
        llm_started = time.perf_counter()
        llm_df = run_parallel_predictions(
            df[llm_input_columns],
            runner=run_single_llm_prediction,
            shot_mode=args.shot_mode,
            max_workers=max(1, args.max_workers),
            session_root=session_root / "pure_llm",
            progress_label="LLM",
        )
        llm_runtime_total = time.perf_counter() - llm_started

    clf_started = time.perf_counter()
    classifier_df = run_classifier_predictions(df[llm_input_columns])
    classifier_runtime_total = time.perf_counter() - clf_started

    hybrid_started = time.perf_counter()
    hybrid_df = run_parallel_predictions(
        df[llm_input_columns],
        runner=run_single_hybrid_prediction,
        shot_mode=args.shot_mode,
        max_workers=max(1, args.max_workers),
        session_root=session_root / "hybrid",
        progress_label="HYBRID",
    )
    hybrid_runtime_total = time.perf_counter() - hybrid_started

    base_columns = [col for col in df.columns if col != "_row_id"]
    merged = (
        df[["_row_id"] + base_columns]
        .merge(llm_df, on=["_row_id", Schema.TITLE], how="left")
        .merge(classifier_df, on=["_row_id", Schema.TITLE], how="left")
        .merge(hybrid_df, on=["_row_id", Schema.TITLE], how="left")
    )

    metrics_summary = build_metrics_summary(
        truth=merged[truth_col],
        llm_pred=merged["LLM_Prediction"],
        classifier_pred=merged["Classifier_Prediction"],
        hybrid_pred=merged["Hybrid_Prediction"],
        llm_runtime_sec=llm_runtime_total,
        classifier_runtime_sec=classifier_runtime_total,
        hybrid_runtime_sec=hybrid_runtime_total,
        llm_runtime_samples=merged["LLM_Runtime_Sec"],
        hybrid_runtime_samples=merged["Hybrid_Runtime_Sec"],
        hybrid_llm_attempted=merged["Hybrid_LLM_Attempted"],
    )
    comparison = build_prediction_comparison(merged, truth_col)
    hybrid_error_breakdown = build_hybrid_error_breakdown(comparison)
    hybrid_decision_sources = build_hybrid_decision_sources(comparison)
    model_disagreements = build_model_disagreements(comparison)

    llm_errors = comparison[comparison["LLM_Correct"] == 0].copy()
    classifier_errors = comparison[comparison["Classifier_Correct"] == 0].copy()
    hybrid_errors = comparison[comparison["Hybrid_Correct"] == 0].copy()

    llm_output_path = outdir / f"urban_sample1000_pure_llm_api_result_{args.tag}.xlsx"
    classifier_output_path = outdir / f"urban_sample1000_classifier_result_{args.tag}.xlsx"
    hybrid_output_path = outdir / f"urban_sample1000_three_stage_hybrid_result_{args.tag}.xlsx"
    report_output_path = outdir / f"urban_sample1000_llm_classifier_hybrid_evaluation_{args.tag}.xlsx"

    llm_export_cols = [
        Schema.TITLE,
        Schema.ABSTRACT,
        truth_col,
        "LLM_Prediction",
        "LLM_Parse_Reason",
        "LLM_Runtime_Sec",
    ]
    classifier_export_cols = [
        Schema.TITLE,
        Schema.ABSTRACT,
        truth_col,
        "Classifier_Prediction",
        "Classifier_Topic_Label",
        "Classifier_Topic_Group",
        "Classifier_Topic_Name",
        "Classifier_Confidence",
        "Classifier_Margin",
        "Classifier_Binary_Probability",
        "Classifier_Matches",
    ]
    hybrid_export_cols = [
        Schema.TITLE,
        Schema.ABSTRACT,
        truth_col,
        "Hybrid_Prediction",
        "Hybrid_Decision_Source",
        "Hybrid_Decision_Reason",
        "Hybrid_Metadata_Route",
        "Hybrid_Metadata_Candidate_Buckets",
        "Hybrid_Topic_Label",
        "Hybrid_Topic_Group",
        "Hybrid_Topic_Name",
        "Hybrid_Topic_Confidence",
        "Hybrid_Topic_Margin",
        "Hybrid_Topic_Confidence_Effective",
        "Hybrid_Topic_Margin_Effective",
        "Hybrid_Topic_Binary_Probability",
        "Hybrid_BERTopic_Status",
        "Hybrid_BERTopic_Topic_ID",
        "Hybrid_BERTopic_Topic_Name",
        "Hybrid_BERTopic_Probability",
        "Hybrid_BERTopic_Is_Outlier",
        "Hybrid_BERTopic_Count",
        "Hybrid_BERTopic_Pos_Rate",
        "Hybrid_BERTopic_Mapped_Label",
        "Hybrid_BERTopic_Mapped_Group",
        "Hybrid_BERTopic_Label_Purity",
        "Hybrid_BERTopic_High_Purity",
        "Hybrid_BERTopic_True_Outlier",
        "Hybrid_BERTopic_Prior_Mode",
        "Hybrid_BERTopic_Confidence_Delta",
        "Hybrid_BERTopic_Margin_Delta",
        "Hybrid_LLM_Attempted",
        "Hybrid_LLM_Used",
        "Hybrid_LLM_Failure_Reason",
        "Hybrid_Runtime_Sec",
    ]

    comparison[llm_export_cols].to_excel(llm_output_path, index=False, engine="openpyxl")
    comparison[classifier_export_cols].to_excel(classifier_output_path, index=False, engine="openpyxl")
    comparison[hybrid_export_cols].to_excel(hybrid_output_path, index=False, engine="openpyxl")

    with pd.ExcelWriter(report_output_path, engine="openpyxl") as writer:
        metrics_summary.to_excel(writer, sheet_name="Metrics_Summary", index=False)
        hybrid_error_breakdown.to_excel(writer, sheet_name="Hybrid_Error_Breakdown", index=False)
        hybrid_decision_sources.to_excel(writer, sheet_name="Hybrid_Decision_Sources", index=False)
        comparison.to_excel(writer, sheet_name="Prediction_Comparison", index=False)
        llm_errors.to_excel(writer, sheet_name="PureLLM_Errors", index=False)
        classifier_errors.to_excel(writer, sheet_name="Classifier_Errors", index=False)
        hybrid_errors.to_excel(writer, sheet_name="Hybrid_Errors", index=False)
        model_disagreements.to_excel(writer, sheet_name="Model_Disagreements", index=False)
        pd.DataFrame(
            [
                {
                    "input": str(input_path),
                    "truth_column": truth_col,
                    "shot_mode": args.shot_mode,
                    "reuse_pure_llm": str(reused_pure_llm_path) if reused_pure_llm_path else "",
                    "max_workers": args.max_workers,
                    "tag": args.tag,
                }
            ]
        ).to_excel(writer, sheet_name="Run_Metadata", index=False)

    print(f"Saved: {llm_output_path}")
    print(f"Saved: {classifier_output_path}")
    print(f"Saved: {hybrid_output_path}")
    print(f"Saved: {report_output_path}")


if __name__ == "__main__":
    main()
