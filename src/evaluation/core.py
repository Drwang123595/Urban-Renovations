import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..runtime.config import Schema
from ..urban.urban_topic_taxonomy import (
    UNKNOWN_TOPIC_GROUP,
    UNKNOWN_TOPIC_LABEL,
    topic_group_for_label,
    topic_name_for_label,
)


SPATIAL_LEVEL_KEYWORDS = {
    "1": ["global", "world", "worldwide", "world-wide"],
    "2": ["multi-country", "cross-border", "continental", "eu ", "asean", "european union"],
    "3": ["country", "nation", "national"],
    "4": ["multi-provincial", "multi-state", "region", "regional"],
    "5": ["provincial", "state", "province"],
    "6": ["multi-city", "megaregion", "urban agglomeration", "bay area", "greater bay"],
    "7": ["single city", "municipal", "city-wide"],
    "8": ["district", "county", "neighborhood"],
    "9": ["micro", "neighborhood", "block", "street", "grid", "community"],
}

SPATIAL_LEVEL_MAP = {
    "全球": 1,
    "洲际": 2,
    "跨国": 2,
    "多国": 2,
    "多国/多地区": 2,
    "全国": 3,
    "地区": 3,
    "国家": 3,
    "全国/地区": 3,
    "多省": 4,
    "多州": 4,
    "省/州": 5,
    "全省/全州": 4,
    "多省/州": 4,
    "省": 5,
    "州": 5,
    "多城市": 6,
    "城市群": 6,
    "单城市": 7,
    "全城市": 7,
    "市": 7,
    "区": 8,
    "县": 8,
    "街道": 8,
    "社区": 8,
    "单街区": 8,
    "多街区": 8,
    "微观": 9,
    "邻里": 9,
}

FIELD_SPECS = [
    (Schema.IS_URBAN_RENEWAL, "Urban Renewal", True),
    (Schema.IS_SPATIAL, "Spatial Study", True),
    (Schema.SPATIAL_LEVEL, "Spatial Level", False),
    (Schema.SPATIAL_DESC, "Spatial Desc", False),
]

METRIC_OUTPUT_COLUMNS = [
    "File",
    "Metric",
    "Accuracy",
    "Correct",
    "Total",
    "TP",
    "TN",
    "FP",
    "FN",
    "Precision",
    "Recall",
    "F1",
]

CHUNK_METRIC_OUTPUT_COLUMNS = [
    "File",
    "Metric",
    "Chunk",
    "Chunk Start",
    "Chunk End",
    "Accuracy",
    "Correct",
    "Total",
    "TP",
    "TN",
    "FP",
    "FN",
    "Precision",
    "Recall",
    "F1",
    "Truth Positive Rate",
    "Predicted Positive Rate",
]

THEME_METRIC_OUTPUT_COLUMNS = [
    "File",
    "Theme",
    "Theme Name",
    "Theme Group",
    "Accuracy",
    "Correct",
    "Total",
    "Truth Support",
    "Pred Support",
    "TP",
    "FP",
    "FN",
    "Precision",
    "Recall",
    "F1",
]

THEME_CONFUSION_OUTPUT_COLUMNS = [
    "File",
    "Truth Theme",
    "Truth Theme Name",
    "Pred Theme",
    "Pred Theme Name",
    "Count",
]

THEME_FAMILY_OUTPUT_COLUMNS = [
    "File",
    "Metric",
    "Accuracy",
    "Correct",
    "Total",
    "TP",
    "TN",
    "FP",
    "FN",
    "Precision",
    "Recall",
    "F1",
]

UNKNOWN_RATE_OUTPUT_COLUMNS = [
    "File",
    "Total Samples",
    "Theme Truth Samples",
    "Predicted Unknown Count",
    "Predicted Unknown Rate",
    "Truth Unknown Count",
    "Truth Unknown Rate",
]

DECISION_SOURCE_OUTPUT_COLUMNS = [
    "File",
    "Decision Source",
    "Total",
    "Accuracy",
    "TP",
    "TN",
    "FP",
    "FN",
    "Precision",
    "Recall",
    "F1",
    "Predicted Positive Rate",
    "Unknown Topic Rate",
]

TOPIC_DISTRIBUTION_OUTPUT_COLUMNS = [
    "File",
    "Topic Final",
    "Topic Name",
    "Topic Group",
    "Count",
    "Share",
    "Truth Positive Rate",
    "FP",
    "FN",
]

BOUNDARY_BUCKET_OUTPUT_COLUMNS = [
    "File",
    "Boundary Bucket",
    "Family Conflict Pattern",
    "Total",
    "Accuracy",
    "TP",
    "TN",
    "FP",
    "FN",
    "Unknown Count",
]

UNKNOWN_CONFLICT_OUTPUT_COLUMNS = [
    "File",
    "Review Reason",
    "Boundary Bucket",
    "Family Conflict Pattern",
    "Count",
    "Rule Family",
    "Local Family",
    "LLM Hint",
]

BOOTSTRAP_CI_OUTPUT_COLUMNS = [
    "File",
    "Metric",
    "Point Estimate",
    "CI Lower",
    "CI Upper",
    "Bootstrap Samples",
]

MCNEMAR_OUTPUT_COLUMNS = [
    "File A",
    "File B",
    "Metric",
    "B",
    "C",
    "Statistic",
    "P Value",
]

EXPLAINABILITY_QUALITY_OUTPUT_COLUMNS = [
    "File",
    "Total",
    "Decision Explanation Coverage",
    "Rule Stack Coverage",
    "Binary Evidence Coverage",
    "Positive Evidence Coverage",
    "Negative Evidence Coverage",
    "Evidence Balance Coverage",
    "Review Trigger Count",
    "Review Trigger Rate",
    "Near Threshold Count",
    "Conflict Count",
]

EVIDENCE_BALANCE_OUTPUT_COLUMNS = [
    "File",
    "Evidence Balance",
    "Total",
    "Accuracy",
    "TP",
    "TN",
    "FP",
    "FN",
    "Precision",
    "Recall",
    "F1",
    "Predicted Positive Rate",
    "Review Trigger Rate",
]


def validate_accuracy_bounds(metrics_df: pd.DataFrame, *, context: str):
    if metrics_df.empty or "Accuracy" not in metrics_df.columns:
        return

    values = pd.to_numeric(metrics_df["Accuracy"], errors="coerce")
    invalid = metrics_df[values.notna() & ((values < 0) | (values > 100))]
    if invalid.empty:
        return

    preview = invalid[["Metric", "Accuracy"]].head(5).to_dict("records")
    raise ValueError(f"{context} contains invalid Accuracy outside 0-100: {preview}")

THEME_TRUTH_ALIASES = [
    "theme_gold",
    "Theme Gold",
    "theme_label",
    "topic_gold",
    "gold_theme",
    "主题金标",
    "主题标签",
]

THEME_PRED_ALIASES = [
    "topic_final",
    "topic_label",
]

COLUMN_ALIASES = {
    Schema.IS_URBAN_RENEWAL: [
        "是否属于城市更新研究(人工)",
        "是否属于城市更新研究（人工）",
    ],
    Schema.IS_SPATIAL: [
        "空间研究/非空间研究(人工)",
        "空间研究/非空间研究（人工）",
    ],
    Schema.SPATIAL_LEVEL: [
        "空间等级(人工)",
        "空间等级（人工）",
    ],
    Schema.SPATIAL_DESC: [
        "具体空间描述(人工)",
        "具体空间描述（人工）",
    ],
}
for _canonical_name in (
    Schema.IS_URBAN_RENEWAL,
    Schema.IS_SPATIAL,
    Schema.SPATIAL_LEVEL,
    Schema.SPATIAL_DESC,
):
    _aliases = COLUMN_ALIASES.setdefault(_canonical_name, [])
    for _suffix in ("_local_v2", "_local"):
        _alias_name = f"{_canonical_name}{_suffix}"
        if _alias_name not in _aliases:
            _aliases.insert(0, _alias_name)


@dataclass
class AlignmentResult:
    merged: pd.DataFrame
    diagnostics: Dict[str, pd.DataFrame]
    summary: Dict[str, float]


def build_key(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .fillna("")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def normalize_binary_value(value):
    text = str(value).strip()
    if text.replace(".0", "") == "1":
        return 1
    if text.replace(".0", "") == "0":
        return 0
    return -1


def normalize_spatial_level(value):
    text = str(value).strip()
    if text in SPATIAL_LEVEL_MAP:
        return SPATIAL_LEVEL_MAP[text]
    lower = text.lower()
    for level, keywords in SPATIAL_LEVEL_KEYWORDS.items():
        for keyword in keywords:
            if keyword in lower:
                return int(level)
    match = re.search(r"\b([1-9])\b", text)
    if match:
        return int(match.group(1))
    return -1


def normalize_spatial_desc(value):
    text = str(value).strip().lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def fuzzy_match_spatial_level(truth_val, pred_val):
    truth = normalize_spatial_level(truth_val)
    pred = normalize_spatial_level(pred_val)
    if truth == -1 or pred == -1:
        return str(truth_val).strip().lower() == str(pred_val).strip().lower()
    return truth == pred


def fuzzy_match_spatial_desc(truth_val, pred_val, threshold=0.6):
    truth = normalize_spatial_desc(truth_val)
    pred = normalize_spatial_desc(pred_val)
    if truth == "not mentioned" or pred == "not mentioned":
        return truth == pred
    truth_words = set(truth.split())
    pred_words = set(pred.split())
    if not truth_words or not pred_words:
        return truth == pred
    intersection = len(truth_words & pred_words)
    union = len(truth_words | pred_words)
    similarity = intersection / union if union else 0
    return similarity >= threshold


def _ensure_required_columns(df: pd.DataFrame, required_cols: List[str], role: str):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{role}缺失必要列: {joined}")


def _deduplicate_on_key(df: pd.DataFrame, key_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dup_mask = df.duplicated(subset=[key_col], keep=False)
    duplicate_rows = df[dup_mask].copy()
    deduped = df.drop_duplicates(subset=[key_col], keep="first").copy()
    return deduped, duplicate_rows


def _normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()
    rename_map = {}
    for canonical_name, aliases in COLUMN_ALIASES.items():
        if canonical_name in renamed.columns:
            continue
        alias_name = next((alias for alias in aliases if alias in renamed.columns), None)
        if alias_name:
            rename_map[alias_name] = canonical_name
    if rename_map:
        renamed = renamed.rename(columns=rename_map)
    return renamed


def _resolve_truth_col(merged_df: pd.DataFrame, field_name: str):
    if f"{field_name}_truth" in merged_df.columns:
        return f"{field_name}_truth"
    return None


def _resolve_pred_col(merged_df: pd.DataFrame, field_name: str):
    if f"{field_name}_pred" in merged_df.columns:
        return f"{field_name}_pred"
    return None


def _resolve_optional_col(
    merged_df: pd.DataFrame,
    candidates: List[str],
    *,
    role: str,
):
    suffix = f"_{role}"
    for candidate in candidates:
        suffixed = f"{candidate}{suffix}"
        if suffixed in merged_df.columns:
            return suffixed
        if candidate in merged_df.columns:
            return candidate
    return None


def _non_empty_series(frame: pd.DataFrame, column: Optional[str]) -> pd.Series:
    if column is None or column not in frame.columns:
        return pd.Series([False] * len(frame), index=frame.index, dtype=bool)
    normalized = frame[column].fillna("").astype(str).str.strip()
    return ~normalized.str.lower().isin({"", "nan", "none", "null"})


def _numeric_flag_series(frame: pd.DataFrame, column: Optional[str]) -> pd.Series:
    if column is None or column not in frame.columns:
        return pd.Series([0] * len(frame), index=frame.index, dtype=int)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)


def _normalize_theme_label(value) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    if text.lower() == UNKNOWN_TOPIC_LABEL.lower():
        return UNKNOWN_TOPIC_LABEL
    upper = text.upper()
    if upper.startswith(("U", "N")) and upper[1:].isdigit():
        return upper
    return text


def _theme_group(label: str) -> str:
    normalized = _normalize_theme_label(label)
    if not normalized:
        return ""
    if normalized == UNKNOWN_TOPIC_LABEL:
        return "unknown"
    return topic_group_for_label(normalized)


def _binary_metrics_from_series(truth: pd.Series, pred: pd.Series) -> Dict[str, float]:
    truth_norm = truth.apply(normalize_binary_value)
    pred_norm = pred.apply(normalize_binary_value)
    tp = int(((truth_norm == 1) & (pred_norm == 1)).sum())
    tn = int(((truth_norm == 0) & (pred_norm == 0)).sum())
    fp = int(((truth_norm == 0) & (pred_norm == 1)).sum())
    fn = int(((truth_norm == 1) & (pred_norm == 0)).sum())
    total = int(len(truth_norm))
    correct = tp + tn
    accuracy = round((correct / total * 100.0) if total else 0.0, 4)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "Accuracy": accuracy,
        "Correct": correct,
        "Total": total,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Precision": round(precision, 6),
        "Recall": round(recall, 6),
        "F1": round(f1, 6),
    }


def align_truth_pred(
    truth_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    strict: bool = False,
    coverage_threshold: float = 0.8,
) -> AlignmentResult:
    required = [Schema.TITLE]
    _ensure_required_columns(truth_df, required, "真值文件")
    _ensure_required_columns(pred_df, required, "预测文件")

    truth = _normalize_metric_columns(truth_df)
    pred = _normalize_metric_columns(pred_df)

    truth["_key"] = build_key(truth[Schema.TITLE])
    pred["_key"] = build_key(pred[Schema.TITLE])

    truth, truth_dup = _deduplicate_on_key(truth, "_key")
    pred, pred_dup = _deduplicate_on_key(pred, "_key")

    merged = pd.merge(
        truth,
        pred,
        on="_key",
        suffixes=("_truth", "_pred"),
        how="inner",
    )

    truth_only = truth.loc[~truth["_key"].isin(pred["_key"])].copy()
    pred_only = pred.loc[~pred["_key"].isin(truth["_key"])].copy()

    truth_count = len(truth)
    pred_count = len(pred)
    match_count = len(merged)
    truth_coverage = match_count / truth_count if truth_count else 0.0
    pred_coverage = match_count / pred_count if pred_count else 0.0

    if strict:
        if len(truth_dup) > 0 or len(pred_dup) > 0:
            raise ValueError("严格模式失败: 检测到重复主键，请先清洗标题重复数据")
        if truth_coverage < coverage_threshold or pred_coverage < coverage_threshold:
            raise ValueError(
                f"严格模式失败: 对齐覆盖率不足 truth={truth_coverage:.2%}, pred={pred_coverage:.2%}, 阈值={coverage_threshold:.2%}"
            )

    return AlignmentResult(
        merged=merged,
        diagnostics={
            "truth_unmatched": truth_only,
            "pred_unmatched": pred_only,
            "truth_duplicates": truth_dup,
            "pred_duplicates": pred_dup,
        },
        summary={
            "truth_count": float(truth_count),
            "pred_count": float(pred_count),
            "match_count": float(match_count),
            "truth_coverage": truth_coverage,
            "pred_coverage": pred_coverage,
        },
    )


def evaluate_merged(
    merged_df: pd.DataFrame,
    source_name: str,
    spatial_desc_threshold: float = 0.6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if merged_df.empty:
        raise ValueError("无可评估样本: 对齐后结果为空")

    merged = merged_df.copy()
    metrics = []

    for field_name, metric_name, is_binary in FIELD_SPECS:
        truth_col = _resolve_truth_col(merged, field_name)
        pred_col = _resolve_pred_col(merged, field_name)
        missing_pair = truth_col is None or pred_col is None

        if missing_pair:
            condition = pd.Series(pd.NA, index=merged.index, dtype="object")
            tp, tn, fp, fn = 0, 0, 0, 0
            precision, recall, f1 = np.nan, np.nan, np.nan
        elif is_binary:
            truth_norm = merged[truth_col].apply(normalize_binary_value)
            pred_norm = merged[pred_col].apply(normalize_binary_value)
            condition = (truth_norm == pred_norm) & (pred_norm.isin([0, 1]))
            tp = ((truth_norm == 1) & (pred_norm == 1)).sum()
            tn = ((truth_norm == 0) & (pred_norm == 0)).sum()
            fp = ((truth_norm == 0) & (pred_norm == 1)).sum()
            fn = ((truth_norm == 1) & (pred_norm == 0)).sum()
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        elif metric_name == "Spatial Level":
            condition = merged.apply(
                lambda row: fuzzy_match_spatial_level(row[truth_col], row[pred_col]),
                axis=1,
            )
            tp, tn, fp, fn = 0, 0, 0, 0
            precision, recall, f1 = np.nan, np.nan, np.nan
        else:
            condition = merged.apply(
                lambda row: fuzzy_match_spatial_desc(row[truth_col], row[pred_col], threshold=spatial_desc_threshold),
                axis=1,
            )
            tp, tn, fp, fn = 0, 0, 0, 0
            precision, recall, f1 = np.nan, np.nan, np.nan

        diff_col = f"Diff_{metric_name}"
        if missing_pair:
            merged[diff_col] = np.nan
            correct = 0
            total = 0
            accuracy = np.nan
        else:
            merged[diff_col] = np.where(condition, 1, 0)
            correct = int(merged[diff_col].sum())
            total = len(merged)
            accuracy = (correct / total * 100.0) if total else np.nan

        metrics.append(
            {
                "File": source_name,
                "Metric": metric_name,
                "Accuracy": round(accuracy, 4) if not np.isnan(accuracy) else np.nan,
                "Correct": correct,
                "Total": total,
                "TP": int(tp),
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "Precision": round(precision, 6) if not np.isnan(precision) else np.nan,
                "Recall": round(recall, 6) if not np.isnan(recall) else np.nan,
                "F1": round(f1, 6) if not np.isnan(f1) else np.nan,
            }
        )

    ordered_metric_pairs = [
        (Schema.IS_URBAN_RENEWAL, "Diff_Urban Renewal"),
        (Schema.IS_SPATIAL, "Diff_Spatial Study"),
        (Schema.SPATIAL_LEVEL, "Diff_Spatial Level"),
        (Schema.SPATIAL_DESC, "Diff_Spatial Desc"),
    ]
    detail_output = pd.DataFrame(index=merged.index)
    for field_name, diff_col in ordered_metric_pairs:
        truth_col = _resolve_truth_col(merged, field_name)
        pred_col = _resolve_pred_col(merged, field_name)
        detail_output[f"{field_name}_truth"] = merged[truth_col] if truth_col else np.nan
        detail_output[f"{field_name}_pred"] = merged[pred_col] if pred_col else np.nan
        detail_output[diff_col] = merged[diff_col] if diff_col in merged.columns else 0

    title_col = f"{Schema.TITLE}_truth" if f"{Schema.TITLE}_truth" in merged.columns else Schema.TITLE
    abstract_col = f"{Schema.ABSTRACT}_truth" if f"{Schema.ABSTRACT}_truth" in merged.columns else Schema.ABSTRACT
    detail_output[Schema.TITLE] = merged[title_col] if title_col in merged.columns else ""
    detail_output[Schema.ABSTRACT] = merged[abstract_col] if abstract_col in merged.columns else ""
    detail_output = detail_output[
        [
            Schema.TITLE,
            Schema.ABSTRACT,
            f"{Schema.IS_URBAN_RENEWAL}_truth",
            f"{Schema.IS_URBAN_RENEWAL}_pred",
            "Diff_Urban Renewal",
            f"{Schema.IS_SPATIAL}_truth",
            f"{Schema.IS_SPATIAL}_pred",
            "Diff_Spatial Study",
            f"{Schema.SPATIAL_LEVEL}_truth",
            f"{Schema.SPATIAL_LEVEL}_pred",
            "Diff_Spatial Level",
            f"{Schema.SPATIAL_DESC}_truth",
            f"{Schema.SPATIAL_DESC}_pred",
            "Diff_Spatial Desc",
        ]
    ]

    detail_df = detail_output

    metrics_df = pd.DataFrame(metrics)
    metrics_df = metrics_df.reindex(columns=METRIC_OUTPUT_COLUMNS)
    validate_accuracy_bounds(metrics_df, context=f"evaluate_merged:{source_name}")
    return metrics_df, detail_df


def summarize_chunked_binary_metrics(
    merged_df: pd.DataFrame,
    source_name: str,
    chunk_size: int = 100,
) -> pd.DataFrame:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if merged_df.empty:
        return pd.DataFrame(columns=CHUNK_METRIC_OUTPUT_COLUMNS)

    merged = merged_df.copy().reset_index(drop=True)
    rows = []

    for field_name, metric_name, is_binary in FIELD_SPECS:
        if not is_binary:
            continue

        truth_col = _resolve_truth_col(merged, field_name)
        pred_col = _resolve_pred_col(merged, field_name)
        if truth_col is None or pred_col is None:
            continue

        truth_norm = merged[truth_col].apply(normalize_binary_value)
        pred_norm = merged[pred_col].apply(normalize_binary_value)

        for chunk_index, start in enumerate(range(0, len(merged), chunk_size), start=1):
            end = min(start + chunk_size, len(merged))
            truth_chunk = truth_norm.iloc[start:end]
            pred_chunk = pred_norm.iloc[start:end]
            total = len(truth_chunk)
            tp = int(((truth_chunk == 1) & (pred_chunk == 1)).sum())
            tn = int(((truth_chunk == 0) & (pred_chunk == 0)).sum())
            fp = int(((truth_chunk == 0) & (pred_chunk == 1)).sum())
            fn = int(((truth_chunk == 1) & (pred_chunk == 0)).sum())
            correct = tp + tn
            accuracy = (correct / total * 100.0) if total else 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            truth_positive_rate = float((truth_chunk == 1).mean()) if total else 0.0
            predicted_positive_rate = float((pred_chunk == 1).mean()) if total else 0.0

            rows.append(
                {
                    "File": source_name,
                    "Metric": metric_name,
                    "Chunk": chunk_index,
                    "Chunk Start": start + 1,
                    "Chunk End": end,
                    "Accuracy": round(accuracy, 4),
                    "Correct": correct,
                    "Total": total,
                    "TP": tp,
                    "TN": tn,
                    "FP": fp,
                    "FN": fn,
                    "Precision": round(precision, 6),
                    "Recall": round(recall, 6),
                    "F1": round(f1, 6),
                    "Truth Positive Rate": round(truth_positive_rate, 6),
                    "Predicted Positive Rate": round(predicted_positive_rate, 6),
                }
            )

    chunk_df = pd.DataFrame(rows)
    chunk_df = chunk_df.reindex(columns=CHUNK_METRIC_OUTPUT_COLUMNS)
    validate_accuracy_bounds(chunk_df, context=f"summarize_chunked_binary_metrics:{source_name}")
    return chunk_df


def summarize_metrics(all_metrics: pd.DataFrame) -> pd.DataFrame:
    if all_metrics.empty:
        return all_metrics

    summary_rows = []
    for metric_name, group in all_metrics.groupby("Metric"):
        total = group["Total"].sum()
        correct = group["Correct"].sum()
        accuracy = (correct / total * 100.0) if total else np.nan

        tp = group["TP"].sum()
        fp = group["FP"].sum()
        fn = group["FN"].sum()
        metric_precision = group["Precision"].dropna()
        if metric_precision.empty:
            precision = np.nan
            recall = np.nan
            f1 = np.nan
        else:
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        summary_rows.append(
            {
                "File": "__GLOBAL__",
                "Metric": metric_name,
                "Accuracy": round(accuracy, 4) if not np.isnan(accuracy) else np.nan,
                "Correct": int(correct),
                "Total": int(total),
                "TP": int(tp),
                "TN": int(group["TN"].sum()),
                "FP": int(fp),
                "FN": int(fn),
                "Precision": round(precision, 6) if not np.isnan(precision) else np.nan,
                "Recall": round(recall, 6) if not np.isnan(recall) else np.nan,
                "F1": round(f1, 6) if not np.isnan(f1) else np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.reindex(columns=METRIC_OUTPUT_COLUMNS)
    validate_accuracy_bounds(summary_df, context="summarize_metrics")
    return summary_df


def summarize_theme_metrics(
    merged_df: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    truth_col = _resolve_optional_col(merged_df, THEME_TRUTH_ALIASES, role="truth")
    pred_col = _resolve_optional_col(merged_df, THEME_PRED_ALIASES, role="pred")
    if truth_col is None or pred_col is None:
        return pd.DataFrame(columns=THEME_METRIC_OUTPUT_COLUMNS)

    working = merged_df[[truth_col, pred_col]].copy()
    working["truth_theme"] = working[truth_col].apply(_normalize_theme_label)
    working["pred_theme"] = working[pred_col].apply(_normalize_theme_label)
    working = working[working["truth_theme"] != ""].copy()
    if working.empty:
        return pd.DataFrame(columns=THEME_METRIC_OUTPUT_COLUMNS)

    labels = sorted(
        {
            label
            for label in pd.concat([working["truth_theme"], working["pred_theme"]]).tolist()
            if label
        }
    )
    rows = []
    overall_correct = int((working["truth_theme"] == working["pred_theme"]).sum())
    rows.append(
        {
            "File": source_name,
            "Theme": "__OVERALL__",
            "Theme Name": "Exact Theme Match",
            "Theme Group": "",
            "Accuracy": round(overall_correct / len(working) * 100.0, 4),
            "Correct": overall_correct,
            "Total": int(len(working)),
            "Truth Support": int(len(working)),
            "Pred Support": int(len(working)),
            "TP": pd.NA,
            "FP": pd.NA,
            "FN": pd.NA,
            "Precision": pd.NA,
            "Recall": pd.NA,
            "F1": pd.NA,
        }
    )

    for label in labels:
        truth_mask = working["truth_theme"] == label
        pred_mask = working["pred_theme"] == label
        tp = int((truth_mask & pred_mask).sum())
        fp = int((~truth_mask & pred_mask).sum())
        fn = int((truth_mask & ~pred_mask).sum())
        truth_support = int(truth_mask.sum())
        pred_support = int(pred_mask.sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "File": source_name,
                "Theme": label,
                "Theme Name": topic_name_for_label(label) if label != UNKNOWN_TOPIC_LABEL else UNKNOWN_TOPIC_LABEL,
                "Theme Group": _theme_group(label),
                "Accuracy": round(tp / truth_support * 100.0, 4) if truth_support else 0.0,
                "Correct": tp,
                "Total": truth_support,
                "Truth Support": truth_support,
                "Pred Support": pred_support,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Precision": round(precision, 6),
                "Recall": round(recall, 6),
                "F1": round(f1, 6),
            }
        )

    return pd.DataFrame(rows).reindex(columns=THEME_METRIC_OUTPUT_COLUMNS)


def summarize_theme_confusion(
    merged_df: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    truth_col = _resolve_optional_col(merged_df, THEME_TRUTH_ALIASES, role="truth")
    pred_col = _resolve_optional_col(merged_df, THEME_PRED_ALIASES, role="pred")
    if truth_col is None or pred_col is None:
        return pd.DataFrame(columns=THEME_CONFUSION_OUTPUT_COLUMNS)

    working = merged_df[[truth_col, pred_col]].copy()
    working["truth_theme"] = working[truth_col].apply(_normalize_theme_label)
    working["pred_theme"] = working[pred_col].apply(_normalize_theme_label)
    working = working[working["truth_theme"] != ""].copy()
    if working.empty:
        return pd.DataFrame(columns=THEME_CONFUSION_OUTPUT_COLUMNS)

    grouped = (
        working.groupby(["truth_theme", "pred_theme"], dropna=False)
        .size()
        .reset_index(name="Count")
        .sort_values(["Count", "truth_theme", "pred_theme"], ascending=[False, True, True])
    )
    grouped["File"] = source_name
    grouped["Truth Theme"] = grouped["truth_theme"]
    grouped["Truth Theme Name"] = grouped["truth_theme"].apply(
        lambda value: topic_name_for_label(value) if value not in {"", UNKNOWN_TOPIC_LABEL} else value or ""
    )
    grouped["Pred Theme"] = grouped["pred_theme"]
    grouped["Pred Theme Name"] = grouped["pred_theme"].apply(
        lambda value: topic_name_for_label(value) if value not in {"", UNKNOWN_TOPIC_LABEL} else value or ""
    )
    return grouped.reindex(columns=THEME_CONFUSION_OUTPUT_COLUMNS)


def summarize_theme_family_metrics(
    merged_df: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    truth_col = _resolve_optional_col(merged_df, THEME_TRUTH_ALIASES, role="truth")
    pred_col = _resolve_optional_col(merged_df, THEME_PRED_ALIASES, role="pred")
    if truth_col is None or pred_col is None:
        return pd.DataFrame(columns=THEME_FAMILY_OUTPUT_COLUMNS)

    working = merged_df[[truth_col, pred_col]].copy()
    working["truth_family"] = working[truth_col].apply(_normalize_theme_label).apply(_theme_group)
    working["pred_family"] = working[pred_col].apply(_normalize_theme_label).apply(_theme_group)
    working = working[working["truth_family"] != ""].copy()
    if working.empty:
        return pd.DataFrame(columns=THEME_FAMILY_OUTPUT_COLUMNS)

    family_truth = working["truth_family"].replace({"urban": 1, "nonurban": 0})
    family_pred = working["pred_family"].replace({"urban": 1, "nonurban": 0})
    valid = family_truth.isin([0, 1]) & family_pred.isin([0, 1])
    if not valid.any():
        return pd.DataFrame(columns=THEME_FAMILY_OUTPUT_COLUMNS)

    metrics = _binary_metrics_from_series(family_truth[valid], family_pred[valid])
    row = {"File": source_name, "Metric": "Theme Family"}
    row.update(metrics)
    return pd.DataFrame([row]).reindex(columns=THEME_FAMILY_OUTPUT_COLUMNS)


def summarize_unknown_rate(
    merged_df: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    pred_col = _resolve_optional_col(merged_df, THEME_PRED_ALIASES, role="pred")
    truth_col = _resolve_optional_col(merged_df, THEME_TRUTH_ALIASES, role="truth")
    if pred_col is None:
        return pd.DataFrame(columns=UNKNOWN_RATE_OUTPUT_COLUMNS)

    pred_theme = merged_df[pred_col].apply(_normalize_theme_label)
    truth_theme = merged_df[truth_col].apply(_normalize_theme_label) if truth_col else pd.Series([""] * len(merged_df))
    theme_truth_samples = int((truth_theme != "").sum())
    predicted_unknown_count = int((pred_theme == UNKNOWN_TOPIC_LABEL).sum())
    predicted_unknown_rate = predicted_unknown_count / len(pred_theme) if len(pred_theme) else 0.0
    truth_unknown_count = int((truth_theme == UNKNOWN_TOPIC_LABEL).sum())
    truth_unknown_rate = truth_unknown_count / theme_truth_samples if theme_truth_samples else 0.0
    row = {
        "File": source_name,
        "Total Samples": int(len(pred_theme)),
        "Theme Truth Samples": theme_truth_samples,
        "Predicted Unknown Count": predicted_unknown_count,
        "Predicted Unknown Rate": round(predicted_unknown_rate, 6),
        "Truth Unknown Count": truth_unknown_count,
        "Truth Unknown Rate": round(truth_unknown_rate, 6),
    }
    return pd.DataFrame([row]).reindex(columns=UNKNOWN_RATE_OUTPUT_COLUMNS)


def summarize_decision_source_metrics(
    merged_df: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    truth_col = _resolve_truth_col(merged_df, Schema.IS_URBAN_RENEWAL)
    pred_col = _resolve_pred_col(merged_df, Schema.IS_URBAN_RENEWAL)
    if truth_col is None or pred_col is None or "decision_source" not in merged_df.columns:
        return pd.DataFrame(columns=DECISION_SOURCE_OUTPUT_COLUMNS)

    topic_col = _resolve_optional_col(merged_df, THEME_PRED_ALIASES, role="pred")
    rows = []
    working = merged_df.copy()
    working["decision_source"] = working["decision_source"].fillna("").replace("", "missing")
    for decision_source, group in working.groupby("decision_source", dropna=False):
        metrics = _binary_metrics_from_series(group[truth_col], group[pred_col])
        pred_norm = group[pred_col].apply(normalize_binary_value)
        unknown_rate = 0.0
        if topic_col is not None:
            unknown_rate = float(
                group[topic_col].apply(_normalize_theme_label).eq(UNKNOWN_TOPIC_LABEL).mean()
            )
        row = {
            "File": source_name,
            "Decision Source": decision_source,
            "Total": metrics["Total"],
            "Accuracy": metrics["Accuracy"],
            "TP": metrics["TP"],
            "TN": metrics["TN"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "Predicted Positive Rate": round(float((pred_norm == 1).mean()), 6) if len(pred_norm) else 0.0,
            "Unknown Topic Rate": round(unknown_rate, 6),
        }
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=DECISION_SOURCE_OUTPUT_COLUMNS)


def summarize_topic_final_distribution(
    merged_df: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    topic_col = _resolve_optional_col(merged_df, THEME_PRED_ALIASES, role="pred")
    truth_col = _resolve_truth_col(merged_df, Schema.IS_URBAN_RENEWAL)
    pred_col = _resolve_pred_col(merged_df, Schema.IS_URBAN_RENEWAL)
    if topic_col is None:
        return pd.DataFrame(columns=TOPIC_DISTRIBUTION_OUTPUT_COLUMNS)

    working = merged_df.copy()
    working["topic_final_norm"] = working[topic_col].apply(_normalize_theme_label).replace("", UNKNOWN_TOPIC_LABEL)
    truth_norm = working[truth_col].apply(normalize_binary_value) if truth_col else pd.Series([-1] * len(working))
    pred_norm = working[pred_col].apply(normalize_binary_value) if pred_col else pd.Series([-1] * len(working))

    rows = []
    total = len(working)
    for topic_label, group in working.groupby("topic_final_norm", dropna=False):
        idx = group.index
        fp = int(((truth_norm.loc[idx] == 0) & (pred_norm.loc[idx] == 1)).sum()) if truth_col and pred_col else 0
        fn = int(((truth_norm.loc[idx] == 1) & (pred_norm.loc[idx] == 0)).sum()) if truth_col and pred_col else 0
        truth_positive_rate = (
            float((truth_norm.loc[idx] == 1).mean()) if truth_col and len(idx) else np.nan
        )
        rows.append(
            {
                "File": source_name,
                "Topic Final": topic_label,
                "Topic Name": topic_name_for_label(topic_label) if topic_label != UNKNOWN_TOPIC_LABEL else UNKNOWN_TOPIC_LABEL,
                "Topic Group": _theme_group(topic_label),
                "Count": int(len(group)),
                "Share": round(len(group) / total, 6) if total else 0.0,
                "Truth Positive Rate": round(truth_positive_rate, 6) if not np.isnan(truth_positive_rate) else np.nan,
                "FP": fp,
                "FN": fn,
            }
        )
    distribution_df = pd.DataFrame(rows)
    if distribution_df.empty:
        return pd.DataFrame(columns=TOPIC_DISTRIBUTION_OUTPUT_COLUMNS)
    return distribution_df.sort_values(["Count", "Topic Final"], ascending=[False, True]).reindex(
        columns=TOPIC_DISTRIBUTION_OUTPUT_COLUMNS
    )


def summarize_boundary_bucket_metrics(
    merged_df: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    truth_col = _resolve_truth_col(merged_df, Schema.IS_URBAN_RENEWAL)
    pred_col = _resolve_pred_col(merged_df, Schema.IS_URBAN_RENEWAL)
    if truth_col is None or pred_col is None or "boundary_bucket" not in merged_df.columns:
        return pd.DataFrame(columns=BOUNDARY_BUCKET_OUTPUT_COLUMNS)

    rows = []
    topic_col = _resolve_optional_col(merged_df, THEME_PRED_ALIASES, role="pred")
    working = merged_df.copy()
    working["boundary_bucket"] = working["boundary_bucket"].fillna("").replace("", "unclassified")
    if "family_conflict_pattern" not in working.columns:
        working["family_conflict_pattern"] = "unknown"
    working["family_conflict_pattern"] = working["family_conflict_pattern"].fillna("").replace("", "unknown")
    for (bucket, pattern), group in working.groupby(["boundary_bucket", "family_conflict_pattern"], dropna=False):
        metrics = _binary_metrics_from_series(group[truth_col], group[pred_col])
        unknown_count = 0
        if topic_col is not None:
            unknown_count = int(group[topic_col].apply(_normalize_theme_label).eq(UNKNOWN_TOPIC_LABEL).sum())
        rows.append(
            {
                "File": source_name,
                "Boundary Bucket": bucket,
                "Family Conflict Pattern": pattern,
                "Total": metrics["Total"],
                "Accuracy": metrics["Accuracy"],
                "TP": metrics["TP"],
                "TN": metrics["TN"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
                "Unknown Count": unknown_count,
            }
        )
    return pd.DataFrame(rows).reindex(columns=BOUNDARY_BUCKET_OUTPUT_COLUMNS)


def summarize_unknown_conflict_analysis(
    merged_df: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    if "review_reason" not in merged_df.columns:
        return pd.DataFrame(columns=UNKNOWN_CONFLICT_OUTPUT_COLUMNS)

    topic_col = _resolve_optional_col(merged_df, THEME_PRED_ALIASES, role="pred")
    working = merged_df.copy()
    if topic_col is not None:
        working = working[working[topic_col].apply(_normalize_theme_label) == UNKNOWN_TOPIC_LABEL].copy()
    else:
        working = working[working["review_reason"].fillna("").astype(str) != ""].copy()
    if working.empty:
        return pd.DataFrame(columns=UNKNOWN_CONFLICT_OUTPUT_COLUMNS)

    if "boundary_bucket" not in working.columns:
        working["boundary_bucket"] = "unclassified"
    if "family_conflict_pattern" not in working.columns:
        working["family_conflict_pattern"] = "unknown"
    if "topic_family_rule" not in working.columns:
        working["topic_family_rule"] = UNKNOWN_TOPIC_GROUP
    if "topic_family_local" not in working.columns:
        working["topic_family_local"] = UNKNOWN_TOPIC_GROUP
    if "llm_family_hint" not in working.columns:
        working["llm_family_hint"] = ""

    grouped = (
        working.groupby(
            [
                "review_reason",
                "boundary_bucket",
                "family_conflict_pattern",
                "topic_family_rule",
                "topic_family_local",
                "llm_family_hint",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="Count")
        .sort_values(["Count", "review_reason"], ascending=[False, True])
    )
    grouped["File"] = source_name
    grouped = grouped.rename(
        columns={
            "review_reason": "Review Reason",
            "boundary_bucket": "Boundary Bucket",
            "family_conflict_pattern": "Family Conflict Pattern",
            "topic_family_rule": "Rule Family",
            "topic_family_local": "Local Family",
            "llm_family_hint": "LLM Hint",
        }
    )
    return grouped.reindex(columns=UNKNOWN_CONFLICT_OUTPUT_COLUMNS)


def summarize_explainability_quality(
    merged_df: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    total = int(len(merged_df))
    explanation_col = _resolve_optional_col(merged_df, ["decision_explanation"], role="pred")
    rule_stack_col = _resolve_optional_col(merged_df, ["decision_rule_stack"], role="pred")
    binary_evidence_col = _resolve_optional_col(merged_df, ["binary_decision_evidence"], role="pred")
    positive_evidence_col = _resolve_optional_col(merged_df, ["primary_positive_evidence"], role="pred")
    negative_evidence_col = _resolve_optional_col(merged_df, ["primary_negative_evidence"], role="pred")
    evidence_balance_col = _resolve_optional_col(merged_df, ["evidence_balance"], role="pred")
    review_flag_col = _resolve_optional_col(merged_df, ["review_flag"], role="pred")
    review_reason_col = _resolve_optional_col(merged_df, ["review_reason"], role="pred")
    consistency_col = _resolve_optional_col(merged_df, ["binary_topic_consistency_flag"], role="pred")

    def coverage(column: Optional[str]) -> float:
        if total == 0:
            return 0.0
        return round(float(_non_empty_series(merged_df, column).mean()), 6)

    review_flags = _numeric_flag_series(merged_df, review_flag_col)
    if review_reason_col and review_reason_col in merged_df.columns:
        review_reason = merged_df[review_reason_col].fillna("").astype(str)
    else:
        review_reason = pd.Series([""] * total, index=merged_df.index, dtype=object)
    near_threshold = review_reason.str.contains("binary_near_threshold", regex=False, na=False)
    conflict_from_reason = review_reason.str.contains("conflict|inconsistency", regex=True, case=False, na=False)
    consistency_flags = _numeric_flag_series(merged_df, consistency_col) > 0
    conflict = conflict_from_reason | consistency_flags

    row = {
        "File": source_name,
        "Total": total,
        "Decision Explanation Coverage": coverage(explanation_col),
        "Rule Stack Coverage": coverage(rule_stack_col),
        "Binary Evidence Coverage": coverage(binary_evidence_col),
        "Positive Evidence Coverage": coverage(positive_evidence_col),
        "Negative Evidence Coverage": coverage(negative_evidence_col),
        "Evidence Balance Coverage": coverage(evidence_balance_col),
        "Review Trigger Count": int((review_flags > 0).sum()),
        "Review Trigger Rate": round(float((review_flags > 0).mean()), 6) if total else 0.0,
        "Near Threshold Count": int(near_threshold.sum()),
        "Conflict Count": int(conflict.sum()),
    }
    return pd.DataFrame([row]).reindex(columns=EXPLAINABILITY_QUALITY_OUTPUT_COLUMNS)


def summarize_evidence_balance_metrics(
    merged_df: pd.DataFrame,
    source_name: str,
) -> pd.DataFrame:
    balance_col = _resolve_optional_col(merged_df, ["evidence_balance"], role="pred")
    truth_col = _resolve_truth_col(merged_df, Schema.IS_URBAN_RENEWAL)
    pred_col = _resolve_pred_col(merged_df, Schema.IS_URBAN_RENEWAL)
    if truth_col is None or pred_col is None:
        return pd.DataFrame(columns=EVIDENCE_BALANCE_OUTPUT_COLUMNS)

    working = merged_df.copy()
    if balance_col is None:
        working["_evidence_balance_norm"] = "missing"
    else:
        normalized = working[balance_col].fillna("").astype(str).str.strip()
        working["_evidence_balance_norm"] = normalized.mask(
            normalized.str.lower().isin({"", "nan", "none", "null"}),
            "missing",
        )

    review_flag_col = _resolve_optional_col(working, ["review_flag"], role="pred")
    review_flags = _numeric_flag_series(working, review_flag_col) > 0
    pred_norm = working[pred_col].apply(normalize_binary_value)

    rows = []
    for balance, group in working.groupby("_evidence_balance_norm", dropna=False):
        metrics = _binary_metrics_from_series(group[truth_col], group[pred_col])
        idx = group.index
        row = {
            "File": source_name,
            "Evidence Balance": balance,
            "Total": metrics["Total"],
            "Accuracy": metrics["Accuracy"],
            "TP": metrics["TP"],
            "TN": metrics["TN"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "Predicted Positive Rate": round(float((pred_norm.loc[idx] == 1).mean()), 6) if len(idx) else 0.0,
            "Review Trigger Rate": round(float(review_flags.loc[idx].mean()), 6) if len(idx) else 0.0,
        }
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=EVIDENCE_BALANCE_OUTPUT_COLUMNS)
    return (
        pd.DataFrame(rows)
        .sort_values(["Total", "Evidence Balance"], ascending=[False, True])
        .reindex(columns=EVIDENCE_BALANCE_OUTPUT_COLUMNS)
    )


def summarize_bootstrap_ci(
    merged_df: pd.DataFrame,
    source_name: str,
    *,
    bootstrap_samples: int = 400,
    random_seed: int = 20260416,
) -> pd.DataFrame:
    truth_col = _resolve_truth_col(merged_df, Schema.IS_URBAN_RENEWAL)
    pred_col = _resolve_pred_col(merged_df, Schema.IS_URBAN_RENEWAL)
    if truth_col is None or pred_col is None or merged_df.empty:
        return pd.DataFrame(columns=BOOTSTRAP_CI_OUTPUT_COLUMNS)

    truth = merged_df[truth_col].apply(normalize_binary_value).to_numpy()
    pred = merged_df[pred_col].apply(normalize_binary_value).to_numpy()
    if len(truth) == 0:
        return pd.DataFrame(columns=BOOTSTRAP_CI_OUTPUT_COLUMNS)

    base_metrics = _binary_metrics_from_series(merged_df[truth_col], merged_df[pred_col])
    rng = np.random.default_rng(random_seed)
    accuracy_samples = []
    f1_samples = []
    idx = np.arange(len(truth))
    for _ in range(int(max(bootstrap_samples, 50))):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        sample_truth = pd.Series(truth[sample_idx])
        sample_pred = pd.Series(pred[sample_idx])
        metrics = _binary_metrics_from_series(sample_truth, sample_pred)
        accuracy_samples.append(float(metrics["Accuracy"]))
        f1_samples.append(float(metrics["F1"]))

    rows = []
    for metric_name, point_estimate, samples in (
        ("Accuracy", float(base_metrics["Accuracy"]), accuracy_samples),
        ("F1", float(base_metrics["F1"]), f1_samples),
    ):
        rows.append(
            {
                "File": source_name,
                "Metric": metric_name,
                "Point Estimate": point_estimate,
                "CI Lower": round(float(np.quantile(samples, 0.025)), 6),
                "CI Upper": round(float(np.quantile(samples, 0.975)), 6),
                "Bootstrap Samples": int(max(bootstrap_samples, 50)),
            }
        )
    return pd.DataFrame(rows).reindex(columns=BOOTSTRAP_CI_OUTPUT_COLUMNS)


def summarize_mcnemar(
    aligned_frames: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    columns = list(MCNEMAR_OUTPUT_COLUMNS)
    if len(aligned_frames) < 2:
        return pd.DataFrame(columns=columns)

    scored = []
    prepared: Dict[str, pd.DataFrame] = {}
    for name, frame in aligned_frames.items():
        truth_col = _resolve_truth_col(frame, Schema.IS_URBAN_RENEWAL)
        pred_col = _resolve_pred_col(frame, Schema.IS_URBAN_RENEWAL)
        if truth_col is None or pred_col is None:
            continue
        subset = frame[[Schema.TITLE, truth_col, pred_col]].copy()
        subset["correct"] = (
            subset[truth_col].apply(normalize_binary_value)
            == subset[pred_col].apply(normalize_binary_value)
        ).astype(int)
        metrics = _binary_metrics_from_series(subset[truth_col], subset[pred_col])
        prepared[name] = subset
        scored.append((name, float(metrics["F1"])))
    if len(scored) < 2:
        return pd.DataFrame(columns=columns)

    best_two = [name for name, _ in sorted(scored, key=lambda item: (-item[1], item[0]))[:2]]
    left = prepared[best_two[0]].rename(columns={"correct": "correct_a"})
    right = prepared[best_two[1]].rename(columns={"correct": "correct_b"})
    merged = left[[Schema.TITLE, "correct_a"]].merge(
        right[[Schema.TITLE, "correct_b"]],
        on=Schema.TITLE,
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)

    b = int(((merged["correct_a"] == 1) & (merged["correct_b"] == 0)).sum())
    c = int(((merged["correct_a"] == 0) & (merged["correct_b"] == 1)).sum())
    statistic = ((abs(b - c) - 1) ** 2 / (b + c)) if (b + c) else 0.0
    p_value = math.erfc(math.sqrt(max(statistic, 0.0) / 2.0)) if statistic > 0 else 1.0
    return pd.DataFrame(
        [
            {
                "File A": best_two[0],
                "File B": best_two[1],
                "Metric": "Urban Renewal",
                "B": b,
                "C": c,
                "Statistic": round(float(statistic), 6),
                "P Value": round(float(p_value), 6),
            }
        ]
    ).reindex(columns=columns)
