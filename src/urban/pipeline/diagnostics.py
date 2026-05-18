"""Read-only diagnostics for urban-renewal prediction outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_urban_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize prediction health without mutating pipeline outputs."""

    total = int(len(frame))
    final_label = _string_series(frame, "final_label")
    urban_flag = _string_series(frame, "urban_flag")
    topic_final = _string_series(frame, "topic_final")
    topic_group = _string_series(frame, "topic_final_group")
    review_flag = _numeric_series(frame, "review_flag")
    review_reason = _string_series(frame, "review_reason")
    decision_source = _string_series(frame, "decision_source")
    dynamic_override = _numeric_series(frame, "dynamic_binary_override_applied")
    consistency_flag = _numeric_series(frame, "binary_topic_consistency_flag")

    effective_binary = final_label.where(final_label.isin({"0", "1"}), urban_flag)
    unknown_topic = topic_final.eq("Unknown") | topic_group.eq("unknown")
    nonurban_topic = topic_group.eq("nonurban") | topic_final.str.match(r"^N\d+", na=False)
    positive_nonurban = effective_binary.eq("1") & nonurban_topic
    review_required = review_flag.gt(0) | review_reason.ne("") | decision_source.str.contains("unknown_review", na=False)

    return {
        "total_rows": total,
        "final_label_counts": _value_counts(effective_binary),
        "topic_final_counts": _value_counts(topic_final),
        "unknown_topic_count": int(unknown_topic.sum()),
        "unknown_topic_rate": _rate(int(unknown_topic.sum()), total),
        "binary_topic_conflict_count": int(consistency_flag.gt(0).sum()),
        "binary_topic_conflict_rate": _rate(int(consistency_flag.gt(0).sum()), total),
        "dynamic_binary_override_count": int(dynamic_override.gt(0).sum()),
        "dynamic_binary_override_rate": _rate(int(dynamic_override.gt(0).sum()), total),
        "high_risk_nonurban_positive_count": int(positive_nonurban.sum()),
        "high_risk_nonurban_positive_rate": _rate(int(positive_nonurban.sum()), total),
        "llm_adjudication_required_count": int(review_required.sum()),
        "llm_adjudication_required_rate": _rate(int(review_required.sum()), total),
    }


def build_urban_diagnostics_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return diagnostics as a two-column table suitable for Excel/CSV export."""

    diagnostics = build_urban_diagnostics(frame)
    rows = [{"metric": key, "value": value} for key, value in diagnostics.items()]
    return pd.DataFrame(rows, columns=["metric", "value"])


def build_urban_error_attribution(frame: pd.DataFrame, *, truth_column: str) -> pd.DataFrame:
    """Group false positives/negatives by strategy boundary fields.

    The report is read-only: it never changes prediction outputs and only uses
    columns already present in a prediction workbook.
    """

    columns = [
        "error_type",
        "decision_source",
        "boundary_bucket",
        "topic_final",
        "binary_policy_action",
        "dynamic_binary_override_applied",
        "count",
    ]
    if frame.empty or truth_column not in frame.columns:
        return pd.DataFrame(columns=columns)

    truth = _binary_label_series(frame, truth_column)
    prediction = _binary_label_series(frame, "final_label")
    fallback_prediction = _binary_label_series(frame, "urban_flag")
    prediction = prediction.where(prediction.isin({"0", "1"}), fallback_prediction)

    valid = truth.isin({"0", "1"}) & prediction.isin({"0", "1"})
    false_positive = valid & truth.eq("0") & prediction.eq("1")
    false_negative = valid & truth.eq("1") & prediction.eq("0")
    error_mask = false_positive | false_negative
    if not bool(error_mask.any()):
        return pd.DataFrame(columns=columns)

    error_rows = pd.DataFrame(
        {
            "error_type": pd.Series("", index=frame.index, dtype=object),
            "decision_source": _string_series(frame, "decision_source"),
            "boundary_bucket": _string_series(frame, "boundary_bucket"),
            "topic_final": _string_series(frame, "topic_final"),
            "binary_policy_action": _string_series(frame, "binary_policy_action"),
            "dynamic_binary_override_applied": _string_series(frame, "dynamic_binary_override_applied"),
        }
    )
    error_rows.loc[false_positive, "error_type"] = "false_positive"
    error_rows.loc[false_negative, "error_type"] = "false_negative"
    error_rows = error_rows.loc[error_mask].copy()

    grouped = (
        error_rows.groupby(columns[:-1], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["error_type", "count"], ascending=[True, False], kind="stable")
        .reset_index(drop=True)
    )
    return grouped[columns]


def _string_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype=object)
    return frame[column].fillna("").astype(str).str.strip()


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([0] * len(frame), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0)


def _value_counts(series: pd.Series) -> dict[str, int]:
    normalized = series.fillna("").astype(str).str.strip()
    normalized = normalized[normalized.ne("")]
    return {str(key): int(value) for key, value in normalized.value_counts(sort=False).to_dict().items()}


def _binary_label_series(frame: pd.DataFrame, column: str) -> pd.Series:
    values = _string_series(frame, column)
    values = values.str.replace(r"\.0$", "", regex=True)
    return values.where(values.isin({"0", "1"}), "")


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(float(count) / float(total), 6)
