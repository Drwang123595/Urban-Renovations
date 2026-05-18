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


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(float(count) / float(total), 6)
