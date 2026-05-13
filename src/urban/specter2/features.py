from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from src.runtime.config import Schema


def records_from_dataframe(
    frame: pd.DataFrame,
    *,
    title_col: str = Schema.TITLE,
    abstract_col: str = Schema.ABSTRACT,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    title_values = frame[title_col] if title_col in frame.columns else pd.Series([""] * len(frame))
    abstract_values = frame[abstract_col] if abstract_col in frame.columns else pd.Series([""] * len(frame))
    for title, abstract in zip(title_values, abstract_values):
        records.append({"title": _clean_text(title), "abstract": _clean_text(abstract)})
    return records


def normalize_record(record: dict[str, object]) -> dict[str, str]:
    title = record.get("title", record.get(Schema.TITLE, ""))
    abstract = record.get("abstract", record.get(Schema.ABSTRACT, ""))
    return {"title": _clean_text(title), "abstract": _clean_text(abstract)}


def build_embedding_matrix(embeddings: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {array.shape}")
    return np.nan_to_num(array, copy=False)


def baseline_probability_from_predictions(values: Iterable[object]) -> np.ndarray:
    probabilities = []
    for value in values:
        probabilities.append(_coerce_probability(value))
    return np.asarray(probabilities, dtype=np.float32)


def build_hybrid_feature_matrix(
    embeddings: Sequence[Sequence[float]] | np.ndarray,
    baseline_probabilities: Iterable[object],
) -> np.ndarray:
    embedding_matrix = build_embedding_matrix(embeddings)
    baseline = baseline_probability_from_predictions(baseline_probabilities).reshape(-1, 1)
    if len(embedding_matrix) != len(baseline):
        raise ValueError(
            "Embedding and baseline feature row counts differ: "
            f"{len(embedding_matrix)} != {len(baseline)}"
        )
    return np.hstack([embedding_matrix, baseline]).astype(np.float32, copy=False)


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _coerce_probability(value: object) -> float:
    if value is None or pd.isna(value):
        return 0.0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "1.0", "true", "yes"}:
            return 1.0
        if text in {"0", "0.0", "false", "no", ""}:
            return 0.0
        try:
            numeric = float(text)
        except ValueError:
            return 0.0
    else:
        numeric = float(value)
    return float(min(max(numeric, 0.0), 1.0))
