from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .features import (
    baseline_probability_from_predictions,
    build_embedding_matrix,
    build_hybrid_feature_matrix,
)


@dataclass(frozen=True)
class AblationThresholds:
    f1_min_exclusive: float = 0.951553
    precision_min: float = 0.956
    recall_min: float = 0.94
    fp_max: int = 34
    fn_max: int = 48


@dataclass(frozen=True)
class AblationResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    gate_summary: dict[str, object]


def evaluate_specter2_ablation(
    *,
    truth: Iterable[object],
    baseline_pred: Iterable[object],
    embeddings: np.ndarray,
    baseline_probability: Iterable[object] | None = None,
    thresholds: AblationThresholds | None = None,
    random_state: int = 42,
    cv_splits: int = 5,
) -> AblationResult:
    y_true = _normalize_binary_array(truth, name="truth")
    baseline_labels = _normalize_binary_array(baseline_pred, name="baseline_pred")
    embedding_matrix = build_embedding_matrix(embeddings)
    if len(embedding_matrix) != len(y_true):
        raise ValueError(f"Embedding row count {len(embedding_matrix)} does not match truth rows {len(y_true)}")
    if len(baseline_labels) != len(y_true):
        raise ValueError(f"Baseline row count {len(baseline_labels)} does not match truth rows {len(y_true)}")

    baseline_prob = (
        baseline_probability_from_predictions(baseline_probability)
        if baseline_probability is not None
        else baseline_labels.astype(np.float32)
    )
    if len(baseline_prob) != len(y_true):
        raise ValueError(f"Baseline probability row count {len(baseline_prob)} does not match truth rows {len(y_true)}")

    specter2_prob = _linear_probabilities(
        embedding_matrix,
        y_true,
        random_state=random_state,
        cv_splits=cv_splits,
    )
    specter2_pred = (specter2_prob >= 0.5).astype(int)

    hybrid_matrix = build_hybrid_feature_matrix(embedding_matrix, baseline_prob)
    hybrid_prob = _linear_probabilities(
        hybrid_matrix,
        y_true,
        random_state=random_state,
        cv_splits=cv_splits,
    )
    hybrid_pred = (hybrid_prob >= 0.5).astype(int)

    metric_rows = [
        compute_binary_metrics(y_true, baseline_labels, group="baseline"),
        compute_binary_metrics(y_true, specter2_pred, group="specter2_only"),
        compute_binary_metrics(y_true, hybrid_pred, group="hybrid"),
    ]
    metrics = pd.DataFrame(metric_rows)
    gate_summary = evaluate_hybrid_gate(metrics, thresholds or AblationThresholds())
    predictions = pd.DataFrame(
        {
            "truth": y_true,
            "baseline_pred": baseline_labels,
            "baseline_probability": baseline_prob,
            "specter2_only_probability": specter2_prob,
            "specter2_only_pred": specter2_pred,
            "hybrid_probability": hybrid_prob,
            "hybrid_pred": hybrid_pred,
        }
    )
    return AblationResult(metrics=metrics, predictions=predictions, gate_summary=gate_summary)


def compute_binary_metrics(truth: Iterable[object], pred: Iterable[object], *, group: str) -> dict[str, object]:
    y_true = _normalize_binary_array(truth, name="truth")
    y_pred = _normalize_binary_array(pred, name="pred")
    if len(y_true) != len(y_pred):
        raise ValueError(f"Metric row count mismatch: {len(y_true)} != {len(y_pred)}")

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    total = int(len(y_true))
    correct = int(tp + tn)
    accuracy = correct / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "Group": group,
        "Total": total,
        "Correct": correct,
        "Accuracy": round(float(accuracy), 6),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Precision": round(float(precision), 6),
        "Recall": round(float(recall), 6),
        "F1": round(float(f1), 6),
    }


def evaluate_hybrid_gate(metrics: pd.DataFrame, thresholds: AblationThresholds) -> dict[str, object]:
    hybrid = metrics.loc[metrics["Group"] == "hybrid"]
    if hybrid.empty:
        return {"passes": False, "reason": "hybrid metrics are missing"}
    row = hybrid.iloc[0]
    checks = {
        "f1_gt_baseline": float(row["F1"]) > thresholds.f1_min_exclusive,
        "precision_min": float(row["Precision"]) >= thresholds.precision_min,
        "recall_min": float(row["Recall"]) >= thresholds.recall_min,
        "fp_max": int(row["FP"]) <= thresholds.fp_max,
        "fn_max": int(row["FN"]) <= thresholds.fn_max,
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "F1": f"> {thresholds.f1_min_exclusive}",
            "Precision": f">= {thresholds.precision_min}",
            "Recall": f">= {thresholds.recall_min}",
            "FP": f"<= {thresholds.fp_max}",
            "FN": f"<= {thresholds.fn_max}",
        },
    }


def _linear_probabilities(
    features: np.ndarray,
    y_true: np.ndarray,
    *,
    random_state: int,
    cv_splits: int,
) -> np.ndarray:
    if len(y_true) == 0:
        return np.zeros(0, dtype=np.float32)
    classes = np.unique(y_true)
    if len(classes) < 2:
        return np.full(len(y_true), float(classes[0]), dtype=np.float32)

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return _centroid_probabilities(features, y_true)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
            solver="liblinear",
        ),
    )
    counts = np.bincount(y_true, minlength=2)
    min_class_count = int(counts[counts > 0].min())
    splits = min(int(cv_splits), min_class_count)

    if splits >= 2:
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_state)
        probabilities = cross_val_predict(model, features, y_true, cv=cv, method="predict_proba")
        return probabilities[:, 1].astype(np.float32, copy=False)

    model.fit(features, y_true)
    return model.predict_proba(features)[:, 1].astype(np.float32, copy=False)


def _centroid_probabilities(features: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    positive = features[y_true == 1]
    negative = features[y_true == 0]
    if len(positive) == 0:
        return np.zeros(len(y_true), dtype=np.float32)
    if len(negative) == 0:
        return np.ones(len(y_true), dtype=np.float32)
    positive_center = positive.mean(axis=0)
    negative_center = negative.mean(axis=0)
    pos_distance = np.linalg.norm(features - positive_center, axis=1)
    neg_distance = np.linalg.norm(features - negative_center, axis=1)
    score = neg_distance - pos_distance
    return (1.0 / (1.0 + np.exp(-score))).astype(np.float32, copy=False)


def _normalize_binary_array(values: Iterable[object], *, name: str) -> np.ndarray:
    normalized = []
    for value in values:
        normalized.append(_normalize_binary_value(value, name=name))
    return np.asarray(normalized, dtype=int)


def _normalize_binary_value(value: object, *, name: str) -> int:
    if value is None or pd.isna(value):
        raise ValueError(f"{name} contains missing binary value")
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes"}:
        return 1
    if text in {"0", "0.0", "false", "no"}:
        return 0
    raise ValueError(f"{name} contains non-binary value: {value!r}")
