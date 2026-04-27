from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ..runtime.config import Config, Schema
from .urban_bertopic_service import BERTopicSignal
from .urban_metadata import UrbanMetadataRecord, normalize_phrase
from .urban_rule_filter import R7_STRONG_RENEWAL_MECHANISMS
from .urban_topic_taxonomy import (
    COMMON_EXISTING_URBAN_OBJECTS,
    COMMON_RENEWAL_ANCHORS,
    UNKNOWN_TOPIC_GROUP,
    topic_group_for_label,
)
from .urban_training_contract import allowed_training_workbooks


MODEL_FEATURE_COLUMNS = [
    "rule_is_urban",
    "rule_is_nonurban",
    "rule_score",
    "rule_margin",
    "local_is_urban",
    "local_is_nonurban",
    "local_confidence",
    "local_margin",
    "topic_binary_probability",
    "bertopic_is_urban",
    "bertopic_is_nonurban",
    "bertopic_probability",
    "bertopic_label_purity",
    "bertopic_mapped_label_share",
    "bertopic_count",
    "risk_tag_count",
    "anchor_hit_count",
    "object_hit_count",
    "mechanism_hit_count",
    "text_token_count",
    "cross_group_conflict",
    "stage1_conflict_flag",
    "llm_hint_urban",
    "llm_hint_nonurban",
]


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def _family_from_group(group: str) -> str:
    normalized = str(group or "").strip().lower()
    if normalized == "urban":
        return "urban"
    if normalized == "nonurban":
        return "nonurban"
    return UNKNOWN_TOPIC_GROUP


def _empty_bertopic_signal() -> BERTopicSignal:
    return BERTopicSignal(
        available=False,
        status="disabled",
        topic_id=-1,
        topic_name="",
        topic_probability=0.0,
        is_outlier=False,
        topic_count=0,
        topic_pos_rate="",
        mapped_label="",
        mapped_group="",
        mapped_name="",
        label_purity=0.0,
        mapped_label_share=0.0,
        top_terms="",
        sample_titles="",
        source_split="",
        reason="",
    )


@dataclass
class FamilyGateDecision:
    rule_family: str
    local_family: str
    final_family: str
    confidence: float
    probability_urban: float
    decision_source: str
    boundary_bucket: str
    family_conflict_pattern: str
    features: Dict[str, float]


class UrbanFamilyGate:
    def __init__(
        self,
        *,
        model_path: Optional[Path] = None,
        enabled: Optional[bool] = None,
    ):
        self.enabled = Config.URBAN_FAMILY_GATE_ENABLED if enabled is None else bool(enabled)
        self.model_path = Path(model_path or Config.URBAN_FAMILY_GATE_MODEL_PATH)
        self.feature_columns = list(MODEL_FEATURE_COLUMNS)
        self.model = None
        self.model_metadata: Dict[str, Any] = {}
        if self.enabled and self.model_path.exists():
            self._load_model()

    def _load_model(self) -> None:
        try:
            from joblib import load
        except Exception:
            self.model = None
            self.model_metadata = {}
            return

        payload = load(self.model_path)
        self.model = payload.get("model")
        self.feature_columns = list(payload.get("feature_columns", MODEL_FEATURE_COLUMNS))
        self.model_metadata = dict(payload.get("metadata", {}))

    def save_model(self, model, *, metadata: Dict[str, Any] | None = None) -> Path:
        from joblib import dump

        payload = {
            "model": model,
            "feature_columns": list(self.feature_columns),
            "metadata": dict(metadata or {}),
        }
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        dump(payload, self.model_path)
        self.model = model
        self.model_metadata = payload["metadata"]
        return self.model_path

    def _count_phrase_hits(self, normalized_text: str, phrases: Iterable[str]) -> int:
        hits = {
            normalize_phrase(phrase).replace("-", " ")
            for phrase in phrases
            if normalize_phrase(phrase).replace("-", " ") in normalized_text
        }
        return len(hits)

    def build_features(
        self,
        *,
        record: UrbanMetadataRecord,
        route_result,
        topic_prediction,
        bertopic_signal: Optional[BERTopicSignal] = None,
        llm_family_hint: Any = "",
    ) -> Dict[str, float]:
        bertopic_signal = bertopic_signal or _empty_bertopic_signal()
        normalized_text = normalize_phrase(record.title_abstract_text()).replace("-", " ")
        rule_family = _family_from_group(route_result.topic_rule_group or topic_group_for_label(route_result.topic_rule))
        local_family = _family_from_group(topic_prediction.topic_group or topic_group_for_label(topic_prediction.topic_label))
        bertopic_family = _family_from_group(bertopic_signal.mapped_group)
        llm_hint = str(llm_family_hint or "").strip().replace(".0", "")

        features = {
            "rule_is_urban": 1.0 if rule_family == "urban" else 0.0,
            "rule_is_nonurban": 1.0 if rule_family == "nonurban" else 0.0,
            "rule_score": float(route_result.topic_rule_score or 0.0),
            "rule_margin": float(route_result.topic_rule_margin or 0.0),
            "local_is_urban": 1.0 if local_family == "urban" else 0.0,
            "local_is_nonurban": 1.0 if local_family == "nonurban" else 0.0,
            "local_confidence": float(topic_prediction.confidence or 0.0),
            "local_margin": float(topic_prediction.margin or 0.0),
            "topic_binary_probability": float(topic_prediction.binary_probability or 0.0),
            "bertopic_is_urban": 1.0 if bertopic_family == "urban" else 0.0,
            "bertopic_is_nonurban": 1.0 if bertopic_family == "nonurban" else 0.0,
            "bertopic_probability": float(bertopic_signal.topic_probability or 0.0),
            "bertopic_label_purity": float(bertopic_signal.label_purity or 0.0),
            "bertopic_mapped_label_share": float(bertopic_signal.mapped_label_share or 0.0),
            "bertopic_count": float(bertopic_signal.topic_count or 0.0),
            "risk_tag_count": float(len(route_result.stage1_risk_tags or [])),
            "anchor_hit_count": float(self._count_phrase_hits(normalized_text, COMMON_RENEWAL_ANCHORS)),
            "object_hit_count": float(self._count_phrase_hits(normalized_text, COMMON_EXISTING_URBAN_OBJECTS)),
            "mechanism_hit_count": float(self._count_phrase_hits(normalized_text, R7_STRONG_RENEWAL_MECHANISMS)),
            "text_token_count": float(len(normalized_text.split())),
            "cross_group_conflict": float(
                rule_family in {"urban", "nonurban"}
                and local_family in {"urban", "nonurban"}
                and rule_family != local_family
            ),
            "stage1_conflict_flag": float(int(route_result.stage1_conflict_flag or 0)),
            "llm_hint_urban": 1.0 if llm_hint == "1" else 0.0,
            "llm_hint_nonurban": 1.0 if llm_hint == "0" else 0.0,
        }
        return features

    def _heuristic_probability(self, features: Dict[str, float]) -> float:
        score = 0.0
        score += (features["topic_binary_probability"] - 0.5) * 3.6
        score += features["rule_is_urban"] * min(features["rule_score"] / 6.5, 1.0) * 1.0
        score -= features["rule_is_nonurban"] * min(features["rule_score"] / 6.5, 1.0) * 1.0
        score += features["local_is_urban"] * features["local_confidence"] * 1.3
        score -= features["local_is_nonurban"] * features["local_confidence"] * 1.3
        score += features["bertopic_is_urban"] * min(features["bertopic_label_purity"], 1.0) * 0.5
        score -= features["bertopic_is_nonurban"] * min(features["bertopic_label_purity"], 1.0) * 0.5
        score += min(features["anchor_hit_count"], 3.0) * 0.12
        score += min(features["object_hit_count"], 3.0) * 0.12
        score += min(features["mechanism_hit_count"], 3.0) * 0.14
        score -= features["risk_tag_count"] * 0.1

        if features["cross_group_conflict"]:
            score *= 0.82
        if features["llm_hint_urban"] and (features["rule_is_urban"] or features["local_is_urban"]):
            score += 0.2
        if features["llm_hint_nonurban"] and (features["rule_is_nonurban"] or features["local_is_nonurban"]):
            score -= 0.2
        return _sigmoid(score)

    def _predict_probability(self, features: Dict[str, float]) -> tuple[float, str]:
        if self.model is not None:
            import pandas as pd

            frame = pd.DataFrame([{name: features.get(name, 0.0) for name in self.feature_columns}])
            prob = float(self.model.predict_proba(frame)[0][1])
            return prob, "family_gate_model"
        return self._heuristic_probability(features), "family_gate_heuristic"

    def describe_conflict(
        self,
        *,
        rule_label: str,
        local_label: str,
        rule_family: str,
        local_family: str,
    ) -> tuple[str, str]:
        family_pattern = f"{rule_label or 'Unknown'}_vs_{local_label or 'Unknown'}"
        if rule_family == "unknown" and local_family == "unknown":
            return "double_unknown", family_pattern
        if rule_family == "urban" and local_family == "nonurban":
            if rule_label in {"U9", "U10"} and local_label in {"N3", "N4", "N8"}:
                return "governance_policy_finance_boundary", family_pattern
            if rule_label in {"U12", "U15", "U3"} and local_label in {"N4", "N5", "N7", "N9"}:
                return "social_impact_boundary", family_pattern
            if rule_label in {"U5"} and local_label in {"N2", "N9", "N10"}:
                return "brownfield_environment_boundary", family_pattern
            return "urban_rule_nonurban_local", family_pattern
        if rule_family == "nonurban" and local_family == "urban":
            if local_label in {"U9", "U10"} and rule_label in {"N3", "N4", "N8"}:
                return "governance_policy_finance_boundary", family_pattern
            if local_label in {"U12", "U15", "U3"} and rule_label in {"N4", "N5", "N7", "N9"}:
                return "social_impact_boundary", family_pattern
            if local_label in {"U5"} and rule_label in {"N2", "N9", "N10"}:
                return "brownfield_environment_boundary", family_pattern
            return "nonurban_rule_urban_local", family_pattern
        if "U10" in family_pattern or "N8" in family_pattern:
            return "method_boundary", family_pattern
        return "same_family_or_single_source", family_pattern

    def predict(
        self,
        *,
        record: UrbanMetadataRecord,
        route_result,
        topic_prediction,
        bertopic_signal: Optional[BERTopicSignal] = None,
        llm_family_hint: Any = "",
    ) -> FamilyGateDecision:
        features = self.build_features(
            record=record,
            route_result=route_result,
            topic_prediction=topic_prediction,
            bertopic_signal=bertopic_signal,
            llm_family_hint=llm_family_hint,
        )
        probability_urban, decision_source = self._predict_probability(features)
        if probability_urban >= float(Config.URBAN_FAMILY_GATE_THRESHOLD_URBAN):
            final_family = "urban"
        elif probability_urban <= float(Config.URBAN_FAMILY_GATE_THRESHOLD_NONURBAN):
            final_family = "nonurban"
        else:
            final_family = UNKNOWN_TOPIC_GROUP

        rule_family = _family_from_group(route_result.topic_rule_group or topic_group_for_label(route_result.topic_rule))
        local_family = _family_from_group(topic_prediction.topic_group or topic_group_for_label(topic_prediction.topic_label))
        boundary_bucket, conflict_pattern = self.describe_conflict(
            rule_label=str(route_result.topic_rule or ""),
            local_label=str(topic_prediction.topic_label or ""),
            rule_family=rule_family,
            local_family=local_family,
        )
        return FamilyGateDecision(
            rule_family=rule_family,
            local_family=local_family,
            final_family=final_family,
            confidence=round(abs(probability_urban - 0.5) * 2.0, 4),
            probability_urban=round(probability_urban, 6),
            decision_source=decision_source,
            boundary_bucket=boundary_bucket,
            family_conflict_pattern=conflict_pattern,
            features=features,
        )

    def fit_from_allowed_workbooks(
        self,
        *,
        rule_filter,
        topic_classifier,
        bertopic_service=None,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        import pandas as pd
        try:
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
        except Exception as error:
            raise RuntimeError(
                "scikit-learn is required to fit the urban family gate calibrator. "
                "Install project dev dependencies or run under the project Py3.13 environment."
            ) from error

        rows = []
        sources = []
        for path in allowed_training_workbooks(Config.TRAIN_DIR):
            try:
                df = pd.read_excel(path, engine="openpyxl")
            except Exception:
                continue
            label_col = topic_classifier._detect_label_column(df)  # noqa: SLF001 - internal contract reuse
            if not label_col or Schema.TITLE not in df.columns or Schema.ABSTRACT not in df.columns:
                continue
            subset = df[[Schema.TITLE, Schema.ABSTRACT, label_col]].copy()
            subset = subset[subset[label_col].isin([0, 1, "0", "1"])].copy()
            if subset.empty:
                continue
            subset[label_col] = subset[label_col].astype(int)
            sources.append(str(path.resolve()))
            for _, row in subset.iterrows():
                record = UrbanMetadataRecord.from_row(row.to_dict())
                route_result = rule_filter.evaluate(record)
                topic_prediction = topic_classifier.predict(record)
                bertopic_signal = bertopic_service.predict(record) if bertopic_service else _empty_bertopic_signal()
                features = self.build_features(
                    record=record,
                    route_result=route_result,
                    topic_prediction=topic_prediction,
                    bertopic_signal=bertopic_signal,
                    llm_family_hint="",
                )
                features["label"] = int(row[label_col])
                rows.append(features)

        train_df = pd.DataFrame(rows)
        if train_df.empty or train_df["label"].nunique() < 2:
            raise ValueError("Not enough allowed training data to fit urban family gate.")

        X = train_df[self.feature_columns]
        y = train_df["label"].astype(int)
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        model.fit(X, y)
        metadata = {
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "training_sources": sources,
            "sample_count": int(len(train_df)),
            "positive_rate": round(float(y.mean()), 6),
        }
        if output_path is not None:
            self.model_path = Path(output_path)
        self.save_model(model, metadata=metadata)
        return metadata
