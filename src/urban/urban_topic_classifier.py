from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

from ..runtime.config import Config, Schema
from .urban_metadata import UrbanMetadataRecord, tokenize_text
from .urban_training_contract import allowed_training_workbooks, assert_training_source_contract
from .urban_topic_taxonomy import (
    TOPIC_DEFINITIONS,
    UNKNOWN_TOPIC_GROUP,
    UNKNOWN_TOPIC_LABEL,
    UNKNOWN_TOPIC_NAME,
    score_all_topics,
    topic_group_for_label,
    topic_name_for_label,
)


@dataclass
class TopicPrediction:
    topic_label: str
    topic_group: str
    topic_name: str
    confidence: float
    matched_terms: List[str]
    binary_score: float
    binary_probability: float
    margin: float = 0.0
    top_candidates: List[str] = field(default_factory=list)
    scored_topics: List[tuple[str, float]] = field(default_factory=list)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


SHORT_TEXT_TOKEN_THRESHOLD = 8
MEDIUM_TEXT_TOKEN_THRESHOLD = 18
UNKNOWN_SCORE_THRESHOLD = 2.4
UNKNOWN_MARGIN_THRESHOLD = 1.1
TOPIC_UNKNOWN_POLICIES: Dict[str, Dict[str, float]] = {
    "N2": {"score": 2.85, "margin": 1.25, "urban_binary_floor": 0.53},
    "N3": {"score": 2.9, "margin": 1.35, "urban_binary_floor": 0.53},
    "N4": {"score": 2.9, "margin": 1.35, "urban_binary_floor": 0.55},
    "N8": {"score": 2.85, "margin": 1.25, "urban_binary_floor": 0.53},
    "U3": {"score": 2.8, "margin": 1.15, "urban_binary_ceiling": 0.56, "binary_ceiling_score_floor": 5.5},
    "U10": {"score": 2.95, "margin": 1.2, "urban_binary_ceiling": 0.58, "binary_ceiling_score_floor": 5.5},
    "U14": {"score": 2.85, "margin": 1.15, "urban_binary_ceiling": 0.57, "binary_ceiling_score_floor": 5.5},
    "U15": {"score": 2.95, "margin": 1.2, "urban_binary_ceiling": 0.58, "binary_ceiling_score_floor": 5.5},
}


class TokenOddsUrbanModel:
    def __init__(self, min_count: int = 2):
        self.min_count = max(1, min_count)
        self.weights: Dict[str, float] = {}
        self.bias = 0.0
        self.fitted = False

    def fit(self, records: Sequence[UrbanMetadataRecord], labels: Sequence[int]):
        pos_counts: Dict[str, int] = {}
        neg_counts: Dict[str, int] = {}
        pos_docs = 0
        neg_docs = 0
        for record, label in zip(records, labels):
            tokens = set(tokenize_text(record.title_abstract_weighted_text()))
            if not tokens:
                continue
            if int(label) == 1:
                pos_docs += 1
                for token in tokens:
                    pos_counts[token] = pos_counts.get(token, 0) + 1
            else:
                neg_docs += 1
                for token in tokens:
                    neg_counts[token] = neg_counts.get(token, 0) + 1

        vocab = {
            token
            for token in set(pos_counts) | set(neg_counts)
            if pos_counts.get(token, 0) + neg_counts.get(token, 0) >= self.min_count
        }
        if not vocab:
            self.weights = {}
            self.bias = 0.0
            self.fitted = True
            return

        self.bias = math.log((pos_docs + 1) / (neg_docs + 1))
        self.weights = {}
        for token in vocab:
            pos = pos_counts.get(token, 0)
            neg = neg_counts.get(token, 0)
            self.weights[token] = math.log((pos + 1) / (pos_docs + 2)) - math.log(
                (neg + 1) / (neg_docs + 2)
            )
        self.fitted = True

    def predict_score(self, record: UrbanMetadataRecord) -> float:
        if not self.fitted:
            return 0.0
        score = self.bias
        for token in set(tokenize_text(record.title_abstract_weighted_text())):
            score += self.weights.get(token, 0.0)
        return score

    def predict_probability(self, record: UrbanMetadataRecord) -> float:
        score = max(min(self.predict_score(record) / 4.0, 12.0), -12.0)
        return _sigmoid(score)


class UrbanTopicClassifier:
    def __init__(
        self,
        *,
        train_dir: Optional[Path] = None,
        master_metadata_path: Optional[Path] = None,
    ):
        self.train_dir = train_dir or Config.TRAIN_DIR
        self.master_metadata_path = master_metadata_path or (self.train_dir / "Urban Renovation V2.0.xlsx")
        self.binary_model = TokenOddsUrbanModel(min_count=2)
        self.training_sources: List[Path] = []
        self._fit_binary_model()

    def _iter_training_files(self) -> Iterable[Path]:
        self.training_sources = [Path(path) for path in allowed_training_workbooks(self.train_dir)]
        return list(self.training_sources)

    def _detect_label_column(self, df: pd.DataFrame) -> Optional[str]:
        preferred = [
            Schema.IS_URBAN_RENEWAL,
            "Label_UrbanRenewal",
            "urban_label",
            "is_urban_renewal",
        ]
        for candidate in preferred:
            if candidate in df.columns:
                return candidate
        for column in df.columns:
            text = str(column)
            lowered = text.lower()
            if lowered in {"urban_parse_reason", "decision_source", "decision_reason"}:
                continue
            if "urban" in lowered and "renew" in lowered:
                return column
        return None

    def _fit_binary_model(self):
        training_records: List[UrbanMetadataRecord] = []
        labels: List[int] = []
        seen_titles = set()
        assert_training_source_contract(self._iter_training_files())

        for path in self._iter_training_files():
            try:
                header = pd.read_excel(path, engine="openpyxl", nrows=5)
            except Exception:
                continue
            if Schema.TITLE not in header.columns or Schema.ABSTRACT not in header.columns:
                continue
            label_col = self._detect_label_column(header)
            if not label_col:
                continue
            try:
                df = pd.read_excel(path, engine="openpyxl")
            except Exception:
                continue

            sub = df[[Schema.TITLE, Schema.ABSTRACT, label_col]].copy()
            sub = sub[sub[label_col].isin([0, 1, "0", "1"])]
            sub[label_col] = sub[label_col].astype(int)

            for _, row in sub.iterrows():
                title = str(row.get(Schema.TITLE, "") or "")
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)
                training_records.append(UrbanMetadataRecord.from_row(row.to_dict()))
                labels.append(int(row[label_col]))

        if training_records and labels:
            self.binary_model.fit(training_records, labels)

    def _score_topics(
        self,
        record: UrbanMetadataRecord,
        *,
        binary_probability: float,
    ) -> List[Dict[str, object]]:
        scored = score_all_topics(title=record.title, abstract=record.abstract)
        binary_alignment = max(min((binary_probability - 0.5) * 2.0, 1.0), -1.0)
        adjusted: List[Dict[str, object]] = []
        for item in scored:
            topic_label = str(item["label"])
            definition = TOPIC_DEFINITIONS[topic_label]
            score = float(item["score"])
            if definition["group"] == "urban":
                score += max(binary_alignment, 0.0) * 1.5
            else:
                score += max(-binary_alignment, 0.0) * 1.5
            adjusted.append(
                {
                    **item,
                    "score": round(score, 4),
                }
            )
        adjusted.sort(key=lambda item: (-float(item["score"]), str(item["label"])))
        return adjusted

    def _compute_confidence(
        self,
        *,
        record: UrbanMetadataRecord,
        best_score: float,
        margin: float,
        matched_terms: Sequence[str],
        binary_probability: float,
        is_unknown: bool,
    ) -> float:
        token_count = len(tokenize_text(record.title_abstract_text()))
        score_signal = min(max(best_score, 0.0) / 8.0, 1.0)
        margin_signal = min(max(margin, 0.0) / 4.0, 1.0)
        binary_signal = min(abs(binary_probability - 0.5) * 2.0, 1.0)

        confidence = 0.16 + 0.36 * score_signal + 0.22 * margin_signal + 0.16 * binary_signal
        if matched_terms:
            confidence += 0.05
        if is_unknown:
            confidence = min(confidence, 0.49)

        if token_count < SHORT_TEXT_TOKEN_THRESHOLD:
            confidence = min(confidence, 0.45)
        elif token_count < MEDIUM_TEXT_TOKEN_THRESHOLD:
            confidence = min(confidence, 0.68)

        return round(min(max(confidence, 0.2), 0.99), 4)

    def _should_mark_unknown(
        self,
        *,
        best_label: str,
        best_score: float,
        margin: float,
        binary_probability: float,
    ) -> bool:
        weak_binary_signal = abs(binary_probability - 0.5) < 0.12
        if (
            best_score < UNKNOWN_SCORE_THRESHOLD
            or (margin < UNKNOWN_MARGIN_THRESHOLD and weak_binary_signal)
            or (best_score <= 0.0 and weak_binary_signal)
        ):
            return True

        policy = TOPIC_UNKNOWN_POLICIES.get(best_label)
        if not policy:
            return False

        if best_score < float(policy.get("score", UNKNOWN_SCORE_THRESHOLD)):
            return True
        if margin < float(policy.get("margin", UNKNOWN_MARGIN_THRESHOLD)):
            return True
        urban_binary_floor = policy.get("urban_binary_floor")
        if urban_binary_floor is not None and binary_probability >= float(urban_binary_floor):
            return True
        urban_binary_ceiling = policy.get("urban_binary_ceiling")
        binary_ceiling_score_floor = float(policy.get("binary_ceiling_score_floor", float("inf")))
        if (
            urban_binary_ceiling is not None
            and binary_probability <= float(urban_binary_ceiling)
            and best_score < binary_ceiling_score_floor
        ):
            return True
        return False

    def predict(self, record: UrbanMetadataRecord) -> TopicPrediction:
        binary_score = self.binary_model.predict_score(record)
        binary_probability = self.binary_model.predict_probability(record)
        topic_scores = self._score_topics(record, binary_probability=binary_probability)

        best = topic_scores[0]
        best_label = str(best["label"])
        best_score = float(best["score"])
        matched_terms = list(best.get("matched_terms", []))
        second_score = float(topic_scores[1]["score"]) if len(topic_scores) > 1 else 0.0
        margin = round(max(best_score - second_score, 0.0), 4)

        should_mark_unknown = self._should_mark_unknown(
            best_label=best_label,
            best_score=best_score,
            margin=margin,
            binary_probability=binary_probability,
        )

        if should_mark_unknown:
            best_label = UNKNOWN_TOPIC_LABEL
            matched_terms = matched_terms[:8]

        confidence = self._compute_confidence(
            record=record,
            best_score=best_score,
            margin=margin,
            matched_terms=matched_terms,
            binary_probability=binary_probability,
            is_unknown=(best_label == UNKNOWN_TOPIC_LABEL),
        )

        if best_label == UNKNOWN_TOPIC_LABEL:
            topic_group = UNKNOWN_TOPIC_GROUP
            topic_name = UNKNOWN_TOPIC_NAME
        else:
            topic_group = topic_group_for_label(best_label)
            topic_name = topic_name_for_label(best_label)

        return TopicPrediction(
            topic_label=best_label,
            topic_group=topic_group,
            topic_name=topic_name,
            confidence=confidence,
            matched_terms=matched_terms[:10],
            binary_score=round(binary_score, 4),
            binary_probability=round(binary_probability, 4),
            margin=margin,
            top_candidates=[
                f"{item['label']}:{float(item['score']):.2f}"
                for item in topic_scores[:3]
            ],
            scored_topics=[
                (str(item["label"]), float(item["score"]))
                for item in topic_scores
            ],
        )
