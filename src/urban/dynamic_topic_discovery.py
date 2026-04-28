from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Sequence

import pandas as pd

from ..runtime.config import Schema
from .urban_metadata import normalize_phrase
from .urban_topic_taxonomy import (
    COMMON_RENEWAL_ANCHORS,
    OPEN_SET_NONURBAN_LABEL,
    OPEN_SET_URBAN_LABEL,
    TOPIC_DEFINITIONS,
    TOPIC_ENGLISH_NAMES,
    UNKNOWN_TOPIC_LABEL,
    topic_group_for_label,
)


DYNAMIC_TOPIC_COLUMNS = [
    "dynamic_topic_id",
    "dynamic_topic_name_zh",
    "dynamic_topic_keywords",
    "dynamic_topic_size",
    "dynamic_topic_confidence",
    "dynamic_topic_source_pool",
    "dynamic_to_fixed_topic_candidate",
    "dynamic_mapping_status",
]

DYNAMIC_BINARY_COLUMNS = [
    "dynamic_binary_candidate_label",
    "dynamic_binary_candidate_confidence",
    "dynamic_binary_candidate_action",
    "dynamic_binary_candidate_reason",
    "dynamic_binary_review_priority",
]

DYNAMIC_TOPIC_DEFAULTS = {
    "dynamic_topic_id": "",
    "dynamic_topic_name_zh": "",
    "dynamic_topic_keywords": "",
    "dynamic_topic_size": "",
    "dynamic_topic_confidence": "",
    "dynamic_topic_source_pool": "",
    "dynamic_to_fixed_topic_candidate": "",
    "dynamic_mapping_status": "",
}

DYNAMIC_BINARY_DEFAULTS = {
    "dynamic_binary_candidate_label": "",
    "dynamic_binary_candidate_confidence": "",
    "dynamic_binary_candidate_action": "",
    "dynamic_binary_candidate_reason": "",
    "dynamic_binary_review_priority": "",
}

SOURCE_UNKNOWN_POOL = "unknown_pool"
SOURCE_NONURBAN_REVIEW_POOL = "nonurban_review_pool"
SOURCE_REVIEW_POOL = "review_pool"
SOURCE_FULL_CORPUS_POOL = "full_corpus_pool"

STATUS_MAPPED_TO_FIXED = "mapped_to_fixed"
STATUS_CANDIDATE_NEW_URBAN = "candidate_new_urban_topic"
STATUS_CANDIDATE_NEW_NONURBAN = "candidate_new_nonurban_topic"
STATUS_NEEDS_REVIEW = "needs_review"


_EMPTY_VALUES = {"", "nan", "none", "null", "-1", "-1.0"}
_TOKEN_RE = re.compile(r"[a-z][a-z0-9_+-]{2,}")
_URBAN_NAME_HINTS = (
    ("brownfield", "\u68d5\u5730\u518d\u5f00\u53d1"),
    ("industrial", "\u5de5\u4e1a\u7528\u5730\u66f4\u65b0"),
    ("urban_village", "\u57ce\u4e2d\u6751\u6539\u9020"),
    ("village", "\u57ce\u4e2d\u6751\u6539\u9020"),
    ("old_community", "\u8001\u65e7\u5c0f\u533a\u6539\u9020"),
    ("neighborhood", "\u793e\u533a\u66f4\u65b0"),
    ("neighbourhood", "\u793e\u533a\u66f4\u65b0"),
    ("heritage", "\u5386\u53f2\u9057\u4ea7\u6d3b\u5316"),
    ("historic", "\u5386\u53f2\u8857\u533a\u66f4\u65b0"),
    ("gentrification", "\u7ec5\u58eb\u5316\u4e0e\u793e\u533a\u53d8\u5316"),
    ("public_space", "\u516c\u5171\u7a7a\u95f4\u66f4\u65b0"),
    ("street", "\u8857\u9053\u66f4\u65b0"),
    ("finance", "\u66f4\u65b0\u878d\u8d44\u4e0e\u653f\u7b56\u5de5\u5177"),
    ("governance", "\u66f4\u65b0\u6cbb\u7406\u4e0e\u653f\u7b56"),
    ("health", "\u66f4\u65b0\u5065\u5eb7\u6548\u5e94"),
    ("resilience", "\u97e7\u6027\u66f4\u65b0"),
    ("energy", "\u80fd\u6e90\u6539\u9020\u4e0e\u5efa\u7b51\u66f4\u65b0"),
)


@dataclass(frozen=True)
class DynamicTopicConfig:
    min_topic_size: int = 20
    max_topics: int = 60
    max_features: int = 5000
    random_state: int = 20260427
    mapping_min_score: float = 0.12
    include_full_corpus: bool = False
    prefer_sklearn: bool = True


class DynamicTopicDiscovery:
    """Offline post-processing layer for dynamic topic evidence.

    This class is intentionally independent from the binary classifier. It only
    appends dynamic_topic_* evidence fields and never mutates topic_final,
    urban_flag, or final_label.
    """

    def __init__(self, config: DynamicTopicConfig | None = None):
        self.config = config or DynamicTopicConfig()
        self._topic_seed_tokens = self._build_topic_seed_tokens()

    def enrich(self, frame: pd.DataFrame, *, include_full_corpus: bool | None = None) -> pd.DataFrame:
        enriched = frame.copy()
        for column, default_value in {**DYNAMIC_TOPIC_DEFAULTS, **DYNAMIC_BINARY_DEFAULTS}.items():
            if column not in enriched.columns:
                enriched[column] = pd.Series([default_value] * len(enriched), index=enriched.index, dtype=object)
            else:
                enriched[column] = enriched[column].astype(object)

        if enriched.empty:
            return enriched

        include_all = self.config.include_full_corpus if include_full_corpus is None else bool(include_full_corpus)
        source_pools = self._source_pools(enriched, include_full_corpus=include_all)
        candidate_mask = source_pools != ""
        if not bool(candidate_mask.any()):
            return enriched

        candidate_index = list(enriched.index[candidate_mask])
        candidate_docs = [
            self._document_for_row(enriched.loc[index])
            for index in candidate_index
        ]
        cluster_rows = self._cluster_documents(candidate_docs)
        for local_position, cluster_row in cluster_rows.iterrows():
            original_index = candidate_index[int(local_position)]
            for column in DYNAMIC_TOPIC_COLUMNS:
                enriched.at[original_index, column] = cluster_row[column]
            enriched.at[original_index, "dynamic_topic_source_pool"] = source_pools.loc[original_index]
        return self._attach_binary_candidates(enriched)

    def _source_pools(self, frame: pd.DataFrame, *, include_full_corpus: bool) -> pd.Series:
        pools = pd.Series([""] * len(frame), index=frame.index, dtype=object)
        topic_final = self._series(frame, ["topic_final", "topic_label"]).str.strip()
        topic_group = self._series(frame, ["topic_final_group", "topic_group"]).str.strip().str.lower()
        taxonomy_status = self._series(frame, ["taxonomy_coverage_status"]).str.strip().str.lower()
        review_reason = self._series(frame, ["review_reason_raw", "review_reason"]).str.lower()
        decision_source = self._series(frame, ["binary_decision_source", "decision_source"]).str.strip().str.lower()
        final_label = self._series(frame, ["final_label", "urban_flag", Schema.IS_URBAN_RENEWAL]).str.strip()

        review_flag_raw = self._numeric_series(frame, ["review_flag_raw", "review_flag"])
        topic_unknown = topic_final.str.lower().eq(UNKNOWN_TOPIC_LABEL.lower()) | topic_group.eq("unknown")
        status_unknown = taxonomy_status.isin({"unknown", "open_set", "binary_resolved"})
        review_triggered = review_flag_raw > 0
        review_text_triggered = review_reason.str.contains(
            "unknown|open_set|near_threshold|conflict|inconsistency|uncertain",
            regex=True,
            na=False,
        )
        source_triggered = decision_source.str.contains(
            "unknown|review|uncertain|anchor_guard",
            regex=True,
            na=False,
        )

        unknown_mask = topic_unknown | taxonomy_status.eq("unknown") | decision_source.eq("unknown_review")
        nonurban_review_mask = (
            (final_label.isin({"0", "0.0"}))
            & (review_triggered | review_text_triggered | source_triggered | status_unknown)
        )
        review_mask = status_unknown | review_triggered | review_text_triggered | source_triggered

        pools.loc[review_mask] = SOURCE_REVIEW_POOL
        pools.loc[nonurban_review_mask] = SOURCE_NONURBAN_REVIEW_POOL
        pools.loc[unknown_mask] = SOURCE_UNKNOWN_POOL
        if include_full_corpus:
            pools.loc[pools == ""] = SOURCE_FULL_CORPUS_POOL
        return pools

    def _cluster_documents(self, documents: Sequence[str]) -> pd.DataFrame:
        if not documents:
            return pd.DataFrame(columns=DYNAMIC_TOPIC_COLUMNS)
        if self.config.prefer_sklearn:
            try:
                return self._cluster_with_sklearn(documents)
            except Exception:
                pass
        return self._cluster_with_keywords(documents)

    def _cluster_with_sklearn(self, documents: Sequence[str]) -> pd.DataFrame:
        from sklearn.cluster import MiniBatchKMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            max_features=max(100, int(self.config.max_features)),
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_+-]{2,}\b",
        )
        matrix = vectorizer.fit_transform(documents)
        feature_names = list(vectorizer.get_feature_names_out())
        if matrix.shape[0] == 1:
            labels = [0]
            centers = matrix.toarray()
        else:
            topic_count = min(
                max(1, int(self.config.max_topics)),
                max(1, math.ceil(matrix.shape[0] / max(1, int(self.config.min_topic_size)))),
                matrix.shape[0],
            )
            if topic_count == 1:
                labels = [0] * matrix.shape[0]
                centers = matrix.mean(axis=0).A
            else:
                model = MiniBatchKMeans(
                    n_clusters=topic_count,
                    random_state=int(self.config.random_state),
                    n_init=10,
                    batch_size=min(2048, max(128, matrix.shape[0])),
                )
                labels = model.fit_predict(matrix).tolist()
                centers = model.cluster_centers_

        keywords_by_label: dict[int, list[str]] = {}
        for label in sorted(set(labels)):
            center = centers[label] if len(centers) > label else matrix[[idx for idx, value in enumerate(labels) if value == label]].mean(axis=0).A1
            order = center.argsort()[::-1]
            keywords = [feature_names[idx].replace(" ", "_") for idx in order[:10] if float(center[idx]) > 0]
            keywords_by_label[int(label)] = keywords or self._top_keyword_terms(
                [documents[idx] for idx, value in enumerate(labels) if int(value) == int(label)]
            )

        return self._rows_from_labels(labels, keywords_by_label, documents)

    def _cluster_with_keywords(self, documents: Sequence[str]) -> pd.DataFrame:
        buckets: dict[str, list[int]] = {}
        keywords_by_bucket: dict[str, list[str]] = {}
        for index, document in enumerate(documents):
            keywords = self._top_keyword_terms([document])
            bucket = keywords[0] if keywords else f"document_{index}"
            buckets.setdefault(bucket, []).append(index)
            keywords_by_bucket[bucket] = keywords

        labels = [0] * len(documents)
        keywords_by_label: dict[int, list[str]] = {}
        for label, bucket in enumerate(sorted(buckets)):
            for index in buckets[bucket]:
                labels[index] = label
            keywords_by_label[label] = self._top_keyword_terms([documents[idx] for idx in buckets[bucket]]) or keywords_by_bucket[bucket]
        return self._rows_from_labels(labels, keywords_by_label, documents)

    def _rows_from_labels(
        self,
        labels: Sequence[int],
        keywords_by_label: dict[int, list[str]],
        documents: Sequence[str],
    ) -> pd.DataFrame:
        counts = Counter(int(label) for label in labels)
        ordered_labels = [label for label, _ in counts.most_common()]
        dynamic_ids = {label: f"DUR_{position:04d}" for position, label in enumerate(ordered_labels, start=1)}
        rows: list[dict[str, Any]] = []
        for row_index, label_raw in enumerate(labels):
            label = int(label_raw)
            keywords = keywords_by_label.get(label) or self._top_keyword_terms([documents[row_index]])
            candidate, status, score = self._map_keywords_to_fixed(keywords)
            size = int(counts[label])
            confidence = self._confidence(size=size, mapping_score=score, keyword_count=len(keywords))
            rows.append(
                {
                    "dynamic_topic_id": dynamic_ids[label],
                    "dynamic_topic_name_zh": self._name_topic_zh(keywords, candidate, status),
                    "dynamic_topic_keywords": "; ".join(keywords[:10]),
                    "dynamic_topic_size": size,
                    "dynamic_topic_confidence": confidence,
                    "dynamic_topic_source_pool": "",
                    "dynamic_to_fixed_topic_candidate": candidate,
                    "dynamic_mapping_status": status,
                }
            )
        return pd.DataFrame(rows).reindex(columns=DYNAMIC_TOPIC_COLUMNS)

    def _map_keywords_to_fixed(self, keywords: Sequence[str]) -> tuple[str, str, float]:
        normalized_keywords = {
            token
            for keyword in keywords
            for token in normalize_phrase(keyword).replace("-", " ").replace("_", " ").split()
            if len(token) >= 3
        }
        best_label = ""
        best_score = 0.0
        for label, seed_tokens in self._topic_seed_tokens.items():
            if not seed_tokens:
                continue
            overlap = len(normalized_keywords & seed_tokens)
            score = overlap / max(5, min(len(seed_tokens), 30))
            if score > best_score:
                best_label = label
                best_score = score

        if best_label and best_score >= float(self.config.mapping_min_score):
            return best_label, STATUS_MAPPED_TO_FIXED, round(best_score, 6)

        urban_anchor_tokens = {
            token
            for phrase in COMMON_RENEWAL_ANCHORS
            for token in normalize_phrase(phrase).replace("-", " ").split()
            if len(token) >= 3
        }
        if normalized_keywords & urban_anchor_tokens:
            return OPEN_SET_URBAN_LABEL, STATUS_CANDIDATE_NEW_URBAN, round(best_score, 6)

        nonurban_terms = {
            "rural",
            "agriculture",
            "transport",
            "mobility",
            "algorithm",
            "model",
            "simulation",
            "pollution",
            "ecology",
            "tourism",
        }
        if normalized_keywords & nonurban_terms:
            return OPEN_SET_NONURBAN_LABEL, STATUS_CANDIDATE_NEW_NONURBAN, round(best_score, 6)
        return "", STATUS_NEEDS_REVIEW, round(best_score, 6)

    def _confidence(self, *, size: int, mapping_score: float, keyword_count: int) -> float:
        size_component = min(1.0, size / max(1, self.config.min_topic_size))
        keyword_component = min(1.0, keyword_count / 8.0)
        confidence = 0.45 * size_component + 0.35 * min(1.0, mapping_score / 0.35) + 0.20 * keyword_component
        return round(float(confidence), 6)

    def _attach_binary_candidates(self, frame: pd.DataFrame) -> pd.DataFrame:
        enriched = frame.copy()
        for column, default_value in DYNAMIC_BINARY_DEFAULTS.items():
            if column not in enriched.columns:
                enriched[column] = pd.Series([default_value] * len(enriched), index=enriched.index, dtype=object)
            else:
                enriched[column] = enriched[column].astype(object)
        for index, row in enriched.iterrows():
            dynamic_id = str(row.get("dynamic_topic_id", "") or "").strip()
            if not dynamic_id:
                continue

            candidate = str(row.get("dynamic_to_fixed_topic_candidate", "") or "").strip()
            status = str(row.get("dynamic_mapping_status", "") or "").strip()
            confidence = self._safe_float(row.get("dynamic_topic_confidence", 0.0))
            current_label = self._normalize_binary_label(
                row.get("final_label", row.get("urban_flag", row.get(Schema.IS_URBAN_RENEWAL, "")))
            )
            keywords = str(row.get("dynamic_topic_keywords", "") or "")
            candidate_label = self._candidate_binary_label(candidate, status, keywords)

            action = "needs_review"
            priority = "medium"
            if candidate_label:
                if candidate_label == current_label:
                    action = "supports_current_label"
                    priority = "low"
                elif candidate_label == "1" and current_label == "0":
                    action = "possible_false_negative_cluster"
                    priority = "high" if confidence >= 0.70 else "medium"
                elif candidate_label == "0" and current_label == "1":
                    action = "possible_false_positive_cluster"
                    priority = "high" if confidence >= 0.70 else "medium"
                else:
                    action = "needs_review"
                    priority = "medium"

            reason = (
                f"dynamic_topic={dynamic_id}; status={status}; candidate={candidate}; "
                f"current_label={current_label}; candidate_label={candidate_label}; "
                f"confidence={confidence:.6f}; keywords={row.get('dynamic_topic_keywords', '')}"
            )
            enriched.at[index, "dynamic_binary_candidate_label"] = candidate_label
            enriched.at[index, "dynamic_binary_candidate_confidence"] = confidence
            enriched.at[index, "dynamic_binary_candidate_action"] = action
            enriched.at[index, "dynamic_binary_candidate_reason"] = reason
            enriched.at[index, "dynamic_binary_review_priority"] = priority
        return enriched

    def _candidate_binary_label(self, candidate: str, status: str, keywords: str = "") -> str:
        if status == STATUS_CANDIDATE_NEW_URBAN:
            return "1"
        if status == STATUS_CANDIDATE_NEW_NONURBAN:
            return "0"
        if status != STATUS_MAPPED_TO_FIXED or not candidate:
            return ""
        group = topic_group_for_label(candidate)
        if group == "urban":
            return "1" if self._has_binary_urban_anchor(keywords) else ""
        if group == "nonurban":
            return "0"
        return ""

    def _has_binary_urban_anchor(self, keywords: str) -> bool:
        text = normalize_phrase(keywords).replace("-", "_").replace(" ", "_")
        anchors = {
            "renewal",
            "regeneration",
            "redevelopment",
            "revitalization",
            "brownfield",
            "adaptive_reuse",
            "slum",
            "rehabilitation",
            "retrofit",
            "urban_village",
            "old_community",
            "old_neighborhood",
            "old_neighbourhood",
            "old_district",
            "old_town",
            "public_space_renewal",
            "street_renewal",
            "estate_regeneration",
        }
        return any(anchor in text for anchor in anchors)

    def _normalize_binary_label(self, value: Any) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        text = str(value).strip()
        if text.replace(".0", "") == "1":
            return "1"
        if text.replace(".0", "") == "0":
            return "0"
        return ""

    def _safe_float(self, value: Any) -> float:
        try:
            if pd.isna(value):
                return 0.0
            return float(value)
        except Exception:
            return 0.0

    def _name_topic_zh(self, keywords: Sequence[str], candidate: str, status: str) -> str:
        normalized = [normalize_phrase(keyword).replace(" ", "_") for keyword in keywords if str(keyword).strip()]
        for token, label in _URBAN_NAME_HINTS:
            if any(token in keyword for keyword in normalized):
                return label
        if candidate and candidate in TOPIC_ENGLISH_NAMES and status == STATUS_MAPPED_TO_FIXED:
            group = topic_group_for_label(candidate)
            prefix = "\u57ce\u5e02\u66f4\u65b0\u6620\u5c04\u4e3b\u9898" if group == "urban" else "\u975e\u57ce\u5e02\u66f4\u65b0\u6620\u5c04\u4e3b\u9898"
            return f"{prefix}: {candidate}"
        preview = "\u3001".join(keywords[:3])
        return f"\u52a8\u6001\u4e3b\u9898: {preview}" if preview else "\u52a8\u6001\u4e3b\u9898"

    def _top_keyword_terms(self, documents: Sequence[str]) -> list[str]:
        counter: Counter[str] = Counter()
        for document in documents:
            tokens = _TOKEN_RE.findall(normalize_phrase(document))
            for token in tokens:
                if token in _STOP_WORDS or len(token) < 3:
                    continue
                counter[token] += 1
            for left, right in zip(tokens, tokens[1:]):
                if left in _STOP_WORDS or right in _STOP_WORDS:
                    continue
                counter[f"{left}_{right}"] += 2
        return [term for term, _ in counter.most_common(10)]

    def _document_for_row(self, row: pd.Series) -> str:
        parts = [
            row.get(Schema.TITLE, ""),
            row.get(Schema.ABSTRACT, ""),
            row.get(Schema.KEYWORDS, ""),
            row.get(Schema.KEYWORDS_PLUS, ""),
            row.get(Schema.AUTHOR_KEYWORDS, ""),
            row.get(Schema.WOS_CATEGORIES, ""),
            row.get(Schema.RESEARCH_AREAS, ""),
        ]
        return " ".join(str(part) for part in parts if self._has_value(part))

    def _build_topic_seed_tokens(self) -> dict[str, set[str]]:
        topic_tokens: dict[str, set[str]] = {}
        for label, definition in TOPIC_DEFINITIONS.items():
            values: list[str] = [str(definition.get("name", "")), str(TOPIC_ENGLISH_NAMES.get(label, ""))]
            for key in ("seeds", "context_terms", "positive_terms", "negative_terms"):
                raw_values = definition.get(key, [])
                if isinstance(raw_values, str):
                    values.append(raw_values)
                else:
                    values.extend(str(value) for value in raw_values)
            tokens = {
                token
                for value in values
                for token in normalize_phrase(value).replace("-", " ").split()
                if len(token) >= 3
            }
            topic_tokens[label] = tokens
        return topic_tokens

    def _series(self, frame: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
        for column in candidates:
            if column in frame.columns:
                return frame[column].fillna("").astype(str)
        return pd.Series([""] * len(frame), index=frame.index, dtype=object)

    def _numeric_series(self, frame: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
        for column in candidates:
            if column in frame.columns:
                return pd.to_numeric(frame[column], errors="coerce").fillna(0)
        return pd.Series([0] * len(frame), index=frame.index, dtype=float)

    def _has_value(self, value: Any) -> bool:
        return str(value or "").strip().lower() not in _EMPTY_VALUES


_STOP_WORDS = {
    "about",
    "after",
    "also",
    "analysis",
    "and",
    "are",
    "based",
    "between",
    "case",
    "city",
    "data",
    "effect",
    "for",
    "from",
    "has",
    "have",
    "into",
    "its",
    "method",
    "model",
    "of",
    "on",
    "paper",
    "research",
    "result",
    "study",
    "the",
    "this",
    "through",
    "urban",
    "using",
    "with",
}
