from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import hmac
import json
import shutil
import sys
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ..runtime.config import Config, Schema
from .urban_metadata import UrbanMetadataRecord, normalize_phrase, tokenize_text
from .urban_topic_classifier import UrbanTopicClassifier
from .urban_topic_taxonomy import TOPIC_DEFINITIONS, TOPIC_ENGLISH_NAMES, TOPIC_ORDER
from .urban_training_contract import allowed_training_workbooks, assert_training_source_contract


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_MIN_TOPIC_SIZE = 20
DEFAULT_AUTO_MAPPING_MIN_PURITY = 0.70
DEFAULT_AUTO_MAPPING_MIN_SHARE = 0.70
DEFAULT_ZEROSHOT_MIN_SIMILARITY = 0.68
ARTIFACT_INTEGRITY_VERSION = 2
ARTIFACT_HASH_ALGO = "sha256"


@dataclass(frozen=True)
class BERTopicSignal:
    available: bool
    status: str
    topic_id: int = -1
    topic_name: str = ""
    topic_probability: float = 0.0
    is_outlier: bool = False
    topic_count: int = 0
    topic_pos_rate: Optional[float] = None
    mapped_label: str = ""
    mapped_group: str = ""
    mapped_name: str = ""
    label_purity: float = 0.0
    mapped_label_share: float = 0.0
    top_terms: str = ""
    sample_titles: str = ""
    source_split: str = ""
    reason: str = ""


class UrbanBERTopicService:
    _CACHE: Dict[str, Tuple[Any, Dict[str, Dict[str, Any]], Dict[str, Any], Lock]] = {}
    _CACHE_LOCK = Lock()

    def __init__(
        self,
        *,
        artifact_dir: Optional[Path] = None,
        train_dir: Optional[Path] = None,
        embedding_model_name: Optional[str] = None,
        min_topic_size: int = DEFAULT_MIN_TOPIC_SIZE,
    ):
        self.artifact_dir = artifact_dir or Config.BERTOPIC_ARTIFACT_DIR
        self.train_dir = train_dir or Config.TRAIN_DIR
        self.embedding_model_name = embedding_model_name or Config.BERTOPIC_EMBEDDING_MODEL
        self.min_topic_size = max(2, int(min_topic_size))

        self._topic_model = None
        self._topic_stats: Dict[str, Dict[str, Any]] = {}
        self._manifest: Dict[str, Any] = {}
        self._predict_lock = Lock()
        self._availability_status = self._detect_availability_status()

    def is_available(self) -> bool:
        return self._availability_status == "available"

    def availability_status(self) -> str:
        return self._availability_status

    def predict(self, record: UrbanMetadataRecord) -> BERTopicSignal:
        if not self.is_available():
            return BERTopicSignal(
                available=False,
                status=self._availability_status,
                reason=self._availability_status,
            )

        document = self._build_topic_document(record)
        if len(tokenize_text(document)) < 4:
            return BERTopicSignal(
                available=True,
                status="short_text",
                topic_id=-1,
                is_outlier=True,
                reason="short_text",
            )

        try:
            self._ensure_ready()
        except Exception as error:
            return BERTopicSignal(
                available=False,
                status=f"init_error:{type(error).__name__}",
                reason=str(error),
            )

        with self._predict_lock:
            topics, probabilities = self._topic_model.transform([document])

        topic_id = int(topics[0])
        stats = self._topic_stats.get(str(topic_id), {})
        return BERTopicSignal(
            available=True,
            status="ready",
            topic_id=topic_id,
            topic_name=str(stats.get("topic_name", f"topic_{topic_id}")),
            topic_probability=round(self._extract_probability(probabilities, topic_id), 4),
            is_outlier=topic_id == -1,
            topic_count=int(stats.get("count", 0) or 0),
            topic_pos_rate=self._coerce_optional_float(stats.get("pos_rate")),
            mapped_label=str(stats.get("mapped_label", "")),
            mapped_group=str(stats.get("mapped_group", "")),
            mapped_name=str(stats.get("mapped_name", "")),
            label_purity=round(float(stats.get("label_purity", 0.0) or 0.0), 4),
            mapped_label_share=round(float(stats.get("mapped_label_share", 0.0) or 0.0), 4),
            top_terms=str(stats.get("top_terms", "")),
            sample_titles=str(stats.get("sample_titles", "")),
            source_split=str(stats.get("source_split", "")),
            reason=str(stats.get("reason", "")),
        )

    def train_primary_artifacts(self, *, force_retrain: bool = False) -> Dict[str, Any]:
        cache_key = str(self._resolve_artifact_dir())
        with self._CACHE_LOCK:
            self._CACHE.pop(cache_key, None)
        self._topic_model = None
        self._topic_stats = {}
        self._manifest = {}

        model, topic_stats, manifest = self._load_or_fit_artifacts(force_retrain=force_retrain)
        self._topic_model = model
        self._topic_stats = topic_stats
        self._manifest = manifest
        return {
            "artifact_dir": str(self._resolve_artifact_dir()),
            "manifest": manifest,
            "topics": len(topic_stats),
        }

    def _detect_availability_status(self) -> str:
        if sys.version_info[:2] >= (3, 14):
            return "python_runtime_unsupported"
        try:
            self._import_stack()
        except ModuleNotFoundError:
            return "bertopic_stack_missing"
        return "available"

    def _ensure_ready(self):
        if self._topic_model is not None:
            return

        cache_key = str(self._resolve_artifact_dir())
        with self._CACHE_LOCK:
            cached = self._CACHE.get(cache_key)
            if cached is not None:
                self._topic_model, self._topic_stats, self._manifest, self._predict_lock = cached
                return

            model, topic_stats, manifest = self._load_or_fit_artifacts()
            shared_predict_lock = Lock()
            self._CACHE[cache_key] = (model, topic_stats, manifest, shared_predict_lock)
            self._topic_model = model
            self._topic_stats = topic_stats
            self._manifest = manifest
            self._predict_lock = shared_predict_lock

    def _load_or_fit_artifacts(self, *, force_retrain: bool = False):
        self._resolve_artifact_dir()
        manifest_path = self._resolve_artifact_child("manifest.json")
        stats_path = self._resolve_artifact_child("topic_stats.json")
        quality_path = self._resolve_artifact_child("topic_quality.json")
        mapping_path = self._resolve_artifact_child("topic_mapping.json")
        training_manifest_path = self._resolve_artifact_child("training_manifest.json")
        model_path = self._resolve_artifact_child("model")
        integrity_path = self._resolve_artifact_child("integrity.json")

        fingerprint, training_paths = self._build_fingerprint()
        required_paths = (
            manifest_path,
            stats_path,
            quality_path,
            mapping_path,
            training_manifest_path,
            model_path,
            integrity_path,
        )
        if not force_retrain and all(path.exists() for path in required_paths):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if self._validate_artifact_bundle(
                    manifest=manifest,
                    fingerprint=fingerprint,
                    manifest_path=manifest_path,
                    stats_path=stats_path,
                    quality_path=quality_path,
                    mapping_path=mapping_path,
                    training_manifest_path=training_manifest_path,
                    model_path=model_path,
                    integrity_path=integrity_path,
                ):
                    model = self._load_model(model_path)
                    topic_stats = json.loads(stats_path.read_text(encoding="utf-8"))
                    return model, topic_stats, manifest
            except Exception as error:
                print(f"[WARN] BERTopic artifact validation failed: {error}. Rebuilding local artifacts.")

        return self._fit_and_save(
            model_path=model_path,
            manifest_path=manifest_path,
            stats_path=stats_path,
            quality_path=quality_path,
            mapping_path=mapping_path,
            training_manifest_path=training_manifest_path,
            integrity_path=integrity_path,
            fingerprint=fingerprint,
            training_paths=training_paths,
        )

    def _fit_and_save(
        self,
        *,
        model_path: Path,
        manifest_path: Path,
        stats_path: Path,
        quality_path: Path,
        mapping_path: Path,
        training_manifest_path: Path,
        integrity_path: Path,
        fingerprint: str,
        training_paths: Sequence[Path],
    ):
        BERTopic, SentenceTransformer, CountVectorizer = self._import_stack()
        records, labels, source_rows, record_sources = self._load_training_records(training_paths)
        if len(records) < self.min_topic_size:
            raise RuntimeError(
                f"Not enough labeled records for BERTopic primary model: {len(records)}"
            )

        classifier = UrbanTopicClassifier()
        classifier_predictions = [classifier.predict(record) for record in records]
        documents = [self._build_topic_document(record) for record in records]

        embedding_model = SentenceTransformer(self.embedding_model_name)
        vectorizer_model = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 3),
            min_df=3,
            max_df=0.92,
        )
        topic_model = BERTopic(
            embedding_model=embedding_model,
            min_topic_size=self.min_topic_size,
            n_gram_range=(1, 3),
            calculate_probabilities=True,
            seed_topic_list=self._seed_topic_list(),
            zeroshot_topic_list=self._zeroshot_topic_list(),
            zeroshot_min_similarity=DEFAULT_ZEROSHOT_MIN_SIMILARITY,
            vectorizer_model=vectorizer_model,
            verbose=False,
        )
        topics, probabilities = topic_model.fit_transform(documents)
        mapping_overrides = self._load_existing_mapping_overrides(mapping_path)
        topic_stats, topic_quality, topic_mapping = self._build_topic_artifacts(
            topic_model=topic_model,
            topics=topics,
            probabilities=probabilities,
            labels=labels,
            classifier_predictions=classifier_predictions,
            records=records,
            record_sources=record_sources,
            mapping_overrides=mapping_overrides,
        )

        self._remove_existing_model_path(model_path)
        topic_model.save(
            model_path,
            serialization="pickle",
            save_embedding_model=self.embedding_model_name,
            save_ctfidf=True,
        )

        stats_path.write_text(json.dumps(topic_stats, ensure_ascii=False, indent=2), encoding="utf-8")
        quality_path.write_text(json.dumps(topic_quality, ensure_ascii=False, indent=2), encoding="utf-8")
        mapping_path.write_text(json.dumps(topic_mapping, ensure_ascii=False, indent=2), encoding="utf-8")

        training_manifest = {
            "manifest_version": ARTIFACT_INTEGRITY_VERSION,
            "fingerprint": fingerprint,
            "embedding_model": self.embedding_model_name,
            "min_topic_size": self.min_topic_size,
            "vectorizer": {
                "stop_words": "english",
                "ngram_range": [1, 3],
                "min_df": 3,
                "max_df": 0.92,
            },
            "zeroshot_min_similarity": DEFAULT_ZEROSHOT_MIN_SIMILARITY,
            "training_rows": len(records),
            "source_rows": source_rows,
            "training_sources": [str(path) for path in training_paths],
            "source_split": dict(Counter(record_sources)),
            "label_distribution": {
                "urban": int(sum(labels)),
                "nonurban": int(len(labels) - sum(labels)),
            },
            "python_version": ".".join(str(part) for part in sys.version_info[:3]),
        }
        training_manifest_path.write_text(
            json.dumps(training_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        manifest = {
            "manifest_version": ARTIFACT_INTEGRITY_VERSION,
            "fingerprint": fingerprint,
            "embedding_model": self.embedding_model_name,
            "min_topic_size": self.min_topic_size,
            "training_rows": len(records),
            "training_sources": [str(path) for path in training_paths],
            "source_rows": source_rows,
            "python_version": ".".join(str(part) for part in sys.version_info[:3]),
            "artifact_hashes": {
                "stats_sha256": self._hash_path(stats_path),
                "quality_sha256": self._hash_path(quality_path),
                "mapping_sha256": self._hash_path(mapping_path),
                "training_manifest_sha256": self._hash_path(training_manifest_path),
                "model_sha256": self._hash_path(model_path),
            },
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        integrity_record = self._build_integrity_record(
            fingerprint=fingerprint,
            manifest_path=manifest_path,
            stats_path=stats_path,
            quality_path=quality_path,
            mapping_path=mapping_path,
            training_manifest_path=training_manifest_path,
            model_path=model_path,
        )
        integrity_path.write_text(json.dumps(integrity_record, ensure_ascii=False, indent=2), encoding="utf-8")
        return topic_model, topic_stats, manifest

    def _build_topic_artifacts(
        self,
        *,
        topic_model,
        topics: Sequence[int],
        probabilities: Any,
        labels: Sequence[int],
        classifier_predictions: Sequence[Any],
        records: Sequence[UrbanMetadataRecord],
        record_sources: Sequence[str],
        mapping_overrides: Dict[str, Dict[str, Any]],
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        topic_info = topic_model.get_topic_info()
        info_by_topic: Dict[int, Dict[str, Any]] = {}
        for _, row in topic_info.iterrows():
            topic_id = int(row["Topic"])
            representation = row.get("Representation", []) or []
            info_by_topic[topic_id] = {
                "topic_name": str(row.get("Name", f"topic_{topic_id}")),
                "representation": [str(item) for item in representation],
                "top_terms": ", ".join(representation),
            }

        per_topic_gold: Dict[int, List[int]] = {}
        per_topic_labels: Dict[int, List[str]] = {}
        per_topic_titles: Dict[int, List[str]] = {}
        per_topic_sources: Dict[int, List[str]] = {}
        for topic_id, gold_label, prediction, record, source_name in zip(
            topics,
            labels,
            classifier_predictions,
            records,
            record_sources,
        ):
            topic_id = int(topic_id)
            per_topic_gold.setdefault(topic_id, []).append(int(gold_label))
            per_topic_labels.setdefault(topic_id, []).append(str(prediction.topic_label))
            per_topic_titles.setdefault(topic_id, []).append(str(record.title))
            per_topic_sources.setdefault(topic_id, []).append(str(source_name))

        stats_rows: Dict[str, Dict[str, Any]] = {}
        quality_rows: Dict[str, Dict[str, Any]] = {}
        mapping_rows: Dict[str, Dict[str, Any]] = {}
        topic_probabilities = self._topic_probability_by_topic(topics=topics, probabilities=probabilities)

        for topic_id, gold_values in per_topic_gold.items():
            count = len(gold_values)
            label_counter = Counter(per_topic_labels.get(topic_id, []))
            majority_label, majority_count = label_counter.most_common(1)[0] if label_counter else ("", 0)
            majority_share = (majority_count / count) if count else 0.0
            info = info_by_topic.get(topic_id, {})
            top_terms = str(info.get("top_terms", ""))
            representation = info.get("representation", [])
            seed_scores = self._compute_seed_overlap_scores(representation)
            auto_mapping = self._select_auto_mapping(
                label_counter=label_counter,
                seed_scores=seed_scores,
                count=count,
            )
            mapped_label = auto_mapping["mapped_label"]
            mapping_source = auto_mapping["mapping_source"]

            manual_override = mapping_overrides.get(str(topic_id))
            if manual_override:
                override_label = str(manual_override.get("mapped_label", "") or "")
                if override_label in TOPIC_DEFINITIONS:
                    mapped_label = override_label
                    mapping_source = "manual_topic_mapping"

            mapped_group = TOPIC_DEFINITIONS.get(mapped_label, {}).get("group", "")
            mapped_name = TOPIC_ENGLISH_NAMES.get(mapped_label, "")
            mapped_label_share = (
                label_counter.get(mapped_label, 0) / count if count and mapped_label else 0.0
            )
            pos_rate = (sum(gold_values) / count) if count else 0.0
            sample_titles = self._sample_titles(per_topic_titles.get(topic_id, []))
            source_split = self._format_source_split(per_topic_sources.get(topic_id, []))
            max_probability = topic_probabilities.get(topic_id, 0.0)

            quality_row = {
                "topic_id": topic_id,
                "topic_name": str(info.get("topic_name", f"topic_{topic_id}")),
                "count": count,
                "pos_rate": round(pos_rate, 6),
                "neg_rate": round(1.0 - pos_rate, 6),
                "majority_label": majority_label,
                "majority_label_share": round(majority_share, 6),
                "mapped_label": mapped_label,
                "mapped_group": mapped_group,
                "mapped_name": mapped_name,
                "label_purity": round(majority_share, 6),
                "mapped_label_share": round(mapped_label_share, 6),
                "topic_probability_max": round(max_probability, 6),
                "top_terms": top_terms,
                "sample_titles": sample_titles,
                "source_split": source_split,
                "seed_scores": seed_scores,
                "reason": mapping_source,
            }

            stats_rows[str(topic_id)] = {
                "topic_id": topic_id,
                "topic_name": quality_row["topic_name"],
                "count": count,
                "pos_rate": quality_row["pos_rate"],
                "neg_rate": quality_row["neg_rate"],
                "mapped_label": mapped_label,
                "mapped_group": mapped_group,
                "mapped_name": mapped_name,
                "label_purity": quality_row["label_purity"],
                "mapped_label_share": quality_row["mapped_label_share"],
                "top_terms": top_terms,
                "sample_titles": sample_titles,
                "source_split": source_split,
                "reason": mapping_source,
            }
            quality_rows[str(topic_id)] = quality_row
            mapping_rows[str(topic_id)] = {
                "topic_id": topic_id,
                "topic_name": quality_row["topic_name"],
                "mapped_label": mapped_label,
                "mapped_group": mapped_group,
                "mapped_name": mapped_name,
                "label_purity": quality_row["label_purity"],
                "mapped_label_share": quality_row["mapped_label_share"],
                "mapping_source": mapping_source,
                "sample_titles": sample_titles,
                "top_terms": top_terms,
            }

        topic_quality = {
            "format_version": ARTIFACT_INTEGRITY_VERSION,
            "embedding_model": self.embedding_model_name,
            "topics": quality_rows,
        }
        topic_mapping = {
            "format_version": ARTIFACT_INTEGRITY_VERSION,
            "embedding_model": self.embedding_model_name,
            "topics": mapping_rows,
        }
        return stats_rows, topic_quality, topic_mapping

    def _build_fingerprint(self) -> Tuple[str, List[Path]]:
        training_paths = self._resolve_training_paths()
        payload = {
            "embedding_model_name": self.embedding_model_name,
            "min_topic_size": self.min_topic_size,
            "primary_enabled": bool(Config.BERTOPIC_PRIMARY_ENABLED),
            "primary_min_support": int(Config.BERTOPIC_PRIMARY_MIN_SUPPORT),
            "primary_min_purity": float(Config.BERTOPIC_PRIMARY_MIN_PURITY),
            "primary_min_prob": float(Config.BERTOPIC_PRIMARY_MIN_PROB),
            "primary_min_mapped_share": float(Config.BERTOPIC_PRIMARY_MIN_MAPPED_SHARE),
            "seed_topic_list": self._seed_topic_list(),
            "zeroshot_topic_list": self._zeroshot_topic_list(),
            "training_sources": [
                {
                    "path": str(path),
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
                for path in training_paths
                if path.exists()
            ],
        }
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return digest, training_paths

    def _resolve_training_paths(self) -> List[Path]:
        unique: List[Path] = []
        seen = set()
        for path in allowed_training_workbooks(self.train_dir):
            resolved = str(path.resolve())
            if resolved in seen or not path.exists():
                continue
            header = self._safe_read_header(path)
            if header is None:
                continue
            if Schema.TITLE not in header.columns or Schema.ABSTRACT not in header.columns:
                continue
            if not self._detect_label_column(header):
                continue
            seen.add(resolved)
            unique.append(path)
        assert_training_source_contract(unique)
        if not unique:
            raise RuntimeError("No labeled training workbook found for BERTopic primary model.")
        return unique

    def _load_training_records(
        self,
        training_paths: Sequence[Path],
    ) -> Tuple[List[UrbanMetadataRecord], List[int], int, List[str]]:
        records: List[UrbanMetadataRecord] = []
        labels: List[int] = []
        record_sources: List[str] = []
        seen_keys = set()
        source_rows = 0

        for path in training_paths:
            df = pd.read_excel(path, engine="openpyxl")
            label_col = self._detect_label_column(df)
            if not label_col:
                continue
            for column in [
                Schema.AUTHOR_KEYWORDS,
                Schema.KEYWORDS_PLUS,
                Schema.KEYWORDS,
                Schema.WOS_CATEGORIES,
                Schema.RESEARCH_AREAS,
            ]:
                if column not in df.columns:
                    df[column] = ""
            sub = df[
                [
                    Schema.TITLE,
                    Schema.ABSTRACT,
                    Schema.AUTHOR_KEYWORDS,
                    Schema.KEYWORDS_PLUS,
                    Schema.KEYWORDS,
                    Schema.WOS_CATEGORIES,
                    Schema.RESEARCH_AREAS,
                    label_col,
                ]
            ].copy()
            sub = sub[sub[label_col].isin([0, 1, "0", "1"])]
            source_rows += int(len(sub))
            for _, row in sub.iterrows():
                record = UrbanMetadataRecord.from_row(row.to_dict())
                if not record.title and not record.abstract:
                    continue
                key = (
                    normalize_phrase(record.title),
                    normalize_phrase(record.abstract[:400]),
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                records.append(record)
                labels.append(int(row[label_col]))
                record_sources.append(path.name)
        return records, labels, source_rows, record_sources

    def _safe_read_header(self, path: Path) -> Optional[pd.DataFrame]:
        try:
            return pd.read_excel(path, engine="openpyxl", nrows=5)
        except Exception:
            return None

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
            if "local_v2" in lowered or "閸╁骸绔堕弴瀛樻煀" in text or lowered == "is_urban_renewal":
                return column
        return None

    def _build_topic_document(self, record: UrbanMetadataRecord) -> str:
        parts = [record.title, record.abstract]
        if record.keyword_tokens:
            parts.append("keywords: " + "; ".join(record.keyword_tokens[:20]))
        return " [SEP] ".join(part for part in parts if part)

    def _label_seed_tokens(self, topic_label: str) -> List[str]:
        definition = TOPIC_DEFINITIONS.get(topic_label, {})
        tokens: List[str] = []
        for phrase in definition.get("seeds", []):
            for token in normalize_phrase(phrase).replace("-", " ").split():
                if token and token not in tokens:
                    tokens.append(token)
        for phrase in definition.get("context_terms", [])[:8]:
            for token in normalize_phrase(phrase).replace("-", " ").split():
                if token and token not in tokens:
                    tokens.append(token)
        return tokens

    def _seed_topic_list(self) -> List[List[str]]:
        return [self._label_seed_tokens(label)[:12] for label in TOPIC_ORDER]

    def _zeroshot_topic_list(self) -> List[str]:
        return [TOPIC_ENGLISH_NAMES[label] for label in TOPIC_ORDER]

    def _compute_seed_overlap_scores(self, representation: Sequence[str]) -> Dict[str, int]:
        rep_tokens = set()
        for phrase in representation:
            for token in normalize_phrase(phrase).replace("-", " ").split():
                if token:
                    rep_tokens.add(token)
        scores: Dict[str, int] = {}
        for topic_label in TOPIC_ORDER:
            seed_tokens = set(self._label_seed_tokens(topic_label))
            overlap = len(rep_tokens & seed_tokens)
            if overlap:
                scores[topic_label] = overlap
        return scores

    def _select_auto_mapping(
        self,
        *,
        label_counter: Counter[str],
        seed_scores: Dict[str, int],
        count: int,
    ) -> Dict[str, Any]:
        if not count:
            return {
                "mapped_label": "",
                "mapping_source": "empty_topic",
            }

        candidates = set(label_counter) | set(seed_scores)
        best_label = ""
        best_combined = -1.0
        best_share = 0.0
        best_seed = 0
        for topic_label in candidates:
            share = label_counter.get(topic_label, 0) / count
            seed_score = int(seed_scores.get(topic_label, 0))
            combined = (share * 5.0) + (min(seed_score, 4) * 0.35)
            if combined > best_combined:
                best_combined = combined
                best_label = topic_label
                best_share = share
                best_seed = seed_score

        if (
            best_label
            and best_share >= DEFAULT_AUTO_MAPPING_MIN_SHARE
            and (best_share >= DEFAULT_AUTO_MAPPING_MIN_PURITY or best_seed >= 2)
        ):
            return {
                "mapped_label": best_label,
                "mapping_source": "auto_combined_majority_seed_mapping",
            }

        return {
            "mapped_label": "",
            "mapping_source": "low_consensus_unknown",
        }

    def _load_existing_mapping_overrides(self, mapping_path: Path) -> Dict[str, Dict[str, Any]]:
        if not mapping_path.exists():
            return {}
        try:
            payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        topics = payload.get("topics")
        if not isinstance(topics, dict):
            return {}
        overrides: Dict[str, Dict[str, Any]] = {}
        for topic_id, entry in topics.items():
            if not isinstance(entry, dict):
                continue
            if entry.get("mapping_source") in {"manual_topic_mapping", "manual_confirmed"} or entry.get("locked"):
                overrides[str(topic_id)] = entry
        return overrides

    def _topic_probability_by_topic(self, *, topics: Sequence[int], probabilities: Any) -> Dict[int, float]:
        result: Dict[int, float] = {}
        if probabilities is None:
            return result
        if hasattr(probabilities, "tolist"):
            probabilities = probabilities.tolist()
        if not isinstance(probabilities, list):
            return result
        for topic_id, row in zip(topics, probabilities):
            if isinstance(row, list):
                numeric_values = [float(value) for value in row if isinstance(value, (int, float))]
                result[int(topic_id)] = max(numeric_values) if numeric_values else 0.0
            elif isinstance(row, (int, float)):
                result[int(topic_id)] = float(row)
        return result

    def _sample_titles(self, titles: Sequence[str], *, limit: int = 5) -> str:
        ordered: List[str] = []
        seen = set()
        for title in titles:
            clean = str(title or "").strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            ordered.append(clean)
            if len(ordered) >= limit:
                break
        return " | ".join(ordered)

    def _format_source_split(self, source_names: Sequence[str]) -> str:
        counter = Counter(source_names)
        return "; ".join(f"{name}:{count}" for name, count in counter.most_common())

    def _load_model(self, model_path: Path):
        model_path = self._ensure_path_within_artifact_dir(model_path)
        BERTopic, SentenceTransformer, _ = self._import_stack()
        embedding_model = SentenceTransformer(self.embedding_model_name)
        return BERTopic.load(str(model_path), embedding_model=embedding_model)

    def _resolve_artifact_dir(self) -> Path:
        artifact_dir = Path(self.artifact_dir).expanduser().resolve(strict=False)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir

    def _resolve_artifact_child(self, *parts: str) -> Path:
        base = self._resolve_artifact_dir()
        child = (base.joinpath(*parts)).resolve(strict=False)
        return self._ensure_path_within_artifact_dir(child, base_dir=base)

    def _ensure_path_within_artifact_dir(
        self,
        path: Path,
        *,
        base_dir: Optional[Path] = None,
    ) -> Path:
        base = base_dir or self._resolve_artifact_dir()
        resolved = Path(path).expanduser().resolve(strict=False)
        try:
            resolved.relative_to(base)
        except ValueError as error:
            raise RuntimeError(f"Artifact path escapes managed directory: {resolved}") from error
        return resolved

    def _remove_existing_model_path(self, model_path: Path):
        validated = self._ensure_path_within_artifact_dir(model_path)
        original = Path(model_path)
        if original.exists() and original.is_symlink():
            raise RuntimeError(f"Refusing to delete symlinked model path: {original}")
        if validated.exists():
            if validated.is_dir():
                shutil.rmtree(validated)
            else:
                validated.unlink()

    def _validate_artifact_bundle(
        self,
        *,
        manifest: Dict[str, Any],
        fingerprint: str,
        manifest_path: Path,
        stats_path: Path,
        quality_path: Path,
        mapping_path: Path,
        training_manifest_path: Path,
        model_path: Path,
        integrity_path: Path,
    ) -> bool:
        if manifest.get("manifest_version") != ARTIFACT_INTEGRITY_VERSION:
            return False
        if manifest.get("fingerprint") != fingerprint:
            return False

        integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
        if integrity.get("format_version") != ARTIFACT_INTEGRITY_VERSION:
            return False
        if integrity.get("hash_algo") != ARTIFACT_HASH_ALGO:
            return False
        if integrity.get("fingerprint") != fingerprint:
            return False

        current_hashes = {
            "manifest_sha256": self._hash_path(manifest_path),
            "stats_sha256": self._hash_path(stats_path),
            "quality_sha256": self._hash_path(quality_path),
            "mapping_sha256": self._hash_path(mapping_path),
            "training_manifest_sha256": self._hash_path(training_manifest_path),
            "model_sha256": self._hash_path(model_path),
        }
        for key, value in current_hashes.items():
            if integrity.get(key) != value:
                return False

        manifest_hashes = manifest.get("artifact_hashes") or {}
        for artifact_key, current_key in {
            "stats_sha256": "stats_sha256",
            "quality_sha256": "quality_sha256",
            "mapping_sha256": "mapping_sha256",
            "training_manifest_sha256": "training_manifest_sha256",
            "model_sha256": "model_sha256",
        }.items():
            if manifest_hashes.get(artifact_key) != current_hashes[current_key]:
                return False

        expected_hmac = self._sign_integrity_payload(integrity)
        if Config.BERTOPIC_INTEGRITY_KEY and integrity.get("hmac_sha256") != expected_hmac:
            return False
        return True

    def _build_integrity_record(
        self,
        *,
        fingerprint: str,
        manifest_path: Path,
        stats_path: Path,
        quality_path: Path,
        mapping_path: Path,
        training_manifest_path: Path,
        model_path: Path,
    ) -> Dict[str, Any]:
        record = {
            "format_version": ARTIFACT_INTEGRITY_VERSION,
            "hash_algo": ARTIFACT_HASH_ALGO,
            "fingerprint": fingerprint,
            "manifest_sha256": self._hash_path(manifest_path),
            "stats_sha256": self._hash_path(stats_path),
            "quality_sha256": self._hash_path(quality_path),
            "mapping_sha256": self._hash_path(mapping_path),
            "training_manifest_sha256": self._hash_path(training_manifest_path),
            "model_sha256": self._hash_path(model_path),
            "integrity_mode": "hmac_sha256" if Config.BERTOPIC_INTEGRITY_KEY else "hash_only",
        }
        record["hmac_sha256"] = self._sign_integrity_payload(record)
        return record

    def _sign_integrity_payload(self, payload: Dict[str, Any]) -> str:
        if not Config.BERTOPIC_INTEGRITY_KEY:
            return ""
        clean_payload = dict(payload)
        clean_payload.pop("hmac_sha256", None)
        message = json.dumps(clean_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hmac.new(
            Config.BERTOPIC_INTEGRITY_KEY.encode("utf-8"),
            message,
            hashlib.sha256,
        ).hexdigest()

    def _hash_path(self, path: Path) -> str:
        validated = self._ensure_path_within_artifact_dir(path)
        if not validated.exists():
            raise FileNotFoundError(f"Missing artifact path for hashing: {validated}")
        if validated.is_symlink():
            raise RuntimeError(f"Symbolic links are not allowed inside artifact bundle: {validated}")
        if validated.is_file():
            return self._hash_file(validated)

        digest = hashlib.sha256()
        for child in sorted(validated.rglob("*"), key=lambda item: item.as_posix()):
            if child.is_symlink():
                raise RuntimeError(f"Symbolic links are not allowed inside artifact bundle: {child}")
            if child.is_dir():
                continue
            relative_name = child.relative_to(validated).as_posix()
            digest.update(relative_name.encode("utf-8"))
            digest.update(self._hash_file(child).encode("utf-8"))
        return digest.hexdigest()

    def _hash_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _import_stack(self):
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from sklearn.feature_extraction.text import CountVectorizer

        return BERTopic, SentenceTransformer, CountVectorizer

    def _extract_probability(self, probabilities: Any, topic_id: int) -> float:
        if probabilities is None:
            return 0.0
        if hasattr(probabilities, "tolist"):
            probabilities = probabilities.tolist()

        if not isinstance(probabilities, list) or not probabilities:
            return 0.0

        first = probabilities[0]
        if isinstance(first, list):
            numeric_values = [float(value) for value in first if isinstance(value, (int, float))]
            return max(numeric_values) if numeric_values else 0.0
        if isinstance(first, (int, float)):
            return float(first)
        return 0.0

    def _coerce_optional_float(self, value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        return round(float(value), 4)

    def is_primary_candidate(self, signal: BERTopicSignal) -> bool:
        if not signal.available or signal.is_outlier:
            return False
        if not signal.mapped_label or not signal.mapped_group:
            return False
        if signal.topic_count < int(Config.BERTOPIC_PRIMARY_MIN_SUPPORT):
            return False
        if signal.label_purity < float(Config.BERTOPIC_PRIMARY_MIN_PURITY):
            return False
        if signal.topic_probability < float(Config.BERTOPIC_PRIMARY_MIN_PROB):
            return False
        if signal.mapped_label_share < float(Config.BERTOPIC_PRIMARY_MIN_MAPPED_SHARE):
            return False
        return True

    def has_high_trust_alignment(self, signal: BERTopicSignal, *, topic_label: str, topic_group: str) -> bool:
        if not self.is_primary_candidate(signal):
            return False
        same_label = bool(signal.mapped_label) and signal.mapped_label == topic_label
        same_group = bool(signal.mapped_group) and signal.mapped_group == topic_group
        return bool(same_label or same_group)

    def has_topic_conflict(self, signal: BERTopicSignal, *, topic_label: str, topic_group: str) -> bool:
        if not self.is_primary_candidate(signal):
            return False
        if signal.mapped_label and signal.mapped_label != topic_label:
            return True
        if signal.mapped_group and signal.mapped_group != topic_group:
            return True
        return False

    def is_high_purity_topic(self, signal: BERTopicSignal) -> bool:
        return self.is_primary_candidate(signal)
