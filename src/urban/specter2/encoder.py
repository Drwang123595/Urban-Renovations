import importlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .cache import EmbeddingCache, make_cache_key
from .config import Specter2Config
from .features import normalize_record


SPECTER2_UNAVAILABLE = "specter2_unavailable"


@dataclass(frozen=True)
class Specter2Availability:
    status: str
    reason: str = ""
    missing_dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class EncodingResult:
    status: str
    embeddings: np.ndarray
    reason: str = ""
    cache_hits: int = 0
    cache_misses: int = 0


def check_availability() -> Specter2Availability:
    missing = []
    for module_name in ("adapters", "transformers", "torch"):
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        return Specter2Availability(
            status=SPECTER2_UNAVAILABLE,
            reason=f"Missing optional SPECTER2 dependencies: {', '.join(missing)}",
            missing_dependencies=tuple(missing),
        )
    return Specter2Availability(status="available")


class Specter2Encoder:
    def __init__(self, config: Specter2Config | None = None, cache: EmbeddingCache | None = None):
        self.config = config or Specter2Config()
        self.cache = cache or EmbeddingCache(self.config.resolved_cache_dir())
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._device = None

    def encode(self, records: Sequence[dict[str, object]]) -> EncodingResult:
        normalized_records = [normalize_record(record) for record in records]
        if not normalized_records:
            return EncodingResult(status="empty", embeddings=np.zeros((0, 0), dtype=np.float32))

        availability = check_availability()
        if availability.status != "available":
            return EncodingResult(
                status=availability.status,
                embeddings=np.zeros((0, 0), dtype=np.float32),
                reason=availability.reason,
            )

        vectors: list[np.ndarray | None] = []
        missing_records: list[tuple[int, dict[str, str], str]] = []
        cache_hits = 0
        for index, record in enumerate(normalized_records):
            key = make_cache_key(
                model_name=self.config.model_name,
                adapter_name=self.config.adapter_name,
                title=record["title"],
                abstract=record["abstract"],
            )
            cached = self.cache.load(key)
            if cached is None:
                vectors.append(None)
                missing_records.append((index, record, key))
            else:
                vectors.append(np.asarray(cached, dtype=np.float32))
                cache_hits += 1

        if missing_records:
            self._ensure_loaded()
            for batch_start in range(0, len(missing_records), self.config.batch_size):
                batch = missing_records[batch_start : batch_start + self.config.batch_size]
                texts = [self._format_text(record) for _, record, _ in batch]
                batch_embeddings = self._encode_texts(texts)
                for vector, (original_index, _, key) in zip(batch_embeddings, batch):
                    vector = np.asarray(vector, dtype=np.float32)
                    vectors[original_index] = vector
                    self.cache.store(key, vector)

        matrix = np.vstack([vector for vector in vectors if vector is not None]).astype(np.float32, copy=False)
        return EncodingResult(
            status="ok",
            embeddings=matrix,
            cache_hits=cache_hits,
            cache_misses=len(missing_records),
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        adapters_module = importlib.import_module("adapters")
        transformers_module = importlib.import_module("transformers")
        torch_module = importlib.import_module("torch")

        auto_adapter_model = getattr(adapters_module, "AutoAdapterModel")
        auto_tokenizer = getattr(transformers_module, "AutoTokenizer")

        tokenizer = auto_tokenizer.from_pretrained(self.config.model_name)
        model = auto_adapter_model.from_pretrained(self.config.model_name)
        model.load_adapter(self.config.adapter_name, source="hf", set_active=True)

        device = self.config.device or ("cuda" if torch_module.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch_module
        self._device = device

    def _format_text(self, record: dict[str, str]) -> str:
        separator = getattr(self._tokenizer, "sep_token", None) or "[SEP]"
        title = record["title"]
        abstract = record["abstract"]
        if title and abstract:
            return f"{title} {separator} {abstract}"
        return title or abstract

    def _encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        inputs = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(self._device) for name, tensor in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.detach().cpu().numpy().astype(np.float32, copy=False)
