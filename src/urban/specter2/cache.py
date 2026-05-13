import hashlib
import json
from pathlib import Path

import numpy as np


def _normalize_cache_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def make_cache_key(
    *,
    model_name: str,
    adapter_name: str,
    title: object,
    abstract: object,
) -> str:
    payload = {
        "abstract": _normalize_cache_text(abstract),
        "adapter_name": _normalize_cache_text(adapter_name),
        "model_name": _normalize_cache_text(model_name),
        "title": _normalize_cache_text(title),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class EmbeddingCache:
    def __init__(self, root_dir: Path | str):
        self.root_dir = Path(root_dir)

    def path_for_key(self, key: str) -> Path:
        return self.root_dir / f"{key}.npy"

    def load(self, key: str) -> np.ndarray | None:
        path = self.path_for_key(key)
        if not path.exists():
            return None
        return np.load(path)

    def store(self, key: str, vector: np.ndarray) -> Path:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for_key(key)
        np.save(path, np.asarray(vector, dtype=np.float32))
        return path
