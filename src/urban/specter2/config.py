from dataclasses import dataclass
from pathlib import Path

from src.runtime.config import Config


DEFAULT_MODEL_NAME = "allenai/specter2_base"
DEFAULT_CLASSIFICATION_ADAPTER = "allenai/specter2_classification"
DEFAULT_CACHE_DIR = Config.MODELS_DIR / "specter2_embeddings"


@dataclass(frozen=True)
class Specter2Config:
    model_name: str = DEFAULT_MODEL_NAME
    adapter_name: str = DEFAULT_CLASSIFICATION_ADAPTER
    cache_dir: Path = DEFAULT_CACHE_DIR
    batch_size: int = 16
    max_length: int = 512
    device: str | None = None

    def resolved_cache_dir(self) -> Path:
        return Path(self.cache_dir)
