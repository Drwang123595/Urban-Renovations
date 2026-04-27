from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from ..runtime.config import Config


EXPLICIT_TRAINING_NAME_MARKERS = (
    "train_only",
    "training_only",
    "calibration_train",
    "family_gate_train",
    "boundary_train",
)


def stable_release_training_workbook() -> Path:
    return Config.TRAIN_DIR / f"{Config.STABLE_RELEASE_DATASET_ID}.xlsx"


def is_stable_release_training_workbook(path: Path) -> bool:
    candidate = Path(path).resolve()
    stable_path = stable_release_training_workbook().resolve()
    return candidate == stable_path or candidate.name == stable_path.name


def is_explicit_training_workbook(path: Path) -> bool:
    lower_name = Path(path).name.lower()
    stem = Path(path).stem.lower()
    if stem == Config.LEGACY_BASELINE_DATASET_ID.lower():
        return True
    return any(marker in lower_name for marker in EXPLICIT_TRAINING_NAME_MARKERS)


def allowed_training_workbooks(train_dir: Path | None = None) -> List[Path]:
    root = Path(train_dir or Config.TRAIN_DIR)
    if not root.exists():
        return []
    candidates = sorted(path for path in root.glob("*.xlsx") if is_explicit_training_workbook(path))
    assert_training_source_contract(candidates)
    return candidates


def assert_training_source_contract(paths: Iterable[Path]) -> None:
    bad = [Path(path) for path in paths if is_stable_release_training_workbook(Path(path))]
    if bad:
        joined = ", ".join(sorted(str(path.resolve()) for path in bad))
        raise ValueError(
            "stable_release workbook must never be used as a training source: "
            f"{joined}"
        )
