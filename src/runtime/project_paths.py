from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR_NAME = "Data"
DEFAULT_STABLE_DATASET_ID = "Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407"


def _existing_child_case_insensitive(parent: Path, name: str) -> Path | None:
    if not parent.exists():
        return None
    expected = name.lower()
    for child in parent.iterdir():
        if child.name.lower() == expected:
            return child
    return None


def data_root(project_root: Path = PROJECT_ROOT) -> Path:
    existing = _existing_child_case_insensitive(project_root, DEFAULT_DATA_DIR_NAME)
    if existing:
        return existing
    return project_root / DEFAULT_DATA_DIR_NAME


@dataclass(frozen=True)
class DatasetPaths:
    dataset_id: str
    dataset_dir: Path
    input_dir: Path
    labels_dir: Path
    legacy_labels_dir: Path
    runs_dir: Path

    @property
    def label_file(self) -> Path:
        preferred = self.labels_dir / f"{self.dataset_id}.xlsx"
        legacy = self.legacy_labels_dir / f"{self.dataset_id}.xlsx"
        if preferred.exists():
            return preferred
        if legacy.exists():
            return legacy
        return preferred


@dataclass(frozen=True)
class RunPaths:
    dataset_id: str
    experiment_track: str
    tag: str
    dataset_dir: Path
    run_dir: Path
    prediction_dir: Path
    report_dir: Path
    review_dir: Path
    log_dir: Path

    def prediction_file(self, stem: str) -> Path:
        return self.prediction_dir / f"{stem}.xlsx"

    def eval_summary_file(self) -> Path:
        return self.report_dir / "Eval_Summary.xlsx"

    def unknown_review_file(self, stem: str) -> Path:
        return self.review_dir / f"Unknown_Review_{stem}.xlsx"

    def run_summary_file(self) -> Path:
        return self.run_dir / "Stable_Run_Summary.json"

    def log_file(self) -> Path:
        return self.log_dir / f"{self.experiment_track}_{self.tag}.log"


def dataset_paths(dataset_id: str, project_root: Path = PROJECT_ROOT) -> DatasetPaths:
    root = data_root(project_root)
    dataset_dir = root / dataset_id
    return DatasetPaths(
        dataset_id=dataset_id,
        dataset_dir=dataset_dir,
        input_dir=dataset_dir / "input",
        labels_dir=dataset_dir / "input" / "labels",
        legacy_labels_dir=dataset_dir / "labels",
        runs_dir=dataset_dir / "runs",
    )


def run_paths(
    dataset_id: str,
    experiment_track: str,
    tag: str,
    project_root: Path = PROJECT_ROOT,
) -> RunPaths:
    paths = dataset_paths(dataset_id, project_root=project_root)
    run_dir = paths.runs_dir / experiment_track / tag
    return RunPaths(
        dataset_id=dataset_id,
        experiment_track=experiment_track,
        tag=tag,
        dataset_dir=paths.dataset_dir,
        run_dir=run_dir,
        prediction_dir=run_dir / "predictions",
        report_dir=run_dir / "reports",
        review_dir=run_dir / "reviews",
        log_dir=run_dir / "logs",
    )


def ensure_dataset_layout(paths: DatasetPaths) -> None:
    paths.input_dir.mkdir(parents=True, exist_ok=True)
    paths.labels_dir.mkdir(parents=True, exist_ok=True)
    paths.runs_dir.mkdir(parents=True, exist_ok=True)


def ensure_run_layout(paths: RunPaths) -> None:
    paths.prediction_dir.mkdir(parents=True, exist_ok=True)
    paths.report_dir.mkdir(parents=True, exist_ok=True)
    paths.review_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)
