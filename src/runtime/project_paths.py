from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR_NAME = "Data"
DEFAULT_TRAIN_DIR_NAME = "train"
DEFAULT_OUTPUT_DIR_NAME = "output"
DEFAULT_STABLE_DATASET_ID = "Urban Renovation V2.0_cleaned_article_sample_1000_local_labeled_v2_20260407"


def _slug_tokenize(value: object) -> list[str]:
    text = str(value or "").lower()
    for suffix in (".xlsx", ".xls", ".xlsm", ".csv"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    return [token for token in re.split(r"[^a-z0-9]+", text) if token]


def dataset_slug(dataset_id: object) -> str:
    """Build a short, stable, ASCII dataset identifier for output filenames."""

    tokens = _slug_tokenize(dataset_id)
    if not tokens:
        return "dataset"

    date_token = ""
    for token in reversed(tokens):
        if re.fullmatch(r"20\d{6}", token):
            date_token = token
            break

    if len(tokens) >= 4 and tokens[:3] == ["urban", "renovation", "v2"]:
        prefix = ["urban", "renovation", "v2", tokens[3]]
    else:
        prefix = tokens[:4]

    selected = list(prefix)
    if date_token and date_token not in selected:
        selected.append(date_token)
    return "_".join(selected) or "dataset"


def safe_file_stem(value: object, *, fallback: str = "output") -> str:
    tokens = _slug_tokenize(value)
    return "_".join(tokens) or fallback


def build_urban_prediction_stem(
    *,
    dataset_id: object,
    urban_method: object,
    run_tag: object,
    shot_mode: object = None,
    llm_assist_enabled: bool | None = None,
) -> str:
    method_parts = [safe_file_stem(urban_method, fallback="method")]
    if shot_mode:
        method_parts.append(safe_file_stem(shot_mode, fallback="shot"))
    if llm_assist_enabled is not None:
        method_parts.append("llm_on" if bool(llm_assist_enabled) else "llm_off")
    return (
        f"{dataset_slug(dataset_id)}__urban_renewal__"
        f"{'_'.join(method_parts)}__{safe_file_stem(run_tag, fallback='run')}"
    )


def build_spatial_prediction_stem(
    *,
    dataset_id: object,
    spatial_shot: object,
    run_tag: object,
) -> str:
    return (
        f"{dataset_slug(dataset_id)}__spatial__"
        f"{safe_file_stem(spatial_shot, fallback='shot')}__{safe_file_stem(run_tag, fallback='run')}"
    )


def build_merged_prediction_stem(
    *,
    dataset_id: object,
    run_tag: object,
) -> str:
    return f"{dataset_slug(dataset_id)}__merged__urban_renewal_spatial__{safe_file_stem(run_tag, fallback='run')}"


def build_unknown_review_stem(
    *,
    dataset_id: object,
    run_tag: object,
) -> str:
    return f"unknown_review__{dataset_slug(dataset_id)}__urban_renewal__{safe_file_stem(run_tag, fallback='run')}"


def validate_path_segment(value: object, *, field_name: str = "path segment") -> str:
    """Validate a single filesystem path segment used inside managed run paths."""

    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty.")
    if text in {".", ".."}:
        raise ValueError(f"{field_name} must be a plain directory name, got {text!r}.")
    if any(ord(ch) < 32 for ch in text):
        raise ValueError(f"{field_name} contains control characters.")
    if "/" in text or "\\" in text:
        raise ValueError(f"{field_name} must not contain path separators: {text!r}")

    windows = PureWindowsPath(text)
    posix = PurePosixPath(text)
    if windows.is_absolute() or windows.drive or posix.is_absolute():
        raise ValueError(f"{field_name} must not be an absolute or drive-qualified path: {text!r}")
    return text


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


def train_root(project_root: Path = PROJECT_ROOT) -> Path:
    return data_root(project_root) / DEFAULT_TRAIN_DIR_NAME


def output_root(project_root: Path = PROJECT_ROOT) -> Path:
    return data_root(project_root) / DEFAULT_OUTPUT_DIR_NAME


def _resolve_path(path_value: str | Path, *, project_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_train_input_path(
    input_value: str | Path,
    *,
    project_root: Path = PROJECT_ROOT,
    must_exist: bool = True,
) -> Path:
    """Resolve a user-supplied input workbook under the managed Data/train root."""

    raw_text = str(input_value or "").strip()
    if not raw_text:
        raise ValueError("Input path must not be empty.")

    root = train_root(project_root).resolve()
    raw_path = Path(raw_text)
    if not raw_path.is_absolute() and raw_path.parent == Path("."):
        candidate = root / raw_path.name
    else:
        candidate = _resolve_path(raw_path, project_root=project_root)
    resolved = candidate.resolve()

    if resolved == root or not _path_is_within(resolved, root):
        raise ValueError(f"Input path must be under Data/train: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Input workbook does not exist: {resolved}")
    return resolved


def resolve_managed_output_path(
    output_value: str | Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Resolve a user-supplied output path under the managed Data/output root."""

    raw_text = str(output_value or "").strip()
    if not raw_text:
        raise ValueError("Output path must not be empty.")

    root = output_root(project_root).resolve()
    resolved = _resolve_path(raw_text, project_root=project_root)
    if resolved == root or not _path_is_within(resolved, root):
        raise ValueError(f"Output path must be under Data/output: {resolved}")
    return resolved


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
        return self.labels_dir / f"{self.dataset_id}.xlsx"


@dataclass(frozen=True)
class RunPaths:
    dataset_id: str
    experiment_track: str
    tag: str
    dataset_dir: Path
    run_dir: Path
    prediction_dir: Path
    urban_prediction_dir: Path
    spatial_prediction_dir: Path
    merged_prediction_dir: Path
    report_dir: Path
    review_dir: Path
    log_dir: Path

    def prediction_file(self, stem: str) -> Path:
        return self.urban_prediction_dir / f"{stem}.xlsx"

    def prediction_dir_for_task(self, task_type: str) -> Path:
        task_key = str(task_type or "").strip().lower()
        if task_key == "urban_renewal":
            return self.urban_prediction_dir
        if task_key == "spatial":
            return self.spatial_prediction_dir
        if task_key == "merged":
            return self.merged_prediction_dir
        raise ValueError(f"Unknown prediction task type: {task_type!r}")

    def eval_summary_file(self) -> Path:
        return self.report_dir / "Eval_Summary.xlsx"

    def unknown_review_file(self, stem: str) -> Path:
        return self.review_dir / f"{stem}.xlsx"

    def run_summary_file(self) -> Path:
        return self.run_dir / "Stable_Run_Summary.json"

    def log_file(self) -> Path:
        return self.log_dir / f"{self.experiment_track}_{self.tag}.log"


def dataset_paths(dataset_id: str, project_root: Path = PROJECT_ROOT) -> DatasetPaths:
    dataset_id = validate_path_segment(dataset_id, field_name="dataset_id")
    dataset_dir = output_root(project_root) / dataset_id
    train_dir = train_root(project_root)
    return DatasetPaths(
        dataset_id=dataset_id,
        dataset_dir=dataset_dir,
        input_dir=train_dir,
        labels_dir=train_dir,
        legacy_labels_dir=train_dir,
        runs_dir=dataset_dir / "runs",
    )


def run_paths(
    dataset_id: str,
    experiment_track: str,
    tag: str,
    project_root: Path = PROJECT_ROOT,
) -> RunPaths:
    dataset_id = validate_path_segment(dataset_id, field_name="dataset_id")
    experiment_track = validate_path_segment(experiment_track, field_name="experiment_track")
    tag = validate_path_segment(tag, field_name="tag")
    paths = dataset_paths(dataset_id, project_root=project_root)
    run_dir = paths.runs_dir / experiment_track / tag
    return RunPaths(
        dataset_id=dataset_id,
        experiment_track=experiment_track,
        tag=tag,
        dataset_dir=paths.dataset_dir,
        run_dir=run_dir,
        prediction_dir=run_dir / "predictions",
        urban_prediction_dir=run_dir / "predictions" / "urban_renewal",
        spatial_prediction_dir=run_dir / "predictions" / "spatial",
        merged_prediction_dir=run_dir / "predictions" / "merged",
        report_dir=run_dir / "reports",
        review_dir=run_dir / "reviews",
        log_dir=run_dir / "logs",
    )


def ensure_dataset_layout(paths: DatasetPaths) -> None:
    paths.input_dir.mkdir(parents=True, exist_ok=True)
    paths.labels_dir.mkdir(parents=True, exist_ok=True)
    paths.runs_dir.mkdir(parents=True, exist_ok=True)


def ensure_run_layout(paths: RunPaths) -> None:
    paths.urban_prediction_dir.mkdir(parents=True, exist_ok=True)
    paths.spatial_prediction_dir.mkdir(parents=True, exist_ok=True)
    paths.merged_prediction_dir.mkdir(parents=True, exist_ok=True)
    paths.report_dir.mkdir(parents=True, exist_ok=True)
    paths.review_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)


def safe_dataset_paths(dataset_id: str, project_root: Path = PROJECT_ROOT) -> DatasetPaths:
    return dataset_paths(dataset_id, project_root=project_root)


def safe_run_paths(
    dataset_id: str,
    experiment_track: str,
    tag: str,
    project_root: Path = PROJECT_ROOT,
) -> RunPaths:
    return run_paths(dataset_id, experiment_track, tag, project_root=project_root)
