from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PROJECT_ROOT))

from src.runtime.project_paths import (
    DEFAULT_STABLE_DATASET_ID,
    PROJECT_ROOT,
    dataset_paths,
    ensure_dataset_layout,
    ensure_run_layout,
    run_paths,
)
from src.prompting.manifest import load_prompt_manifest

DEFAULT_DATASET_ID = DEFAULT_STABLE_DATASET_ID
STABLE_MODEL_NAME = "deepseek-v4-flash"
DEFAULT_TAG = "20260427_deepseek_v4_flash_stable"
DEFAULT_ORDER_ID = "canonical_title_order"
DEFAULT_MAX_SAMPLES_PER_WINDOW = 50
EXPECTED_FULL_SAMPLE_COUNT = 1000


@dataclass(frozen=True)
class StableThresholds:
    required_model_name: str = STABLE_MODEL_NAME
    min_accuracy: float = 88.0
    min_precision: float = 0.956
    min_recall: float = 0.94
    min_f1: float = 0.948
    max_fp: int = 34
    max_fn: int = 48
    max_unknown_count: int = 38
    max_unknown_rate: float = 0.038
    min_unknown_hint_resolution_accuracy: float = 92.0
    max_llm_used_sum: int = 0
    min_explanation_coverage: float = 1.0
    min_rule_stack_coverage: float = 1.0
    min_binary_evidence_coverage: float = 1.0


@dataclass(frozen=True)
class StablePaths:
    dataset_id: str
    tag: str
    run_dir: Path
    labels_file: Path
    output_dir: Path
    result_dir: Path
    review_dir: Path
    log_dir: Path
    prediction_file: Path
    eval_summary_file: Path
    unknown_review_file: Path
    run_summary_file: Path
    log_file: Path


def resolve_python() -> Path:
    preferred = PROJECT_ROOT / ".venv-bertopic313" / "Scripts" / "python.exe"
    if preferred.exists():
        return preferred
    return Path(sys.executable)


def build_paths(dataset_id: str = DEFAULT_DATASET_ID, tag: str = DEFAULT_TAG) -> StablePaths:
    dataset_layout = dataset_paths(dataset_id)
    run_layout = run_paths(dataset_id, "stable_release", tag)
    stem = f"urban_renewal_three_stage_hybrid_few_llm_on_{tag}"
    labels_file = dataset_layout.label_file
    prediction_file = run_layout.prediction_file(stem)
    eval_summary_file = run_layout.eval_summary_file()
    unknown_review_file = run_layout.review_dir / f"Unknown_Review_hybrid_llm_on_{tag}.xlsx"
    run_summary_file = run_layout.run_summary_file()
    log_file = run_layout.log_file()
    return StablePaths(
        dataset_id=dataset_id,
        tag=tag,
        run_dir=run_layout.run_dir,
        labels_file=labels_file,
        output_dir=run_layout.prediction_dir,
        result_dir=run_layout.report_dir,
        review_dir=run_layout.review_dir,
        log_dir=run_layout.log_dir,
        prediction_file=prediction_file,
        eval_summary_file=eval_summary_file,
        unknown_review_file=unknown_review_file,
        run_summary_file=run_summary_file,
        log_file=log_file,
    )


def build_classification_command(
    python_path: Path,
    paths: StablePaths,
    *,
    limit: int | None = None,
    max_samples_per_window: int = DEFAULT_MAX_SAMPLES_PER_WINDOW,
) -> list[str]:
    command = [
        str(python_path),
        str(PROJECT_ROOT / "scripts" / "pipeline" / "main_py313.py"),
        "--non-interactive",
        "--task",
        "urban_renewal",
        "--experiment-track",
        "stable_release",
        "--input",
        str(paths.labels_file),
        "--output",
        str(paths.prediction_file),
        "--urban-method",
        "three_stage_hybrid",
        "--hybrid-llm-assist",
        "on",
        "--urban-shot",
        "few",
        "--dataset-id",
        paths.dataset_id,
        "--truth-file",
        str(paths.labels_file),
        "--order-id",
        DEFAULT_ORDER_ID,
        "--max-samples-per-window",
        str(max_samples_per_window),
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    return command


def build_evaluate_command(python_path: Path, paths: StablePaths) -> list[str]:
    return [
        str(python_path),
        str(PROJECT_ROOT / "scripts" / "evaluation" / "evaluate.py"),
        "--experiment-track",
        "stable_release",
        "--truth",
        str(paths.labels_file),
        "--pred",
        str(paths.prediction_file),
        "--report-dir",
        str(paths.result_dir),
        "--pred-scope",
        "urban_renewal",
        "--strict",
        "--strict-truth-match",
    ]


def build_unknown_review_command(python_path: Path, paths: StablePaths) -> list[str]:
    return [
        str(python_path),
        str(PROJECT_ROOT / "scripts" / "evaluation" / "export_unknown_review.py"),
        "--pred",
        str(paths.prediction_file),
        "--truth",
        str(paths.labels_file),
        "--output",
        str(paths.unknown_review_file),
    ]


def command_for_display(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def stable_child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["LLM_MODEL_NAME"] = STABLE_MODEL_NAME
    env["DEEPSEEK_MODEL"] = STABLE_MODEL_NAME
    return env


def run_logged(
    label: str,
    command: list[str],
    *,
    log_file: Path,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
) -> None:
    line = f"\n[{datetime.now().isoformat(timespec='seconds')}] {label}\n{command_for_display(command)}\n"
    print(line.strip())
    if dry_run:
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            check=False,
        )
        handle.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] exit_code={completed.returncode}\n")
    if completed.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {completed.returncode}. See log: {log_file}")


def _find_urban_metric(summary_path: Path, file_stem: str) -> pd.Series:
    all_metrics = pd.read_excel(summary_path, sheet_name="All Metrics", engine="openpyxl")
    rows = all_metrics[(all_metrics["Metric"] == "Urban Renewal") & (all_metrics["File"] == file_stem)]
    if rows.empty:
        rows = all_metrics[all_metrics["Metric"] == "Urban Renewal"]
    if rows.empty:
        raise ValueError("Eval_Summary.xlsx does not contain an Urban Renewal row.")
    return rows.iloc[0]


def collect_stable_metrics(paths: StablePaths) -> dict[str, Any]:
    file_stem = paths.prediction_file.stem
    pred_df = pd.read_excel(paths.prediction_file, engine="openpyxl")
    manifest = load_prompt_manifest(paths.prediction_file) or {}
    runtime = manifest.get("runtime") or {}
    urban = _find_urban_metric(paths.eval_summary_file, file_stem)
    unknown_df = pd.read_excel(paths.eval_summary_file, sheet_name="Unknown Rate", engine="openpyxl")
    unknown_rows = unknown_df[unknown_df["File"] == file_stem]
    if unknown_rows.empty:
        raise ValueError("Eval_Summary.xlsx does not contain the expected Unknown Rate row.")
    unknown = unknown_rows.iloc[0]
    decision_df = pd.read_excel(paths.eval_summary_file, sheet_name="Decision Source Metrics", engine="openpyxl")
    decisions = decision_df[decision_df["File"] == file_stem].copy()
    unknown_hint = decisions[decisions["Decision Source"] == "unknown_hint_resolution"]
    unknown_review = decisions[decisions["Decision Source"] == "unknown_review"]
    explainability_df = pd.read_excel(paths.eval_summary_file, sheet_name="Explainability Quality", engine="openpyxl")
    explainability_rows = explainability_df[explainability_df["File"] == file_stem]
    if explainability_rows.empty:
        raise ValueError("Eval_Summary.xlsx does not contain the expected Explainability Quality row.")
    explainability = explainability_rows.iloc[0]
    llm_used_sum = int(pd.to_numeric(pred_df.get("llm_used", pd.Series(dtype=int)), errors="coerce").fillna(0).sum())
    llm_attempted_sum = int(
        pd.to_numeric(pred_df.get("llm_attempted", pd.Series(dtype=int)), errors="coerce").fillna(0).sum()
    )
    return {
        "file": file_stem,
        "model_name": str(runtime.get("model_name", "")),
        "rows": int(len(pred_df)),
        "accuracy": float(urban["Accuracy"]),
        "correct": int(urban["Correct"]),
        "total": int(urban["Total"]),
        "tp": int(urban["TP"]),
        "tn": int(urban["TN"]),
        "fp": int(urban["FP"]),
        "fn": int(urban["FN"]),
        "precision": float(urban["Precision"]),
        "recall": float(urban["Recall"]),
        "f1": float(urban["F1"]),
        "predicted_unknown_count": int(unknown["Predicted Unknown Count"]),
        "predicted_unknown_rate": float(unknown["Predicted Unknown Rate"]),
        "unknown_hint_resolution_total": int(unknown_hint.iloc[0]["Total"]) if not unknown_hint.empty else 0,
        "unknown_hint_resolution_accuracy": float(unknown_hint.iloc[0]["Accuracy"]) if not unknown_hint.empty else 0.0,
        "unknown_review_total": int(unknown_review.iloc[0]["Total"]) if not unknown_review.empty else 0,
        "decision_sources": decisions["Decision Source"].astype(str).tolist(),
        "llm_used_sum": llm_used_sum,
        "llm_attempted_sum": llm_attempted_sum,
        "explanation_coverage": float(explainability["Decision Explanation Coverage"]),
        "rule_stack_coverage": float(explainability["Rule Stack Coverage"]),
        "binary_evidence_coverage": float(explainability["Binary Evidence Coverage"]),
    }


def validate_gates(metrics: dict[str, Any], thresholds: StableThresholds, *, expected_rows: int) -> list[str]:
    failures = []
    checks = [
        (
            metrics.get("model_name") == thresholds.required_model_name,
            f"model_name expected {thresholds.required_model_name}, got {metrics.get('model_name') or 'missing'}",
        ),
        (metrics["rows"] == expected_rows, f"rows expected {expected_rows}, got {metrics['rows']}"),
        (metrics["total"] == expected_rows, f"evaluated total expected {expected_rows}, got {metrics['total']}"),
        (metrics["accuracy"] >= thresholds.min_accuracy, "accuracy below stable threshold"),
        (metrics["precision"] >= thresholds.min_precision, "precision below stable threshold"),
        (metrics["recall"] >= thresholds.min_recall, "recall below stable threshold"),
        (metrics["f1"] >= thresholds.min_f1, "f1 below stable threshold"),
        (metrics["fp"] <= thresholds.max_fp, "FP above stable threshold"),
        (metrics["fn"] <= thresholds.max_fn, "FN above stable threshold"),
        (
            metrics["predicted_unknown_count"] <= thresholds.max_unknown_count,
            "Predicted Unknown Count above stable threshold",
        ),
        (
            metrics["predicted_unknown_rate"] <= thresholds.max_unknown_rate,
            "Predicted Unknown Rate above stable threshold",
        ),
        (
            metrics["unknown_hint_resolution_accuracy"] >= thresholds.min_unknown_hint_resolution_accuracy,
            "unknown_hint_resolution accuracy below stable threshold",
        ),
        (metrics["llm_used_sum"] <= thresholds.max_llm_used_sum, "llm_used violates stable release contract"),
        (
            metrics["explanation_coverage"] >= thresholds.min_explanation_coverage,
            "decision explanation coverage below stable threshold",
        ),
        (
            metrics["rule_stack_coverage"] >= thresholds.min_rule_stack_coverage,
            "decision rule stack coverage below stable threshold",
        ),
        (
            metrics["binary_evidence_coverage"] >= thresholds.min_binary_evidence_coverage,
            "binary decision evidence coverage below stable threshold",
        ),
    ]
    for passed, message in checks:
        if not passed:
            failures.append(message)
    return failures


def write_run_summary(
    paths: StablePaths,
    metrics: dict[str, Any],
    thresholds: StableThresholds,
    commands: dict[str, list[str]],
    gate_status: str,
    gate_failures: list[str],
) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_id": paths.dataset_id,
        "tag": paths.tag,
        "paths": {key: str(value) for key, value in asdict(paths).items() if isinstance(value, Path)},
        "metrics": metrics,
        "thresholds": asdict(thresholds),
        "gate_status": gate_status,
        "gate_failures": gate_failures,
        "commands": {name: command_for_display(command) for name, command in commands.items()},
    }
    paths.run_summary_file.parent.mkdir(parents=True, exist_ok=True)
    paths.run_summary_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the locked stable urban-renewal release pipeline.")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="Stable run tag used under baseline_<tag> directories")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="Locked stable dataset identifier")
    parser.add_argument("--python", default=str(resolve_python()), help="Python executable for child scripts")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test row limit")
    parser.add_argument(
        "--max-samples-per-window",
        type=int,
        default=DEFAULT_MAX_SAMPLES_PER_WINDOW,
        help="Window size recorded in the manifest",
    )
    parser.add_argument("--skip-classification", action="store_true", help="Use an existing prediction workbook")
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing prediction workbook")
    parser.add_argument("--no-gate", action="store_true", help="Allow smoke/partial artifacts without enforcing stable gates")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and paths without running them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["LLM_MODEL_NAME"] = STABLE_MODEL_NAME
    os.environ["DEEPSEEK_MODEL"] = STABLE_MODEL_NAME
    paths = build_paths(dataset_id=args.dataset_id, tag=args.tag)
    python_path = Path(args.python)
    thresholds = StableThresholds()
    dataset_layout = dataset_paths(args.dataset_id)
    ensure_dataset_layout(dataset_layout)
    ensure_run_layout(run_paths(args.dataset_id, "stable_release", args.tag))

    if not paths.labels_file.exists():
        raise FileNotFoundError(f"Missing stable label workbook: {paths.labels_file}")
    if args.no_gate and args.limit is None:
        raise ValueError("--no-gate is only allowed for partial smoke runs with --limit.")
    if not args.skip_classification and paths.prediction_file.exists() and not args.force and not args.dry_run:
        raise FileExistsError(
            f"Prediction file already exists: {paths.prediction_file}. "
            "Use --skip-classification to evaluate it or --force to overwrite it."
        )

    commands = {
        "classification": build_classification_command(
            python_path,
            paths,
            limit=args.limit,
            max_samples_per_window=args.max_samples_per_window,
        ),
        "evaluation": build_evaluate_command(python_path, paths),
        "unknown_review": build_unknown_review_command(python_path, paths),
    }

    print(f"Stable dataset: {paths.dataset_id}")
    print(f"Output: {paths.prediction_file}")
    print(f"Result dir: {paths.result_dir}")
    print(f"Log: {paths.log_file}")

    if args.skip_classification:
        print("Skipping classification; using existing prediction workbook.")
    else:
        run_logged(
            "classification",
            commands["classification"],
            log_file=paths.log_file,
            dry_run=args.dry_run,
            env=stable_child_env(),
        )
    run_logged("evaluation", commands["evaluation"], log_file=paths.log_file, dry_run=args.dry_run, env=stable_child_env())
    run_logged(
        "unknown_review",
        commands["unknown_review"],
        log_file=paths.log_file,
        dry_run=args.dry_run,
        env=stable_child_env(),
    )

    if args.dry_run:
        return

    metrics = collect_stable_metrics(paths)
    gate_failures: list[str] = []
    if args.no_gate:
        gate_status = "skipped_by_request"
    elif args.limit is not None:
        gate_status = "skipped_partial_run"
    else:
        gate_failures = validate_gates(metrics, thresholds, expected_rows=EXPECTED_FULL_SAMPLE_COUNT)
        gate_status = "passed" if not gate_failures else "failed"
    write_run_summary(paths, metrics, thresholds, commands, gate_status, gate_failures)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Stable run summary saved to: {paths.run_summary_file}")
    if gate_failures:
        raise RuntimeError("Stable gates failed: " + "; ".join(gate_failures))
    if gate_status == "passed":
        print("Stable gates passed.")
    else:
        print(f"Stable gates {gate_status}.")


if __name__ == "__main__":
    main()
