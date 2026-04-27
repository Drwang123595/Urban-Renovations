from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.config import Config, Schema


RESULT_COLUMNS = [
    Schema.IS_URBAN_RENEWAL,
    Schema.IS_SPATIAL,
    Schema.SPATIAL_LEVEL,
    Schema.SPATIAL_DESC,
]


def _normalise_filters(strategies: Optional[Iterable[str]]) -> set[str]:
    return {str(item).strip().lower() for item in strategies or [] if str(item).strip()}


def _matches_strategy(file_path: Path, filters: set[str]) -> bool:
    if not filters:
        return True
    stem = file_path.stem.lower()
    parts = stem.split("_")
    candidates = {stem}
    candidates.update(parts)
    if len(parts) >= 2:
        candidates.add("_".join(parts[:2]))
    return bool(candidates & filters) or any(stem.startswith(f"{item}_") for item in filters)


def _discover_latest_prediction_dir(task_dir: Path) -> Optional[Path]:
    runs_dir = task_dir / "runs"
    if not runs_dir.exists():
        return None
    candidates = [path for path in runs_dir.glob("*/*/predictions") if path.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _resolve_prediction_dir(
    task_name: Optional[str],
    *,
    run_dir: Optional[Path] = None,
    prediction_dir: Optional[Path] = None,
) -> Path:
    if prediction_dir is not None:
        return prediction_dir
    if run_dir is not None:
        return run_dir / "predictions"
    if not task_name:
        raise ValueError("task_name is required unless --run-dir or --pred-dir is provided")

    task_dir = Config.DATA_DIR / task_name
    legacy_output = task_dir / "output"
    if legacy_output.exists():
        return legacy_output

    discovered = _discover_latest_prediction_dir(task_dir)
    if discovered is not None:
        return discovered
    return legacy_output


def _resolve_result_dir(prediction_dir: Path, result_dir: Optional[Path] = None) -> Path:
    if result_dir is not None:
        return result_dir
    if prediction_dir.name == "predictions":
        return prediction_dir.parent / "reports"
    return prediction_dir.parent / "Result"


def _prefix_for_file(file_path: Path) -> str:
    return f"[{file_path.stem.upper()}]"


def _result_columns_present(df: pd.DataFrame) -> list[str]:
    return [column for column in RESULT_COLUMNS if column in df.columns]


def merge_results(
    task_name: Optional[str] = None,
    strategies: Optional[List[str]] = None,
    *,
    run_dir: Optional[Path] = None,
    prediction_dir: Optional[Path] = None,
    result_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Merge prediction workbooks from one task or run into a comparison workbook."""
    output_dir = _resolve_prediction_dir(
        task_name,
        run_dir=Path(run_dir) if run_dir is not None else None,
        prediction_dir=Path(prediction_dir) if prediction_dir is not None else None,
    )
    reports_dir = _resolve_result_dir(output_dir, Path(result_dir) if result_dir is not None else None)

    if not output_dir.exists():
        print(f"Prediction directory not found: {output_dir}")
        return None

    filters = _normalise_filters(strategies)
    files = sorted(
        file_path
        for file_path in output_dir.glob("*.xlsx")
        if _matches_strategy(file_path, filters)
    )
    if not files:
        suffix = f" for strategies={sorted(filters)}" if filters else ""
        print(f"No prediction files found in {output_dir}{suffix}.")
        return None

    print(f"Prediction directory: {output_dir}")
    print(f"Found {len(files)} result files.")

    merged_df = None
    for file_path in files:
        print(f"Loading {file_path.name}...")
        df = pd.read_excel(file_path, engine="openpyxl")
        prefix = _prefix_for_file(file_path)
        rename_map = {column: f"{prefix} {column}" for column in RESULT_COLUMNS}

        if merged_df is None:
            merged_df = df.rename(columns=rename_map)
            continue

        result_cols = _result_columns_present(df)
        if not result_cols:
            print(f"[WARN] No known result columns found in {file_path.name}; skipping.")
            continue
        subset = df[result_cols].rename(columns=rename_map)
        merged_df = pd.concat([merged_df, subset], axis=1)

    if merged_df is None:
        print("Merge failed.")
        return None

    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = reports_dir / f"merged_comparison_{timestamp}.xlsx"
    merged_df.to_excel(output_path, index=False, engine="openpyxl")
    print(f"Successfully merged results to: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge prediction workbooks from a legacy task output folder or a canonical run directory."
    )
    parser.add_argument("task_name", nargs="?", default="test1", help="Task folder under Data/, e.g. test1")
    parser.add_argument("strategies", nargs="*", help="Optional strategy/file-stem filters")
    parser.add_argument("--run-dir", type=Path, default=None, help="Canonical run directory containing predictions/")
    parser.add_argument("--pred-dir", type=Path, default=None, help="Prediction directory to merge")
    parser.add_argument("--result-dir", type=Path, default=None, help="Directory for merged comparison workbook")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_results(
        args.task_name,
        args.strategies,
        run_dir=args.run_dir,
        prediction_dir=args.pred_dir,
        result_dir=args.result_dir,
    )


if __name__ == "__main__":
    main()
