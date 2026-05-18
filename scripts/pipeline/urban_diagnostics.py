from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.urban.pipeline.diagnostics import build_urban_diagnostics, build_urban_diagnostics_frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize urban-renewal prediction diagnostics.")
    parser.add_argument("input", type=Path, help="Prediction workbook or CSV file to inspect.")
    parser.add_argument("--output", type=Path, default=None, help="Optional Excel/CSV diagnostics output path.")
    parser.add_argument("--sheet", default=0, help="Excel sheet name or index to read. Defaults to the first sheet.")
    args = parser.parse_args(argv)

    frame = _read_frame(args.input, sheet=args.sheet)
    diagnostics = build_urban_diagnostics(frame)

    if args.output is not None:
        _write_diagnostics(args.output, frame)
    else:
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2, default=str))
    return 0


def _read_frame(path: Path, *, sheet: Any = 0) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        sheet_arg: Any = sheet
        if isinstance(sheet, str) and sheet.strip().isdigit():
            sheet_arg = int(sheet)
        return pd.read_excel(path, sheet_name=sheet_arg, engine="openpyxl")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported diagnostics input format: {path.suffix}")


def _write_diagnostics(path: Path, frame: pd.DataFrame) -> None:
    diagnostics_frame = build_urban_diagnostics_frame(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        diagnostics_frame.to_excel(path, index=False, engine="openpyxl")
        return
    if suffix == ".csv":
        diagnostics_frame.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported diagnostics output format: {path.suffix}")


if __name__ == "__main__":
    raise SystemExit(main())
