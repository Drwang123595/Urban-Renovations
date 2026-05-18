"""Urban-renewal pipeline I/O helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ...runtime.config import Schema
from .contracts import URBAN_RESULT_COLUMNS, apply_urban_output_defaults


def build_urban_output_row(
    title: str,
    abstract: str,
    result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one urban prediction row according to the output contract."""

    result = result or {}
    urban_label = result.get(Schema.IS_URBAN_RENEWAL, "0")
    if urban_label in (None, ""):
        urban_label = result.get("final_label", "0")
    urban_label = "0" if urban_label in (None, "") else str(urban_label)
    urban_flag = result.get("urban_flag", urban_label)
    urban_flag = urban_label if urban_flag in (None, "") else urban_flag
    output = {
        Schema.TITLE: title,
        Schema.ABSTRACT: abstract,
        Schema.IS_URBAN_RENEWAL: urban_label,
        "urban_flag": urban_flag,
        "urban_parse_reason": result.get("urban_parse_reason", "missing_parse_reason"),
    }
    for column in URBAN_RESULT_COLUMNS:
        if column in result:
            output[column] = result[column]
    apply_urban_output_defaults(output)
    if "final_label" in output and output.get("final_label") in (None, ""):
        output["final_label"] = urban_label
    return output


def build_urban_prediction_frame(rows: list[Dict[str, Any]]) -> pd.DataFrame:
    """Create the urban prediction frame from output rows."""

    return pd.DataFrame(rows)


def write_urban_prediction_checkpoint(rows: list[Dict[str, Any]], output_path: Path) -> None:
    """Write the current urban prediction rows as a checkpoint workbook."""

    build_urban_prediction_frame(rows).to_excel(output_path, index=False, engine="openpyxl")
