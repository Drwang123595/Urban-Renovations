from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.config import Schema


DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "Data"
    / "full_task_sample_10000_20260420"
    / "input"
    / "full_task_sample_10000_seed20260420.xlsx"
)
DEFAULT_SEED = 20260420
DEFAULT_SAMPLE_SIZE = 10000
DEFAULT_MIN_ABSTRACT_CHARS = 20

OUTPUT_COLUMNS = [
    Schema.TITLE,
    Schema.KEYWORDS_PLUS,
    Schema.ABSTRACT,
    Schema.WOS_CATEGORIES,
    Schema.RESEARCH_AREAS,
]


def _clean_text_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in OUTPUT_COLUMNS:
        cleaned[column] = cleaned[column].fillna("").astype(str).str.strip()
    return cleaned


def build_clean_sample(
    input_path: Path,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    min_abstract_chars: int = DEFAULT_MIN_ABSTRACT_CHARS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_excel(input_path, engine="openpyxl")
    missing = [column for column in OUTPUT_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    working = _clean_text_columns(raw[OUTPUT_COLUMNS])
    original_rows = len(working)
    title_abstract_present = (
        working[Schema.TITLE].ne("")
        & working[Schema.ABSTRACT].ne("")
    )
    short_abstract = working[Schema.ABSTRACT].str.len() < min_abstract_chars

    eligible = working[title_abstract_present & ~short_abstract].copy()
    before_dedup_rows = len(eligible)
    clean = eligible.drop_duplicates([Schema.TITLE, Schema.ABSTRACT], keep="first")
    clean_rows = len(clean)
    if clean_rows < sample_size:
        raise ValueError(
            f"Not enough clean rows for sample_size={sample_size}: clean_rows={clean_rows}"
        )

    sample = clean.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    summary = pd.DataFrame(
        [
            {"metric": "source_file", "value": str(input_path.resolve())},
            {"metric": "sample_seed", "value": seed},
            {"metric": "sample_size", "value": sample_size},
            {"metric": "min_abstract_chars", "value": min_abstract_chars},
            {"metric": "original_rows", "value": original_rows},
            {"metric": "title_or_abstract_blank_rows", "value": int((~title_abstract_present).sum())},
            {"metric": "short_abstract_rows", "value": int(short_abstract.sum())},
            {"metric": "rows_after_required_and_length_filter", "value": before_dedup_rows},
            {"metric": "exact_title_abstract_duplicate_rows_removed", "value": before_dedup_rows - clean_rows},
            {"metric": "clean_rows", "value": clean_rows},
            {"metric": "selected_rows", "value": len(sample)},
            {"metric": "missing_wos_categories_selected", "value": int(sample[Schema.WOS_CATEGORIES].eq("").sum())},
            {"metric": "missing_research_areas_selected", "value": int(sample[Schema.RESEARCH_AREAS].eq("").sum())},
        ]
    )
    return sample[OUTPUT_COLUMNS], summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a cleaned reproducible 10000-paper task sample.")
    parser.add_argument("--input", required=True, help="Source workbook path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output workbook path")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-abstract-chars", type=int, default=DEFAULT_MIN_ABSTRACT_CHARS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample, summary = build_clean_sample(
        input_path,
        sample_size=args.sample_size,
        seed=args.seed,
        min_abstract_chars=args.min_abstract_chars,
    )
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        sample.to_excel(writer, sheet_name="sample", index=False)
        summary.to_excel(writer, sheet_name="cleaning_summary", index=False)

    print(f"Saved sample workbook: {output_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
