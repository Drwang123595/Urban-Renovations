from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.spatial_blind_label_common import parse_batch_json, validate_label
from src.runtime.config import Schema


DEFAULT_INPUT = PROJECT_ROOT / "Data" / "spatial_sample_2000_20260428" / "input" / "spatial_sample_2000_seed20260428.xlsx"
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "Data" / "spatial_sample_2000_20260428" / "analysis"
DEFAULT_OUTPUT = DEFAULT_ANALYSIS_DIR / "codex_gpt_blind_labels_2000_20260428.xlsx"
DEFAULT_CHECKPOINT = DEFAULT_ANALYSIS_DIR / "codex_gpt_blind_labels_2000_20260428.jsonl"
DEFAULT_MODEL = "gpt-5.4-mini"


def load_checkpoint(path: Path) -> dict[int, Dict[str, Any]]:
    records: dict[int, Dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_index = record.get("row_index")
            if row_index is None:
                continue
            records[int(row_index)] = record
    return records


def append_checkpoint(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_batches(items: list[Dict[str, Any]], batch_size: int) -> Iterable[list[Dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_prompt(batch: list[Dict[str, Any]], retry_reason: str = "") -> str:
    compact_records = [
        {
            "row_index": item["row_index"],
            "source_row_0based": item["source_row_0based"],
            "title": item["title"],
            "abstract": item["abstract"],
        }
        for item in batch
    ]
    return f"""
You are simulating an expert human coder for bibliometric spatial research annotation.

Blind labeling rules:
- Use only each record's title and abstract. Do not use any pipeline prediction.
- Spatial research is true only when the core objective, empirical analysis, data collection, case study, or policy recommendation is anchored in a named or specifically identifiable real geographic area on Earth.
- If a text implies a site/city/case exists but does not identify it, label non-spatial.
- Never infer an unnamed city, site, neighborhood, project area, brownfield site, or "case study context".
- Restricted implicit country/region is allowed only when explicit institutions, policies, or national/regional programs make it identifiable; never use this to infer an unnamed city/site/neighborhood.

Scale levels:
1. Global Scale
2. Multi-national / Continental Scale
3. National / Single-country Scale
4. Multi-provincial / Sub-national Regional Scale
5. Single-provincial / State Scale
6. Multi-city / Megaregion Scale
7. Single-city / Municipal Scale
8. District / County Scale
9. Micro / Neighborhood / Block Scale

Return a strict JSON array with exactly one object per input record and no extra text.
Each object must have exactly these keys:
- row_index
- source_row_0based
- gpt_is_spatial: 1 or 0
- gpt_spatial_scale_level: one of the 9 scale strings above, or null
- gpt_specific_study_area: named/specific area from title/abstract, or null

If gpt_is_spatial is 0, both scale and area must be null.
Do not output placeholders such as "unspecified city", "study area", "a brownfield site", "unknown site", "case study context".

{f"Previous batch output was invalid because: {retry_reason}. Correct the JSON array." if retry_reason else ""}

INPUT_RECORDS_JSON:
{json.dumps(compact_records, ensure_ascii=False, indent=2)}
""".strip()


def run_codex_exec(prompt: str, model: str, workdir: Path, timeout: int) -> tuple[str, str, int]:
    with tempfile.TemporaryDirectory(prefix="codex_spatial_label_") as tmpdir:
        output_path = Path(tmpdir) / "last_message.txt"
        cmd = [
            "codex.cmd",
            "exec",
            "--model",
            model,
            "--sandbox",
            "read-only",
            "--cd",
            str(workdir),
            "--color",
            "never",
            "--output-last-message",
            str(output_path),
            "-",
        ]
        completed = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
        )
        last_message = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
        combined_log = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
        return last_message, combined_log, completed.returncode


def is_usage_limit_error(text: str) -> bool:
    lowered = text.lower()
    return "usage limit" in lowered or "try again" in lowered and "codex" in lowered


def label_batch(
    batch: list[Dict[str, Any]],
    model: str,
    timeout: int,
    max_retries: int,
) -> tuple[list[Dict[str, Any]], str]:
    retry_reason = ""
    for attempt in range(max_retries + 1):
        prompt = build_prompt(batch, retry_reason=retry_reason)
        raw, log, return_code = run_codex_exec(prompt, model=model, workdir=PROJECT_ROOT, timeout=timeout)
        if is_usage_limit_error(raw) or is_usage_limit_error(log):
            raise RuntimeError("codex_usage_limit: Codex CLI reports usage limit; resume after quota reset.")
        if return_code != 0 and not raw:
            retry_reason = f"codex_cli_return_code_{return_code}"
            time.sleep(2 * (attempt + 1))
            continue
        try:
            parsed = parse_batch_json(raw)
            by_row = {record["row_index"]: record for record in parsed}
            records: list[Dict[str, Any]] = []
            for item in batch:
                record = by_row.get(item["row_index"])
                if record is None:
                    raise ValueError(f"missing_row_{item['row_index']}")
                status, error = validate_label(record)
                records.append(
                    {
                        "row_index": item["row_index"],
                        "source_row_0based": item["source_row_0based"],
                        Schema.TITLE: item["title"],
                        Schema.ABSTRACT: item["abstract"],
                        **record,
                        "label_status": status,
                        "label_error": error,
                        "label_model": model,
                        "label_attempts": attempt + 1,
                        "raw_response": raw,
                    }
                )
            if any(record["label_status"] != "valid" for record in records):
                retry_reason = ";".join(
                    sorted({record["label_error"] for record in records if record["label_error"]})
                )
                if attempt < max_retries:
                    continue
            return records, raw
        except Exception as error:  # noqa: BLE001 - retry parser/model formatting issues.
            retry_reason = f"{type(error).__name__}: {error}"
            if attempt < max_retries:
                time.sleep(2 * (attempt + 1))
                continue
    failed = []
    for item in batch:
        failed.append(
            {
                "row_index": item["row_index"],
                "source_row_0based": item["source_row_0based"],
                Schema.TITLE: item["title"],
                Schema.ABSTRACT: item["abstract"],
                "gpt_is_spatial": "",
                "gpt_spatial_scale_level": "",
                "gpt_specific_study_area": "",
                "label_status": "failed",
                "label_error": retry_reason or "codex_batch_failed",
                "label_model": model,
                "label_attempts": max_retries + 1,
                "raw_response": "",
            }
        )
    return failed, ""


def dataframe_to_items(df: pd.DataFrame, start: int = 0, limit: Optional[int] = None) -> list[Dict[str, Any]]:
    selected = df.iloc[start:]
    if limit is not None:
        selected = selected.head(limit)
    items: list[Dict[str, Any]] = []
    for row_index, row in selected.iterrows():
        source_row = row.get("source_row_0based", row_index)
        items.append(
            {
                "row_index": int(row_index),
                "source_row_0based": int(source_row) if pd.notna(source_row) else int(row_index),
                "title": str(row.get(Schema.TITLE, "") or ""),
                "abstract": str(row.get(Schema.ABSTRACT, "") or ""),
            }
        )
    return items


def save_output(records: list[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).sort_values("row_index").to_excel(output, index=False, engine="openpyxl")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blindly label spatial research fields with Codex CLI GPT.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", default=os.environ.get("CODEX_GPT_LABEL_MODEL", DEFAULT_MODEL))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--probe-only", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.batch_size < 1:
        raise RuntimeError("--batch-size must be >= 1")

    df = pd.read_excel(args.input, engine="openpyxl")
    missing = sorted({Schema.TITLE, Schema.ABSTRACT} - set(df.columns))
    if missing:
        raise RuntimeError(f"Input missing required columns: {missing}")
    if len(df) != 2000 and args.limit is None:
        raise RuntimeError(f"Expected 2000 input rows for full run, got {len(df)}")

    items = dataframe_to_items(df, start=args.start, limit=args.limit)
    if args.probe_only:
        items = items[:1]
        args.batch_size = 1

    checkpoint = load_checkpoint(args.checkpoint)
    todo = [item for item in items if checkpoint.get(item["row_index"], {}).get("label_status") != "valid"]
    print(f"[INFO] total_target={len(items)} valid_checkpoint={len(items)-len(todo)} todo={len(todo)}")

    try:
        for batch_no, batch in enumerate(iter_batches(todo, args.batch_size), start=1):
            records, _ = label_batch(
                batch,
                model=args.model,
                timeout=args.timeout,
                max_retries=args.max_retries,
            )
            for record in records:
                append_checkpoint(args.checkpoint, record)
                checkpoint[int(record["row_index"])] = record
            valid_count = sum(1 for record in checkpoint.values() if record.get("label_status") == "valid")
            print(f"[INFO] batch={batch_no} wrote={len(records)} valid_checkpoint={valid_count}")
            if args.probe_only:
                break
    except RuntimeError as error:
        if "codex_usage_limit" in str(error):
            print(f"[ERROR] {error}")
            save_output(list(checkpoint.values()), args.output)
            return 2
        raise

    final_records = [checkpoint[item["row_index"]] for item in items if item["row_index"] in checkpoint]
    save_output(final_records, args.output)
    status_counts = pd.DataFrame(final_records)["label_status"].value_counts(dropna=False).to_dict() if final_records else {}
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "checkpoint": str(args.checkpoint),
        "model": args.model,
        "target_rows": len(items),
        "output_rows": len(final_records),
        "status_counts": status_counts,
    }
    write_json(args.output.with_suffix(".summary.json"), summary)
    print(f"[INFO] labels saved to {args.output}")
    print(f"[INFO] checkpoint saved to {args.checkpoint}")
    print(f"[INFO] status_counts={status_counts}")
    if args.probe_only and status_counts.get("valid", 0) != 1:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
