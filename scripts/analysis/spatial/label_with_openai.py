from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.config import Schema
from src.runtime.llm_client import LLMQuotaExceededError


DEFAULT_INPUT = PROJECT_ROOT / "Data" / "spatial_sample_2000_20260428" / "input" / "spatial_sample_2000_seed20260428.xlsx"
DEFAULT_ANALYSIS_DIR = PROJECT_ROOT / "Data" / "spatial_sample_2000_20260428" / "analysis"
DEFAULT_OUTPUT = DEFAULT_ANALYSIS_DIR / "gpt_blind_labels_2000_20260428.xlsx"
DEFAULT_CHECKPOINT = DEFAULT_ANALYSIS_DIR / "gpt_blind_labels_2000_20260428.jsonl"
DEFAULT_MODEL = "gpt-5.4-mini"
QUOTA_ERROR_TOKENS = (
    "DAILY_LIMIT_EXCEEDED",
    "USAGE_LIMIT_EXCEEDED",
    "QUOTA_EXCEEDED",
    "INSUFFICIENT_QUOTA",
    "DAILY USAGE LIMIT EXCEEDED",
    "TOO MANY REQUESTS",
)

SCALE_LEVELS = {
    "1": "1. Global Scale",
    "2": "2. Multi-national / Continental Scale",
    "3": "3. National / Single-country Scale",
    "4": "4. Multi-provincial / Sub-national Regional Scale",
    "5": "5. Single-provincial / State Scale",
    "6": "6. Multi-city / Megaregion Scale",
    "7": "7. Single-city / Municipal Scale",
    "8": "8. District / County Scale",
    "9": "9. Micro / Neighborhood / Block Scale",
}

PLACEHOLDER_AREA_TERMS = (
    "unspecified",
    "unknown",
    "unnamed",
    "not specified",
    "case study context",
    "implicit city",
    "unknown site",
)

GENERIC_AREA_PATTERN = re.compile(
    r"^(?:an?\s+|the\s+)?"
    r"(?:(?:selected|local|urban|brownfield|ecologically sensitive|contentious|"
    r"case study|study)\s+)*"
    r"(?:city|site|case study|study area|urban area|project area|municipality|"
    r"neighbou?rhood|district|block|corridor|development|area)"
    r"(?:\s+(?:under study|in\s+(?:an?\s+|the\s+)?"
    r"(?:city|municipality|site|study area|case study context|urban context)))?$",
    re.IGNORECASE,
)


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    text = re.sub(r"[\u2018\u2019\u201c\u201d]", "'", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def normalize_spatial_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value != value:
            return False
        return bool(value)
    normalized = normalize_text(value).strip("\"'")
    return normalized in {"1", "true", "yes", "y"}


def normalize_scale(value: Any) -> Optional[str]:
    text = normalize_text(value)
    if text in {"", "none", "null", "not mentioned", "n/a", "na", "nan"}:
        return None
    match = re.match(r"^([1-9])(?:\.|\b)", text)
    if match:
        return SCALE_LEVELS.get(match.group(1))
    for level in SCALE_LEVELS.values():
        label = level.split(".", 1)[1].strip().lower()
        if text == label or label in text:
            return level
    return None


def is_placeholder_area(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip().strip("\"'")
    normalized = normalize_text(text).strip(" .;:")
    if normalized in {"", "none", "null", "not mentioned", "n/a", "na", "nan"}:
        return True
    if any(term in normalized for term in PLACEHOLDER_AREA_TERMS):
        return True
    return bool(GENERIC_AREA_PATTERN.fullmatch(normalized))


def extract_first_json(text: str) -> Dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ValueError("no_json_object")
    decoder = json.JSONDecoder()
    data, _ = decoder.raw_decode(text[start:])
    if not isinstance(data, dict):
        raise ValueError("json_not_object")
    return data


def parse_label_json(text: str) -> Dict[str, Any]:
    data = extract_first_json(text)
    is_spatial = normalize_spatial_flag(data.get("Is_Spatial_Research", False))
    scale = normalize_scale(data.get("Spatial_Scale_Level"))
    area = data.get("Specific_Study_Area")
    area_text = None if area is None else str(area).strip()
    if not is_spatial:
        scale = None
        area_text = None
    return {
        "gpt_is_spatial": int(is_spatial),
        "gpt_spatial_scale_level": scale,
        "gpt_specific_study_area": area_text,
    }


def validate_label(parsed: Dict[str, Any]) -> tuple[str, str]:
    is_spatial = int(parsed.get("gpt_is_spatial", 0) or 0)
    scale = parsed.get("gpt_spatial_scale_level")
    area = parsed.get("gpt_specific_study_area")
    if not is_spatial:
        return "valid", ""
    if not scale:
        return "invalid", "gpt_label_invalid_missing_scale"
    if is_placeholder_area(area):
        return "invalid", "gpt_placeholder_or_missing_area"
    return "valid", ""


def build_messages(title: str, abstract: str, retry_reason: str = "") -> list[dict[str, str]]:
    system = (
        "You are simulating an expert human coder for bibliometric spatial research annotation. "
        "Use only TITLE and ABSTRACT as evidence. Do not infer unnamed cities, sites, neighborhoods, "
        "or projects. Output one strict JSON object only."
    )
    user = f"""
Task: Blindly label the paper's true spatial research status.

Spatial research means the core research objective, empirical analysis, data collection, case study, or policy recommendations are substantially anchored in a named or specifically identifiable real geographic area on Earth.
Non-spatial means theoretical, methodological, generic urban discussion, or cases where a place is implied but not identifiable from TITLE/ABSTRACT.

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

Rules:
- If spatial=false, Spatial_Scale_Level and Specific_Study_Area must be null.
- If spatial=true, Specific_Study_Area must be a named or specifically identifiable area from TITLE/ABSTRACT.
- Never output placeholders such as "unspecified city", "study area", "a brownfield site", "unknown site", or "case study context".
- Restricted implicit country/region is allowed only when explicit institutions, policies, or national/regional programs make it identifiable; never use this to infer an unnamed city/site/neighborhood.

Return exactly this JSON shape:
{{
  "Is_Spatial_Research": true,
  "Spatial_Scale_Level": "7. Single-city / Municipal Scale",
  "Specific_Study_Area": "New York City"
}}

{f"Previous output was invalid because: {retry_reason}. Correct it." if retry_reason else ""}

[TITLE]
{title}

[ABSTRACT]
{abstract}
""".strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def ensure_openai_gpt_config() -> tuple[str, Optional[str], str]:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    model = os.environ.get("OPENAI_GPT_LABEL_MODEL", "").strip() or DEFAULT_MODEL
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; GPT blind labeling cannot run.")
    if not model.startswith("gpt-"):
        raise RuntimeError(f"OPENAI_GPT_LABEL_MODEL must be a GPT model name, got: {model}")
    if base_url and "deepseek" in base_url.lower():
        raise RuntimeError("OPENAI_BASE_URL points to DeepSeek; refusing to run GPT blind labeling.")
    return api_key, base_url, model


def make_client(api_key: str, base_url: Optional[str]) -> OpenAI:
    kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": 90}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _error_status_code(error: Exception) -> Optional[int]:
    status = getattr(error, "status_code", None)
    if status is None:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _error_payload_text(error: Exception) -> str:
    parts = [str(error)]
    body = getattr(error, "body", None)
    if body:
        try:
            parts.append(json.dumps(body, ensure_ascii=False))
        except TypeError:
            parts.append(str(body))
    response = getattr(error, "response", None)
    if response is not None:
        text = getattr(response, "text", "")
        if text:
            parts.append(str(text))
    return " ".join(part for part in parts if part)


def raise_if_quota_exceeded(error: Exception) -> None:
    status_code = _error_status_code(error)
    payload = _error_payload_text(error)
    upper = payload.upper()
    if status_code != 429 or not any(token in upper for token in QUOTA_ERROR_TOKENS):
        return
    code = "USAGE_LIMIT_EXCEEDED" if "USAGE_LIMIT_EXCEEDED" in upper else "RATE_LIMIT_429"
    reason = "DAILY_LIMIT_EXCEEDED" if "DAILY_LIMIT_EXCEEDED" in upper else "RATE_LIMIT_429"
    if "TOO MANY REQUESTS" in upper:
        reason = "TOO_MANY_REQUESTS"
    raise LLMQuotaExceededError(
        f"OpenAI quota exhausted: status={status_code} code={code} reason={reason}",
        status_code=status_code,
        code=code,
        reason=reason,
    ) from error


def call_openai_label(
    client: OpenAI,
    model: str,
    title: str,
    abstract: str,
    retry_reason: str = "",
) -> tuple[str, Dict[str, Any], str, str]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=build_messages(title, abstract, retry_reason=retry_reason),
            temperature=0.0,
            max_tokens=500,
        )
    except Exception as error:
        raise_if_quota_exceeded(error)
        raise
    raw = response.choices[0].message.content or ""
    parsed = parse_label_json(raw)
    status, error = validate_label(parsed)
    return raw, parsed, status, error


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
            row_index = int(record.get("row_index", -1))
            if row_index >= 0:
                records[row_index] = record
    return records


def append_checkpoint(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_target_rows(df: pd.DataFrame, start: int = 0, limit: Optional[int] = None) -> Iterable[tuple[int, pd.Series]]:
    selected = df.iloc[start:]
    if limit is not None:
        selected = selected.head(limit)
    for row_index, row in selected.iterrows():
        yield int(row_index), row


def label_dataframe(
    df: pd.DataFrame,
    client: OpenAI,
    model: str,
    checkpoint_path: Path,
    start: int = 0,
    limit: Optional[int] = None,
    max_retries: int = 2,
) -> pd.DataFrame:
    checkpoint = load_checkpoint(checkpoint_path)
    results: list[Dict[str, Any]] = []
    for row_index, row in iter_target_rows(df, start=start, limit=limit):
        existing = checkpoint.get(row_index)
        if existing and existing.get("label_status") == "valid":
            results.append(existing)
            continue

        title = str(row.get(Schema.TITLE, "") or "")
        abstract = str(row.get(Schema.ABSTRACT, "") or "")
        source_row = row.get("source_row_0based", row_index)
        retry_reason = ""
        final_record: Dict[str, Any] = {}
        for attempt in range(max_retries + 1):
            try:
                raw, parsed, status, error = call_openai_label(
                    client,
                    model,
                    title,
                    abstract,
                    retry_reason=retry_reason,
                )
                final_record = {
                    "row_index": row_index,
                    "source_row_0based": int(source_row) if pd.notna(source_row) else row_index,
                    Schema.TITLE: title,
                    Schema.ABSTRACT: abstract,
                    **parsed,
                    "label_status": status,
                    "label_error": error,
                    "label_model": model,
                    "label_attempts": attempt + 1,
                    "raw_response": raw,
                }
                if status == "valid":
                    break
                retry_reason = error
            except LLMQuotaExceededError as error:
                final_record = {
                    "row_index": row_index,
                    "source_row_0based": int(source_row) if pd.notna(source_row) else row_index,
                    Schema.TITLE: title,
                    Schema.ABSTRACT: abstract,
                    "gpt_is_spatial": "",
                    "gpt_spatial_scale_level": "",
                    "gpt_specific_study_area": "",
                    "label_status": "quota_exhausted",
                    "label_error": f"{type(error).__name__}: {error}",
                    "label_model": model,
                    "label_attempts": attempt + 1,
                    "raw_response": "",
                }
                append_checkpoint(checkpoint_path, final_record)
                checkpoint[row_index] = final_record
                results.append(final_record)
                raise
            except Exception as error:  # noqa: BLE001 - keep batch resilient and auditable.
                final_record = {
                    "row_index": row_index,
                    "source_row_0based": int(source_row) if pd.notna(source_row) else row_index,
                    Schema.TITLE: title,
                    Schema.ABSTRACT: abstract,
                    "gpt_is_spatial": "",
                    "gpt_spatial_scale_level": "",
                    "gpt_specific_study_area": "",
                    "label_status": "failed",
                    "label_error": f"{type(error).__name__}: {error}",
                    "label_model": model,
                    "label_attempts": attempt + 1,
                    "raw_response": "",
                }
                retry_reason = "api_or_parse_failure"
                time.sleep(min(10, 2 * (attempt + 1)))

        append_checkpoint(checkpoint_path, final_record)
        checkpoint[row_index] = final_record
        results.append(final_record)
        if (len(results) % 25) == 0:
            print(f"[INFO] labeled_or_loaded={len(results)} row_index={row_index}")
    return pd.DataFrame(results).sort_values("row_index").reset_index(drop=True)


def probe_openai(client: OpenAI, model: str) -> None:
    raw, parsed, status, error = call_openai_label(
        client,
        model,
        "Spatial case in New York City",
        "This study analyzes pandemic governance in New York City using local public health documents.",
    )
    if status != "valid" or parsed.get("gpt_is_spatial") != 1:
        raise RuntimeError(f"OpenAI probe returned invalid label: status={status}, error={error}, raw={raw[:200]}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blindly label spatial research fields with OpenAI GPT.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--probe-only", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    api_key, base_url, env_model = ensure_openai_gpt_config()
    model = args.model or env_model
    if not model.startswith("gpt-"):
        raise RuntimeError(f"Model must start with gpt-, got: {model}")
    client = make_client(api_key, base_url)
    try:
        probe_openai(client, model)
    except LLMQuotaExceededError as exc:
        print(f"[WARN] OpenAI quota exhausted during probe: {exc}")
        return 2
    print(f"[INFO] OpenAI GPT probe passed with model={model}")
    if args.probe_only:
        return 0

    df = pd.read_excel(args.input, engine="openpyxl")
    required = {Schema.TITLE, Schema.ABSTRACT}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Input missing required columns: {missing}")
    if len(df) != 2000 and args.limit is None:
        raise RuntimeError(f"Expected 2000 input rows for full run, got {len(df)}")

    try:
        labels = label_dataframe(
            df,
            client=client,
            model=model,
            checkpoint_path=args.checkpoint,
            start=args.start,
            limit=args.limit,
            max_retries=args.max_retries,
        )
    except LLMQuotaExceededError as exc:
        checkpoint = load_checkpoint(args.checkpoint)
        labels = pd.DataFrame(checkpoint.values())
        if not labels.empty and "row_index" in labels.columns:
            labels = labels.sort_values("row_index").reset_index(drop=True)
        if not labels.empty:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            labels.to_excel(args.output, index=False, engine="openpyxl")
        valid_count = int((labels.get("label_status", pd.Series(dtype=object)) == "valid").sum()) if not labels.empty else 0
        print(
            "[WARN] OpenAI quota exhausted; saved current checkpoint and partial labels. "
            f"valid_rows={valid_count} checkpoint={args.checkpoint} output={args.output} error={exc}"
        )
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    labels.to_excel(args.output, index=False, engine="openpyxl")
    status_counts = labels["label_status"].value_counts(dropna=False).to_dict()
    print(f"[INFO] labels saved to {args.output}")
    print(f"[INFO] checkpoint saved to {args.checkpoint}")
    print(f"[INFO] status_counts={status_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
