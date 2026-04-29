from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, Optional


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


def normalize_area_text(value: Any) -> str:
    text = normalize_text(value)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9&/,\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,;-")


def normalize_spatial_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return False
        return bool(int(value))
    normalized = normalize_text(value).strip("\"'")
    return normalized in {"1", "true", "yes", "y", "spatial"}


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


def extract_first_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ValueError("no_json_object")
    decoder = json.JSONDecoder()
    data, _ = decoder.raw_decode(text[start:])
    if not isinstance(data, dict):
        raise ValueError("json_not_object")
    return data


def extract_first_json_array(text: str) -> list[Dict[str, Any]]:
    starts = [idx for idx in (text.find("["), text.find("{")) if idx >= 0]
    if not starts:
        raise ValueError("no_json_payload")
    start = min(starts)
    decoder = json.JSONDecoder()
    data, _ = decoder.raw_decode(text[start:])
    if isinstance(data, dict):
        records = data.get("labels")
        if isinstance(records, list):
            data = records
    if not isinstance(data, list):
        raise ValueError("json_not_array")
    if not all(isinstance(item, dict) for item in data):
        raise ValueError("json_array_contains_non_object")
    return data


def parse_single_label(data: Dict[str, Any]) -> Dict[str, Any]:
    is_spatial = normalize_spatial_flag(
        data.get("gpt_is_spatial", data.get("Is_Spatial_Research", False))
    )
    scale = normalize_scale(
        data.get("gpt_spatial_scale_level", data.get("Spatial_Scale_Level"))
    )
    area = data.get("gpt_specific_study_area", data.get("Specific_Study_Area"))
    area_text = None if area is None else str(area).strip()
    if not is_spatial:
        scale = None
        area_text = None
    return {
        "gpt_is_spatial": int(is_spatial),
        "gpt_spatial_scale_level": scale,
        "gpt_specific_study_area": area_text,
    }


def parse_label_json(text: str) -> Dict[str, Any]:
    return parse_single_label(extract_first_json_object(text))


def parse_batch_json(text: str) -> list[Dict[str, Any]]:
    parsed: list[Dict[str, Any]] = []
    for item in extract_first_json_array(text):
        row_index = item.get("row_index")
        source_row = item.get("source_row_0based", row_index)
        label = parse_single_label(item)
        record = {
            "row_index": int(row_index) if row_index is not None else None,
            "source_row_0based": int(source_row) if source_row is not None else None,
            **label,
        }
        parsed.append(record)
    return parsed


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


def area_match_type(left: Any, right: Any) -> str:
    left_norm = normalize_area_text(left)
    right_norm = normalize_area_text(right)
    if not left_norm or not right_norm:
        return "missing"
    if left_norm == right_norm:
        return "exact"
    if left_norm in right_norm or right_norm in left_norm:
        return "containment"
    stopwords = {"the", "and", "of", "in", "at", "for", "city", "area"}
    left_tokens = {
        tok
        for tok in re.split(r"[\s,;/&-]+", left_norm)
        if tok and tok not in stopwords and (len(tok) > 2 or tok.isdigit())
    }
    right_tokens = {
        tok
        for tok in re.split(r"[\s,;/&-]+", right_norm)
        if tok and tok not in stopwords and (len(tok) > 2 or tok.isdigit())
    }
    if not left_tokens or not right_tokens:
        return "different"
    overlap = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
    return "token_overlap" if overlap >= 0.6 else "different"


def binary_metrics(gold: Iterable[int], pred: Iterable[int]) -> Dict[str, float | int]:
    pairs = [(int(g), int(p)) for g, p in zip(gold, pred)]
    tp = sum(1 for g, p in pairs if g == 1 and p == 1)
    tn = sum(1 for g, p in pairs if g == 0 and p == 0)
    fp = sum(1 for g, p in pairs if g == 0 and p == 1)
    fn = sum(1 for g, p in pairs if g == 1 and p == 0)
    n = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / n if n else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    pred_counts = Counter(p for _, p in pairs)
    gold_counts = Counter(g for g, _ in pairs)
    expected = sum(gold_counts[k] * pred_counts[k] for k in {0, 1}) / (n * n) if n else 0.0
    kappa = (accuracy - expected) / (1 - expected) if n and expected < 1 else 0.0
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "cohen_kappa": kappa,
        "agreement_rate": accuracy,
    }
