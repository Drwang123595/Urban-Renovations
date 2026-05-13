import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation.core import COLUMN_ALIASES, align_truth_pred
from src.runtime.config import Schema
from src.urban.specter2.config import Specter2Config
from src.urban.specter2.encoder import Specter2Encoder
from src.urban.specter2.evaluator import (
    AblationThresholds,
    compute_binary_metrics,
    evaluate_specter2_ablation,
)
from src.urban.specter2.features import records_from_dataframe


SUMMARY_XLSX = "Specter2_Ablation_Summary.xlsx"
MANIFEST_JSON = "specter2_ablation_manifest.json"


def run_ablation(
    *,
    truth_workbook: Path,
    prediction_workbook: Path,
    output_dir: Path,
    limit: int | None = None,
    batch_size: int = 16,
) -> dict[str, object]:
    truth_df = pd.read_excel(truth_workbook)
    pred_df = pd.read_excel(prediction_workbook)
    aligned = align_truth_pred(truth_df, pred_df, strict=False)
    merged = aligned.merged.copy()
    if limit is not None:
        merged = merged.head(int(limit)).copy()

    truth_col = _resolve_binary_column(
        merged,
        _field_role_candidates(Schema.IS_URBAN_RENEWAL, "truth"),
        role="truth",
    )
    pred_col = _resolve_binary_column(
        merged,
        _field_role_candidates(Schema.IS_URBAN_RENEWAL, "pred")
        + ["final_label_pred", "urban_flag_pred", "final_label", "urban_flag"],
        role="pred",
    )
    title_col = _resolve_column(merged, [f"{Schema.TITLE}_truth", f"{Schema.TITLE}_pred", Schema.TITLE, "_key"])
    abstract_col = _resolve_column(
        merged,
        [f"{Schema.ABSTRACT}_truth", f"{Schema.ABSTRACT}_pred", Schema.ABSTRACT],
        required=False,
    )

    model_input = pd.DataFrame(
        {
            Schema.TITLE: merged[title_col],
            Schema.ABSTRACT: merged[abstract_col] if abstract_col else "",
        }
    )
    records = records_from_dataframe(model_input)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_XLSX
    manifest_path = output_dir / MANIFEST_JSON

    encoder = Specter2Encoder(Specter2Config(batch_size=batch_size))
    encoding = encoder.encode(records)
    manifest = {
        "truth_workbook": str(truth_workbook),
        "prediction_workbook": str(prediction_workbook),
        "output_workbook": str(summary_path),
        "rows": int(len(merged)),
        "alignment_summary": aligned.summary,
        "encoding_status": encoding.status,
        "encoding_reason": encoding.reason,
        "cache_hits": int(encoding.cache_hits),
        "cache_misses": int(encoding.cache_misses),
    }

    if encoding.status != "ok":
        metrics = pd.DataFrame(
            [
                compute_binary_metrics(
                    merged[truth_col],
                    merged[pred_col],
                    group="baseline",
                )
            ]
        )
        with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
            metrics.to_excel(writer, sheet_name="Summary", index=False)
            pd.DataFrame([manifest]).to_excel(writer, sheet_name="Manifest", index=False)
        manifest["groups_evaluated"] = ["baseline"]
        manifest["gate_summary"] = {"passes": False, "reason": encoding.status}
    else:
        result = evaluate_specter2_ablation(
            truth=merged[truth_col],
            baseline_pred=merged[pred_col],
            embeddings=encoding.embeddings,
            thresholds=AblationThresholds(),
        )
        predictions = pd.concat(
            [
                model_input.reset_index(drop=True),
                result.predictions.reset_index(drop=True),
            ],
            axis=1,
        )
        with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
            result.metrics.to_excel(writer, sheet_name="Summary", index=False)
            predictions.to_excel(writer, sheet_name="Predictions", index=False)
            pd.DataFrame([result.gate_summary]).to_excel(writer, sheet_name="Gate", index=False)
            pd.DataFrame([aligned.summary]).to_excel(writer, sheet_name="Alignment", index=False)
        manifest["groups_evaluated"] = ["baseline", "specter2_only", "hybrid"]
        manifest["gate_summary"] = result.gate_summary

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _resolve_column(frame: pd.DataFrame, candidates: list[str], *, required: bool = True) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    if required:
        raise ValueError(f"None of the required columns were found: {candidates}")
    return None


def _field_role_candidates(field_name: str, role: str) -> list[str]:
    aliases = [field_name]
    aliases.extend(alias for alias in COLUMN_ALIASES.get(field_name, []) if alias not in aliases)
    return [f"{alias}_{role}" for alias in aliases] + aliases


def _resolve_binary_column(frame: pd.DataFrame, candidates: list[str], *, role: str) -> str:
    resolved = _resolve_column(frame, candidates, required=False)
    if resolved is not None:
        return resolved

    suffix = f"_{role}"
    excluded = {
        f"{Schema.TITLE}_{role}",
        f"{Schema.ABSTRACT}_{role}",
        Schema.TITLE,
        Schema.ABSTRACT,
        "_key",
    }
    for column in frame.columns:
        if column in excluded or not str(column).endswith(suffix):
            continue
        if _looks_binary(frame[column]):
            return str(column)
    raise ValueError(f"Could not resolve {role} binary column from candidates: {candidates}")


def _looks_binary(series: pd.Series) -> bool:
    values = series.dropna().astype(str).str.strip().str.lower()
    if values.empty:
        return False
    return values.isin({"0", "0.0", "1", "1.0", "true", "false", "yes", "no"}).all()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SPECTER2 urban-renewal ablation evaluation.")
    parser.add_argument("--truth-workbook", type=Path, required=True)
    parser.add_argument("--prediction-workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    manifest = run_ablation(
        truth_workbook=args.truth_workbook,
        prediction_workbook=args.prediction_workbook,
        output_dir=args.output_dir,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
