from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runtime.config import Config, Schema


DEFAULT_DATASET_ID = Config.STABLE_RELEASE_DATASET_ID
DEFAULT_LABEL_WORKBOOK = Config.STABLE_RELEASE_LABEL_FILE
DEFAULT_SPATIAL_PRED = (
    Config.DATA_DIR
    / DEFAULT_DATASET_ID
    / "output"
    / "spatial_few_20260415_174044.xlsx"
)
DEFAULT_OUTPUT = (
    Config.DATA_DIR
    / DEFAULT_DATASET_ID
    / "Result"
    / "spatial_annotation_subset_300_seed20260416.xlsx"
)


def _normalize_binary(value) -> str:
    text = str(value).strip().replace(".0", "")
    if text in {"0", "1"}:
        return text
    return ""


def _largest_remainder_allocation(counts: pd.Series, sample_size: int) -> dict[str, int]:
    if counts.empty:
        return {}
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    total = int(counts.sum())
    if sample_size >= total:
        return {str(key): int(value) for key, value in counts.items()}

    quotas = counts / total * sample_size
    base = quotas.astype(int)
    remainder = quotas - base

    nonzero_keys = [str(key) for key, value in counts.items() if int(value) > 0]
    allocation = {str(key): int(base.loc[key]) for key in counts.index}

    if sample_size >= len(nonzero_keys):
        for key in nonzero_keys:
            if allocation[key] == 0:
                allocation[key] = 1

    current = sum(allocation.values())
    if current > sample_size:
        removable = sorted(
            [key for key in allocation if allocation[key] > 1],
            key=lambda key: (remainder.loc[key], counts.loc[key]),
        )
        while current > sample_size and removable:
            key = removable.pop(0)
            allocation[key] -= 1
            current -= 1

    if current < sample_size:
        expandable = sorted(
            [str(key) for key in counts.index if allocation[str(key)] < int(counts.loc[key])],
            key=lambda key: (remainder.loc[key], counts.loc[key]),
            reverse=True,
        )
        idx = 0
        while current < sample_size and expandable:
            key = expandable[idx % len(expandable)]
            if allocation[key] < int(counts.loc[key]):
                allocation[key] += 1
                current += 1
            idx += 1
            if idx > len(expandable) * (sample_size + 1):
                break

    return allocation


def load_label_frame(workbook_path: Path) -> pd.DataFrame:
    df = pd.read_excel(workbook_path, engine="openpyxl")
    required = [
        Schema.TITLE,
        Schema.ABSTRACT,
        Schema.KEYWORDS_PLUS,
        Schema.WOS_CATEGORIES,
        Schema.RESEARCH_AREAS,
        Schema.IS_URBAN_RENEWAL,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in label workbook: {', '.join(missing)}")

    label_df = df[required].copy()
    label_df["urban_truth"] = label_df[Schema.IS_URBAN_RENEWAL].map(_normalize_binary)
    label_df["_key"] = label_df[Schema.TITLE].astype(str).str.strip().str.lower()
    return label_df.drop_duplicates("_key", keep="first")


def load_spatial_predictions(pred_path: Path) -> pd.DataFrame:
    df = pd.read_excel(pred_path, engine="openpyxl")
    required = [
        Schema.TITLE,
        Schema.IS_SPATIAL,
        Schema.SPATIAL_LEVEL,
        Schema.SPATIAL_DESC,
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in spatial prediction workbook: {', '.join(missing)}")

    pred_df = df.copy()
    pred_df["_key"] = pred_df[Schema.TITLE].astype(str).str.strip().str.lower()
    pred_df["spatial_pred"] = pred_df[Schema.IS_SPATIAL].map(_normalize_binary)
    pred_df = pred_df.rename(
        columns={
            Schema.SPATIAL_LEVEL: "spatial_level_pred",
            Schema.SPATIAL_DESC: "spatial_desc_pred",
            "Reasoning": "spatial_reasoning",
            "Confidence": "spatial_confidence",
        }
    )
    keep_columns = [
        "_key",
        "spatial_pred",
        "spatial_level_pred",
        "spatial_desc_pred",
        "spatial_reasoning",
        "spatial_confidence",
    ]
    return pred_df[keep_columns].drop_duplicates("_key", keep="first")


def build_annotation_frame(
    label_df: pd.DataFrame,
    dataset_id: str,
    seed: int,
    sample_size: int,
    spatial_pred_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    working = label_df.copy()
    if spatial_pred_df is not None:
        working = working.merge(spatial_pred_df, on="_key", how="left")
    else:
        working["spatial_pred"] = ""
        working["spatial_level_pred"] = ""
        working["spatial_desc_pred"] = ""
        working["spatial_reasoning"] = ""
        working["spatial_confidence"] = ""

    working["stratum"] = working["urban_truth"].replace("", "unknown")
    if "spatial_pred" in working.columns:
        working["stratum"] = (
            "urban="
            + working["urban_truth"].replace("", "unknown")
            + "|spatial_pred="
            + working["spatial_pred"].replace("", "blank")
        )

    counts = working["stratum"].value_counts().sort_index()
    allocation = _largest_remainder_allocation(counts, sample_size)

    sampled_frames: list[pd.DataFrame] = []
    for offset, (stratum, count) in enumerate(sorted(allocation.items())):
        if count <= 0:
            continue
        group = working[working["stratum"] == stratum].copy()
        sampled = group.sample(n=count, random_state=seed + offset).copy()
        sampled_frames.append(sampled)

    if not sampled_frames:
        raise ValueError("No rows selected for annotation subset")

    sampled_df = pd.concat(sampled_frames, ignore_index=True)
    sampled_df = sampled_df.sort_values(["stratum", Schema.TITLE], kind="stable").reset_index(drop=True)
    sampled_df.insert(0, "review_id", [f"SPA-{idx:03d}" for idx in range(1, len(sampled_df) + 1)])
    sampled_df.insert(1, "dataset_id", dataset_id)
    sampled_df.insert(2, "sample_seed", seed)
    sampled_df.insert(3, "sample_stratum", sampled_df["stratum"])

    sampled_df["annotator_a_spatial"] = ""
    sampled_df["annotator_a_level"] = ""
    sampled_df["annotator_a_desc"] = ""
    sampled_df["annotator_a_notes"] = ""
    sampled_df["annotator_b_spatial"] = ""
    sampled_df["annotator_b_level"] = ""
    sampled_df["annotator_b_desc"] = ""
    sampled_df["annotator_b_notes"] = ""
    sampled_df["adjudicated_spatial"] = ""
    sampled_df["adjudicated_level"] = ""
    sampled_df["adjudicated_desc"] = ""
    sampled_df["adjudication_notes"] = ""
    sampled_df["agreement_spatial"] = ""
    sampled_df["agreement_level"] = ""
    sampled_df["agreement_desc"] = ""

    annotation_columns = [
        "review_id",
        "dataset_id",
        "sample_seed",
        "sample_stratum",
        Schema.TITLE,
        Schema.ABSTRACT,
        Schema.KEYWORDS_PLUS,
        Schema.WOS_CATEGORIES,
        Schema.RESEARCH_AREAS,
        "urban_truth",
        "spatial_pred",
        "spatial_level_pred",
        "spatial_desc_pred",
        "spatial_reasoning",
        "spatial_confidence",
        "annotator_a_spatial",
        "annotator_a_level",
        "annotator_a_desc",
        "annotator_a_notes",
        "annotator_b_spatial",
        "annotator_b_level",
        "annotator_b_desc",
        "annotator_b_notes",
        "adjudicated_spatial",
        "adjudicated_level",
        "adjudicated_desc",
        "adjudication_notes",
        "agreement_spatial",
        "agreement_level",
        "agreement_desc",
    ]

    summary_df = counts.rename_axis("sample_stratum").reset_index(name="population_count")
    summary_df["selected_count"] = summary_df["sample_stratum"].map(lambda key: allocation.get(str(key), 0))
    summary_df["selected_rate"] = (
        summary_df["selected_count"] / summary_df["population_count"]
    ).round(6)
    summary_df.insert(0, "dataset_id", dataset_id)
    summary_df.insert(1, "sample_seed", seed)

    return sampled_df[annotation_columns], summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a stratified spatial annotation subset with double-review columns."
    )
    parser.add_argument(
        "--labels",
        default=str(DEFAULT_LABEL_WORKBOOK),
        help="Label workbook path",
    )
    parser.add_argument(
        "--spatial-pred",
        default=str(DEFAULT_SPATIAL_PRED),
        help="Optional spatial prediction workbook path",
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help="Dataset identifier written into the workbook",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
        help="Number of papers to sample",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260416,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output workbook path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    label_path = Path(args.labels).resolve()
    spatial_pred_path = Path(args.spatial_pred).resolve() if args.spatial_pred else None
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not label_path.exists():
        raise FileNotFoundError(f"Label workbook not found: {label_path}")

    label_df = load_label_frame(label_path)
    spatial_pred_df = None
    if spatial_pred_path and spatial_pred_path.exists():
        spatial_pred_df = load_spatial_predictions(spatial_pred_path)

    annotation_df, summary_df = build_annotation_frame(
        label_df=label_df,
        dataset_id=args.dataset_id,
        seed=args.seed,
        sample_size=args.sample_size,
        spatial_pred_df=spatial_pred_df,
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        annotation_df.to_excel(writer, sheet_name="annotation_samples", index=False)
        summary_df.to_excel(writer, sheet_name="sampling_summary", index=False)

    print(f"Saved annotation workbook: {output_path}")
    print(f"Rows selected: {len(annotation_df)}")
    print("Strata:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
