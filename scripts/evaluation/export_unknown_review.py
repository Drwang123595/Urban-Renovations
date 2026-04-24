from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


TITLE_COL = "Article Title"
ABSTRACT_COL = "Abstract"
URBAN_TRUTH_COL = "urban_flag_truth"
THEME_TRUTH_COL = "theme_gold_truth"
DEFAULT_OUTPUT_COLUMNS = [
    TITLE_COL,
    ABSTRACT_COL,
    URBAN_TRUTH_COL,
    THEME_TRUTH_COL,
    "topic_rule",
    "topic_rule_score",
    "topic_rule_margin",
    "topic_local_label",
    "topic_local_confidence",
    "topic_local_margin",
    "topic_final",
    "decision_source",
    "decision_reason",
    "review_reason",
    "unknown_recovery_path",
    "unknown_recovery_evidence",
    "bertopic_hint_label",
    "bertopic_hint_group",
    "bertopic_cluster_quality",
    "llm_attempted",
    "llm_family_hint",
    "llm_family_hint_reason",
    "reviewed_theme",
    "reviewed_urban_flag",
    "review_notes",
]


def detect_truth_label_column(df: pd.DataFrame) -> Optional[str]:
    excluded_exact = {
        TITLE_COL,
        ABSTRACT_COL,
        "Author Keywords",
        "Keywords Plus",
        "Keywords",
        "WoS Categories",
        "Research Areas",
        "theme_gold",
        "theme_gold_source",
        "review_status",
    }
    for column in df.columns:
        if column in excluded_exact:
            continue
        text = str(column).lower()
        if any(token in text for token in ("keyword", "abstract", "wos", "research", "theme", "review")):
            continue
        return str(column)
    return None


def build_unknown_review(pred_path: Path, truth_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred_df = pd.read_excel(pred_path, engine="openpyxl")
    truth_df = pd.read_excel(truth_path, engine="openpyxl")
    if TITLE_COL not in pred_df.columns:
        raise ValueError(f"Prediction file missing required column: {TITLE_COL}")
    if TITLE_COL not in truth_df.columns:
        raise ValueError(f"Truth file missing required column: {TITLE_COL}")

    truth_label_col = detect_truth_label_column(truth_df)
    if not truth_label_col:
        raise ValueError("Unable to detect the urban truth column in the label workbook.")

    truth_columns = [TITLE_COL, truth_label_col]
    if "theme_gold" in truth_df.columns:
        truth_columns.append("theme_gold")
    merged = pred_df.merge(
        truth_df[truth_columns],
        on=TITLE_COL,
        how="left",
    )

    unknown_mask = merged.get("topic_final", pd.Series(index=merged.index, dtype=object)).astype(str).eq("Unknown")
    review_df = merged.loc[unknown_mask].copy()
    review_df.rename(
        columns={
            truth_label_col: URBAN_TRUTH_COL,
            "theme_gold": THEME_TRUTH_COL,
        },
        inplace=True,
    )
    for column in ("reviewed_theme", "reviewed_urban_flag", "review_notes"):
        if column not in review_df.columns:
            review_df[column] = ""

    ordered_columns = [column for column in DEFAULT_OUTPUT_COLUMNS if column in review_df.columns]
    remaining_columns = [column for column in review_df.columns if column not in ordered_columns]
    review_df = review_df[ordered_columns + remaining_columns]

    summary_rows = [
        {"Metric": "Total Samples", "Value": int(len(pred_df))},
        {"Metric": "Unknown Samples", "Value": int(len(review_df))},
        {"Metric": "Unknown Rate", "Value": round((len(review_df) / len(pred_df)) if len(pred_df) else 0.0, 4)},
    ]
    if URBAN_TRUTH_COL in review_df.columns:
        truth_series = review_df[URBAN_TRUTH_COL].astype(str).str.replace(".0", "", regex=False)
        summary_rows.extend(
            [
                {"Metric": "Unknown Truth Positive Count", "Value": int(truth_series.eq("1").sum())},
                {"Metric": "Unknown Truth Negative Count", "Value": int(truth_series.eq("0").sum())},
                {
                    "Metric": "Unknown Truth Positive Rate",
                    "Value": round(float(truth_series.eq("1").mean()) if len(review_df) else 0.0, 4),
                },
            ]
        )
    summary_df = pd.DataFrame(summary_rows)

    if not review_df.empty:
        reason_summary_df = (
            review_df["review_reason"]
            .fillna("")
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis("Review Reason")
            .reset_index(name="Count")
        )
    else:
        reason_summary_df = pd.DataFrame(columns=["Review Reason", "Count"])

    return review_df, summary_df, reason_summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Unknown samples for manual review.")
    parser.add_argument("--pred", required=True, help="Path to the prediction workbook")
    parser.add_argument("--truth", required=True, help="Path to the truth/label workbook")
    parser.add_argument("--output", required=True, help="Path to the output review workbook")
    args = parser.parse_args()

    pred_path = Path(args.pred)
    truth_path = Path(args.truth)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    review_df, summary_df, reason_summary_df = build_unknown_review(pred_path, truth_path)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        review_df.to_excel(writer, index=False, sheet_name="Unknown_Review")
        summary_df.to_excel(writer, index=False, sheet_name="Summary")
        reason_summary_df.to_excel(writer, index=False, sheet_name="Reason_Summary")
    print(f"Unknown review workbook saved to: {output_path}")


if __name__ == "__main__":
    main()
