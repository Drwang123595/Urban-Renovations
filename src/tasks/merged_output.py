from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..runtime.config import Schema
from ..urban.urban_topic_taxonomy import topic_name_for_label, topic_name_zh_for_label


REVIEW_PREDICT_URBAN_COLUMN = "\u9884\u6d4b_\u662f\u5426\u5c5e\u4e8e\u57ce\u5e02\u66f4\u65b0\u7814\u7a76"
REVIEW_PREDICT_SPATIAL_COLUMN = "\u9884\u6d4b_\u7a7a\u95f4\u7814\u7a76/\u975e\u7a7a\u95f4\u7814\u7a76"
REVIEW_PREDICT_SPATIAL_LEVEL_COLUMN = "\u9884\u6d4b_\u7a7a\u95f4\u7b49\u7ea7"
REVIEW_PREDICT_SPATIAL_DESC_COLUMN = "\u9884\u6d4b_\u5177\u4f53\u7a7a\u95f4\u63cf\u8ff0"
REVIEW_URBAN_CONFIDENCE_COLUMN = "\u57ce\u5e02\u66f4\u65b0\u5224\u5b9a\u7f6e\u4fe1\u5ea6(confidence)"
REVIEW_REASONING_COLUMN = "\u7a7a\u95f4\u63d0\u53d6\u4f9d\u636e(Reasoning)"
REVIEW_SPATIAL_CONFIDENCE_COLUMN = "\u7a7a\u95f4\u63d0\u53d6\u7f6e\u4fe1\u5ea6(Confidence)"
REVIEW_DECISION_EXPLANATION_COLUMN = "\u57ce\u5e02\u66f4\u65b0\u5224\u5b9a\u8bf4\u660e(decision_explanation)"
REVIEW_POSITIVE_EVIDENCE_COLUMN = "\u4e3b\u8981\u652f\u6301\u8bc1\u636e(primary_positive_evidence)"
REVIEW_NEGATIVE_EVIDENCE_COLUMN = "\u4e3b\u8981\u6392\u9664\u8bc1\u636e(primary_negative_evidence)"
REVIEW_EVIDENCE_BALANCE_COLUMN = "\u8bc1\u636e\u503e\u5411(evidence_balance)"
REVIEW_RULE_STACK_COLUMN = "\u89c4\u5219\u94fe\u8def(decision_rule_stack)"
REVIEW_BINARY_EVIDENCE_COLUMN = "\u4e8c\u5206\u7c7b\u6253\u5206\u4f9d\u636e(binary_decision_evidence)"
REVIEW_UNKNOWN_RECOVERY_PATH_COLUMN = "\u672a\u77e5\u6062\u590d\u8def\u5f84(unknown_recovery_path)"
REVIEW_UNKNOWN_RECOVERY_EVIDENCE_COLUMN = "\u672a\u77e5\u6062\u590d\u8bc1\u636e(unknown_recovery_evidence)"
REVIEW_DYNAMIC_TOPIC_ID_COLUMN = "\u52a8\u6001\u4e3b\u9898\u7f16\u53f7(dynamic_topic_id)"
REVIEW_DYNAMIC_TOPIC_NAME_COLUMN = "\u52a8\u6001\u4e3b\u9898\u540d\u79f0(dynamic_topic_name_zh)"
REVIEW_DYNAMIC_TOPIC_KEYWORDS_COLUMN = "\u52a8\u6001\u4e3b\u9898\u5173\u952e\u8bcd(dynamic_topic_keywords)"
REVIEW_DYNAMIC_TOPIC_SIZE_COLUMN = "\u52a8\u6001\u4e3b\u9898\u6837\u672c\u91cf(dynamic_topic_size)"
REVIEW_DYNAMIC_TOPIC_CONFIDENCE_COLUMN = "\u52a8\u6001\u4e3b\u9898\u7f6e\u4fe1\u5ea6(dynamic_topic_confidence)"
REVIEW_DYNAMIC_TOPIC_SOURCE_POOL_COLUMN = "\u52a8\u6001\u4e3b\u9898\u6765\u6e90\u6c60(dynamic_topic_source_pool)"
REVIEW_DYNAMIC_FIXED_CANDIDATE_COLUMN = "\u56fa\u5b9a\u4e3b\u9898\u5019\u9009(dynamic_to_fixed_topic_candidate)"
REVIEW_DYNAMIC_MAPPING_STATUS_COLUMN = "\u52a8\u6001\u6620\u5c04\u72b6\u6001(dynamic_mapping_status)"
REVIEW_DYNAMIC_BINARY_LABEL_COLUMN = "\u52a8\u6001\u4e8c\u5206\u7c7b\u5019\u9009(dynamic_binary_candidate_label)"
REVIEW_DYNAMIC_BINARY_CONFIDENCE_COLUMN = "\u52a8\u6001\u4e8c\u5206\u7c7b\u7f6e\u4fe1\u5ea6(dynamic_binary_candidate_confidence)"
REVIEW_DYNAMIC_BINARY_ACTION_COLUMN = "\u52a8\u6001\u4e8c\u5206\u7c7b\u6821\u51c6\u52a8\u4f5c(dynamic_binary_candidate_action)"
REVIEW_DYNAMIC_BINARY_REASON_COLUMN = "\u52a8\u6001\u4e8c\u5206\u7c7b\u6821\u51c6\u7406\u7531(dynamic_binary_candidate_reason)"
REVIEW_DYNAMIC_BINARY_PRIORITY_COLUMN = "\u52a8\u6001\u4e8c\u5206\u7c7b\u590d\u6838\u4f18\u5148\u7ea7(dynamic_binary_review_priority)"

REVIEW_INPUT_COLUMNS = [
    Schema.TITLE,
    "Publication Year",
    Schema.KEYWORDS_PLUS,
    Schema.ABSTRACT,
    Schema.WOS_CATEGORIES,
    Schema.RESEARCH_AREAS,
    Schema.IS_URBAN_RENEWAL,
]

REVIEW_DERIVED_COLUMNS = [
    REVIEW_PREDICT_URBAN_COLUMN,
    REVIEW_PREDICT_SPATIAL_COLUMN,
    REVIEW_PREDICT_SPATIAL_LEVEL_COLUMN,
    REVIEW_PREDICT_SPATIAL_DESC_COLUMN,
    "topic_final",
    "topic_final_name_en",
    "topic_final_name_zh",
    REVIEW_URBAN_CONFIDENCE_COLUMN,
    REVIEW_REASONING_COLUMN,
    REVIEW_SPATIAL_CONFIDENCE_COLUMN,
    "review_flag",
    "review_reason",
    REVIEW_DECISION_EXPLANATION_COLUMN,
    REVIEW_POSITIVE_EVIDENCE_COLUMN,
    REVIEW_NEGATIVE_EVIDENCE_COLUMN,
    REVIEW_EVIDENCE_BALANCE_COLUMN,
    REVIEW_RULE_STACK_COLUMN,
    REVIEW_BINARY_EVIDENCE_COLUMN,
    REVIEW_UNKNOWN_RECOVERY_PATH_COLUMN,
    REVIEW_UNKNOWN_RECOVERY_EVIDENCE_COLUMN,
    REVIEW_DYNAMIC_TOPIC_ID_COLUMN,
    REVIEW_DYNAMIC_TOPIC_NAME_COLUMN,
    REVIEW_DYNAMIC_TOPIC_KEYWORDS_COLUMN,
    REVIEW_DYNAMIC_TOPIC_SIZE_COLUMN,
    REVIEW_DYNAMIC_TOPIC_CONFIDENCE_COLUMN,
    REVIEW_DYNAMIC_TOPIC_SOURCE_POOL_COLUMN,
    REVIEW_DYNAMIC_FIXED_CANDIDATE_COLUMN,
    REVIEW_DYNAMIC_MAPPING_STATUS_COLUMN,
    REVIEW_DYNAMIC_BINARY_LABEL_COLUMN,
    REVIEW_DYNAMIC_BINARY_CONFIDENCE_COLUMN,
    REVIEW_DYNAMIC_BINARY_ACTION_COLUMN,
    REVIEW_DYNAMIC_BINARY_REASON_COLUMN,
    REVIEW_DYNAMIC_BINARY_PRIORITY_COLUMN,
]


def load_task_input_frame(task_dir: Path) -> Optional[pd.DataFrame]:
    for labels_dir in (task_dir / "input" / "labels", task_dir / "labels"):
        if not labels_dir.exists():
            continue

        preferred = labels_dir / f"{task_dir.name}.xlsx"
        if preferred.exists():
            input_df = pd.read_excel(preferred, engine="openpyxl")
            return _enrich_publication_year(input_df, task_dir)

        candidates = sorted(labels_dir.glob("*.xlsx"))
        if candidates:
            input_df = pd.read_excel(candidates[0], engine="openpyxl")
            return _enrich_publication_year(input_df, task_dir)
    return None


def _normalized_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([""] * len(frame), index=frame.index, dtype=object)
    return frame[column].fillna("").astype(str).str.strip().str.lower()


def _build_alignment_keys(frame: pd.DataFrame) -> pd.DataFrame:
    keyed = frame.copy()
    title_key = _normalized_series(keyed, Schema.TITLE)
    abstract_key = _normalized_series(keyed, Schema.ABSTRACT)
    keyed["_merge_key"] = title_key + "\x1f" + abstract_key
    keyed["_dup_index"] = keyed.groupby("_merge_key", sort=False).cumcount()
    return keyed


def _publication_year_lookup(task_dir: Path) -> dict[str, object]:
    train_path = task_dir.parent / "train" / "Urban Renovation V2.0.xlsx"
    if not train_path.exists():
        return {}

    train_df = pd.read_excel(train_path, engine="openpyxl")
    if "Publication Year" not in train_df.columns:
        return {}

    columns = [column for column in [Schema.TITLE, Schema.ABSTRACT, "Publication Year"] if column in train_df.columns]
    source = _build_alignment_keys(train_df[columns])
    return (
        source.groupby("_merge_key", sort=False)["Publication Year"]
        .agg(lambda series: series.dropna().iloc[0] if not series.dropna().empty else "")
        .to_dict()
    )


def _enrich_publication_year(input_df: pd.DataFrame, task_dir: Path) -> pd.DataFrame:
    year_lookup = _publication_year_lookup(task_dir)
    if not year_lookup:
        return input_df

    enriched = input_df.copy()
    keys = _normalized_series(enriched, Schema.TITLE) + "\x1f" + _normalized_series(enriched, Schema.ABSTRACT)
    mapped_years = keys.map(year_lookup).fillna("")

    if "Publication Year" not in enriched.columns:
        insert_at = 1 if len(enriched.columns) >= 1 else 0
        enriched.insert(insert_at, "Publication Year", mapped_years)
        return enriched

    publication_year = enriched["Publication Year"]
    needs_fill = publication_year.isna() | publication_year.astype(str).str.strip().eq("")
    enriched.loc[needs_fill, "Publication Year"] = mapped_years.loc[needs_fill]
    return enriched


def _align_input_frame_to_merged(merged: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
    source = _build_alignment_keys(input_df)
    target = _build_alignment_keys(merged[[column for column in merged.columns if column in {Schema.TITLE, Schema.ABSTRACT}]])
    input_columns = list(input_df.columns)
    aligned = target[["_merge_key", "_dup_index"]].merge(
        source[input_columns + ["_merge_key", "_dup_index"]],
        on=["_merge_key", "_dup_index"],
        how="left",
        sort=False,
    )
    return aligned[input_columns].copy()


def _select_series(working: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for candidate in candidates:
        if candidate in working.columns:
            return working[candidate]
    return pd.Series([""] * len(working), index=working.index, dtype=object)


def _build_review_input_frame(merged: pd.DataFrame, input_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if input_df is not None and not input_df.empty:
        source = _align_input_frame_to_merged(merged, input_df)
    else:
        source = merged.copy()

    review_input = pd.DataFrame(index=merged.index)
    for column in REVIEW_INPUT_COLUMNS:
        review_input[column] = _select_series(source, [column])
    return review_input


def build_review_ready_merged_frame(
    merged: pd.DataFrame,
    input_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    working = merged.copy()
    review_input = _build_review_input_frame(working, input_df)

    derived = pd.DataFrame(index=working.index)
    derived[REVIEW_PREDICT_URBAN_COLUMN] = _select_series(
        working,
        ["final_label", "urban_flag", Schema.IS_URBAN_RENEWAL],
    )
    derived[REVIEW_PREDICT_SPATIAL_COLUMN] = _select_series(
        working,
        [Schema.IS_SPATIAL, f"{Schema.IS_SPATIAL}_spatial"],
    )
    derived[REVIEW_PREDICT_SPATIAL_LEVEL_COLUMN] = _select_series(
        working,
        [Schema.SPATIAL_LEVEL, f"{Schema.SPATIAL_LEVEL}_spatial"],
    )
    derived[REVIEW_PREDICT_SPATIAL_DESC_COLUMN] = _select_series(
        working,
        [Schema.SPATIAL_DESC, f"{Schema.SPATIAL_DESC}_spatial"],
    )

    topic_final = _select_series(working, ["topic_final"]).fillna("").astype(str).str.strip()
    derived["topic_final"] = topic_final
    derived["topic_final_name_en"] = topic_final.apply(topic_name_for_label)
    derived["topic_final_name_zh"] = topic_final.apply(topic_name_zh_for_label)
    derived[REVIEW_URBAN_CONFIDENCE_COLUMN] = _select_series(working, ["confidence"])
    derived[REVIEW_REASONING_COLUMN] = _select_series(working, ["Reasoning", "Reasoning_spatial"])
    derived[REVIEW_SPATIAL_CONFIDENCE_COLUMN] = _select_series(working, ["Confidence", "Confidence_spatial"])
    derived["review_flag"] = _select_series(working, ["review_flag"])
    derived["review_reason"] = _select_series(working, ["review_reason"])
    derived[REVIEW_DECISION_EXPLANATION_COLUMN] = _select_series(working, ["decision_explanation"])
    derived[REVIEW_POSITIVE_EVIDENCE_COLUMN] = _select_series(working, ["primary_positive_evidence"])
    derived[REVIEW_NEGATIVE_EVIDENCE_COLUMN] = _select_series(working, ["primary_negative_evidence"])
    derived[REVIEW_EVIDENCE_BALANCE_COLUMN] = _select_series(working, ["evidence_balance"])
    derived[REVIEW_RULE_STACK_COLUMN] = _select_series(working, ["decision_rule_stack"])
    derived[REVIEW_BINARY_EVIDENCE_COLUMN] = _select_series(working, ["binary_decision_evidence"])
    derived[REVIEW_UNKNOWN_RECOVERY_PATH_COLUMN] = _select_series(working, ["unknown_recovery_path"])
    derived[REVIEW_UNKNOWN_RECOVERY_EVIDENCE_COLUMN] = _select_series(working, ["unknown_recovery_evidence"])
    dynamic_column_map = {
        REVIEW_DYNAMIC_TOPIC_ID_COLUMN: "dynamic_topic_id",
        REVIEW_DYNAMIC_TOPIC_NAME_COLUMN: "dynamic_topic_name_zh",
        REVIEW_DYNAMIC_TOPIC_KEYWORDS_COLUMN: "dynamic_topic_keywords",
        REVIEW_DYNAMIC_TOPIC_SIZE_COLUMN: "dynamic_topic_size",
        REVIEW_DYNAMIC_TOPIC_CONFIDENCE_COLUMN: "dynamic_topic_confidence",
        REVIEW_DYNAMIC_TOPIC_SOURCE_POOL_COLUMN: "dynamic_topic_source_pool",
        REVIEW_DYNAMIC_FIXED_CANDIDATE_COLUMN: "dynamic_to_fixed_topic_candidate",
        REVIEW_DYNAMIC_MAPPING_STATUS_COLUMN: "dynamic_mapping_status",
        REVIEW_DYNAMIC_BINARY_LABEL_COLUMN: "dynamic_binary_candidate_label",
        REVIEW_DYNAMIC_BINARY_CONFIDENCE_COLUMN: "dynamic_binary_candidate_confidence",
        REVIEW_DYNAMIC_BINARY_ACTION_COLUMN: "dynamic_binary_candidate_action",
        REVIEW_DYNAMIC_BINARY_REASON_COLUMN: "dynamic_binary_candidate_reason",
        REVIEW_DYNAMIC_BINARY_PRIORITY_COLUMN: "dynamic_binary_review_priority",
    }
    for review_column, source_column in dynamic_column_map.items():
        derived[review_column] = _select_series(working, [source_column])

    return pd.concat(
        [
            review_input[REVIEW_INPUT_COLUMNS].reset_index(drop=True),
            derived[REVIEW_DERIVED_COLUMNS].reset_index(drop=True),
        ],
        axis=1,
    )
