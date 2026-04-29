from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..runtime.config import Schema
from ..reporting.metric_name_catalog import display_name_for_field, metric_dictionary_frame
from ..urban.urban_topic_taxonomy import topic_name_for_label, topic_name_zh_for_label


REVIEW_PREDICT_URBAN_COLUMN = display_name_for_field("final_label")
REVIEW_PREDICT_SPATIAL_COLUMN = display_name_for_field(Schema.IS_SPATIAL)
REVIEW_PREDICT_SPATIAL_LEVEL_COLUMN = display_name_for_field(Schema.SPATIAL_LEVEL)
REVIEW_PREDICT_SPATIAL_DESC_COLUMN = display_name_for_field(Schema.SPATIAL_DESC)
REVIEW_TOPIC_FINAL_COLUMN = display_name_for_field("topic_final")
REVIEW_TOPIC_FINAL_NAME_EN_COLUMN = display_name_for_field("topic_final_name_en")
REVIEW_TOPIC_FINAL_NAME_ZH_COLUMN = display_name_for_field("topic_final_name_zh")
REVIEW_TAXONOMY_COVERAGE_COLUMN = display_name_for_field("taxonomy_coverage_status")
REVIEW_URBAN_CONFIDENCE_COLUMN = display_name_for_field("confidence")
REVIEW_REASONING_COLUMN = display_name_for_field("Reasoning")
REVIEW_SPATIAL_CONFIDENCE_COLUMN = display_name_for_field("Confidence")
REVIEW_SPATIAL_VALIDATION_STATUS_COLUMN = display_name_for_field(Schema.SPATIAL_VALIDATION_STATUS)
REVIEW_SPATIAL_VALIDATION_REASON_COLUMN = display_name_for_field(Schema.SPATIAL_VALIDATION_REASON)
REVIEW_SPATIAL_AREA_EVIDENCE_COLUMN = display_name_for_field(Schema.SPATIAL_AREA_EVIDENCE)
REVIEW_REVIEW_FLAG_COLUMN = display_name_for_field("review_flag")
REVIEW_REVIEW_REASON_COLUMN = display_name_for_field("review_reason")
REVIEW_DECISION_EXPLANATION_COLUMN = display_name_for_field("decision_explanation")
REVIEW_POSITIVE_EVIDENCE_COLUMN = display_name_for_field("primary_positive_evidence")
REVIEW_NEGATIVE_EVIDENCE_COLUMN = display_name_for_field("primary_negative_evidence")
REVIEW_EVIDENCE_BALANCE_COLUMN = display_name_for_field("evidence_balance")
REVIEW_RULE_STACK_COLUMN = display_name_for_field("decision_rule_stack")
REVIEW_BINARY_EVIDENCE_COLUMN = display_name_for_field("binary_decision_evidence")
REVIEW_UNKNOWN_RECOVERY_PATH_COLUMN = display_name_for_field("unknown_recovery_path")
REVIEW_UNKNOWN_RECOVERY_EVIDENCE_COLUMN = display_name_for_field("unknown_recovery_evidence")
REVIEW_DYNAMIC_TOPIC_ID_COLUMN = display_name_for_field("dynamic_topic_id")
REVIEW_DYNAMIC_TOPIC_NAME_COLUMN = display_name_for_field("dynamic_topic_name_zh")
REVIEW_DYNAMIC_TOPIC_KEYWORDS_COLUMN = display_name_for_field("dynamic_topic_keywords")
REVIEW_DYNAMIC_TOPIC_SIZE_COLUMN = display_name_for_field("dynamic_topic_size")
REVIEW_DYNAMIC_TOPIC_CONFIDENCE_COLUMN = display_name_for_field("dynamic_topic_confidence")
REVIEW_DYNAMIC_TOPIC_SOURCE_POOL_COLUMN = display_name_for_field("dynamic_topic_source_pool")
REVIEW_DYNAMIC_FIXED_CANDIDATE_COLUMN = display_name_for_field("dynamic_to_fixed_topic_candidate")
REVIEW_DYNAMIC_MAPPING_STATUS_COLUMN = display_name_for_field("dynamic_mapping_status")
REVIEW_DYNAMIC_BINARY_LABEL_COLUMN = display_name_for_field("dynamic_binary_candidate_label")
REVIEW_DYNAMIC_BINARY_CONFIDENCE_COLUMN = display_name_for_field("dynamic_binary_candidate_confidence")
REVIEW_DYNAMIC_BINARY_ACTION_COLUMN = display_name_for_field("dynamic_binary_candidate_action")
REVIEW_DYNAMIC_BINARY_REASON_COLUMN = display_name_for_field("dynamic_binary_candidate_reason")
REVIEW_DYNAMIC_BINARY_PRIORITY_COLUMN = display_name_for_field("dynamic_binary_review_priority")
REVIEW_DYNAMIC_BINARY_OVERRIDE_APPLIED_COLUMN = display_name_for_field("dynamic_binary_override_applied")
REVIEW_DYNAMIC_BINARY_OVERRIDE_LABEL_COLUMN = display_name_for_field("dynamic_binary_override_label")
REVIEW_DYNAMIC_BINARY_OVERRIDE_TOPIC_COLUMN = display_name_for_field("dynamic_binary_override_topic")
REVIEW_DYNAMIC_BINARY_OVERRIDE_REASON_COLUMN = display_name_for_field("dynamic_binary_override_reason")
REVIEW_BINARY_POLICY_ACTION_COLUMN = display_name_for_field("binary_policy_action")
REVIEW_BINARY_POLICY_REASON_COLUMN = display_name_for_field("binary_policy_reason")
REVIEW_BINARY_POLICY_CONFLICT_TYPE_COLUMN = display_name_for_field("binary_policy_conflict_type")
REVIEW_LLM_ADJUDICATION_REQUIRED_COLUMN = display_name_for_field("llm_adjudication_required")
REVIEW_LLM_ADJUDICATION_LABEL_COLUMN = display_name_for_field("llm_adjudication_label")
REVIEW_LLM_ADJUDICATION_CONFIDENCE_COLUMN = display_name_for_field("llm_adjudication_confidence")
REVIEW_LLM_ADJUDICATION_REASON_COLUMN = display_name_for_field("llm_adjudication_reason")
REVIEW_LLM_USED_COLUMN = display_name_for_field("llm_used")
REVIEW_LLM_ATTEMPTED_COLUMN = display_name_for_field("llm_attempted")

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
    REVIEW_TOPIC_FINAL_COLUMN,
    REVIEW_TOPIC_FINAL_NAME_EN_COLUMN,
    REVIEW_TOPIC_FINAL_NAME_ZH_COLUMN,
    REVIEW_TAXONOMY_COVERAGE_COLUMN,
    REVIEW_URBAN_CONFIDENCE_COLUMN,
    REVIEW_REASONING_COLUMN,
    REVIEW_SPATIAL_CONFIDENCE_COLUMN,
    REVIEW_SPATIAL_VALIDATION_STATUS_COLUMN,
    REVIEW_SPATIAL_VALIDATION_REASON_COLUMN,
    REVIEW_SPATIAL_AREA_EVIDENCE_COLUMN,
    REVIEW_REVIEW_FLAG_COLUMN,
    REVIEW_REVIEW_REASON_COLUMN,
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
    REVIEW_DYNAMIC_BINARY_OVERRIDE_APPLIED_COLUMN,
    REVIEW_DYNAMIC_BINARY_OVERRIDE_LABEL_COLUMN,
    REVIEW_DYNAMIC_BINARY_OVERRIDE_TOPIC_COLUMN,
    REVIEW_DYNAMIC_BINARY_OVERRIDE_REASON_COLUMN,
    REVIEW_BINARY_POLICY_ACTION_COLUMN,
    REVIEW_BINARY_POLICY_REASON_COLUMN,
    REVIEW_BINARY_POLICY_CONFLICT_TYPE_COLUMN,
    REVIEW_LLM_ADJUDICATION_REQUIRED_COLUMN,
    REVIEW_LLM_ADJUDICATION_LABEL_COLUMN,
    REVIEW_LLM_ADJUDICATION_CONFIDENCE_COLUMN,
    REVIEW_LLM_ADJUDICATION_REASON_COLUMN,
    REVIEW_LLM_USED_COLUMN,
    REVIEW_LLM_ATTEMPTED_COLUMN,
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
    derived[REVIEW_TOPIC_FINAL_COLUMN] = topic_final
    derived[REVIEW_TOPIC_FINAL_NAME_EN_COLUMN] = topic_final.apply(topic_name_for_label)
    derived[REVIEW_TOPIC_FINAL_NAME_ZH_COLUMN] = topic_final.apply(topic_name_zh_for_label)
    derived[REVIEW_TAXONOMY_COVERAGE_COLUMN] = _select_series(working, ["taxonomy_coverage_status"])
    derived[REVIEW_URBAN_CONFIDENCE_COLUMN] = _select_series(working, ["confidence"])
    derived[REVIEW_REASONING_COLUMN] = _select_series(working, ["Reasoning", "Reasoning_spatial"])
    derived[REVIEW_SPATIAL_CONFIDENCE_COLUMN] = _select_series(working, ["Confidence", "Confidence_spatial"])
    derived[REVIEW_SPATIAL_VALIDATION_STATUS_COLUMN] = _select_series(
        working,
        [Schema.SPATIAL_VALIDATION_STATUS, f"{Schema.SPATIAL_VALIDATION_STATUS}_spatial"],
    )
    derived[REVIEW_SPATIAL_VALIDATION_REASON_COLUMN] = _select_series(
        working,
        [Schema.SPATIAL_VALIDATION_REASON, f"{Schema.SPATIAL_VALIDATION_REASON}_spatial"],
    )
    derived[REVIEW_SPATIAL_AREA_EVIDENCE_COLUMN] = _select_series(
        working,
        [Schema.SPATIAL_AREA_EVIDENCE, f"{Schema.SPATIAL_AREA_EVIDENCE}_spatial"],
    )
    derived[REVIEW_REVIEW_FLAG_COLUMN] = _select_series(working, ["review_flag"])
    derived[REVIEW_REVIEW_REASON_COLUMN] = _select_series(working, ["review_reason"])
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
        REVIEW_DYNAMIC_BINARY_OVERRIDE_APPLIED_COLUMN: "dynamic_binary_override_applied",
        REVIEW_DYNAMIC_BINARY_OVERRIDE_LABEL_COLUMN: "dynamic_binary_override_label",
        REVIEW_DYNAMIC_BINARY_OVERRIDE_TOPIC_COLUMN: "dynamic_binary_override_topic",
        REVIEW_DYNAMIC_BINARY_OVERRIDE_REASON_COLUMN: "dynamic_binary_override_reason",
        REVIEW_BINARY_POLICY_ACTION_COLUMN: "binary_policy_action",
        REVIEW_BINARY_POLICY_REASON_COLUMN: "binary_policy_reason",
        REVIEW_BINARY_POLICY_CONFLICT_TYPE_COLUMN: "binary_policy_conflict_type",
        REVIEW_LLM_ADJUDICATION_REQUIRED_COLUMN: "llm_adjudication_required",
        REVIEW_LLM_ADJUDICATION_LABEL_COLUMN: "llm_adjudication_label",
        REVIEW_LLM_ADJUDICATION_CONFIDENCE_COLUMN: "llm_adjudication_confidence",
        REVIEW_LLM_ADJUDICATION_REASON_COLUMN: "llm_adjudication_reason",
        REVIEW_LLM_USED_COLUMN: "llm_used",
        REVIEW_LLM_ATTEMPTED_COLUMN: "llm_attempted",
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


def build_metric_dictionary_frame() -> pd.DataFrame:
    return metric_dictionary_frame()
