from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional

import pandas as pd

from ...runtime.config import Schema
from ..urban_metadata import normalize_phrase
from ..taxonomy.core import (
    COMMON_RENEWAL_ANCHORS,
    COMMON_RURAL_ANCHORS,
    CORE_RENEWAL_ANCHORS,
    OPEN_SET_NONURBAN_LABEL,
    OPEN_SET_URBAN_LABEL,
    UNKNOWN_TOPIC_LABEL,
    legacy_topic_for_label,
    topic_group_for_label,
    topic_name_for_label,
    urban_flag_for_topic_label,
)


DYNAMIC_BINARY_REFINEMENT_COLUMNS = [
    "dynamic_binary_override_applied",
    "dynamic_binary_override_label",
    "dynamic_binary_override_topic",
    "dynamic_binary_override_reason",
    "dynamic_binary_override_source",
]

DYNAMIC_BINARY_FINAL_MUTATION_FIELDS = {
    Schema.IS_URBAN_RENEWAL,
    "urban_flag",
    "final_label",
    "topic_final",
    "topic_final_group",
    "topic_final_name",
    "topic_label",
    "topic_group",
    "topic_name",
    "legacy_topic_label",
    "legacy_topic_group",
    "legacy_topic_name",
    "taxonomy_coverage_status",
    "binary_decision_source",
    "decision_source",
    "decision_explanation",
    "binary_decision_evidence",
}


def _normalize_binary_label(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    if text.replace(".0", "") == "1":
        return "1"
    if text.replace(".0", "") == "0":
        return "0"
    return ""


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


_WS_RE = re.compile(r"\s+")


def _document_text(row: pd.Series) -> str:
    title = str(row.get(Schema.TITLE, "") or "")
    abstract = str(row.get(Schema.ABSTRACT, "") or "")
    combined = f"{title} {abstract}".strip()
    if not combined:
        return ""
    normalized = normalize_phrase(combined)
    return _WS_RE.sub(" ", normalized).strip()


def _contains_core_anchor(text: str) -> bool:
    if not text:
        return False
    lowered = normalize_phrase(text)
    return any(anchor in lowered for anchor in CORE_RENEWAL_ANCHORS)


def _contains_common_anchor(text: str) -> bool:
    if not text:
        return False
    lowered = normalize_phrase(text)
    return any(anchor in lowered for anchor in COMMON_RENEWAL_ANCHORS)


def _contains_rural_anchor(text: str) -> bool:
    if not text:
        return False
    lowered = normalize_phrase(text)
    return any(anchor in lowered for anchor in COMMON_RURAL_ANCHORS)


@dataclass(frozen=True)
class DynamicBinaryRefinementConfig:
    enabled: bool = True
    mutate_final_fields: bool = True
    unknown_only: bool = True
    allow_flip_existing: bool = False
    force_flip_existing_mutation: bool = False
    require_review_flag_for_flip: bool = True
    near_threshold_margin: float = 0.08
    min_topic_confidence: float = 0.72
    min_topic_size: int = 20
    require_anchor_for_positive: bool = True

    @classmethod
    def from_context(cls, context: dict | None) -> "DynamicBinaryRefinementConfig":
        ctx = context or {}

        def _ctx_bool(key: str, default: bool) -> bool:
            raw = ctx.get(key)
            if raw in ("", None):
                return default
            if isinstance(raw, bool):
                return bool(raw)
            return str(raw).strip().lower() in {"1", "true", "yes", "on"}

        def _ctx_float(key: str, default: float) -> float:
            raw = ctx.get(key)
            if raw in ("", None):
                return float(default)
            try:
                return float(raw)
            except Exception:
                return float(default)

        def _ctx_int(key: str, default: int) -> int:
            raw = ctx.get(key)
            if raw in ("", None):
                return int(default)
            try:
                return int(float(raw))
            except Exception:
                return int(default)

        return cls(
            enabled=_ctx_bool("dynamic_binary_refinement_enabled", True),
            mutate_final_fields=_ctx_bool("dynamic_binary_refinement_mutate", True),
            unknown_only=_ctx_bool("dynamic_binary_refinement_unknown_only", True),
            allow_flip_existing=_ctx_bool("dynamic_binary_refinement_allow_flip", False),
            force_flip_existing_mutation=_ctx_bool(
                "dynamic_binary_refinement_force_flip_existing_mutation",
                False,
            ),
            require_review_flag_for_flip=_ctx_bool(
                "dynamic_binary_refinement_require_review_flag_for_flip",
                True,
            ),
            near_threshold_margin=_ctx_float("dynamic_binary_refinement_near_threshold_margin", 0.08),
            min_topic_confidence=_ctx_float("dynamic_binary_refinement_min_topic_confidence", 0.72),
            min_topic_size=_ctx_int("dynamic_binary_refinement_min_topic_size", 20),
            require_anchor_for_positive=_ctx_bool("dynamic_binary_refinement_require_anchor_for_positive", True),
        )


@dataclass(frozen=True)
class DynamicBinaryCandidateContext:
    """Precomputed series used by dynamic binary refinement."""

    candidate_norm: pd.Series
    topic_confidence: pd.Series
    topic_size: pd.Series
    current_norm: pd.Series
    unknown_mask: pd.Series
    conflict_mask: pd.Series


class DynamicBinaryRefiner:
    """Deterministic binary refinement driven by dynamic topic evidence.

    This post-processing step is designed to be applied after running the
    existing hybrid classifier. It never calls LLM/API and only uses fields
    already present in the prediction workbook (plus the dynamic_topic_* fields).
    """

    def __init__(self, config: DynamicBinaryRefinementConfig | None = None):
        self.config = config or DynamicBinaryRefinementConfig()

    def refine(
        self,
        frame: pd.DataFrame,
        *,
        mutate_final_fields: Optional[bool] = None,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()

        cfg = self.config
        if not cfg.enabled:
            return frame.copy()

        mutate = cfg.mutate_final_fields if mutate_final_fields is None else bool(mutate_final_fields)

        working = self._prepare_working_frame(frame)

        context = self._build_candidate_context(working)
        target_mask = self._build_target_mask(working, context)
        target_mask = self._apply_positive_anchor_constraints(working, target_mask, context)

        if not bool(target_mask.any()):
            return working

        self._apply_overrides(
            working,
            target_mask,
            context,
            mutate_final_fields=mutate,
        )
        return working

    def _prepare_working_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Copy the frame and normalize columns that may receive overrides."""

        working = frame.copy()
        for column in DYNAMIC_BINARY_REFINEMENT_COLUMNS:
            if column not in working.columns:
                working[column] = pd.Series([""] * len(working), index=working.index, dtype=object)
            else:
                working[column] = working[column].astype(object)

        # Pandas 2.2+ becomes strict about dtype upcasting (e.g. int -> str).
        # The refinement step may write string labels into these columns, so
        # keep them as object for robustness even when the frame is loaded from Excel.
        for col in DYNAMIC_BINARY_FINAL_MUTATION_FIELDS:
            if col in working.columns:
                working[col] = working[col].astype(object)
        return working

    def _build_candidate_context(self, working: pd.DataFrame) -> DynamicBinaryCandidateContext:
        """Precompute candidate labels and masks shared by refinement stages."""

        candidate_label_series = working.get("dynamic_binary_candidate_label", pd.Series([""] * len(working)))
        candidate_label_series = candidate_label_series.fillna("").astype(str).str.strip()
        topic_confidence_series = pd.to_numeric(
            working.get("dynamic_topic_confidence", 0.0), errors="coerce"
        ).fillna(0.0)
        topic_size_series = pd.to_numeric(working.get("dynamic_topic_size", 0), errors="coerce").fillna(0).astype(int)

        current_label_raw = working.get("final_label")
        if current_label_raw is None:
            current_label_raw = working.get("urban_flag")
        if current_label_raw is None:
            current_label_raw = working.get(Schema.IS_URBAN_RENEWAL)
        if current_label_raw is None:
            current_label_raw = pd.Series([""] * len(working), index=working.index, dtype=object)
        current_label_series = current_label_raw.fillna("").astype(str).str.strip()

        current_norm = current_label_series.apply(_normalize_binary_label)
        candidate_norm = candidate_label_series.apply(_normalize_binary_label)

        topic_final_series = working.get("topic_final", pd.Series([""] * len(working), index=working.index, dtype=object))
        taxonomy_status_series = working.get(
            "taxonomy_coverage_status",
            pd.Series([""] * len(working), index=working.index, dtype=object),
        )
        topic_unknown = topic_final_series.fillna("").astype(str).str.strip().eq(UNKNOWN_TOPIC_LABEL)
        taxonomy_unknown = taxonomy_status_series.fillna("").astype(str).str.strip().eq("unknown")

        unknown_mask = (~current_norm.isin({"0", "1"})) | topic_unknown | taxonomy_unknown
        conflict_mask = current_norm.isin({"0", "1"}) & (candidate_norm != current_norm) & (~unknown_mask)
        return DynamicBinaryCandidateContext(
            candidate_norm=candidate_norm,
            topic_confidence=topic_confidence_series,
            topic_size=topic_size_series,
            current_norm=current_norm,
            unknown_mask=unknown_mask,
            conflict_mask=conflict_mask,
        )

    def _build_target_mask(
        self,
        working: pd.DataFrame,
        context: DynamicBinaryCandidateContext,
    ) -> pd.Series:
        """Select rows eligible for dynamic binary overrides."""

        cfg = self.config
        eligible = (
            context.candidate_norm.isin({"0", "1"})
            & (context.topic_confidence >= float(cfg.min_topic_confidence))
            & (context.topic_size >= int(cfg.min_topic_size))
        )
        if not bool(eligible.any()):
            return pd.Series(False, index=working.index, dtype=bool)

        if cfg.unknown_only:
            target_mask = eligible & context.unknown_mask
        else:
            target_mask = eligible & (
                context.unknown_mask | (context.conflict_mask if cfg.allow_flip_existing else False)
            )

        if not bool(target_mask.any()):
            return target_mask

        # Dynamic topics are allowed to increase recall (0->1) or resolve
        # unlabeled rows, but they must not reduce recall by flipping an
        # existing binary positive to 0. This applies even when topic_final is
        # Unknown; the binary decision is the final contract.
        negative_flip = (context.candidate_norm == "0") & (context.current_norm == "1")
        target_mask = target_mask & (~negative_flip)

        if cfg.allow_flip_existing and cfg.require_review_flag_for_flip:
            review_flag = pd.to_numeric(
                working.get("review_flag_raw", working.get("review_flag", 0)),
                errors="coerce",
            ).fillna(0).astype(int)
            score = pd.to_numeric(working.get("urban_probability_score", 0.0), errors="coerce").fillna(0.0)
            threshold = pd.to_numeric(
                working.get("binary_decision_threshold", float("nan")),
                errors="coerce",
            )
            threshold = threshold.fillna(float(threshold.median()) if threshold.notna().any() else 0.5)
            near_threshold = (score - threshold).abs() <= float(cfg.near_threshold_margin)
            flip_gate = (review_flag > 0) | near_threshold
            target_mask = target_mask & (context.unknown_mask | (~context.unknown_mask & flip_gate))
        return target_mask

    def _apply_positive_anchor_constraints(
        self,
        working: pd.DataFrame,
        target_mask: pd.Series,
        context: DynamicBinaryCandidateContext,
    ) -> pd.Series:
        """Keep positive dynamic overrides behind the existing anchor gates."""

        if not bool(target_mask.any()):
            return target_mask
        if self.config.require_anchor_for_positive:
            positive_mask = target_mask & (context.candidate_norm == "1")
            if bool(positive_mask.any()):
                uncertain_action = (
                    working.get(
                        "uncertain_nonurban_guard_action",
                        pd.Series([""] * len(working), index=working.index, dtype=object),
                    )
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .str.lower()
                )

                doc_text = pd.Series([""] * len(working), index=working.index, dtype=object)
                doc_text.loc[positive_mask] = working.loc[positive_mask].apply(_document_text, axis=1)

                core_anchor = doc_text.loc[positive_mask].apply(_contains_core_anchor)
                common_anchor = doc_text.loc[positive_mask].apply(_contains_common_anchor)
                rural_anchor = doc_text.loc[positive_mask].apply(_contains_rural_anchor)

                # For unknown rows, keep the strict core-anchor requirement (and
                # block obviously rural contexts).
                # For existing 0->1 flips, allow additional gates:
                # - uncertain_nonurban_guard_action=keep_0 (high precision in eval)
                # - uncertain_nonurban_guard_action=review + common renewal anchor
                current_pos = context.current_norm.loc[positive_mask]
                unknown_pos = context.unknown_mask.loc[positive_mask]
                action_pos = uncertain_action.loc[positive_mask]

                positive_allow = unknown_pos & core_anchor & (~rural_anchor)
                allow_keep0 = (
                    (~unknown_pos)
                    & (current_pos == "0")
                    & action_pos.eq("keep_0")
                    & (~rural_anchor)
                )
                allow_review = (
                    (~unknown_pos)
                    & (current_pos == "0")
                    & action_pos.eq("review")
                    & common_anchor
                    & (~rural_anchor)
                )
                positive_allow = positive_allow | allow_keep0 | allow_review

                allowed = pd.Series(False, index=working.index, dtype=bool)
                allowed.loc[positive_mask] = positive_allow
                target_mask = target_mask & ((context.candidate_norm != "1") | allowed)
        return target_mask

    def _apply_overrides(
        self,
        working: pd.DataFrame,
        target_mask: pd.Series,
        context: DynamicBinaryCandidateContext,
        *,
        mutate_final_fields: bool,
    ) -> None:
        """Record dynamic override evidence and optionally mutate final fields."""

        dynamic_topic_id = working.get("dynamic_topic_id", pd.Series([""] * len(working))).fillna("").astype(str).str.strip()
        mapping_status = working.get("dynamic_mapping_status", pd.Series([""] * len(working))).fillna("").astype(str).str.strip()
        fixed_candidate = working.get("dynamic_to_fixed_topic_candidate", pd.Series([""] * len(working))).fillna("").astype(str).str.strip()
        binary_action = working.get("dynamic_binary_candidate_action", pd.Series([""] * len(working))).fillna("").astype(str).str.strip()

        for idx in working.index[target_mask]:
            cand_label = str(context.candidate_norm.loc[idx] or "").strip()
            if cand_label not in {"0", "1"}:
                continue

            candidate_topic = str(fixed_candidate.loc[idx] or "").strip()
            status = str(mapping_status.loc[idx] or "").strip()
            if not candidate_topic:
                candidate_topic = OPEN_SET_URBAN_LABEL if cand_label == "1" else OPEN_SET_NONURBAN_LABEL
            else:
                group = topic_group_for_label(candidate_topic)
                if cand_label == "1" and group != "urban":
                    candidate_topic = OPEN_SET_URBAN_LABEL
                if cand_label == "0" and group != "nonurban":
                    candidate_topic = OPEN_SET_NONURBAN_LABEL

            confidence = float(context.topic_confidence.loc[idx])
            size = int(context.topic_size.loc[idx])
            source = "dynamic_topic_refiner"
            is_existing_flip = bool(context.conflict_mask.loc[idx] and not context.unknown_mask.loc[idx])
            if context.unknown_mask.loc[idx]:
                source = "dynamic_topic_refiner_unknown"
            elif is_existing_flip:
                source = "dynamic_topic_refiner_flip"

            force_flip = bool(self.config.force_flip_existing_mutation)
            review_only_flip = bool(is_existing_flip and not force_flip)
            if review_only_flip:
                source = "dynamic_topic_refiner_flip_review"

            reason = (
                f"dynamic_topic={dynamic_topic_id.loc[idx]}; "
                f"candidate_label={cand_label}; candidate_topic={candidate_topic}; "
                f"status={status}; action={binary_action.loc[idx]}; "
                f"topic_size={size}; topic_confidence={confidence:.6f}"
            )

            self._record_override_evidence(
                working,
                idx,
                candidate_label=cand_label,
                candidate_topic=candidate_topic,
                reason=reason,
                source=source,
                applied=0 if review_only_flip else 1,
            )

            if mutate_final_fields and not review_only_flip:
                self._mutate_final_fields(
                    working,
                    idx,
                    candidate_label=cand_label,
                    candidate_topic=candidate_topic,
                    source=source,
                    dynamic_topic_id=str(dynamic_topic_id.loc[idx]),
                )

    def _record_override_evidence(
        self,
        working: pd.DataFrame,
        idx: Any,
        *,
        candidate_label: str,
        candidate_topic: str,
        reason: str,
        source: str,
        applied: int = 1,
    ) -> None:
        """Record dynamic binary evidence without touching final decision fields."""

        working.at[idx, "dynamic_binary_override_applied"] = int(applied)
        working.at[idx, "dynamic_binary_override_label"] = candidate_label
        working.at[idx, "dynamic_binary_override_topic"] = candidate_topic
        working.at[idx, "dynamic_binary_override_reason"] = reason
        working.at[idx, "dynamic_binary_override_source"] = source

    def _mutate_final_fields(
        self,
        working: pd.DataFrame,
        idx: Any,
        *,
        candidate_label: str,
        candidate_topic: str,
        source: str,
        dynamic_topic_id: str,
    ) -> None:
        """Apply the explicit final-field mutation allowed by the batch policy."""

        previous_topic = str(working.at[idx, "topic_final"]) if "topic_final" in working.columns else ""
        previous_source = str(working.at[idx, "binary_decision_source"]) if "binary_decision_source" in working.columns else ""

        working.at[idx, Schema.IS_URBAN_RENEWAL] = candidate_label
        if "urban_flag" in working.columns:
            working.at[idx, "urban_flag"] = candidate_label
        if "final_label" in working.columns:
            working.at[idx, "final_label"] = candidate_label

        if "topic_final" in working.columns:
            working.at[idx, "topic_final"] = candidate_topic
        if "topic_final_group" in working.columns:
            working.at[idx, "topic_final_group"] = topic_group_for_label(candidate_topic)
        if "topic_final_name" in working.columns:
            working.at[idx, "topic_final_name"] = topic_name_for_label(candidate_topic)
        if "topic_label" in working.columns:
            working.at[idx, "topic_label"] = candidate_topic
        if "topic_group" in working.columns:
            working.at[idx, "topic_group"] = topic_group_for_label(candidate_topic)
        if "topic_name" in working.columns:
            working.at[idx, "topic_name"] = topic_name_for_label(candidate_topic)

        if "legacy_topic_label" in working.columns:
            legacy_label, legacy_group, legacy_name = legacy_topic_for_label(candidate_topic)
            working.at[idx, "legacy_topic_label"] = legacy_label
            if "legacy_topic_group" in working.columns:
                working.at[idx, "legacy_topic_group"] = legacy_group
            if "legacy_topic_name" in working.columns:
                working.at[idx, "legacy_topic_name"] = legacy_name

        if "taxonomy_coverage_status" in working.columns:
            if candidate_topic in {OPEN_SET_URBAN_LABEL, OPEN_SET_NONURBAN_LABEL}:
                working.at[idx, "taxonomy_coverage_status"] = "open_set"
            elif candidate_topic != UNKNOWN_TOPIC_LABEL:
                working.at[idx, "taxonomy_coverage_status"] = "binary_resolved"

        if "binary_decision_source" in working.columns:
            joined = "|".join(part for part in [previous_source, source] if part)
            working.at[idx, "binary_decision_source"] = joined or source

        if "decision_source" in working.columns:
            prior = str(working.at[idx, "decision_source"] or "")
            joined = "|".join(part for part in [prior, source] if part)
            working.at[idx, "decision_source"] = joined or source

        if "decision_explanation" in working.columns:
            prior = str(working.at[idx, "decision_explanation"] or "")
            suffix = f"; dynamic_refine={source}; topic={candidate_topic}; prev_topic={previous_topic}"
            working.at[idx, "decision_explanation"] = f"{prior}{suffix}" if prior else suffix.lstrip("; ").strip()

        if "binary_decision_evidence" in working.columns:
            prior = str(working.at[idx, "binary_decision_evidence"] or "")
            note = f"dynamic_refine={dynamic_topic_id}:{candidate_topic}"
            joined = "; ".join(part for part in [prior, note] if part)
            working.at[idx, "binary_decision_evidence"] = joined
