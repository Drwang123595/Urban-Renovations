from __future__ import annotations

from typing import Any

import pandas as pd

from ...runtime.config import Config, Schema
from ...runtime.llm_client import DeepSeekClient
from .llm_adjudicator import LlmAdjudicationResult, LlmAdjudicator
from .llm_triage import HARD_NEGATIVE_REASONS, LlmTriageDecision, LlmTriagePolicy


LLM_BINARY_V2_WORKFLOW = "llm_binary_v2"
LLM_OVERRIDE_CONFIDENCE_FLOOR = 0.80
LLM_INFORMATIVE_CONFIDENCE_FLOOR = 0.65

LLM_BINARY_V2_DEFAULTS: dict[str, Any] = {
    "pre_llm_label": "",
    "pre_llm_score": "",
    "pre_llm_decision_source": "",
    "pre_llm_decision_reason": "",
    "llm_triage_action": "",
    "llm_triage_reasons": "",
    "llm_adjudication_attempted": 0,
    "llm_adjudication_used": 0,
    "llm_adjudication_status": "",
    "llm_adjudication_label": "",
    "llm_adjudication_confidence": "",
    "llm_adjudication_decision_type": "",
    "llm_adjudication_reason": "",
    "llm_adjudication_evidence": "",
    "llm_adjudication_failure_reason": "",
    "llm_adjudication_prompt_version": "",
    "binary_final_score": "",
    "binary_final_source": "",
    "binary_final_reason": "",
}


class LlmBinaryFinalizer:
    """Finalize binary labels with selective structured LLM adjudication."""

    def __init__(
        self,
        *,
        llm_client: DeepSeekClient | None = None,
        llm_enabled: bool = False,
        triage_policy: LlmTriagePolicy | None = None,
        adjudicator: LlmAdjudicator | None = None,
        override_confidence_floor: float = LLM_OVERRIDE_CONFIDENCE_FLOOR,
        informative_confidence_floor: float = LLM_INFORMATIVE_CONFIDENCE_FLOOR,
    ):
        self.llm_enabled = bool(llm_enabled and llm_client is not None)
        self.triage_policy = triage_policy or LlmTriagePolicy()
        self.adjudicator = adjudicator or LlmAdjudicator(llm_client)
        self.override_confidence_floor = float(override_confidence_floor)
        self.informative_confidence_floor = float(informative_confidence_floor)

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()

        working = frame.copy()
        for column, default in LLM_BINARY_V2_DEFAULTS.items():
            if column not in working.columns:
                working[column] = pd.Series([default] * len(working), index=working.index, dtype=object)
            else:
                working[column] = working[column].astype(object)

        for idx, row in working.iterrows():
            self._capture_pre_llm_state(working, idx, row)
            triage = self.triage_policy.evaluate(row)
            self._write_triage(working, idx, triage)
            if not triage.should_call or not self.llm_enabled:
                self._finalize_without_llm(working, idx, row, triage)
                continue
            adjudication = self.adjudicator.adjudicate(row)
            final_label, final_score, source, reason = self._resolve_label(row, triage, adjudication)
            self._write_adjudication(working, idx, adjudication)
            self._set_binary_result(working, idx, final_label=final_label, final_score=final_score, source=source, reason=reason)

        return working

    def _capture_pre_llm_state(self, frame: pd.DataFrame, idx: Any, row: pd.Series) -> None:
        frame.at[idx, "pre_llm_label"] = _label(row.get("final_label", row.get("urban_flag", "")))
        frame.at[idx, "pre_llm_score"] = _safe_float(row.get("urban_probability_score"), "")
        frame.at[idx, "pre_llm_decision_source"] = _text(row.get("decision_source") or row.get("binary_decision_source"))
        frame.at[idx, "pre_llm_decision_reason"] = _text(row.get("decision_reason") or row.get("binary_policy_reason"))

    def _write_triage(self, frame: pd.DataFrame, idx: Any, triage: LlmTriageDecision) -> None:
        frame.at[idx, "llm_triage_action"] = triage.action
        frame.at[idx, "llm_triage_reasons"] = ";".join(triage.reasons)

    def _finalize_without_llm(self, frame: pd.DataFrame, idx: Any, row: pd.Series, triage: LlmTriageDecision) -> None:
        label = _label(row.get("final_label", row.get("urban_flag", ""))) or "0"
        score = _score_for_label(row, label)
        reason = triage.action
        if triage.should_call and not self.llm_enabled:
            frame.at[idx, "llm_adjudication_status"] = "skipped_disabled"
            reason = f"{reason}:llm_disabled"
        else:
            frame.at[idx, "llm_adjudication_status"] = "not_required"
        self._set_binary_result(
            frame,
            idx,
            final_label=label,
            final_score=score,
            source="deterministic_binary",
            reason=reason,
        )

    def _resolve_label(
        self,
        row: pd.Series,
        triage: LlmTriageDecision,
        adjudication: LlmAdjudicationResult,
    ) -> tuple[str, float, str, str]:
        pre_label = _label(row.get("final_label", row.get("urban_flag", ""))) or "0"
        pre_score = _score_for_label(row, pre_label)
        if _text(row.get("metadata_route_reason")) in HARD_NEGATIVE_REASONS:
            return pre_label, pre_score, "deterministic_binary", "protected_hard_negative"
        if adjudication.status != "valid" or adjudication.label not in {"0", "1"}:
            return pre_label, pre_score, "deterministic_binary", f"llm_ignored:{adjudication.status or 'invalid'}"
        if adjudication.confidence >= self.override_confidence_floor:
            final_score = adjudication.confidence if adjudication.label == "1" else 1.0 - adjudication.confidence
            return adjudication.label, round(final_score, 6), LLM_BINARY_V2_WORKFLOW, (
                f"llm_override:{adjudication.decision_type}:{adjudication.reason}"
            )
        if adjudication.confidence >= self.informative_confidence_floor:
            return pre_label, pre_score, "deterministic_binary", (
                f"llm_informative_only:{adjudication.decision_type}:{adjudication.reason}"
            )
        return pre_label, pre_score, "deterministic_binary", f"llm_low_confidence:{adjudication.confidence:.4f}"

    def _write_adjudication(self, frame: pd.DataFrame, idx: Any, adjudication: LlmAdjudicationResult) -> None:
        frame.at[idx, "llm_adjudication_attempted"] = int(adjudication.attempted)
        frame.at[idx, "llm_adjudication_status"] = _status_for_result(adjudication, self)
        frame.at[idx, "llm_adjudication_label"] = adjudication.label
        frame.at[idx, "llm_adjudication_confidence"] = adjudication.confidence if adjudication.attempted else ""
        frame.at[idx, "llm_adjudication_decision_type"] = adjudication.decision_type
        frame.at[idx, "llm_adjudication_reason"] = adjudication.reason
        frame.at[idx, "llm_adjudication_evidence"] = "; ".join(adjudication.evidence)
        frame.at[idx, "llm_adjudication_failure_reason"] = adjudication.failure_reason
        frame.at[idx, "llm_adjudication_prompt_version"] = adjudication.prompt_version
        frame.at[idx, "llm_adjudication_used"] = int(
            adjudication.status == "valid" and adjudication.confidence >= self.override_confidence_floor
        )
        if "llm_attempted" in frame.columns:
            frame.at[idx, "llm_attempted"] = max(_int(frame.at[idx, "llm_attempted"]), int(adjudication.attempted))
        if "llm_used" in frame.columns:
            frame.at[idx, "llm_used"] = max(_int(frame.at[idx, "llm_used"]), 0)

    def _set_binary_result(
        self,
        frame: pd.DataFrame,
        idx: Any,
        *,
        final_label: str,
        final_score: float,
        source: str,
        reason: str,
    ) -> None:
        normalized = "1" if str(final_label) == "1" else "0"
        for column in (Schema.IS_URBAN_RENEWAL, "final_label", "urban_flag"):
            if column in frame.columns:
                frame.at[idx, column] = normalized
        frame.at[idx, "binary_final_score"] = round(float(final_score), 6)
        frame.at[idx, "binary_final_source"] = source
        frame.at[idx, "binary_final_reason"] = reason
        if source == LLM_BINARY_V2_WORKFLOW and "decision_source" in frame.columns:
            prior = _text(frame.at[idx, "decision_source"])
            frame.at[idx, "decision_source"] = _append_pipe(prior, source)
        if source == LLM_BINARY_V2_WORKFLOW and "decision_reason" in frame.columns:
            prior = _text(frame.at[idx, "decision_reason"])
            frame.at[idx, "decision_reason"] = f"{prior};{reason}".strip(";")


def workflow_enabled(context: dict[str, Any]) -> bool:
    return str(context.get("urban_binary_workflow_version", "stable_v1") or "stable_v1").strip() == LLM_BINARY_V2_WORKFLOW


def summarize_llm_binary_v2(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "llm_adjudication_attempted" not in frame.columns:
        return {
            "urban_binary_workflow_version": LLM_BINARY_V2_WORKFLOW,
            "llm_adjudication_attempted_count": 0,
            "llm_adjudication_used_count": 0,
            "llm_adjudication_invalid_count": 0,
        }
    attempted = pd.to_numeric(frame.get("llm_adjudication_attempted"), errors="coerce").fillna(0).astype(int)
    used = pd.to_numeric(frame.get("llm_adjudication_used"), errors="coerce").fillna(0).astype(int)
    statuses = frame.get("llm_adjudication_status", pd.Series(dtype=object)).fillna("").astype(str)
    return {
        "urban_binary_workflow_version": LLM_BINARY_V2_WORKFLOW,
        "llm_adjudication_attempted_count": int(attempted.sum()),
        "llm_adjudication_used_count": int(used.sum()),
        "llm_adjudication_invalid_count": int(statuses.str.startswith(("invalid", "unsupported", "incoherent")).sum()),
        "llm_adjudication_status_counts": statuses.value_counts().to_dict(),
    }


def _status_for_result(adjudication: LlmAdjudicationResult, finalizer: LlmBinaryFinalizer) -> str:
    if adjudication.status != "valid":
        return adjudication.status
    if adjudication.confidence >= finalizer.override_confidence_floor:
        return "used"
    if adjudication.confidence >= finalizer.informative_confidence_floor:
        return "informative_only"
    return "ignored_low_confidence"


def _label(value: Any) -> str:
    text = _text(value)
    if text.endswith(".0"):
        text = text[:-2]
    return text if text in {"0", "1"} else ""


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _safe_float(value: Any, default: Any) -> Any:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_for_label(row: pd.Series, label: str) -> float:
    score = _safe_float(row.get("urban_probability_score"), None)
    if score is None:
        return 0.5 if label == "" else (0.75 if label == "1" else 0.25)
    return round(float(score), 6)


def _int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _append_pipe(prior: str, value: str) -> str:
    parts = [part for part in prior.split("|") if part]
    if value not in parts:
        parts.append(value)
    return "|".join(parts)
