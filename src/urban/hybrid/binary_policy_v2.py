from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from ...runtime.config import Schema
from ...runtime.llm_client import DeepSeekClient
from ..core.metadata import normalize_phrase
from ..taxonomy.core import (
    COMMON_EXISTING_URBAN_OBJECTS,
    COMMON_METHOD_ANCHORS,
    COMMON_RENEWAL_ANCHORS,
    COMMON_RURAL_ANCHORS,
    CORE_RENEWAL_ANCHORS,
    UNKNOWN_TOPIC_LABEL,
    topic_group_for_label,
)


BINARY_POLICY_V2_COLUMNS = [
    "binary_policy_action",
    "binary_policy_reason",
    "binary_policy_conflict_type",
    "llm_adjudication_required",
    "llm_adjudication_label",
    "llm_adjudication_confidence",
    "llm_adjudication_reason",
]

BINARY_POLICY_V2_DEFAULTS = {
    "binary_policy_action": "",
    "binary_policy_reason": "",
    "binary_policy_conflict_type": "",
    "llm_adjudication_required": 0,
    "llm_adjudication_label": "",
    "llm_adjudication_confidence": "",
    "llm_adjudication_reason": "",
}

HIGH_RISK_NONURBAN_TOPICS = {"N1", "N3", "N4", "N5", "N7", "N9", "N10"}
HARD_NEGATIVE_REASONS = {"math_term_misuse", "rural_nonurban"}
CONFLICT_EVIDENCE_BALANCES = {"conflict_positive"}
LLM_CONFIDENCE_FLOOR = 0.75

GENTRIFICATION_ANCHORS = (
    "gentrification",
    "green gentrification",
    "state-led gentrification",
    "climate gentrification",
)

URBAN_OBJECT_EXTENSIONS = (
    "built environment",
    "neighborhood",
    "neighbourhood",
    "community",
    "settlement",
    "informal settlement",
    "district",
    "industrial district",
    "housing",
    "public housing",
    "public space",
    "street",
    "block",
    "urban area",
    "city",
)

URBAN_CONTEXT_TERMS = (
    "urban",
    "city",
    "cities",
    "municipal",
    "metropolitan",
    "downtown",
    "inner city",
    "inner-city",
)


@dataclass(frozen=True)
class PolicyDecision:
    label: str
    action: str
    reason: str
    conflict_type: str = ""
    llm_required: int = 0


class UrbanBinaryPolicyV2:
    """Final binary policy for urban-renewal extraction.

    The policy is intentionally evidence based: it never uses row ids, titles as
    fixtures, or ground-truth values. It reconciles the binary score, fixed topic
    family, dynamic-topic evidence, and optional LLM adjudication for conflict
    rows.
    """

    def __init__(
        self,
        *,
        llm_client: Optional[DeepSeekClient] = None,
        llm_enabled: bool = False,
        llm_confidence_floor: float = LLM_CONFIDENCE_FLOOR,
        evidence_only: bool = False,
    ):
        self.llm_client = llm_client
        self.llm_enabled = bool(llm_enabled and llm_client is not None)
        self.llm_confidence_floor = float(llm_confidence_floor)
        self.evidence_only = bool(evidence_only)

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()

        working = frame.copy()
        for column, default_value in BINARY_POLICY_V2_DEFAULTS.items():
            if column not in working.columns:
                working[column] = pd.Series([default_value] * len(working), index=working.index, dtype=object)
            else:
                working[column] = working[column].astype(object)

        for column in (
            Schema.IS_URBAN_RENEWAL,
            "urban_flag",
            "final_label",
            "llm_used",
            "llm_attempted",
            "binary_decision_source",
            "decision_source",
            "decision_explanation",
            "binary_decision_evidence",
        ):
            if column in working.columns:
                working[column] = working[column].astype(object)

        for idx, row in working.iterrows():
            decision = self._decide_without_llm(row)
            self._apply_decision(working, idx, decision)
            if decision.llm_required and self.llm_enabled:
                self._adjudicate_with_llm(working, idx)

        return working

    def _decide_without_llm(self, row: pd.Series) -> PolicyDecision:
        current_label = _normalize_binary_label(
            row.get("final_label", row.get("urban_flag", row.get(Schema.IS_URBAN_RENEWAL, "")))
        )
        final_topic = str(row.get("topic_final", "") or "").strip()
        topic_group = str(row.get("topic_final_group", "") or "").strip().lower()
        if not topic_group:
            topic_group = topic_group_for_label(final_topic)
        score = _safe_float(row.get("urban_probability_score"), 0.0)
        evidence_balance = str(row.get("evidence_balance", "") or "").strip().lower()

        signals = _EvidenceSignals.from_row(row)

        if self._is_hard_negative(row, signals):
            return PolicyDecision(
                label="0",
                action="protected_negative",
                reason=f"hard_negative:{signals.hard_negative_reason or 'risk'}",
                conflict_type="hard_negative",
                llm_required=0,
            )

        if current_label != "1":
            if self._strong_positive_support(topic_group, score, signals, final_topic):
                return PolicyDecision(
                    label="1",
                    action="accept_positive",
                    reason="strong_positive_support_from_anchor_and_topic",
                    conflict_type="false_negative_recovery",
                    llm_required=0,
                )
            return PolicyDecision(
                label=current_label or "0",
                action="accept_negative",
                reason="binary_negative_or_unresolved_without_strong_positive_support",
            )

        if topic_group == "urban":
            if signals.method_only_risk and not signals.strong_positive:
                return PolicyDecision(
                    label="1",
                    action="conflict_review",
                    reason="urban_topic_binary_positive_with_method_context_requires_review",
                    conflict_type="method_only_positive",
                    llm_required=1,
                )
            return PolicyDecision(
                label="1",
                action="accept_positive",
                reason="urban_topic_binary_positive",
            )

        conflict = self._conflict_type(row, topic_group, evidence_balance, signals, final_topic, score)
        if conflict:
            if signals.strong_positive and not signals.rural_risk:
                return PolicyDecision(
                    label="1",
                    action="accept_positive",
                    reason=f"conflict_resolved_by_core_anchor_and_existing_urban_object:{conflict}",
                    conflict_type=conflict,
                    llm_required=0,
                )
            return PolicyDecision(
                label="1",
                action="conflict_review",
                reason=f"binary_positive_retained_for_recall_pending_review:{conflict}",
                conflict_type=conflict,
                llm_required=1,
            )

        return PolicyDecision(
            label="1",
            action="accept_positive",
            reason="binary_positive_without_policy_conflict",
        )

    def _is_hard_negative(self, row: pd.Series, signals: "_EvidenceSignals") -> bool:
        route_reason = str(row.get("metadata_route_reason", "") or "").strip()
        binary_source = str(row.get("binary_decision_source", "") or "")
        if route_reason in HARD_NEGATIVE_REASONS:
            signals.hard_negative_reason = route_reason
            return True
        if "binary_hard_negative_override" in binary_source:
            signals.hard_negative_reason = "binary_hard_negative_override"
            return True
        return False

    def _strong_positive_support(
        self,
        topic_group: str,
        score: float,
        signals: "_EvidenceSignals",
        final_topic: str,
    ) -> bool:
        if signals.strong_positive:
            return True
        if topic_group == "urban" and score >= 0.50 and not signals.method_only_risk:
            return True
        if final_topic in HIGH_RISK_NONURBAN_TOPICS:
            return False
        return False

    def _conflict_type(
        self,
        row: pd.Series,
        topic_group: str,
        evidence_balance: str,
        signals: "_EvidenceSignals",
        final_topic: str,
        score: float,
    ) -> str:
        conflicts: list[str] = []
        consistency = str(row.get("binary_topic_consistency_flag", "") or "").strip()
        if consistency in {"1", "1.0", "true", "True"}:
            conflicts.append("binary_topic_inconsistency")
        if evidence_balance in CONFLICT_EVIDENCE_BALANCES:
            conflicts.append(evidence_balance)
        if topic_group in {"nonurban", "unknown"}:
            conflicts.append(f"binary_positive_{topic_group}_topic")
        if final_topic in HIGH_RISK_NONURBAN_TOPICS:
            conflicts.append(f"high_risk_nonurban_topic_{final_topic}")
        if signals.method_only_risk:
            conflicts.append("method_only_or_background_context")
        threshold = _safe_float(row.get("binary_decision_threshold"), 0.45)
        if abs(score - threshold) <= 0.03:
            conflicts.append("near_threshold")
        dynamic_label = _normalize_binary_label(row.get("dynamic_binary_candidate_label", ""))
        if dynamic_label == "0":
            conflicts.append("dynamic_topic_negative_candidate")
        return "|".join(dict.fromkeys(conflicts))

    def _apply_decision(self, frame: pd.DataFrame, idx: Any, decision: PolicyDecision) -> None:
        frame.at[idx, "binary_policy_action"] = decision.action
        frame.at[idx, "binary_policy_reason"] = decision.reason
        frame.at[idx, "binary_policy_conflict_type"] = decision.conflict_type
        frame.at[idx, "llm_adjudication_required"] = int(bool(decision.llm_required))
        if not self.evidence_only:
            self._set_final_label(frame, idx, decision.label)
        self._append_source(frame, idx, "binary_policy_v2")
        self._append_explanation(frame, idx, f"policy_v2={decision.action}:{decision.reason}")

    def _set_final_label(self, frame: pd.DataFrame, idx: Any, label: str) -> None:
        label = "1" if str(label) == "1" else "0"
        for column in (Schema.IS_URBAN_RENEWAL, "urban_flag", "final_label"):
            if column in frame.columns:
                frame.at[idx, column] = label

    def _append_source(self, frame: pd.DataFrame, idx: Any, source: str) -> None:
        for column in ("binary_decision_source", "decision_source"):
            if column not in frame.columns:
                continue
            prior = str(frame.at[idx, column] or "").strip()
            parts = [part for part in prior.split("|") if part]
            if source not in parts:
                parts.append(source)
            frame.at[idx, column] = "|".join(parts)

    def _append_explanation(self, frame: pd.DataFrame, idx: Any, note: str) -> None:
        if "decision_explanation" in frame.columns:
            prior = str(frame.at[idx, "decision_explanation"] or "").strip()
            frame.at[idx, "decision_explanation"] = f"{prior}; {note}" if prior else note
        if "binary_decision_evidence" in frame.columns:
            prior = str(frame.at[idx, "binary_decision_evidence"] or "").strip()
            frame.at[idx, "binary_decision_evidence"] = f"{prior}; {note}" if prior else note

    def _adjudicate_with_llm(self, frame: pd.DataFrame, idx: Any) -> None:
        if self.llm_client is None:
            return
        row = frame.loc[idx]
        if "llm_attempted" in frame.columns:
            frame.at[idx, "llm_attempted"] = 1

        raw = self.llm_client.chat_completion(self._messages_for_row(row), temperature=0.0, max_retries=2)
        label, confidence, reason = self._parse_llm_adjudication(raw)
        frame.at[idx, "llm_adjudication_label"] = label
        frame.at[idx, "llm_adjudication_confidence"] = confidence if label else ""
        frame.at[idx, "llm_adjudication_reason"] = reason
        if label not in {"0", "1"} or confidence < self.llm_confidence_floor:
            return

        if not self.evidence_only:
            self._set_final_label(frame, idx, label)
        if "llm_used" in frame.columns:
            frame.at[idx, "llm_used"] = 1
        self._append_source(frame, idx, "llm_adjudication")
        self._append_explanation(frame, idx, f"llm_adjudication={label}:{confidence:.2f}:{reason}")

    def _messages_for_row(self, row: pd.Series) -> list[dict[str, str]]:
        evidence = {
            "current_label": row.get("final_label", ""),
            "topic_final": row.get("topic_final", ""),
            "topic_final_group": row.get("topic_final_group", ""),
            "urban_probability_score": row.get("urban_probability_score", ""),
            "evidence_balance": row.get("evidence_balance", ""),
            "binary_policy_conflict_type": row.get("binary_policy_conflict_type", ""),
            "primary_positive_evidence": row.get("primary_positive_evidence", ""),
            "primary_negative_evidence": row.get("primary_negative_evidence", ""),
            "dynamic_topic": row.get("dynamic_topic_name_zh", ""),
            "dynamic_keywords": row.get("dynamic_topic_keywords", ""),
        }
        user = (
            "Decide whether the paper is an urban renewal study. "
            "Urban renewal means renewal/regeneration/redevelopment/revitalization/rehabilitation/retrofit/adaptive reuse/"
            "gentrification/upgrading of existing urban built environments, communities, brownfields, old districts, "
            "housing estates, industrial heritage, public spaces, or informal settlements. "
            "Return 0 for generic urban policy, rural regeneration, pure methods, transport/ecology/tourism, or background-only mentions.\n\n"
            f"Title: {row.get(Schema.TITLE, row.get('Article Title', ''))}\n"
            f"Abstract: {row.get(Schema.ABSTRACT, row.get('Abstract', ''))}\n"
            f"Keywords: {row.get(Schema.KEYWORDS, '')} {row.get(Schema.KEYWORDS_PLUS, '')} {row.get(Schema.AUTHOR_KEYWORDS, '')}\n"
            f"Rule evidence: {json.dumps(evidence, ensure_ascii=False)}\n\n"
            "Return strict JSON only: {\"label\":\"0 or 1\",\"confidence\":0.0-1.0,\"reason\":\"short evidence\"}"
        )
        return [
            {
                "role": "system",
                "content": "You are a strict binary adjudicator for academic urban renewal literature screening.",
            },
            {"role": "user", "content": user},
        ]

    def _parse_llm_adjudication(self, raw: Optional[str]) -> tuple[str, float, str]:
        text = str(raw or "").strip()
        if not text:
            return "", 0.0, "empty_response"
        payload = text
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            payload = match.group(0)
        try:
            data = json.loads(payload)
            label = _normalize_binary_label(data.get("label", ""))
            confidence = _safe_float(data.get("confidence"), 0.0)
            reason = str(data.get("reason", "json_response") or "json_response").strip()
            return label, min(max(confidence, 0.0), 1.0), reason
        except Exception:
            first_digit = re.search(r"\b([01])\b", text)
            if first_digit:
                return first_digit.group(1), 0.75, "fallback_digit_response"
            return "", 0.0, "parse_failure"


@dataclass
class _EvidenceSignals:
    text: str
    action_anchor: bool
    object_anchor: bool
    core_anchor: bool
    rural_risk: bool
    method_risk: bool
    method_only_risk: bool
    urban_context: bool
    strong_positive: bool
    hard_negative_reason: str = ""

    @classmethod
    def from_row(cls, row: pd.Series) -> "_EvidenceSignals":
        text = _document_text(row)
        norm = normalize_phrase(text)
        action_anchor = _contains_any(norm, COMMON_RENEWAL_ANCHORS + GENTRIFICATION_ANCHORS)
        core_anchor = _contains_any(norm, CORE_RENEWAL_ANCHORS + GENTRIFICATION_ANCHORS)
        object_anchor = _contains_any(norm, COMMON_EXISTING_URBAN_OBJECTS + URBAN_OBJECT_EXTENSIONS)
        rural_risk = _contains_any(norm, COMMON_RURAL_ANCHORS)
        method_risk = _contains_any(norm, COMMON_METHOD_ANCHORS)
        urban_context = _contains_any(norm, URBAN_CONTEXT_TERMS)
        risk_tags = str(row.get("stage1_risk_tags", "") or "").lower()
        method_only_risk = (
            method_risk
            or "generic_technical" in risk_tags
            or "background_support" in risk_tags
            or "explicit_renewal_wording_but_other_object" in risk_tags
        ) and not (core_anchor and object_anchor)
        strong_positive = bool(core_anchor and object_anchor and not rural_risk and not method_only_risk)
        return cls(
            text=text,
            action_anchor=action_anchor,
            object_anchor=object_anchor,
            core_anchor=core_anchor,
            rural_risk=rural_risk,
            method_risk=method_risk,
            method_only_risk=method_only_risk,
            urban_context=urban_context,
            strong_positive=strong_positive,
        )


def _document_text(row: pd.Series) -> str:
    parts = []
    for column in (
        Schema.TITLE,
        Schema.ABSTRACT,
        Schema.AUTHOR_KEYWORDS,
        Schema.KEYWORDS_PLUS,
        Schema.KEYWORDS,
        Schema.WOS_CATEGORIES,
        Schema.RESEARCH_AREAS,
    ):
        value = row.get(column, "")
        if value not in (None, ""):
            parts.append(str(value))
    return " ".join(parts)


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(normalize_phrase(phrase) in text for phrase in phrases if phrase)


def _normalize_binary_label(value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text in {"0", "1"}:
        return text
    lowered = text.lower()
    if lowered in {"true", "yes", "urban", "positive"}:
        return "1"
    if lowered in {"false", "no", "nonurban", "negative"}:
        return "0"
    return ""


def _safe_float(value: Any, default: float) -> float:
    try:
        if value in (None, ""):
            return float(default)
        return float(value)
    except Exception:
        return float(default)
