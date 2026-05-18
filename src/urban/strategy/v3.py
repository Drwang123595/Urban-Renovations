"""Compatibility wrappers for legacy strategy_v3 imports.

The canonical implementation is the stable strategy pipeline in
``src.urban.strategy.pipeline``.
"""

from __future__ import annotations

import pandas as pd

from .decision import decide_stable_strategy
from .evidence import DecisionResult
from .output import STRATEGY_V3_DEFAULTS
from .pipeline import apply_stable_strategy, build_evidence_bundle_from_row


def apply_strategy_v3_shadow(frame: pd.DataFrame) -> pd.DataFrame:
    return apply_stable_strategy(frame, mutate_final_fields=False)


def decide_strategy_v3(evidence) -> DecisionResult:
    decision = decide_stable_strategy(evidence)
    return DecisionResult(
        label=decision.final_label,
        topic=decision.topic_final,
        status=decision.status,
        reason=_legacy_reason(decision.reason, decision.status),
        evidence=decision.positive_evidence,
        confidence=decision.confidence,
    )


def _legacy_reason(reason: str, status) -> str:
    token = str(getattr(status, "value", status) or "")
    if token == "unknown_review":
        return f"weak_or_unmapped_positive:{reason}"
    if token == "accepted_positive" and "renewal_action_and_existing_urban_object" in reason:
        return f"strong_positive:{reason}"
    return reason
