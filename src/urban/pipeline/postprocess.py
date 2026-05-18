"""Batch post-processing orchestration for urban-renewal predictions."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ...runtime.llm_client import DeepSeekClient
from ..dynamic.binary_refinement import DynamicBinaryRefinementConfig, DynamicBinaryRefiner
from ..dynamic.topic_discovery import DynamicTopicConfig, DynamicTopicDiscovery
from ..hybrid.binary_policy_v2 import UrbanBinaryPolicyV2
from ..strategy import apply_stable_strategy


def postprocess_urban_predictions(
    frame: pd.DataFrame,
    *,
    run_context: dict[str, Any] | None = None,
    llm_client: DeepSeekClient | None = None,
    hybrid_llm_assist_enabled: bool = False,
    urban_method: Any = None,
) -> pd.DataFrame:
    """Apply batch-only urban-renewal post-processing in contract order.

    Dynamic topic discovery only appends evidence columns. Dynamic binary
    refinement may mutate final labels only when explicitly enabled. The
    binary policy remains the production reconciliation layer for legacy
    stability; the stable strategy layer appends one explainable decision
    candidate by default and can mutate final fields only by explicit context.
    """

    context = run_context or {}
    enriched = frame

    # 1) Evidence-only layer: dynamic_topic_* / dynamic_binary_candidate_*.
    if _dynamic_topics_enabled(context) or _dynamic_binary_refinement_enabled(context):
        enriched = _append_dynamic_topic_evidence(enriched, context)

    # 2) Explicit label-mutation layer: may rewrite final_label / urban_flag.
    if _dynamic_binary_refinement_enabled(context):
        enriched = _apply_dynamic_binary_refinement(enriched, context)

    # 3) Legacy binary reconciliation layer: preserves existing production
    # final_label / urban_flag behavior unless callers disable it.
    if _binary_policy_v2_enabled(context):
        enriched = _apply_binary_policy_v2(
            enriched,
            context=context,
            llm_client=llm_client,
            hybrid_llm_assist_enabled=hybrid_llm_assist_enabled,
            urban_method=urban_method,
        )

    # 4) Stable strategy layer: centralizes evidence -> decision explanation.
    # By default it appends strategy_* fields only; explicit context may allow
    # it to rewrite final_label / urban_flag / topic_final.
    if _stable_strategy_enabled(context):
        enriched = _apply_stable_strategy_layer(enriched, context)

    return enriched


def _dynamic_topics_enabled(context: dict[str, Any]) -> bool:
    """Return whether dynamic topic evidence should be appended."""

    return bool(context.get("dynamic_topics_enabled", False))


def _dynamic_binary_refinement_enabled(context: dict[str, Any]) -> bool:
    """Return whether dynamic binary refinement is enabled."""

    return bool(context.get("dynamic_binary_refinement_enabled", False))


def _binary_policy_v2_enabled(context: dict[str, Any]) -> bool:
    """Return whether the final binary policy layer should run."""

    return bool(context.get("urban_binary_policy_v2_enabled", True))


def _stable_strategy_enabled(context: dict[str, Any]) -> bool:
    """Return whether the stable strategy decision should be appended."""

    raw = context.get("urban_stable_strategy_enabled", None)
    if raw is not None:
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    raw = context.get("urban_strategy_v3_shadow_enabled", None)
    if raw is not None:
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    version = str(context.get("urban_strategy_version", "stable") or "").strip().lower()
    return version not in {"0", "false", "off", "disabled", "none", "legacy"}


def _stable_strategy_mutates_final_fields(context: dict[str, Any]) -> bool:
    raw = context.get("urban_stable_strategy_mutate_final_fields", False)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _append_dynamic_topic_evidence(frame: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    """Append dynamic-topic evidence without changing final labels."""

    try:
        prefer_sklearn = not bool(context.get("dynamic_topics_keyword_fallback_only", False))
        config = DynamicTopicConfig(
            min_topic_size=int(context.get("dynamic_topics_min_topic_size", 20) or 20),
            max_topics=int(context.get("dynamic_topics_max_topics", 60) or 60),
            mapping_min_score=float(context.get("dynamic_topics_mapping_min_score", 0.12) or 0.12),
            include_full_corpus=bool(context.get("dynamic_topics_include_full_corpus", False)),
            prefer_sklearn=prefer_sklearn,
        )
        discovery = DynamicTopicDiscovery(config)
        return discovery.enrich(
            frame,
            include_full_corpus=bool(context.get("dynamic_topics_include_full_corpus", False)),
        )
    except Exception as exc:
        print(f"[WARN] Dynamic topic enrichment failed, continuing without it: {type(exc).__name__}: {exc}")
        return frame


def _apply_dynamic_binary_refinement(frame: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    """Apply the explicitly enabled layer that may mutate final binary fields."""

    try:
        refiner = DynamicBinaryRefiner(DynamicBinaryRefinementConfig.from_context(context))
        return refiner.refine(frame, mutate_final_fields=True)
    except Exception as exc:
        print(f"[WARN] Dynamic binary refinement failed, continuing without it: {type(exc).__name__}: {exc}")
        return frame


def _apply_binary_policy_v2(
    frame: pd.DataFrame,
    *,
    context: dict[str, Any],
    llm_client: DeepSeekClient | None,
    hybrid_llm_assist_enabled: bool,
    urban_method: Any,
) -> pd.DataFrame:
    """Apply the legacy binary reconciliation policy."""

    try:
        method_value = getattr(urban_method, "value", str(urban_method or ""))
        llm_adjudication_enabled = (
            str(context.get("experiment_track") or "").strip() == "research_matrix"
            and bool(context.get("hybrid_llm_assist_enabled", hybrid_llm_assist_enabled))
            and method_value == "three_stage_hybrid"
        )
        policy = UrbanBinaryPolicyV2(
            llm_client=llm_client,
            llm_enabled=llm_adjudication_enabled,
        )
        return policy.apply(frame)
    except Exception as exc:
        print(f"[WARN] Urban binary policy failed, continuing without it: {type(exc).__name__}: {exc}")
        return frame


def _apply_stable_strategy_layer(frame: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    """Append or apply the stable evidence strategy."""

    try:
        return apply_stable_strategy(
            frame,
            mutate_final_fields=_stable_strategy_mutates_final_fields(context),
        )
    except Exception as exc:
        print(f"[WARN] Urban stable strategy failed, continuing without it: {type(exc).__name__}: {exc}")
        return frame
