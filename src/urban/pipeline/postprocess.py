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

    context = run_context if run_context is not None else {}
    enriched = frame

    # 1) Evidence-only layer: dynamic_topic_* / dynamic_binary_candidate_*.
    if _dynamic_topics_enabled(context) or _dynamic_binary_refinement_enabled(context):
        enriched = _append_dynamic_topic_evidence(enriched, context)
    else:
        _record_postprocess_layer(context, "dynamic_topic_evidence", "skipped")

    # 2) Explicit label-mutation layer: may rewrite final_label / urban_flag.
    if _dynamic_binary_refinement_enabled(context):
        enriched = _apply_dynamic_binary_refinement(enriched, context)
    else:
        _record_postprocess_layer(context, "dynamic_binary_refinement", "skipped")

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
    else:
        _record_postprocess_layer(context, "binary_policy_v2", "skipped")

    # 4) Stable strategy layer: centralizes evidence -> decision explanation.
    # By default it appends strategy_* fields only; explicit context may allow
    # it to rewrite final_label / urban_flag / topic_final.
    if _stable_strategy_enabled(context):
        enriched = _apply_stable_strategy_layer(enriched, context)
    else:
        _record_postprocess_layer(context, "stable_strategy", "skipped")

    return enriched


def _context_flag(context: dict[str, Any], key: str, default: bool) -> bool:
    raw = context.get(key, default)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _postprocess_strict(context: dict[str, Any]) -> bool:
    if "urban_postprocess_strict" in context:
        return _context_flag(context, "urban_postprocess_strict", False)
    return str(context.get("experiment_track") or "").strip() == "stable_release"


def _record_postprocess_layer(
    context: dict[str, Any],
    layer: str,
    status: str,
    *,
    error: str = "",
) -> None:
    layers = context.setdefault("urban_postprocess_layers", [])
    if isinstance(layers, list):
        item = {"layer": layer, "status": status}
        if error:
            item["error"] = error
        layers.append(item)


def _handle_postprocess_failure(
    context: dict[str, Any],
    layer: str,
    message: str,
    exc: Exception,
) -> None:
    error = f"{type(exc).__name__}: {exc}"
    _record_postprocess_layer(context, layer, "failed", error=error)
    if _postprocess_strict(context):
        raise RuntimeError(f"{message}: {error}") from exc
    print(f"[WARN] {message}, continuing without it: {error}")


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
        enriched = discovery.enrich(
            frame,
            include_full_corpus=bool(context.get("dynamic_topics_include_full_corpus", False)),
        )
        _record_postprocess_layer(context, "dynamic_topic_evidence", "applied")
        return enriched
    except Exception as exc:
        _handle_postprocess_failure(context, "dynamic_topic_evidence", "Dynamic topic enrichment failed", exc)
        return frame


def _apply_dynamic_binary_refinement(frame: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    """Apply the explicitly enabled layer that may mutate final binary fields."""

    try:
        refiner = DynamicBinaryRefiner(DynamicBinaryRefinementConfig.from_context(context))
        enriched = refiner.refine(frame, mutate_final_fields=True)
        _record_postprocess_layer(context, "dynamic_binary_refinement", "applied")
        return enriched
    except Exception as exc:
        _handle_postprocess_failure(
            context,
            "dynamic_binary_refinement",
            "Dynamic binary refinement failed",
            exc,
        )
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
        enriched = policy.apply(frame)
        _record_postprocess_layer(context, "binary_policy_v2", "applied")
        return enriched
    except Exception as exc:
        _handle_postprocess_failure(context, "binary_policy_v2", "Urban binary policy failed", exc)
        return frame


def _apply_stable_strategy_layer(frame: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    """Append or apply the stable evidence strategy."""

    try:
        enriched = apply_stable_strategy(
            frame,
            mutate_final_fields=_stable_strategy_mutates_final_fields(context),
        )
        _record_postprocess_layer(context, "stable_strategy", "applied")
        return enriched
    except Exception as exc:
        _handle_postprocess_failure(context, "stable_strategy", "Urban stable strategy failed", exc)
        return frame
