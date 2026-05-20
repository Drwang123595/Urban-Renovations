"""Batch post-processing orchestration for urban-renewal predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
from ...runtime.llm_client import DeepSeekClient, LLMQuotaExceededError
from ...runtime.resume import ResumeCheckpoint
from ..dynamic.binary_refinement import DynamicBinaryRefinementConfig, DynamicBinaryRefiner
from ..dynamic.topic_discovery import DynamicTopicConfig, DynamicTopicDiscovery
from ..hybrid.binary_finalizer import LlmBinaryFinalizer, workflow_enabled as llm_binary_v2_enabled
from ..hybrid.binary_policy_v2 import UrbanBinaryPolicyV2
from ..strategy import apply_stable_strategy, build_llm_semantic_analyzer


@dataclass(frozen=True)
class PostprocessLayer:
    name: str
    enabled: Callable[[dict[str, Any]], bool]
    apply: Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]


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
    refinement records candidate evidence but does not directly mutate final
    labels. The binary policy appends conflict/risk evidence. The stable
    strategy is the final decision layer for topic_final/final_label.
    """

    context = run_context if run_context is not None else {}
    enriched = frame
    for layer in _postprocess_layers(
        llm_client=llm_client,
        hybrid_llm_assist_enabled=hybrid_llm_assist_enabled,
        urban_method=urban_method,
    ):
        if layer.enabled(context):
            enriched = layer.apply(enriched, context)
        else:
            _record_postprocess_layer(context, layer.name, "skipped")

    return enriched


def _postprocess_layers(
    *,
    llm_client: DeepSeekClient | None,
    hybrid_llm_assist_enabled: bool,
    urban_method: Any,
) -> list[PostprocessLayer]:
    return [
        PostprocessLayer(
            "dynamic_topic_evidence",
            lambda context: _dynamic_topics_enabled(context) or _dynamic_binary_refinement_enabled(context),
            _append_dynamic_topic_evidence,
        ),
        PostprocessLayer(
            "dynamic_binary_refinement",
            _dynamic_binary_refinement_enabled,
            _apply_dynamic_binary_refinement,
        ),
        PostprocessLayer(
            "binary_policy_v2",
            _binary_policy_v2_enabled,
            lambda frame, context: _apply_binary_policy_v2_evidence_only(
                frame,
                context=context,
                llm_client=llm_client,
                hybrid_llm_assist_enabled=hybrid_llm_assist_enabled,
                urban_method=urban_method,
            ),
        ),
        PostprocessLayer(
            "llm_binary_v2",
            llm_binary_v2_enabled,
            lambda frame, context: _apply_llm_binary_v2_finalizer(
                frame,
                context=context,
                llm_client=llm_client,
                hybrid_llm_assist_enabled=hybrid_llm_assist_enabled,
                urban_method=urban_method,
            ),
        ),
        PostprocessLayer(
            "stable_strategy",
            _stable_strategy_enabled,
            _apply_stable_strategy_layer,
        ),
    ]


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
    if isinstance(exc, LLMQuotaExceededError):
        raise
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
    raw = context.get("urban_stable_strategy_mutate_final_fields", True)
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
        enriched = refiner.refine(frame, mutate_final_fields=False)
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
            and not llm_binary_v2_enabled(context)
        )
        policy = UrbanBinaryPolicyV2(
            llm_client=llm_client,
            llm_enabled=llm_adjudication_enabled,
            evidence_only=bool(context.get("urban_binary_policy_v2_evidence_only", False)),
        )
        enriched = policy.apply(frame)
        _record_postprocess_layer(context, "binary_policy_v2", "applied")
        return enriched
    except Exception as exc:
        _handle_postprocess_failure(context, "binary_policy_v2", "Urban binary policy failed", exc)
        return frame


def _apply_binary_policy_v2_evidence_only(
    frame: pd.DataFrame,
    *,
    context: dict[str, Any],
    llm_client: DeepSeekClient | None,
    hybrid_llm_assist_enabled: bool,
    urban_method: Any,
) -> pd.DataFrame:
    original_evidence_only = context.get("urban_binary_policy_v2_evidence_only")
    context["urban_binary_policy_v2_evidence_only"] = True
    try:
        return _apply_binary_policy_v2(
            frame,
            context=context,
            llm_client=llm_client,
            hybrid_llm_assist_enabled=False,
            urban_method=urban_method,
        )
    finally:
        if original_evidence_only is None:
            context.pop("urban_binary_policy_v2_evidence_only", None)
        else:
            context["urban_binary_policy_v2_evidence_only"] = original_evidence_only


def _apply_llm_binary_v2_finalizer(
    frame: pd.DataFrame,
    *,
    context: dict[str, Any],
    llm_client: DeepSeekClient | None,
    hybrid_llm_assist_enabled: bool,
    urban_method: Any,
) -> pd.DataFrame:
    """Apply the experimental binary-first selective LLM workflow."""

    try:
        method_value = getattr(urban_method, "value", str(urban_method or ""))
        llm_enabled = bool(context.get("hybrid_llm_assist_enabled", hybrid_llm_assist_enabled))
        if method_value and method_value != "three_stage_hybrid":
            llm_enabled = False
        finalizer = LlmBinaryFinalizer(llm_client=llm_client, llm_enabled=llm_enabled)
        enriched = finalizer.apply(
            frame,
            **_resume_kwargs_for_layer(context, "llm_binary_v2"),
        )
        _record_postprocess_layer(context, "llm_binary_v2", "applied")
        return enriched
    except Exception as exc:
        _handle_postprocess_failure(context, "llm_binary_v2", "LLM binary v2 finalization failed", exc)
        return frame


def _apply_stable_strategy_layer(frame: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    """Append or apply the stable evidence strategy."""

    try:
        enriched = apply_stable_strategy(
            frame,
            mutate_final_fields=_stable_strategy_mutates_final_fields(context),
            llm_analyzer=_stable_strategy_llm_analyzer(context),
            **_resume_kwargs_for_layer(context, "stable_strategy"),
        )
        _record_postprocess_layer(context, "stable_strategy", "applied")
        return enriched
    except Exception as exc:
        _handle_postprocess_failure(context, "stable_strategy", "Urban stable strategy failed", exc)
        return frame


def _stable_strategy_llm_analyzer(context: dict[str, Any]):
    strategy = context.get("urban_stable_strategy_llm_strategy")
    enabled = _context_flag(context, "urban_stable_strategy_llm_enabled", bool(strategy))
    if strategy is None or not enabled:
        return None
    return build_llm_semantic_analyzer(strategy, enabled=enabled)


def _resume_kwargs_for_layer(context: dict[str, Any], layer: str) -> dict[str, Any]:
    raw_path = context.get("resume_checkpoint")
    if not raw_path:
        return {}
    checkpoint = ResumeCheckpoint(raw_path)
    task_type = str(context.get("resume_task_type") or "urban_renewal")
    return {
        "resume_checkpoint": checkpoint,
        "resume_task_type": f"{task_type}:{layer}",
        "resume_run_id": str(context.get("resume_run_id") or ""),
        "resume_input_fingerprint": str(context.get("resume_input_fingerprint") or ""),
    }
