from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .strategy_registry import PromptStrategyDefinition, PromptStrategyRegistry
from ..runtime.config import Config


MANIFEST_VERSION = "2.0"


@dataclass(frozen=True)
class StrategySnapshot:
    key: str
    name: str
    theme: str
    template_file: str
    template_path: str
    template_sha256: str
    enabled: bool
    version: str
    lifecycle: str
    owner: str
    change_summary: str


@dataclass(frozen=True)
class RunPromptManifest:
    manifest_version: str
    generated_at: str
    task_mode: str
    active_tasks: list[str]
    input_file: str
    runtime: Dict[str, Any]
    experiment: Dict[str, Any]
    registry: Dict[str, Any]
    strategies: Dict[str, Dict[str, Any]]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_text(path.read_text(encoding="utf-8"))


def manifest_path_for_output(output_path: Path) -> Path:
    output_path = Path(output_path)
    return output_path.with_name(f"{output_path.stem}.prompt_manifest.json")


def build_strategy_snapshot(
    registry: PromptStrategyRegistry,
    template_root: Path,
    theme: str,
    strategy_or_alias: str,
) -> StrategySnapshot:
    definition = registry.get_definition(theme, strategy_or_alias)
    if not definition:
        raise ValueError(f"Strategy not found: theme={theme}, strategy={strategy_or_alias}")

    template_path = Path(template_root) / definition.theme / definition.template_file
    if not template_path.exists():
        raise FileNotFoundError(f"Missing template file: {template_path}")

    return StrategySnapshot(
        key=definition.key,
        name=definition.name,
        theme=definition.theme,
        template_file=definition.template_file,
        template_path=str(template_path.resolve()),
        template_sha256=sha256_file(template_path),
        enabled=definition.enabled,
        version=definition.version,
        lifecycle=definition.lifecycle,
        owner=definition.owner,
        change_summary=definition.change_summary,
    )


def build_run_prompt_manifest(
    task_mode: str,
    active_tasks: Iterable[str],
    input_file: Path,
    registry_path: Path,
    strategy_snapshots: Dict[str, StrategySnapshot],
    experiment_context: Optional[Dict[str, Any]] = None,
    runtime_context: Optional[Dict[str, Any]] = None,
) -> RunPromptManifest:
    registry_path = Path(registry_path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing strategy registry: {registry_path}")

    runtime_context = dict(runtime_context or {})
    runtime_context.setdefault("python_version", f"{sys.version_info[0]}.{sys.version_info[1]}")
    runtime_context.setdefault("python_executable", sys.executable)
    runtime_context.setdefault("entrypoint", "")

    experiment_context = dict(experiment_context or {})
    experiment_context.setdefault("experiment_track", "stable_release")
    experiment_context.setdefault("dataset_id", Path(input_file).stem)
    experiment_context.setdefault("truth_file", "")
    experiment_context.setdefault("session_policy", "per_paper_isolated")
    experiment_context.setdefault("order_id", "canonical_title_order")
    experiment_context.setdefault("order_seed", None)
    experiment_context.setdefault("max_samples_per_window", None)
    experiment_context.setdefault("pred_scope", task_mode)
    experiment_context.setdefault("urban_method", "")
    experiment_context.setdefault("hybrid_llm_assist_enabled", None)

    return RunPromptManifest(
        manifest_version=MANIFEST_VERSION,
        generated_at=utc_now_iso(),
        task_mode=task_mode,
        active_tasks=list(active_tasks),
        input_file=str(Path(input_file).resolve()),
        runtime={
            "model_name": Config.MODEL_NAME,
            "base_url": Config.BASE_URL,
            "temperature": Config.TEMPERATURE,
            "max_tokens": Config.MAX_TOKENS,
            "timeout": Config.TIMEOUT,
            "python_version": runtime_context["python_version"],
            "python_executable": runtime_context["python_executable"],
            "entrypoint": runtime_context["entrypoint"],
        },
        experiment=experiment_context,
        registry={
            "path": str(registry_path.resolve()),
            "sha256": sha256_file(registry_path),
        },
        strategies={task_name: asdict(snapshot) for task_name, snapshot in strategy_snapshots.items()},
    )


def write_prompt_manifest(output_path: Path, manifest: RunPromptManifest) -> Path:
    path = manifest_path_for_output(Path(output_path))
    path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def load_prompt_manifest(prediction_path: Path) -> Optional[Dict[str, Any]]:
    path = manifest_path_for_output(Path(prediction_path))
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def strategy_payload_for_comparison(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        task_name: {
            "key": payload.get("key"),
            "version": payload.get("version"),
            "template_sha256": payload.get("template_sha256"),
            "lifecycle": payload.get("lifecycle"),
        }
        for task_name, payload in sorted((manifest.get("strategies") or {}).items())
    }


def build_comparability_payload(
    manifest: Optional[Dict[str, Any]],
    truth_file: Path,
) -> Dict[str, Any]:
    if manifest is None:
        return {
            "truth_file": str(Path(truth_file).resolve()),
            "missing_manifest": True,
        }

    runtime = manifest.get("runtime") or {}
    experiment = manifest.get("experiment") or {}
    return {
        "truth_file": str(Path(truth_file).resolve()),
        "input_file": manifest.get("input_file"),
        "model_name": runtime.get("model_name"),
        "base_url": runtime.get("base_url"),
        "temperature": runtime.get("temperature"),
        "max_tokens": runtime.get("max_tokens"),
        "python_version": runtime.get("python_version"),
        "entrypoint": runtime.get("entrypoint"),
        "task_mode": manifest.get("task_mode"),
        "experiment_track": experiment.get("experiment_track"),
        "dataset_id": experiment.get("dataset_id"),
        "session_policy": experiment.get("session_policy"),
        "order_id": experiment.get("order_id"),
        "order_seed": experiment.get("order_seed"),
        "max_samples_per_window": experiment.get("max_samples_per_window"),
        "pred_scope": experiment.get("pred_scope"),
        "urban_method": experiment.get("urban_method"),
        "hybrid_llm_assist_enabled": experiment.get("hybrid_llm_assist_enabled"),
        "registry_sha256": (manifest.get("registry") or {}).get("sha256"),
        "strategies": strategy_payload_for_comparison(manifest),
    }


def build_comparability_signature(
    manifest: Optional[Dict[str, Any]],
    truth_file: Path,
) -> str:
    payload = build_comparability_payload(manifest, truth_file)
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def build_long_context_group_payload(
    manifest: Optional[Dict[str, Any]],
    truth_file: Path,
) -> Dict[str, Any]:
    payload = build_comparability_payload(manifest, truth_file)
    payload.pop("order_id", None)
    payload.pop("order_seed", None)
    return payload


def build_long_context_group_signature(
    manifest: Optional[Dict[str, Any]],
    truth_file: Path,
) -> str:
    payload = build_long_context_group_payload(manifest, truth_file)
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def compare_manifests(
    baseline_manifest: Optional[Dict[str, Any]],
    candidate_manifest: Optional[Dict[str, Any]],
    baseline_truth: Path,
    candidate_truth: Path,
) -> list[str]:
    baseline_payload = build_comparability_payload(baseline_manifest, baseline_truth)
    candidate_payload = build_comparability_payload(candidate_manifest, candidate_truth)
    mismatches = []
    for key in sorted(set(baseline_payload.keys()) | set(candidate_payload.keys())):
        if baseline_payload.get(key) != candidate_payload.get(key):
            mismatches.append(key)
    return mismatches


def ensure_strategy_runnable(
    definition: PromptStrategyDefinition,
    allow_candidate: bool = False,
):
    if not definition.enabled:
        raise ValueError(f"Strategy is not enabled: {definition.name}")
    if definition.lifecycle == "deprecated":
        raise ValueError(f"Strategy is deprecated and cannot be used: {definition.name}")
    if definition.lifecycle == "candidate" and not allow_candidate:
        raise ValueError(
            f"Strategy requires explicit candidate opt-in: {definition.name}. "
            f"Run with --allow-candidate for experimental prompt versions."
        )
