import json
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.prompt_manifest import (
    build_comparability_signature,
    build_long_context_group_signature,
    build_run_prompt_manifest,
    build_strategy_snapshot,
    compare_manifests,
    ensure_strategy_runnable,
    load_prompt_manifest,
    write_prompt_manifest,
)
from src.prompt_strategy_registry import PromptStrategyRegistry


def make_entry(
    key,
    name,
    theme,
    template_file,
    *,
    enabled=True,
    aliases=None,
    description="",
    version="1.0.0",
    lifecycle="stable",
    owner="tester",
    change_summary="initial",
):
    return {
        "key": key,
        "name": name,
        "theme": theme,
        "template_file": template_file,
        "enabled": enabled,
        "aliases": aliases or [],
        "description": description,
        "version": version,
        "lifecycle": lifecycle,
        "owner": owner,
        "change_summary": change_summary,
    }


def build_registry_workspace(tmp_path: Path):
    template_root = tmp_path / "templates"
    (template_root / "urban_renewal").mkdir(parents=True, exist_ok=True)
    (template_root / "spatial").mkdir(parents=True, exist_ok=True)
    (template_root / "urban_renewal" / "few.yaml").write_text("system_prompt: urban few\n", encoding="utf-8")
    (template_root / "spatial" / "zero.yaml").write_text("system_prompt: spatial zero\n", encoding="utf-8")
    registry_raw = {
        "themes": ["urban_renewal", "spatial"],
        "strategies": [
            make_entry("urban_renewal.few", "few", "urban_renewal", "few.yaml", version="2.0.0"),
            make_entry("spatial.zero", "zero", "spatial", "zero.yaml"),
        ],
    }
    registry_path = template_root / "strategy_registry.yaml"
    registry_path.write_text(yaml.safe_dump(registry_raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return template_root, registry_path, PromptStrategyRegistry.load_from_file(registry_path)


def test_manifest_roundtrip_and_comparability(tmp_path: Path):
    template_root, registry_path, registry = build_registry_workspace(tmp_path)
    snapshot = build_strategy_snapshot(registry, template_root, "urban_renewal", "few")
    manifest = build_run_prompt_manifest(
        task_mode="urban_renewal",
        active_tasks=["urban_renewal"],
        input_file=tmp_path / "input.xlsx",
        registry_path=registry_path,
        strategy_snapshots={"urban_renewal": snapshot},
    )
    output_path = tmp_path / "urban_renewal_few.xlsx"
    output_path.write_text("placeholder", encoding="utf-8")
    manifest_path = write_prompt_manifest(output_path, manifest)
    assert manifest_path.exists()

    loaded = load_prompt_manifest(output_path)
    assert loaded is not None
    assert loaded["manifest_version"] == "2.0"
    assert loaded["strategies"]["urban_renewal"]["version"] == "2.0.0"
    assert loaded["experiment"]["experiment_track"] == "stable_release"
    assert loaded["runtime"]["python_version"]

    signature_a = build_comparability_signature(loaded, tmp_path / "truth.xlsx")
    signature_b = build_comparability_signature(loaded, tmp_path / "truth.xlsx")
    assert signature_a == signature_b


def test_compare_manifests_detects_strategy_differences(tmp_path: Path):
    template_root, registry_path, registry = build_registry_workspace(tmp_path)
    snapshot = build_strategy_snapshot(registry, template_root, "urban_renewal", "few")
    manifest_a = build_run_prompt_manifest(
        task_mode="urban_renewal",
        active_tasks=["urban_renewal"],
        input_file=tmp_path / "input_a.xlsx",
        registry_path=registry_path,
        strategy_snapshots={"urban_renewal": snapshot},
    )
    manifest_b = build_run_prompt_manifest(
        task_mode="urban_renewal",
        active_tasks=["urban_renewal"],
        input_file=tmp_path / "input_b.xlsx",
        registry_path=registry_path,
        strategy_snapshots={"urban_renewal": snapshot},
    )
    manifest_b_dict = json.loads(json.dumps(manifest_b, default=lambda value: value.__dict__))
    manifest_a_dict = json.loads(json.dumps(manifest_a, default=lambda value: value.__dict__))
    manifest_b_dict["strategies"]["urban_renewal"]["version"] = "3.0.0"
    mismatches = compare_manifests(
        manifest_a_dict,
        manifest_b_dict,
        tmp_path / "truth.xlsx",
        tmp_path / "truth.xlsx",
    )
    assert "strategies" in mismatches


def test_long_context_group_signature_ignores_order_fields(tmp_path: Path):
    template_root, registry_path, registry = build_registry_workspace(tmp_path)
    snapshot = build_strategy_snapshot(registry, template_root, "urban_renewal", "few")
    common_kwargs = {
        "task_mode": "urban_renewal",
        "active_tasks": ["urban_renewal"],
        "input_file": tmp_path / "input.xlsx",
        "registry_path": registry_path,
        "strategy_snapshots": {"urban_renewal": snapshot},
    }
    manifest_a = build_run_prompt_manifest(
        **common_kwargs,
        experiment_context={
            "experiment_track": "research_matrix",
            "dataset_id": "demo",
            "session_policy": "cross_paper_long_context",
            "order_id": "canonical_title_order",
            "order_seed": None,
            "max_samples_per_window": 50,
            "pred_scope": "urban_renewal",
            "urban_method": "pure_llm_api",
            "hybrid_llm_assist_enabled": False,
        },
    )
    manifest_b = build_run_prompt_manifest(
        **common_kwargs,
        experiment_context={
            "experiment_track": "research_matrix",
            "dataset_id": "demo",
            "session_policy": "cross_paper_long_context",
            "order_id": "shuffle_seed_20260415_a",
            "order_seed": 20260415,
            "max_samples_per_window": 50,
            "pred_scope": "urban_renewal",
            "urban_method": "pure_llm_api",
            "hybrid_llm_assist_enabled": False,
        },
    )
    manifest_a_dict = json.loads(json.dumps(manifest_a, default=lambda value: value.__dict__))
    manifest_b_dict = json.loads(json.dumps(manifest_b, default=lambda value: value.__dict__))
    assert (
        build_comparability_signature(manifest_a_dict, tmp_path / "truth.xlsx")
        != build_comparability_signature(manifest_b_dict, tmp_path / "truth.xlsx")
    )
    assert (
        build_long_context_group_signature(manifest_a_dict, tmp_path / "truth.xlsx")
        == build_long_context_group_signature(manifest_b_dict, tmp_path / "truth.xlsx")
    )


def test_candidate_strategy_requires_explicit_opt_in():
    entry = make_entry(
        "urban_renewal.few",
        "few",
        "urban_renewal",
        "few.yaml",
        lifecycle="candidate",
    )
    definition = PromptStrategyRegistry._from_dict(
        {"themes": ["urban_renewal"], "strategies": [entry]}
    ).get_definition("urban_renewal", "few")
    try:
        ensure_strategy_runnable(definition, allow_candidate=False)
        assert False, "expected candidate prompt rejection"
    except ValueError as exc:
        assert "candidate opt-in" in str(exc)
    ensure_strategy_runnable(definition, allow_candidate=True)
