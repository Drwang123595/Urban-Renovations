import sys
import subprocess
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

import scripts.main as main_module
from scripts.main import (
    build_argument_parser,
    build_run_context,
    choose_task_mode,
    get_enabled_shot_modes,
    normalize_hybrid_llm_assist,
    normalize_shot_mode,
    normalize_urban_method,
    prepare_experiment_args,
    resolve_hybrid_llm_assist,
    select_hybrid_llm_assist,
    select_output_mode,
    select_shot_mode,
    select_task_mode,
    select_urban_method,
    task_requires_api_key,
    urban_method_requires_prompt,
)
from src.runtime.config import Config
from src.prompting.strategy_registry import PromptStrategyRegistry
from src.tasks.router import TaskType, UrbanMethod


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


def _build_registry():
    raw = {
        "themes": ["urban_renewal", "spatial"],
        "strategies": [
            make_entry("urban_renewal.zero", "zero", "urban_renewal", "zero.yaml"),
            make_entry(
                "urban_renewal.cot",
                "cot",
                "urban_renewal",
                "cot.yaml",
                aliases=["chain_of_thought"],
            ),
            make_entry(
                "urban_renewal.few",
                "few",
                "urban_renewal",
                "few.yaml",
                lifecycle="candidate",
                aliases=["fewshot"],
            ),
            make_entry(
                "urban_renewal.reflection",
                "reflection",
                "urban_renewal",
                "reflection.yaml",
                enabled=False,
                lifecycle="deprecated",
            ),
            make_entry("spatial.zero", "zero", "spatial", "zero.yaml"),
        ],
    }
    return PromptStrategyRegistry._from_dict(raw)


def test_normalize_shot_mode_supports_alias():
    registry = _build_registry()
    enabled_modes = registry.list_enabled_strategies(theme="urban_renewal", allow_candidate=False)
    assert normalize_shot_mode(
        "chain_of_thought",
        registry,
        enabled_modes,
        theme="urban_renewal",
        allow_candidate=False,
    ) == "cot"


def test_normalize_shot_mode_rejects_candidate_without_flag():
    registry = _build_registry()
    enabled_modes = registry.list_enabled_strategies(theme="urban_renewal", allow_candidate=False)
    with pytest.raises(ValueError, match="candidate opt-in"):
        normalize_shot_mode(
            "few",
            registry,
            enabled_modes,
            theme="urban_renewal",
            allow_candidate=False,
        )


def test_normalize_shot_mode_accepts_candidate_with_flag():
    registry = _build_registry()
    enabled_modes = registry.list_enabled_strategies(theme="urban_renewal", allow_candidate=True)
    assert normalize_shot_mode(
        "few",
        registry,
        enabled_modes,
        theme="urban_renewal",
        allow_candidate=True,
    ) == "few"


def test_normalize_shot_mode_rejects_invalid_strategy():
    registry = _build_registry()
    enabled_modes = registry.list_enabled_strategies(theme="urban_renewal", allow_candidate=False)
    with pytest.raises(ValueError, match="Invalid strategy"):
        normalize_shot_mode(
            "unknown",
            registry,
            enabled_modes,
            theme="urban_renewal",
            allow_candidate=False,
        )


def test_select_shot_mode_uses_default_on_enter(monkeypatch):
    registry = _build_registry()
    enabled_modes = registry.list_enabled_strategies(theme="urban_renewal", allow_candidate=False)
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert (
        select_shot_mode(
            registry,
            enabled_modes,
            default_mode="zero",
            theme="urban_renewal",
            label="urban_renewal",
            allow_candidate=False,
        )
        == "zero"
    )


def test_select_shot_mode_accepts_alias_input(monkeypatch):
    registry = _build_registry()
    enabled_modes = registry.list_enabled_strategies(theme="urban_renewal", allow_candidate=False)
    monkeypatch.setattr("builtins.input", lambda _: "chain_of_thought")
    assert (
        select_shot_mode(
            registry,
            enabled_modes,
            default_mode="zero",
            theme="urban_renewal",
            label="urban_renewal",
            allow_candidate=False,
        )
        == "cot"
    )


def test_get_enabled_shot_modes_is_registry_driven():
    registry = _build_registry()
    assert get_enabled_shot_modes(registry, theme="urban_renewal", allow_candidate=False) == ["zero", "cot"]
    assert get_enabled_shot_modes(registry, theme="urban_renewal", allow_candidate=True) == ["zero", "cot", "few"]
    assert get_enabled_shot_modes(registry, theme="spatial", allow_candidate=False) == ["zero"]


def test_select_task_mode_uses_default_on_enter(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert select_task_mode(default_task=TaskType.SPATIAL) == TaskType.SPATIAL


def test_normalize_urban_method_supports_alias():
    assert normalize_urban_method("classifier") == UrbanMethod.LOCAL_TOPIC_CLASSIFIER
    assert normalize_urban_method("hybrid") == UrbanMethod.THREE_STAGE_HYBRID


def test_normalize_hybrid_llm_assist_supports_on_off_aliases():
    assert normalize_hybrid_llm_assist("on") is True
    assert normalize_hybrid_llm_assist("off") is False
    assert normalize_hybrid_llm_assist("1") is True
    assert normalize_hybrid_llm_assist("0") is False


def test_resolve_hybrid_llm_assist_prefers_cli_over_config(monkeypatch):
    monkeypatch.setattr(Config, "URBAN_HYBRID_LLM_ASSIST_ENABLED", True)
    assert resolve_hybrid_llm_assist("off") is False
    monkeypatch.setattr(Config, "URBAN_HYBRID_LLM_ASSIST_ENABLED", False)
    assert resolve_hybrid_llm_assist("on") is True


def test_resolve_hybrid_llm_assist_uses_config_default(monkeypatch):
    monkeypatch.setattr(Config, "URBAN_HYBRID_LLM_ASSIST_ENABLED", True)
    assert resolve_hybrid_llm_assist() is True
    monkeypatch.setattr(Config, "URBAN_HYBRID_LLM_ASSIST_ENABLED", False)
    assert resolve_hybrid_llm_assist() is False


def test_task_requires_api_key_respects_hybrid_toggle():
    assert (
        task_requires_api_key(
            TaskType.URBAN_RENEWAL,
            UrbanMethod.THREE_STAGE_HYBRID,
            hybrid_llm_assist_enabled=False,
        )
        is False
    )
    assert (
        task_requires_api_key(
            TaskType.URBAN_RENEWAL,
            UrbanMethod.THREE_STAGE_HYBRID,
            hybrid_llm_assist_enabled=True,
        )
        is True
    )
    assert (
        task_requires_api_key(
            TaskType.URBAN_RENEWAL,
            UrbanMethod.LOCAL_TOPIC_CLASSIFIER,
            hybrid_llm_assist_enabled=True,
        )
        is False
    )
    assert (
        task_requires_api_key(
            TaskType.SPATIAL,
            UrbanMethod.LOCAL_TOPIC_CLASSIFIER,
            hybrid_llm_assist_enabled=False,
        )
        is True
    )


def test_urban_method_requires_prompt_skips_hybrid_prompt_when_llm_assist_off():
    assert (
        urban_method_requires_prompt(
            UrbanMethod.THREE_STAGE_HYBRID,
            hybrid_llm_assist_enabled=False,
        )
        is False
    )
    assert (
        urban_method_requires_prompt(
            UrbanMethod.THREE_STAGE_HYBRID,
            hybrid_llm_assist_enabled=True,
        )
        is True
    )


def test_build_run_context_records_flow_audit_toggle():
    context = build_run_context(
        experiment_track="research_matrix",
        dataset_id="demo_dataset",
        truth_file="truth.xlsx",
        task_mode=TaskType.URBAN_RENEWAL,
        urban_method=UrbanMethod.THREE_STAGE_HYBRID,
        hybrid_llm_assist_enabled=True,
        session_policy="per_paper_isolated",
        order_id="canonical_title_order",
        order_seed=None,
        max_samples_per_window=50,
        dynamic_topics_enabled=False,
        dynamic_topics_include_full_corpus=False,
        dynamic_binary_refinement_enabled=False,
        dynamic_binary_refinement_unknown_only=True,
        dynamic_binary_refinement_allow_flip=False,
        urban_flow_audit_enabled=False,
    )

    assert context["urban_flow_audit_enabled"] is False


def test_argument_parser_accepts_flow_audit_toggle():
    args = build_argument_parser().parse_args(["--urban-flow-audit", "off"])

    assert args.urban_flow_audit == "off"


def test_select_urban_method_uses_default_on_enter(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert select_urban_method(default_method=UrbanMethod.PURE_LLM_API) == UrbanMethod.PURE_LLM_API


def test_select_hybrid_llm_assist_uses_default_on_enter(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    assert select_hybrid_llm_assist(default_enabled=False) is False


def test_select_output_mode_custom_path(monkeypatch):
    answers = iter(["2", "custom_out.xlsx"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    assert select_output_mode(task=TaskType.BOTH, preset_output=None) == "custom_out.xlsx"


def test_prepare_experiment_args_prompts_for_default_research_track(monkeypatch):
    args = build_argument_parser().parse_args([])
    called_defaults = []
    monkeypatch.setattr(main_module, "default_train_input", lambda: str(Config.TRAIN_DIR / "sample.xlsx"))
    monkeypatch.setattr(main_module, "is_train_scoped_input", lambda _: True)
    monkeypatch.setattr(
        main_module,
        "select_experiment_track",
        lambda default_track: called_defaults.append(default_track) or "stable_release",
    )

    prepare_experiment_args(args)

    assert called_defaults == ["research_matrix"]
    assert args.experiment_track == "stable_release"


def test_prepare_experiment_args_respects_explicit_track(monkeypatch):
    args = build_argument_parser().parse_args(["--experiment-track", "legacy_archive"])
    monkeypatch.setattr(main_module, "select_experiment_track", lambda _: pytest.fail("should not prompt"))

    prepare_experiment_args(args)

    assert args.experiment_track == "legacy_archive"


def test_choose_task_mode_defaults_stable_release_to_urban(monkeypatch):
    args = build_argument_parser().parse_args([])
    args.experiment_track = "stable_release"
    monkeypatch.setattr("builtins.input", lambda _: "")

    choose_task_mode(args)

    assert args.task == TaskType.URBAN_RENEWAL


def test_script_main_help_is_available_from_root_entrypoint():
    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "main.py"), "--help"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--urban-shot" in result.stdout
    assert "legacy compatibility wrapper" not in result.stdout
