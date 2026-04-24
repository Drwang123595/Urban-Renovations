import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.prompts import PromptGenerator


def test_urban_renewal_four_strategies_loadable():
    for shot_mode in ["zero", "one", "few"]:
        prompt = PromptGenerator(shot_mode=shot_mode).get_single_system_prompt()
        assert isinstance(prompt, str)
        assert prompt.strip()
        assert "WoS Categories" in prompt
        assert "Do NOT require the literal phrases" in prompt

    cot_prompt = PromptGenerator(shot_mode="zero").get_cot_system_prompt()
    assert isinstance(cot_prompt, str)
    assert cot_prompt.strip()
    assert "WoS Categories" in cot_prompt
    assert "Do NOT require the literal phrases" in cot_prompt


def test_spatial_four_strategies_loadable():
    for shot_mode in ["zero", "one", "few", "cot"]:
        prompt = PromptGenerator(shot_mode=shot_mode).get_spatial_system_prompt()
        assert isinstance(prompt, str)
        assert prompt.strip()
        assert "generic city/site/context is not enough" in prompt
        assert "unspecified city" in prompt
        assert "case study context" in prompt
        assert "Specific_Study_Area_Note" not in prompt


def test_reflection_prompt_loads_from_registry_managed_theme_template():
    prompt_gen = PromptGenerator(shot_mode="zero")
    system_prompt = prompt_gen.get_reflection_system_prompt()
    critique_prompt = prompt_gen.get_reflection_critique_prompt()
    assert "second-pass self-audit" in system_prompt
    assert "review your previous answer carefully" in critique_prompt.lower()


def test_invalid_strategy_returns_available_options():
    with pytest.raises(ValueError) as exc:
        PromptGenerator(shot_mode="invalid")
    message = str(exc.value)
    assert "Invalid strategy" in message
    assert "Available strategies" in message


def test_missing_template_reports_theme_strategy_and_path(tmp_path: Path):
    theme_dir = tmp_path / "urban_renewal"
    theme_dir.mkdir(parents=True, exist_ok=True)
    (theme_dir / "zero.yaml").write_text("system_prompt: test", encoding="utf-8")

    prompt_gen = PromptGenerator(shot_mode="zero")
    prompt_gen.template_root = tmp_path

    with pytest.raises(FileNotFoundError) as exc:
        prompt_gen._get_system_prompt("urban_renewal", "one")

    message = str(exc.value)
    assert "theme=urban_renewal" in message
    assert "strategy=one" in message
    assert "path=" in message


def test_step_prompt_marks_auxiliary_signals_as_weak_and_supports_title_abstract_only_mode():
    prompt_gen = PromptGenerator(shot_mode="few")
    prompt = prompt_gen.get_step_prompt(
        1,
        "Street regeneration and public health",
        "Studies intervention in an existing neighborhood.",
        metadata={},
        auxiliary_context={
            "topic_label": "U4",
            "hard_case_reasons": ["low_confidence", "high_noise_topic_bucket"],
        },
    )
    assert "[TITLE_ABSTRACT_ONLY MODE]" in prompt
    assert "[AUXILIARY SIGNALS - WEAK HINTS ONLY]" in prompt
    assert "must NOT override clear evidence from the TITLE and ABSTRACT" in prompt
