import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.prompt_strategy_registry import PromptStrategyRegistry
from src.prompts import PromptGenerator


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


def test_registry_drives_available_strategies_by_theme():
    prompt_gen = PromptGenerator(shot_mode="zero")
    assert prompt_gen._available_strategies("urban_renewal") == ["zero", "one", "few", "cot"]
    assert prompt_gen._available_strategies("spatial") == ["zero", "one", "few", "cot"]


def test_registry_alias_can_initialize_prompt_generator():
    prompt_gen = PromptGenerator(shot_mode="chain_of_thought")
    assert prompt_gen.shot_mode == "cot"
    prompt = prompt_gen.get_single_system_prompt()
    assert isinstance(prompt, str)
    assert prompt.strip()


def test_registry_rejects_conflicting_alias():
    with pytest.raises(ValueError) as exc:
        PromptStrategyRegistry._from_dict(
            {
                "themes": ["urban_renewal", "spatial"],
                "strategies": [
                    make_entry(
                        "urban_renewal.zero",
                        "zero",
                        "urban_renewal",
                        "zero.yaml",
                        aliases=["shared_alias"],
                    ),
                    make_entry(
                        "urban_renewal.few",
                        "few",
                        "urban_renewal",
                        "few.yaml",
                        aliases=["shared_alias"],
                    ),
                ],
            }
        )
    assert "duplicate alias" in str(exc.value)


def test_registry_rejects_missing_required_fields():
    with pytest.raises(ValueError) as exc:
        PromptStrategyRegistry._from_dict(
            {
                "themes": ["urban_renewal", "spatial"],
                "strategies": [
                    {
                        "key": "urban_renewal.zero",
                        "name": "zero",
                        "theme": "urban_renewal",
                        "enabled": True,
                        "aliases": [],
                    }
                ],
            }
        )
    message = str(exc.value)
    assert "missing required fields" in message
    assert "template_file" in message
    assert "version" in message
    assert "lifecycle" in message


def test_registry_rejects_invalid_lifecycle():
    with pytest.raises(ValueError) as exc:
        PromptStrategyRegistry._from_dict(
            {
                "themes": ["urban_renewal"],
                "strategies": [
                    make_entry(
                        "urban_renewal.zero",
                        "zero",
                        "urban_renewal",
                        "zero.yaml",
                        lifecycle="draft",
                    )
                ],
            }
        )
    assert "lifecycle must be one of" in str(exc.value)


def test_registry_rejects_alias_conflict_with_other_strategy_name_in_same_theme():
    with pytest.raises(ValueError) as exc:
        PromptStrategyRegistry._from_dict(
            {
                "themes": ["urban_renewal", "spatial"],
                "strategies": [
                    make_entry(
                        "urban_renewal.zero",
                        "zero",
                        "urban_renewal",
                        "zero.yaml",
                        aliases=["few"],
                    ),
                    make_entry(
                        "urban_renewal.few",
                        "few",
                        "urban_renewal",
                        "few.yaml",
                    ),
                ],
            }
        )
    assert "alias conflicts with strategy name" in str(exc.value)


def test_registry_allows_same_alias_across_different_themes():
    registry = PromptStrategyRegistry._from_dict(
        {
            "themes": ["urban_renewal", "spatial"],
            "strategies": [
                make_entry(
                    "urban_renewal.zero",
                    "zero",
                    "urban_renewal",
                    "zero.yaml",
                    aliases=["shared_alias"],
                ),
                make_entry(
                    "spatial.zero",
                    "zero",
                    "spatial",
                    "zero.yaml",
                    aliases=["shared_alias"],
                ),
            ],
        }
    )
    assert registry.resolve_strategy("shared_alias", "urban_renewal") == "urban_renewal.zero"
    assert registry.resolve_strategy("shared_alias", "spatial") == "spatial.zero"
