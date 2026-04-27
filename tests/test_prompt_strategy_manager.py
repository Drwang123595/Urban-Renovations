import sys
from pathlib import Path

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.prompt_strategy_manager import PromptStrategyManager


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


def _init_workspace(tmp_path: Path) -> PromptStrategyManager:
    template_root = tmp_path / "templates"
    (template_root / "urban_renewal").mkdir(parents=True, exist_ok=True)
    (template_root / "spatial").mkdir(parents=True, exist_ok=True)

    raw = {
        "themes": ["urban_renewal", "spatial"],
        "strategies": [
            make_entry("urban_renewal.zero", "zero", "urban_renewal", "zero.yaml"),
            make_entry("spatial.zero", "zero", "spatial", "zero.yaml"),
        ],
    }
    (template_root / "strategy_registry.yaml").write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    for theme in ["urban_renewal", "spatial"]:
        (template_root / theme / "zero.yaml").write_text(
            "system_prompt: zero strategy\n",
            encoding="utf-8",
        )
    return PromptStrategyManager(template_root=template_root)


def test_manager_crud_and_consistency_flow(tmp_path: Path):
    manager = _init_workspace(tmp_path)

    add_report = manager.add_strategy(
        name="reflection",
        theme="urban_renewal",
        template_file="reflection.yaml",
        aliases=["self_check"],
        description="Reflection template",
        version="0.9.0",
        lifecycle="candidate",
        owner="tester",
        change_summary="add reflection candidate",
        template_payload={"system_prompt": "urban reflection"},
    )
    assert add_report.ok is True

    strategy = manager.get_strategy("self_check", theme="urban_renewal")
    assert strategy["name"] == "reflection"
    assert strategy["enabled"] is True
    assert strategy["lifecycle"] == "candidate"

    update_report = manager.update_strategy(
        name_or_alias="reflection",
        theme="urban_renewal",
        description="Updated reflection",
        aliases=["reflection_alias"],
        version="1.0.0",
        owner="tester",
        change_summary="promote reflection candidate",
    )
    assert update_report.ok is True
    updated = manager.get_strategy("reflection_alias", theme="urban_renewal")
    assert updated["description"] == "Updated reflection"
    assert updated["version"] == "1.0.0"

    promote_report = manager.promote_strategy(
        "reflection",
        theme="urban_renewal",
        owner="tester",
        change_summary="mark stable",
    )
    assert promote_report.ok is True
    promoted = manager.get_strategy("reflection", theme="urban_renewal")
    assert promoted["lifecycle"] == "stable"
    assert promoted["enabled"] is True

    deprecate_report = manager.deprecate_strategy(
        "reflection",
        theme="urban_renewal",
        owner="tester",
        change_summary="retire reflection",
    )
    assert deprecate_report.ok is True
    deprecated = manager.get_strategy("reflection", theme="urban_renewal")
    assert deprecated["lifecycle"] == "deprecated"
    assert deprecated["enabled"] is False

    delete_report = manager.delete_strategy("reflection", theme="urban_renewal", remove_templates=True)
    assert delete_report.ok is True
    with pytest.raises(ValueError):
        manager.get_strategy("reflection", theme="urban_renewal")

    remaining = [item["key"] for item in manager.list_strategies()]
    assert sorted(remaining) == ["spatial.zero", "urban_renewal.zero"]


def test_add_rejects_missing_system_prompt(tmp_path: Path):
    manager = _init_workspace(tmp_path)
    with pytest.raises(ValueError):
        manager.add_strategy(
            name="bad",
            theme="urban_renewal",
            template_file="bad.yaml",
            version="1.0.0",
            lifecycle="candidate",
            owner="tester",
            change_summary="bad payload",
            template_payload={"x": "y"},
        )


def test_update_rejects_invalid_payload(tmp_path: Path):
    manager = _init_workspace(tmp_path)
    with pytest.raises(ValueError) as exc:
        manager.update_strategy(
            name_or_alias="zero",
            theme="urban_renewal",
            owner="tester",
            change_summary="invalid payload",
            version="1.0.1",
            template_payload={"spatial": {"system_prompt": "not allowed"}},
        )
    assert "missing non-empty system_prompt" in str(exc.value)


def test_consistency_detects_missing_template_file(tmp_path: Path):
    manager = _init_workspace(tmp_path)
    (tmp_path / "templates" / "spatial" / "zero.yaml").unlink()
    report = manager.check_consistency()
    assert report.ok is False
    assert any(item.code == "template.missing" for item in report.diagnostics)


def test_consistency_detects_root_legacy_template(tmp_path: Path):
    manager = _init_workspace(tmp_path)
    (tmp_path / "templates" / "single.yaml").write_text("system_prompt: legacy\n", encoding="utf-8")
    report = manager.check_consistency()
    assert report.ok is False
    assert any(item.code == "legacy.root_template" for item in report.diagnostics)
