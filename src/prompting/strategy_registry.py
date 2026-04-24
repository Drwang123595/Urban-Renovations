from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml


ALLOWED_LIFECYCLES = {"candidate", "stable", "deprecated"}


@dataclass(frozen=True)
class PromptStrategyDefinition:
    key: str
    name: str
    theme: str
    template_file: str
    enabled: bool
    aliases: List[str]
    description: str
    version: str
    lifecycle: str
    owner: str
    change_summary: str


class PromptStrategyRegistry:
    def __init__(
        self,
        themes: List[str],
        strategies: Dict[str, PromptStrategyDefinition],
        alias_index: Dict[str, Dict[str, str]],
        name_index: Dict[str, Dict[str, str]],
    ):
        self.themes = themes
        self.strategies = strategies
        self.alias_index = alias_index
        self.name_index = name_index

    @classmethod
    def load_from_file(cls, registry_path: Path) -> "PromptStrategyRegistry":
        if not registry_path.exists():
            raise FileNotFoundError(f"Missing strategy registry: path={registry_path}")

        with registry_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: Dict) -> "PromptStrategyRegistry":
        themes = raw.get("themes")
        raw_strategies = raw.get("strategies")

        if not isinstance(themes, list) or not themes:
            raise ValueError("Invalid strategy registry: themes must be a non-empty list")
        if not isinstance(raw_strategies, list) or not raw_strategies:
            raise ValueError("Invalid strategy registry: strategies must be a non-empty list")

        strategy_defs: Dict[str, PromptStrategyDefinition] = {}
        alias_index: Dict[str, Dict[str, str]] = {theme: {} for theme in themes}
        name_index: Dict[str, Dict[str, str]] = {theme: {} for theme in themes}

        for item in raw_strategies:
            if not isinstance(item, dict):
                raise ValueError("Invalid strategy registry: every strategy definition must be a mapping")

            name = item.get("name")
            key = item.get("key")
            theme = item.get("theme")
            template_file = item.get("template_file")
            enabled = item.get("enabled")
            aliases = item.get("aliases", [])
            description = item.get("description", "")
            version = item.get("version")
            lifecycle = item.get("lifecycle")
            owner = item.get("owner")
            change_summary = item.get("change_summary")

            missing_fields = []
            for field_name, value in (
                ("key", key),
                ("name", name),
                ("theme", theme),
                ("template_file", template_file),
                ("enabled", enabled),
                ("version", version),
                ("lifecycle", lifecycle),
                ("owner", owner),
                ("change_summary", change_summary),
            ):
                if value is None or (isinstance(value, str) and not value.strip()):
                    missing_fields.append(field_name)
            if missing_fields:
                raise ValueError(
                    f"Invalid strategy registry: missing required fields "
                    f"strategy={name or '<unknown>'}, fields={', '.join(missing_fields)}"
                )

            if key in strategy_defs:
                raise ValueError(f"Invalid strategy registry: duplicate strategy key={key}")
            if not isinstance(theme, str):
                raise ValueError(f"Invalid strategy registry: theme must be a string strategy={name}")
            if theme not in themes:
                raise ValueError(f"Invalid strategy registry: unknown theme strategy={name}, theme={theme}")
            if not isinstance(template_file, str) or not template_file.strip():
                raise ValueError(
                    f"Invalid strategy registry: template_file must be a non-empty string strategy={name}"
                )
            if Path(template_file).name != template_file:
                raise ValueError(
                    f"Invalid strategy registry: template_file must be a file name within its theme directory "
                    f"strategy={name}, template_file={template_file}"
                )
            if not isinstance(enabled, bool):
                raise ValueError(f"Invalid strategy registry: enabled must be bool strategy={name}")
            if not isinstance(aliases, list):
                raise ValueError(f"Invalid strategy registry: aliases must be a list strategy={name}")
            if "." not in key:
                raise ValueError(f"Invalid strategy registry: key must include theme prefix key={key}")
            key_theme = key.split(".", 1)[0]
            if key_theme != theme:
                raise ValueError(
                    f"Invalid strategy registry: key/theme mismatch key={key}, theme={theme}"
                )
            if lifecycle not in ALLOWED_LIFECYCLES:
                allowed = ", ".join(sorted(ALLOWED_LIFECYCLES))
                raise ValueError(
                    f"Invalid strategy registry: lifecycle must be one of {allowed} "
                    f"strategy={name}, lifecycle={lifecycle}"
                )
            if name in name_index[theme]:
                raise ValueError(f"Invalid strategy registry: duplicate strategy name theme={theme}, name={name}")
            if key in aliases:
                raise ValueError(f"Invalid strategy registry: alias cannot equal key key={key}")

            normalized_aliases: List[str] = []
            seen_aliases = set()
            for alias in aliases:
                if not isinstance(alias, str) or not alias.strip():
                    raise ValueError(
                        f"Invalid strategy registry: aliases must be non-empty strings strategy={name}"
                    )
                alias_value = alias.strip()
                if alias_value in seen_aliases:
                    continue
                seen_aliases.add(alias_value)
                normalized_aliases.append(alias_value)

            strategy_defs[key] = PromptStrategyDefinition(
                key=key,
                name=name,
                theme=theme,
                template_file=template_file,
                enabled=enabled,
                aliases=normalized_aliases,
                description=str(description or ""),
                version=str(version).strip(),
                lifecycle=lifecycle,
                owner=str(owner).strip(),
                change_summary=str(change_summary).strip(),
            )
            name_index[theme][name] = key

        for key, definition in strategy_defs.items():
            scoped_aliases = alias_index[definition.theme]
            for alias in definition.aliases:
                if alias in name_index[definition.theme]:
                    raise ValueError(
                        f"Invalid strategy registry: alias conflicts with strategy name "
                        f"alias={alias}, theme={definition.theme}"
                    )
                existing_owner = scoped_aliases.get(alias)
                if existing_owner and existing_owner != key:
                    raise ValueError(
                        f"Invalid strategy registry: duplicate alias alias={alias}, "
                        f"theme={definition.theme}, owners={existing_owner}|{key}"
                    )
                scoped_aliases[alias] = key

        return cls(
            themes=themes,
            strategies=strategy_defs,
            alias_index=alias_index,
            name_index=name_index,
        )

    def resolve_strategy(self, name_or_alias: str, theme: str) -> Optional[str]:
        definition = self.strategies.get(name_or_alias)
        if definition and definition.theme == theme:
            return definition.key
        key_by_name = self.name_index.get(theme, {}).get(name_or_alias)
        if key_by_name:
            return key_by_name
        return self.alias_index.get(theme, {}).get(name_or_alias)

    def list_enabled_strategies(
        self,
        theme: Optional[str] = None,
        allow_candidate: bool = True,
    ) -> List[str]:
        if theme is not None and theme not in self.themes:
            raise ValueError(f"Unknown theme: {theme}. Available themes: {', '.join(self.themes)}")

        items: List[str] = []
        for definition in self.strategies.values():
            if theme is not None and definition.theme != theme:
                continue
            if not definition.enabled:
                continue
            if definition.lifecycle == "deprecated":
                continue
            if definition.lifecycle == "candidate" and not allow_candidate:
                continue
            items.append(definition.name)
        return items

    def get_definition(self, theme: str, strategy_or_alias: str) -> Optional[PromptStrategyDefinition]:
        key = self.resolve_strategy(strategy_or_alias, theme)
        if not key:
            return None
        return self.strategies.get(key)

    def get_template_file(self, theme: str, strategy_or_alias: str) -> Optional[str]:
        definition = self.get_definition(theme, strategy_or_alias)
        if not definition:
            return None
        return definition.template_file
