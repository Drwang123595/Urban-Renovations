from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Sequence

import yaml

from .strategy_registry import ALLOWED_LIFECYCLES, PromptStrategyRegistry


ROOT_ALLOWED_FILES = {"strategy_registry.yaml"}


@dataclass(frozen=True)
class DiagnosticItem:
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class ConsistencyReport:
    ok: bool
    diagnostics: List[DiagnosticItem]


class PromptStrategyManager:
    def __init__(self, template_root: Path):
        self.template_root = Path(template_root)
        self.registry_path = self.template_root / "strategy_registry.yaml"

    def list_strategies(
        self,
        theme: Optional[str] = None,
        include_disabled: bool = True,
    ) -> List[Dict[str, Any]]:
        registry = self._load_registry()
        result: List[Dict[str, Any]] = []
        for definition in registry.strategies.values():
            if theme and definition.theme != theme:
                continue
            if not include_disabled and not definition.enabled:
                continue
            result.append(self._definition_to_dict(definition))
        return result

    def get_strategy(self, name_or_alias: str, theme: str) -> Dict[str, Any]:
        raw = self._load_registry_raw()
        registry = PromptStrategyRegistry._from_dict(raw)
        key = registry.resolve_strategy(name_or_alias, theme=theme)
        if not key:
            raise ValueError(f"Strategy not found: theme={theme}, strategy={name_or_alias}")
        for item in raw["strategies"]:
            if item.get("key") == key:
                return item
        raise ValueError(f"Strategy not found: theme={theme}, strategy={name_or_alias}")

    def add_strategy(
        self,
        name: str,
        theme: str,
        template_file: Optional[str] = None,
        enabled: bool = True,
        aliases: Optional[Sequence[str]] = None,
        description: str = "",
        version: str = "",
        lifecycle: str = "candidate",
        owner: str = "",
        change_summary: str = "",
        template_payload: Optional[Dict[str, Any]] = None,
    ) -> ConsistencyReport:
        raw = self._load_registry_raw()
        registry = PromptStrategyRegistry._from_dict(raw)
        if registry.resolve_strategy(name, theme=theme):
            raise ValueError(f"Strategy already exists: theme={theme}, strategy={name}")

        key = self._build_key(theme, name)
        if key in registry.strategies:
            raise ValueError(f"Strategy key already exists: {key}")

        normalized_theme = self._normalize_theme(theme, raw["themes"])
        normalized_template_file = self._normalize_template_file(template_file or f"{name}.yaml")
        normalized_aliases = self._normalize_aliases(aliases or [])
        normalized_version = self._require_text(version, "version")
        normalized_lifecycle = self._normalize_lifecycle(lifecycle)
        normalized_owner = self._require_text(owner, "owner")
        normalized_change_summary = self._require_text(change_summary, "change_summary")
        payload = self._normalize_template_payload(
            payload=template_payload,
            strategy_name=name,
            theme=normalized_theme,
        )

        new_entry = {
            "key": key,
            "name": name,
            "theme": normalized_theme,
            "template_file": normalized_template_file,
            "enabled": enabled,
            "aliases": normalized_aliases,
            "description": description,
            "version": normalized_version,
            "lifecycle": normalized_lifecycle,
            "owner": normalized_owner,
            "change_summary": normalized_change_summary,
        }
        raw["strategies"].append(new_entry)
        PromptStrategyRegistry._from_dict(raw)

        template_path = self.template_root / normalized_theme / normalized_template_file
        if template_path.exists():
            raise ValueError(f"Template file already exists: {template_path}")

        text_changes = {
            template_path: yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            self.registry_path: yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        }
        self._apply_text_changes(text_changes)
        return self.check_consistency()

    def update_strategy(
        self,
        name_or_alias: str,
        theme: str,
        template_file: Optional[str] = None,
        enabled: Optional[bool] = None,
        aliases: Optional[Sequence[str]] = None,
        description: Optional[str] = None,
        version: Optional[str] = None,
        lifecycle: Optional[str] = None,
        owner: Optional[str] = None,
        change_summary: Optional[str] = None,
        template_payload: Optional[Dict[str, Any]] = None,
    ) -> ConsistencyReport:
        if owner is None:
            raise ValueError("owner is required for update_strategy")
        if change_summary is None:
            raise ValueError("change_summary is required for update_strategy")
        if template_payload is not None and version is None:
            raise ValueError("version must be updated when template_payload changes")

        raw = self._load_registry_raw()
        registry = PromptStrategyRegistry._from_dict(raw)
        key = registry.resolve_strategy(name_or_alias, theme=theme)
        if not key:
            raise ValueError(f"Strategy not found: theme={theme}, strategy={name_or_alias}")

        index = self._find_strategy_index(raw, key)
        entry = dict(raw["strategies"][index])

        if enabled is not None:
            entry["enabled"] = enabled
        if aliases is not None:
            entry["aliases"] = self._normalize_aliases(aliases)
        if description is not None:
            entry["description"] = description
        if template_file is not None:
            entry["template_file"] = self._normalize_template_file(template_file)
        if version is not None:
            entry["version"] = self._require_text(version, "version")
        if lifecycle is not None:
            entry["lifecycle"] = self._normalize_lifecycle(lifecycle)
        entry["owner"] = self._require_text(owner, "owner")
        entry["change_summary"] = self._require_text(change_summary, "change_summary")

        raw["strategies"][index] = entry
        PromptStrategyRegistry._from_dict(raw)

        text_changes: Dict[Path, str] = {
            self.registry_path: yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)
        }
        if template_payload is not None:
            payload = self._normalize_template_payload(
                payload=template_payload,
                strategy_name=entry["name"],
                theme=entry["theme"],
            )
            template_path = self.template_root / entry["theme"] / entry["template_file"]
            text_changes[template_path] = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)

        self._apply_text_changes(text_changes)
        return self.check_consistency()

    def promote_strategy(
        self,
        name_or_alias: str,
        theme: str,
        *,
        owner: str,
        change_summary: str,
    ) -> ConsistencyReport:
        return self.update_strategy(
            name_or_alias=name_or_alias,
            theme=theme,
            lifecycle="stable",
            enabled=True,
            owner=owner,
            change_summary=change_summary,
        )

    def deprecate_strategy(
        self,
        name_or_alias: str,
        theme: str,
        *,
        owner: str,
        change_summary: str,
    ) -> ConsistencyReport:
        return self.update_strategy(
            name_or_alias=name_or_alias,
            theme=theme,
            lifecycle="deprecated",
            enabled=False,
            owner=owner,
            change_summary=change_summary,
        )

    def delete_strategy(
        self,
        name_or_alias: str,
        theme: str,
        remove_templates: bool = False,
        archive_dir: Optional[Path] = None,
    ) -> ConsistencyReport:
        raw = self._load_registry_raw()
        registry = PromptStrategyRegistry._from_dict(raw)
        key = registry.resolve_strategy(name_or_alias, theme=theme)
        if not key:
            raise ValueError(f"Strategy not found: theme={theme}, strategy={name_or_alias}")

        index = self._find_strategy_index(raw, key)
        entry = raw["strategies"].pop(index)
        PromptStrategyRegistry._from_dict(raw)

        text_changes: Dict[Path, str] = {
            self.registry_path: yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)
        }
        delete_paths: List[Path] = []
        template_path = self.template_root / entry["theme"] / entry["template_file"]
        if template_path.exists():
            if remove_templates:
                delete_paths.append(template_path)
            else:
                base_archive = archive_dir or (self.template_root / "_archived")
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_path = base_archive / stamp / entry["theme"] / entry["template_file"]
                text_changes[archive_path] = template_path.read_text(encoding="utf-8")
                delete_paths.append(template_path)

        self._apply_text_changes(text_changes, delete_paths=delete_paths)
        return self.check_consistency()

    def check_consistency(self) -> ConsistencyReport:
        diagnostics: List[DiagnosticItem] = []
        raw = self._load_registry_raw()
        try:
            registry = PromptStrategyRegistry._from_dict(raw)
            diagnostics.append(
                DiagnosticItem(
                    severity="info",
                    code="registry.valid",
                    message=f"Registry validation passed: {self.registry_path}",
                )
            )
        except Exception as exc:
            diagnostics.append(
                DiagnosticItem(
                    severity="error",
                    code="registry.invalid",
                    message=str(exc),
                )
            )
            return ConsistencyReport(ok=False, diagnostics=diagnostics)

        for theme in registry.themes:
            theme_dir = self.template_root / theme
            if not theme_dir.exists():
                diagnostics.append(
                    DiagnosticItem(
                        severity="error",
                        code="theme.missing",
                        message=f"Missing theme directory: {theme_dir}",
                    )
                )

        diagnostics.extend(self._check_root_legacy_templates())

        for definition in registry.strategies.values():
            template_path = self.template_root / definition.theme / definition.template_file
            if not template_path.exists():
                diagnostics.append(
                    DiagnosticItem(
                        severity="error",
                        code="template.missing",
                        message=f"Missing template file: key={definition.key}, path={template_path}",
                    )
                )
                continue
            try:
                payload = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
            except Exception as exc:
                diagnostics.append(
                    DiagnosticItem(
                        severity="error",
                        code="template.parse_error",
                        message=f"Template parse failed: key={definition.key}, path={template_path}, error={exc}",
                    )
                )
                continue
            try:
                self._validate_template_payload(payload, definition.name, definition.theme)
            except Exception as exc:
                diagnostics.append(
                    DiagnosticItem(
                        severity="error",
                        code="template.invalid",
                        message=str(exc),
                    )
                )

        ok = all(item.severity != "error" for item in diagnostics)
        if ok:
            diagnostics.append(
                DiagnosticItem(
                    severity="info",
                    code="consistency.ok",
                    message="Strategy registry and template files are consistent",
                )
            )
        return ConsistencyReport(ok=ok, diagnostics=diagnostics)

    def _load_registry(self) -> PromptStrategyRegistry:
        return PromptStrategyRegistry.load_from_file(self.registry_path)

    def _load_registry_raw(self) -> Dict[str, Any]:
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Missing strategy registry: path={self.registry_path}")
        raw = yaml.safe_load(self.registry_path.read_text(encoding="utf-8")) or {}
        themes = raw.get("themes")
        strategies = raw.get("strategies")
        if not isinstance(themes, list):
            raise ValueError("Invalid strategy registry: themes must be a list")
        if not isinstance(strategies, list):
            raise ValueError("Invalid strategy registry: strategies must be a list")
        return {"themes": themes, "strategies": strategies}

    def _definition_to_dict(self, definition) -> Dict[str, Any]:
        return {
            "key": definition.key,
            "name": definition.name,
            "theme": definition.theme,
            "enabled": definition.enabled,
            "aliases": list(definition.aliases),
            "description": definition.description,
            "template_file": definition.template_file,
            "version": definition.version,
            "lifecycle": definition.lifecycle,
            "owner": definition.owner,
            "change_summary": definition.change_summary,
        }

    def _build_key(self, theme: str, name: str) -> str:
        return f"{theme}.{name}"

    def _normalize_theme(self, theme: str, valid_themes: Sequence[str]) -> str:
        value = str(theme).strip()
        if not value:
            raise ValueError("Theme cannot be empty")
        if value not in valid_themes:
            raise ValueError(f"Unknown theme: {value}. Available themes: {', '.join(valid_themes)}")
        return value

    def _normalize_template_file(self, template_file: str) -> str:
        file_name = str(template_file).strip()
        if not file_name:
            raise ValueError("Template file name cannot be empty")
        if Path(file_name).name != file_name:
            raise ValueError(f"Template file name must stay inside theme directory: {file_name}")
        if not file_name.endswith(".yaml"):
            raise ValueError(f"Template file must end with .yaml: {file_name}")
        return file_name

    def _normalize_aliases(self, aliases: Sequence[str]) -> List[str]:
        normalized = [str(alias).strip() for alias in aliases if str(alias).strip()]
        unique: List[str] = []
        seen = set()
        for item in normalized:
            if item in seen:
                continue
            seen.add(item)
            unique.append(item)
        return unique

    def _normalize_lifecycle(self, lifecycle: str) -> str:
        value = str(lifecycle).strip()
        if value not in ALLOWED_LIFECYCLES:
            raise ValueError(
                f"Lifecycle must be one of {', '.join(sorted(ALLOWED_LIFECYCLES))}: {value}"
            )
        return value

    def _require_text(self, value: str, field_name: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError(f"{field_name} cannot be empty")
        return text

    def _normalize_template_payload(
        self,
        payload: Optional[Dict[str, Any]],
        strategy_name: str,
        theme: str,
    ) -> Dict[str, Any]:
        value = payload or {"system_prompt": f"TODO: fill prompt for strategy={strategy_name}, theme={theme}"}
        self._validate_template_payload(value, strategy_name, theme)
        return value

    def _validate_template_payload(self, payload: Dict[str, Any], strategy: str, theme: str):
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid template payload: strategy={strategy}, theme={theme}, expected mapping")
        system_prompt = payload.get("system_prompt")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError(
                f"Invalid template payload: strategy={strategy}, theme={theme}, missing non-empty system_prompt"
            )

    def _check_root_legacy_templates(self) -> List[DiagnosticItem]:
        diagnostics: List[DiagnosticItem] = []
        if not self.template_root.exists():
            return diagnostics
        for path in sorted(self.template_root.glob("*.yaml")):
            if path.name in ROOT_ALLOWED_FILES:
                continue
            diagnostics.append(
                DiagnosticItem(
                    severity="error",
                    code="legacy.root_template",
                    message=f"Legacy root template must be archived or migrated: {path}",
                )
            )
        return diagnostics

    def _find_strategy_index(self, raw: Dict[str, Any], strategy_key: str) -> int:
        for index, item in enumerate(raw["strategies"]):
            if item.get("key") == strategy_key:
                return index
        raise ValueError(f"Strategy not found: {strategy_key}")

    def _apply_text_changes(
        self,
        text_changes: Dict[Path, str],
        delete_paths: Optional[List[Path]] = None,
    ):
        backups: Dict[Path, Optional[str]] = {}
        changed: List[Path] = []
        deleted: List[Path] = []
        targets = sorted(text_changes.keys(), key=lambda path: str(path))
        delete_targets = sorted(delete_paths or [], key=lambda path: str(path))
        try:
            for path in targets:
                path.parent.mkdir(parents=True, exist_ok=True)
                old = path.read_text(encoding="utf-8") if path.exists() else None
                backups[path] = old
                self._write_text_atomic(path, text_changes[path])
                changed.append(path)

            for path in delete_targets:
                if path not in backups:
                    backups[path] = path.read_text(encoding="utf-8") if path.exists() else None
                if path.exists():
                    path.unlink()
                    deleted.append(path)
        except Exception:
            for path in reversed(changed):
                previous = backups.get(path)
                if previous is None:
                    if path.exists():
                        path.unlink()
                else:
                    self._write_text_atomic(path, previous)
            for path in deleted:
                previous = backups.get(path)
                if previous is not None:
                    self._write_text_atomic(path, previous)
            raise

    def _write_text_atomic(self, path: Path, content: str):
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            dir=str(path.parent),
        ) as handle:
            handle.write(content)
            temp_path = Path(handle.name)
        temp_path.replace(path)
