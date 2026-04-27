from pathlib import Path
from typing import Dict

import yaml

from .strategy_registry import PromptStrategyRegistry


class PromptGenerator:
    THEMES = ("urban_renewal", "spatial")

    def __init__(self, shot_mode: str = "zero", default_theme: str = "urban_renewal"):
        self.template_root = Path(__file__).resolve().parents[1] / "templates"
        self.registry = self._load_strategy_registry()
        self.THEMES = tuple(self.registry.themes)
        self.default_theme = default_theme
        self._validate_theme(self.default_theme)
        self.shot_mode = self._validate_strategy(shot_mode, theme=self.default_theme)

    def _load_strategy_registry(self) -> PromptStrategyRegistry:
        registry_path = self.template_root / "strategy_registry.yaml"
        return PromptStrategyRegistry.load_from_file(registry_path)

    def _validate_theme(self, theme: str):
        if theme not in self.THEMES:
            raise ValueError(f"Invalid theme: {theme}. Available themes: {', '.join(self.THEMES)}")

    def _available_strategies(self, theme: str):
        return self.registry.list_enabled_strategies(theme=theme, allow_candidate=True)

    def _load_template_payload(
        self,
        theme: str,
        strategy_or_alias: str,
        *,
        allow_disabled: bool = False,
        allow_deprecated: bool = False,
    ) -> Dict:
        self._validate_theme(theme)
        definition = self.registry.get_definition(theme, strategy_or_alias)
        if not definition:
            available = ", ".join(self._available_strategies(theme))
            raise ValueError(f"Invalid strategy: {strategy_or_alias}. Available strategies: {available}")

        if not allow_disabled and not definition.enabled:
            available = ", ".join(self._available_strategies(theme))
            raise ValueError(f"Strategy is not enabled: {strategy_or_alias}. Available strategies: {available}")

        if not allow_deprecated and definition.lifecycle == "deprecated":
            available = ", ".join(self._available_strategies(theme))
            raise ValueError(f"Strategy is deprecated: {strategy_or_alias}. Available strategies: {available}")

        template_path = self.template_root / theme / definition.template_file
        if not template_path.exists():
            available = ", ".join(self._available_strategies(theme))
            raise FileNotFoundError(
                f"Missing template file: theme={theme}, strategy={strategy_or_alias}, "
                f"path={template_path}, available={available}"
            )

        with template_path.open("r", encoding="utf-8") as handle:
            content = yaml.safe_load(handle) or {}
        if not isinstance(content, dict):
            raise ValueError(
                f"Invalid template payload: theme={theme}, strategy={strategy_or_alias}, path={template_path}"
            )
        return content

    def _validate_strategy(self, strategy: str, theme: str) -> str:
        definition = self.registry.get_definition(theme, strategy)
        available = self._available_strategies(theme)
        if not definition:
            raise ValueError(f"Invalid strategy: {strategy}. Available strategies: {', '.join(available)}")
        if not definition.enabled or definition.lifecycle == "deprecated":
            raise ValueError(f"Strategy is not enabled: {strategy}. Available strategies: {', '.join(available)}")
        return definition.name

    def _get_system_prompt(self, theme: str, strategy: str) -> str:
        template = self._load_template_payload(theme, strategy)
        system_prompt = str(template.get("system_prompt", "") or "").strip()
        if not system_prompt:
            definition = self.registry.get_definition(theme, strategy)
            file_name = definition.template_file if definition else "unknown"
            template_path = self.template_root / theme / file_name
            raise ValueError(
                f"Template content is empty: theme={theme}, strategy={strategy}, path={template_path}"
            )
        return system_prompt

    def get_system_prompt(self) -> str:
        return self.get_single_system_prompt()

    def get_single_system_prompt(self) -> str:
        return self._get_system_prompt("urban_renewal", self.shot_mode)

    def get_step_system_prompt(self) -> str:
        return self._get_system_prompt("urban_renewal", self.shot_mode)

    def get_cot_system_prompt(self) -> str:
        return self._get_system_prompt("urban_renewal", "cot")

    def get_reflection_system_prompt(self) -> str:
        template = self._load_template_payload(
            "urban_renewal",
            "reflection",
            allow_disabled=True,
            allow_deprecated=True,
        )
        system_prompt = str(template.get("system_prompt", "") or "").strip()
        if not system_prompt:
            raise ValueError("Reflection template is missing system_prompt")
        return system_prompt

    def get_reflection_critique_prompt(self) -> str:
        template = self._load_template_payload(
            "urban_renewal",
            "reflection",
            allow_disabled=True,
            allow_deprecated=True,
        )
        return str(template.get("reflection_critique", "") or "").strip()

    def get_spatial_system_prompt(self) -> str:
        return self._get_system_prompt("spatial", self.shot_mode)

    def get_spatial_user_prompt(self, title: str, abstract: str) -> str:
        return f"[TITLE] {title}\n[ABSTRACT] {abstract}"

    def get_round_prompt(self, round_num: int, title: str, abstract: str, metadata: Dict | None = None) -> str:
        return self.get_single_prompt(title, abstract, metadata=metadata)

    def get_single_prompt(self, title: str, abstract: str, metadata: Dict | None = None) -> str:
        metadata = metadata or {}
        sections = [f"[TITLE] {title}", f"[ABSTRACT] {abstract}"]
        field_map = [
            ("Author Keywords", metadata.get("Author Keywords", "")),
            ("Keywords Plus", metadata.get("Keywords Plus", "")),
            ("Keywords", metadata.get("Keywords", "")),
            ("WoS Categories", metadata.get("WoS Categories", "")),
            ("Research Areas", metadata.get("Research Areas", "")),
        ]
        for label, value in field_map:
            value = str(value or "").strip()
            if value:
                sections.append(f"[{label}] {value}")
        return "\n".join(sections)

    def _has_metadata_support(self, metadata: Dict | None = None) -> bool:
        metadata = metadata or {}
        field_names = [
            "Author Keywords",
            "Keywords Plus",
            "Keywords",
            "WoS Categories",
            "Research Areas",
        ]
        return any(str(metadata.get(field, "") or "").strip() for field in field_names)

    def _format_auxiliary_context(self, auxiliary_context: Dict | None = None) -> str:
        auxiliary_context = auxiliary_context or {}
        hints = [
            f"{key}={value}"
            for key, value in auxiliary_context.items()
            if str(value or "").strip()
        ]
        if not hints:
            return ""
        return (
            "[AUXILIARY SIGNALS - WEAK HINTS ONLY] "
            "The following classifier / BERTopic signals are fallible and must NOT override clear evidence from the TITLE and ABSTRACT. "
            "Use them only to break genuine ties, and do NOT output 0 merely because these signals say low_confidence, high_noise, conflict, or outlier. "
            + "; ".join(hints)
        )

    def get_step_prompt(
        self,
        step_num: int,
        title: str,
        abstract: str,
        include_context: bool = True,
        metadata: Dict | None = None,
        auxiliary_context: Dict | None = None,
    ) -> str:
        if include_context:
            base = self.get_single_prompt(title, abstract, metadata=metadata) + "\n"
        else:
            base = ""

        if not self._has_metadata_support(metadata):
            base += (
                "[TITLE_ABSTRACT_ONLY MODE] "
                "Only TITLE and ABSTRACT should be treated as primary evidence. "
                "Do NOT require exact phrases such as 'urban renewal' or 'urban regeneration'. "
                "If the abstract clearly studies intervention, governance, consequences, evaluation, design, conservation, demolition, reuse, street/public-realm change, highway removal, densification for regeneration, or other restructuring of an existing built environment, output 1 even when the title foregrounds another mechanism.\n"
            )

        auxiliary_block = self._format_auxiliary_context(auxiliary_context)
        if auxiliary_block:
            base += auxiliary_block + "\n"

        if step_num == 1:
            return base + "Step 1: Urban renewal study? Output only 1 or 0."
        return base
