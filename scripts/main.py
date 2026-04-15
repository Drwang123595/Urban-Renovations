import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.prompt_manifest import (
    build_run_prompt_manifest,
    build_strategy_snapshot,
    ensure_strategy_runnable,
    write_prompt_manifest,
)
from src.prompt_strategy_registry import PromptStrategyDefinition, PromptStrategyRegistry
from src.task_router import TaskRouter, TaskType, UrbanMethod


TASK_THEME_MAP = {
    TaskType.URBAN_RENEWAL: "urban_renewal",
    TaskType.SPATIAL: "spatial",
}

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "src" / "templates" / "strategy_registry.yaml"
TEMPLATE_ROOT = REGISTRY_PATH.parent


def print_startup_overview(urban_modes, spatial_modes):
    print(f"\n{'=' * 60}")
    print("Prompt Runtime Overview")
    print(f"{'=' * 60}")
    print("Task modes:")
    print("  1) urban_renewal")
    print("  2) spatial")
    print("  3) both")
    print("Urban execution methods:")
    print("  1) pure_llm_api")
    print("  2) local_topic_classifier")
    print("  3) three_stage_hybrid")
    print("Available prompt strategies:")
    print(f"  - urban_renewal: {', '.join(urban_modes)}")
    print(f"  - spatial:       {', '.join(spatial_modes)}")
    print("Output:")
    print("  - AUTO: Data/<task>/output/<task>_<shot>_<timestamp>.xlsx")
    print("  - Each run also writes <output>.prompt_manifest.json")
    print(f"{'=' * 60}")


def select_input_file(default_input: str = None):
    train_dir = Config.TRAIN_DIR
    if not train_dir.exists():
        if default_input:
            print(f"Train directory not found: {train_dir}, using preset input: {default_input}")
            return default_input
        print(f"Train directory not found: {train_dir}")
        return None

    files = list(train_dir.glob("*.xlsx"))
    if not files:
        if default_input:
            print(f"No Excel files found in {train_dir}, using preset input: {default_input}")
            return default_input
        print(f"No Excel files found in {train_dir}")
        return None

    print(f"\nFound {len(files)} Excel files in {train_dir}:")
    for index, file_path in enumerate(files, start=1):
        print(f"{index}: {file_path.name}")
    if default_input:
        print(f"Preset input: {default_input}")

    while True:
        choice = input("\nSelect input file number (or press Enter to keep preset/default): ").strip()
        if not choice:
            return default_input
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return str(files[idx])
            print("Invalid number.")
        except ValueError:
            print("Please enter a number.")


def load_prompt_strategy_registry() -> PromptStrategyRegistry:
    return PromptStrategyRegistry.load_from_file(REGISTRY_PATH)


def get_enabled_shot_modes(
    registry: PromptStrategyRegistry,
    theme: str,
    allow_candidate: bool = False,
):
    return registry.list_enabled_strategies(theme=theme, allow_candidate=allow_candidate)


def normalize_shot_mode(
    shot_mode: str,
    registry: PromptStrategyRegistry,
    enabled_modes,
    theme: str,
    allow_candidate: bool = False,
) -> str:
    canonical = registry.resolve_strategy(shot_mode, theme=theme)
    if not canonical:
        raise ValueError(f"Invalid strategy: {shot_mode}. Available strategies: {', '.join(enabled_modes)}")
    definition = registry.strategies.get(canonical)
    if not definition:
        raise ValueError(f"Invalid strategy: {shot_mode}. Available strategies: {', '.join(enabled_modes)}")
    ensure_strategy_runnable(definition, allow_candidate=allow_candidate)
    if definition.name not in enabled_modes:
        raise ValueError(f"Strategy is not selectable: {shot_mode}. Available strategies: {', '.join(enabled_modes)}")
    return definition.name


def resolve_definition(
    registry: PromptStrategyRegistry,
    theme: str,
    strategy_or_alias: str,
    allow_candidate: bool = False,
) -> PromptStrategyDefinition:
    definition = registry.get_definition(theme, strategy_or_alias)
    if not definition:
        available = registry.list_enabled_strategies(theme=theme, allow_candidate=allow_candidate)
        raise ValueError(f"Invalid strategy: {strategy_or_alias}. Available strategies: {', '.join(available)}")
    ensure_strategy_runnable(definition, allow_candidate=allow_candidate)
    return definition


def select_shot_mode(
    registry: PromptStrategyRegistry,
    enabled_modes,
    default_mode: str,
    theme: str,
    label: str,
    allow_candidate: bool = False,
):
    print(f"\nSelect shot mode ({label}):")
    for index, mode in enumerate(enabled_modes, start=1):
        print(f"{index} = {mode}")

    while True:
        choice = input(
            f"Enter choice (1-{len(enabled_modes)} / strategy / alias, press Enter for {default_mode}): "
        ).strip()
        if not choice:
            return default_mode

        if choice.isdigit():
            selected_index = int(choice) - 1
            if 0 <= selected_index < len(enabled_modes):
                return enabled_modes[selected_index]
            print("Invalid number.")
            continue

        try:
            return normalize_shot_mode(
                choice,
                registry,
                enabled_modes,
                theme=theme,
                allow_candidate=allow_candidate,
            )
        except ValueError as exc:
            print(exc)


def select_task_mode(default_task: TaskType = None):
    print("\nSelect task mode:")
    print("1 = urban_renewal")
    print("2 = spatial")
    print("3 = both")
    if default_task:
        print(f"Current preset: {default_task.value}")

    choice = input("Enter choice (1/2/3, press Enter to keep preset): ").strip()
    if not choice and default_task:
        return default_task
    if choice == "1":
        return TaskType.URBAN_RENEWAL
    if choice == "2":
        return TaskType.SPATIAL
    return TaskType.BOTH


def normalize_urban_method(method: str) -> UrbanMethod:
    aliases = {
        "1": UrbanMethod.PURE_LLM_API,
        "2": UrbanMethod.LOCAL_TOPIC_CLASSIFIER,
        "3": UrbanMethod.THREE_STAGE_HYBRID,
        "pure_llm": UrbanMethod.PURE_LLM_API,
        "pure_llm_api": UrbanMethod.PURE_LLM_API,
        "llm": UrbanMethod.PURE_LLM_API,
        "local_classifier": UrbanMethod.LOCAL_TOPIC_CLASSIFIER,
        "local_topic_classifier": UrbanMethod.LOCAL_TOPIC_CLASSIFIER,
        "classifier": UrbanMethod.LOCAL_TOPIC_CLASSIFIER,
        "hybrid": UrbanMethod.THREE_STAGE_HYBRID,
        "three_stage_hybrid": UrbanMethod.THREE_STAGE_HYBRID,
    }
    resolved = aliases.get(str(method or "").strip().lower())
    if resolved is None:
        valid = ", ".join(item.value for item in UrbanMethod)
        raise ValueError(f"Invalid urban method: {method}. Available methods: {valid}")
    return resolved

def normalize_hybrid_llm_assist(value: str) -> bool:
    aliases = {
        "on": True,
        "off": False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    resolved = aliases.get(str(value or "").strip().lower())
    if resolved is None:
        raise ValueError("Invalid hybrid LLM assist value: expected on/off")
    return resolved


def resolve_hybrid_llm_assist(cli_value: str = None) -> bool:
    if cli_value is None:
        return bool(Config.URBAN_HYBRID_LLM_ASSIST_ENABLED)
    return normalize_hybrid_llm_assist(cli_value)


def urban_method_requires_llm(
    urban_method: UrbanMethod,
    *,
    hybrid_llm_assist_enabled: bool,
) -> bool:
    if urban_method == UrbanMethod.PURE_LLM_API:
        return True
    if urban_method == UrbanMethod.THREE_STAGE_HYBRID:
        return bool(hybrid_llm_assist_enabled)
    return False


def urban_method_requires_prompt(
    urban_method: UrbanMethod,
    *,
    hybrid_llm_assist_enabled: bool,
) -> bool:
    return urban_method_requires_llm(
        urban_method,
        hybrid_llm_assist_enabled=hybrid_llm_assist_enabled,
    )


def task_requires_api_key(
    task: TaskType,
    urban_method: UrbanMethod,
    *,
    hybrid_llm_assist_enabled: bool,
) -> bool:
    if task in {TaskType.SPATIAL, TaskType.BOTH}:
        return True
    return urban_method_requires_llm(
        urban_method,
        hybrid_llm_assist_enabled=hybrid_llm_assist_enabled,
    )


def select_urban_method(default_method: UrbanMethod = UrbanMethod.THREE_STAGE_HYBRID) -> UrbanMethod:
    print("\nSelect urban execution method:")
    print(f"1 = {UrbanMethod.PURE_LLM_API.value}")
    print(f"2 = {UrbanMethod.LOCAL_TOPIC_CLASSIFIER.value}")
    print(f"3 = {UrbanMethod.THREE_STAGE_HYBRID.value}")
    print(f"Current preset: {default_method.value}")

    while True:
        choice = input("Enter choice (1/2/3/method, press Enter to keep preset): ").strip()
        if not choice:
            return default_method
        try:
            return normalize_urban_method(choice)
        except ValueError as exc:
            print(exc)


def select_output_mode(task: TaskType, preset_output: str = None):
    print("\nSelect output mode:")
    print("1 = auto output path")
    print("2 = custom output path")
    if preset_output:
        print(f"Current preset output: {preset_output}")
    if task == TaskType.BOTH:
        print("Note: in both mode, a custom output path applies to the merged file.")

    choice = input("Enter choice (1/2, press Enter to keep preset/default): ").strip()
    if not choice:
        return preset_output
    if choice == "2":
        path = input("Enter output file path (.xlsx): ").strip()
        return path or preset_output
    return None


def render_strategy_proof(requested: str, definition: PromptStrategyDefinition) -> str:
    return (
        f"requested={requested}, resolved={definition.name}, template={definition.template_file}, "
        f"version={definition.version}, lifecycle={definition.lifecycle}"
    )


def write_manifests_for_results(
    *,
    task_mode: TaskType,
    input_file: str,
    urban_definition: PromptStrategyDefinition,
    spatial_definition: PromptStrategyDefinition,
    results,
):
    urban_snapshot = build_strategy_snapshot(
        registry=load_prompt_strategy_registry(),
        template_root=TEMPLATE_ROOT,
        theme="urban_renewal",
        strategy_or_alias=urban_definition.name,
    )
    spatial_snapshot = build_strategy_snapshot(
        registry=load_prompt_strategy_registry(),
        template_root=TEMPLATE_ROOT,
        theme="spatial",
        strategy_or_alias=spatial_definition.name,
    )

    if task_mode == TaskType.URBAN_RENEWAL:
        manifest = build_run_prompt_manifest(
            task_mode=task_mode.value,
            active_tasks=["urban_renewal"],
            input_file=Path(input_file),
            registry_path=REGISTRY_PATH,
            strategy_snapshots={"urban_renewal": urban_snapshot},
        )
        write_prompt_manifest(results, manifest)
        return

    if task_mode == TaskType.SPATIAL:
        manifest = build_run_prompt_manifest(
            task_mode=task_mode.value,
            active_tasks=["spatial"],
            input_file=Path(input_file),
            registry_path=REGISTRY_PATH,
            strategy_snapshots={"spatial": spatial_snapshot},
        )
        write_prompt_manifest(results, manifest)
        return

    urban_manifest = build_run_prompt_manifest(
        task_mode=task_mode.value,
        active_tasks=["urban_renewal"],
        input_file=Path(input_file),
        registry_path=REGISTRY_PATH,
        strategy_snapshots={"urban_renewal": urban_snapshot},
    )
    spatial_manifest = build_run_prompt_manifest(
        task_mode=task_mode.value,
        active_tasks=["spatial"],
        input_file=Path(input_file),
        registry_path=REGISTRY_PATH,
        strategy_snapshots={"spatial": spatial_snapshot},
    )
    merged_manifest = build_run_prompt_manifest(
        task_mode=task_mode.value,
        active_tasks=["urban_renewal", "spatial"],
        input_file=Path(input_file),
        registry_path=REGISTRY_PATH,
        strategy_snapshots={
            "urban_renewal": urban_snapshot,
            "spatial": spatial_snapshot,
        },
    )
    write_prompt_manifest(results["urban_renewal"], urban_manifest)
    write_prompt_manifest(results["spatial"], spatial_manifest)
    if results.get("merged"):
        write_prompt_manifest(results["merged"], merged_manifest)


def main():
    Config.load_env()
    Config.validate_runtime_environment(require_py313=True, warn_on_minor_drift=True)

    try:
        strategy_registry = load_prompt_strategy_registry()
    except Exception as exc:
        print(f"Error: failed to load strategy registry: {exc}")
        return

    parser = argparse.ArgumentParser(
        description="Urban Renovation Literature Auto-Identification System"
    )
    parser.add_argument("--shot", default=None, help="Fallback shot mode for selected task(s)")
    parser.add_argument("--urban-shot", default=None, help="Shot mode for urban_renewal task")
    parser.add_argument("--spatial-shot", default=None, help="Shot mode for spatial task")
    parser.add_argument(
        "--task",
        choices=["urban_renewal", "spatial", "both"],
        default=None,
        help="Task type",
    )
    parser.add_argument("--strategy", help="Legacy option for backward compatibility. Use --task instead.")
    parser.add_argument("--limit", type=int, help="Limit number of papers to process")
    parser.add_argument("--input", help="Input Excel file path")
    parser.add_argument("--output", help="Output Excel file path")
    parser.add_argument(
        "--urban-method",
        choices=[item.value for item in UrbanMethod],
        default=None,
        help="Urban execution method: pure_llm_api / local_topic_classifier / three_stage_hybrid",
    )
    parser.add_argument(
        "--hybrid-llm-assist",
        choices=["on", "off"],
        default=None,
        help="Enable or disable LLM assist for three_stage_hybrid (default: on, can also be set by env)",
    )
    parser.add_argument(
        "--allow-candidate",
        action="store_true",
        help="Allow candidate prompt strategies for experiment runs",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Run in batch mode without interactive prompts",
    )

    args = parser.parse_args()

    urban_enabled = get_enabled_shot_modes(
        strategy_registry,
        theme="urban_renewal",
        allow_candidate=args.allow_candidate,
    )
    spatial_enabled = get_enabled_shot_modes(
        strategy_registry,
        theme="spatial",
        allow_candidate=args.allow_candidate,
    )
    if not urban_enabled:
        print("Error: no selectable urban_renewal strategies in registry.")
        return
    if not spatial_enabled:
        print("Error: no selectable spatial strategies in registry.")
        return

    print_startup_overview(urban_enabled, spatial_enabled)

    default_task = TaskType(args.task) if args.task else TaskType.BOTH
    if args.non_interactive:
        args.task = default_task
    else:
        args.task = select_task_mode(default_task=default_task)

    default_urban_method = (
        normalize_urban_method(args.urban_method)
        if args.urban_method
        else UrbanMethod.THREE_STAGE_HYBRID
    )
    if args.task in {TaskType.URBAN_RENEWAL, TaskType.BOTH}:
        args.urban_method = (
            default_urban_method
            if args.non_interactive
            else select_urban_method(default_method=default_urban_method)
        )
    else:
        args.urban_method = default_urban_method

    args.hybrid_llm_assist = resolve_hybrid_llm_assist(args.hybrid_llm_assist)

    if task_requires_api_key(
        args.task,
        args.urban_method,
        hybrid_llm_assist_enabled=args.hybrid_llm_assist,
    ) and not Config.API_KEY:
        print("Error: LLM API key not found, but the selected task configuration requires LLM access.")
        print("Please create a .env file in the project root with DEEPSEEK_API_KEY=your_key")
        return

    urban_default = Config.DEFAULT_SHOT_MODE if Config.DEFAULT_SHOT_MODE in urban_enabled else urban_enabled[0]
    spatial_default = Config.DEFAULT_SHOT_MODE if Config.DEFAULT_SHOT_MODE in spatial_enabled else spatial_enabled[0]

    try:
        if args.task == TaskType.URBAN_RENEWAL:
            chosen = args.urban_shot or args.shot
            if chosen:
                urban_shot = normalize_shot_mode(
                    chosen,
                    strategy_registry,
                    urban_enabled,
                    theme="urban_renewal",
                    allow_candidate=args.allow_candidate,
                )
            elif urban_method_requires_prompt(
                args.urban_method,
                hybrid_llm_assist_enabled=args.hybrid_llm_assist,
            ):
                urban_shot = (
                    urban_default
                    if args.non_interactive
                    else select_shot_mode(
                        registry=strategy_registry,
                        enabled_modes=urban_enabled,
                        default_mode=urban_default,
                        theme="urban_renewal",
                        label="urban_renewal",
                        allow_candidate=args.allow_candidate,
                    )
                )
            else:
                urban_shot = urban_default
            spatial_shot = spatial_default
        elif args.task == TaskType.SPATIAL:
            chosen = args.spatial_shot or args.shot
            if chosen:
                spatial_shot = normalize_shot_mode(
                    chosen,
                    strategy_registry,
                    spatial_enabled,
                    theme="spatial",
                    allow_candidate=args.allow_candidate,
                )
            else:
                spatial_shot = (
                    spatial_default
                    if args.non_interactive
                    else select_shot_mode(
                        registry=strategy_registry,
                        enabled_modes=spatial_enabled,
                        default_mode=spatial_default,
                        theme="spatial",
                        label="spatial",
                        allow_candidate=args.allow_candidate,
                    )
                )
            urban_shot = urban_default
        else:
            urban_chosen = args.urban_shot or args.shot
            spatial_chosen = args.spatial_shot or args.shot
            if urban_chosen:
                urban_shot = normalize_shot_mode(
                    urban_chosen,
                    strategy_registry,
                    urban_enabled,
                    theme="urban_renewal",
                    allow_candidate=args.allow_candidate,
                )
            elif not urban_method_requires_prompt(
                args.urban_method,
                hybrid_llm_assist_enabled=args.hybrid_llm_assist,
            ):
                urban_shot = urban_default
            else:
                urban_shot = (
                    urban_default
                    if args.non_interactive
                    else select_shot_mode(
                        registry=strategy_registry,
                        enabled_modes=urban_enabled,
                        default_mode=urban_default,
                        theme="urban_renewal",
                        label="urban_renewal",
                        allow_candidate=args.allow_candidate,
                    )
                )
            if spatial_chosen:
                spatial_shot = normalize_shot_mode(
                    spatial_chosen,
                    strategy_registry,
                    spatial_enabled,
                    theme="spatial",
                    allow_candidate=args.allow_candidate,
                )
            else:
                spatial_shot = (
                    spatial_default
                    if args.non_interactive
                    else select_shot_mode(
                        registry=strategy_registry,
                        enabled_modes=spatial_enabled,
                        default_mode=spatial_default,
                        theme="spatial",
                        label="spatial",
                        allow_candidate=args.allow_candidate,
                    )
                )
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    try:
        urban_definition = resolve_definition(
            strategy_registry,
            theme="urban_renewal",
            strategy_or_alias=urban_shot,
            allow_candidate=args.allow_candidate,
        )
        spatial_definition = resolve_definition(
            strategy_registry,
            theme="spatial",
            strategy_or_alias=spatial_shot,
            allow_candidate=args.allow_candidate,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    preset_input = args.input
    if not preset_input:
        default_file = Config.INPUT_FILE
        if default_file and default_file.exists():
            preset_input = str(default_file)
    args.input = preset_input if args.non_interactive else select_input_file(default_input=preset_input)
    if args.input:
        print(f"Selected input file: {args.input}")
    else:
        print("No input file selected or default file not found.")
        return

    if not args.non_interactive:
        args.output = select_output_mode(task=args.task, preset_output=args.output)

    if args.strategy is not None:
        print("[WARN] --strategy is deprecated. Use --task instead.")
        print("[WARN] Ignoring --strategy and using --task mode.")

    print(f"\n{'=' * 60}")
    print("Urban Renovation Literature Auto-Identification System")
    print(f"{'=' * 60}")
    print(f"Input: {args.input}")
    if args.task == TaskType.BOTH:
        print("Isolation: strict serial (urban -> spatial)")
        print(f"Urban Method: {args.urban_method.value}")
        if args.urban_method == UrbanMethod.THREE_STAGE_HYBRID:
            print(f"Hybrid LLM Assist: {'on' if args.hybrid_llm_assist else 'off'}")
        print(f"Urban Mode: {urban_definition.name}")
        urban_strategy_proof = render_strategy_proof(args.urban_shot or args.shot or urban_shot, urban_definition)
        if not urban_method_requires_prompt(
            args.urban_method,
            hybrid_llm_assist_enabled=args.hybrid_llm_assist,
        ):
            urban_strategy_proof += " (unused by current urban method)"
        print(f"Urban Strategy Proof: {urban_strategy_proof}")
        print(f"Spatial Mode: {spatial_definition.name}")
        print(
            f"Spatial Strategy Proof: {render_strategy_proof(args.spatial_shot or args.shot or spatial_shot, spatial_definition)}"
        )
    elif args.task == TaskType.URBAN_RENEWAL:
        print(f"Urban Method: {args.urban_method.value}")
        if args.urban_method == UrbanMethod.THREE_STAGE_HYBRID:
            print(f"Hybrid LLM Assist: {'on' if args.hybrid_llm_assist else 'off'}")
        print(f"Mode: {urban_definition.name}")
        urban_strategy_proof = render_strategy_proof(args.urban_shot or args.shot or urban_shot, urban_definition)
        if not urban_method_requires_prompt(
            args.urban_method,
            hybrid_llm_assist_enabled=args.hybrid_llm_assist,
        ):
            urban_strategy_proof += " (unused by current urban method)"
        print(f"Strategy Proof: {urban_strategy_proof}")
    else:
        print(f"Mode: {spatial_definition.name}")
        print(f"Strategy Proof: {render_strategy_proof(args.spatial_shot or args.shot or spatial_shot, spatial_definition)}")
    print(f"Task: {args.task.value}")
    print(f"Output: {args.output or 'AUTO'}")

    router = TaskRouter(
        shot_mode=urban_definition.name,
        urban_shot_mode=urban_definition.name,
        spatial_shot_mode=spatial_definition.name,
        urban_method=args.urban_method,
        hybrid_llm_assist_enabled=args.hybrid_llm_assist,
    )

    try:
        if args.task == TaskType.URBAN_RENEWAL:
            print("Running Urban Renewal Classification only...")
            result = router.run_urban_renewal(
                input_file=args.input,
                output_file=args.output,
                limit=args.limit,
            )
        elif args.task == TaskType.SPATIAL:
            print("Running Spatial Attribute Extraction only...")
            result = router.run_spatial(
                input_file=args.input,
                output_file=args.output,
                limit=args.limit,
            )
        else:
            print("Running both tasks (Urban Renewal + Spatial)...")
            result = router.run_both(
                input_file=args.input,
                output_file=args.output,
                limit=args.limit,
            )
            print(f"\n{'=' * 60}")
            print("Results:")
            print(f"  Urban Renewal: {result['urban_renewal']}")
            print(f"  Spatial: {result['spatial']}")
            if result.get("merged"):
                print(f"  Merged: {result['merged']}")

        write_manifests_for_results(
            task_mode=args.task,
            input_file=args.input,
            urban_definition=urban_definition,
            spatial_definition=spatial_definition,
            results=result,
        )
    except Exception as exc:
        print(f"Error: run failed: {exc}")
        raise

    print(f"{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
