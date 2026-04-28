import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prompting.manifest import (
    build_run_prompt_manifest,
    build_strategy_snapshot,
    ensure_strategy_runnable,
    write_prompt_manifest,
)
from src.prompting.strategy_registry import PromptStrategyDefinition, PromptStrategyRegistry
from src.runtime.config import Config
from src.tasks.task_router import TaskRouter, TaskType, UrbanMethod


TASK_THEME_MAP = {
    TaskType.URBAN_RENEWAL: "urban_renewal",
    TaskType.SPATIAL: "spatial",
}

REGISTRY_PATH = PROJECT_ROOT / "src" / "templates" / "strategy_registry.yaml"
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
    print("Experiment tracks:")
    print(f"  - {', '.join(Config.EXPERIMENT_TRACKS)}")
    print("Output:")
    print("  - AUTO: Data/<dataset>/runs/<track>/<run_tag>/predictions/<task>_<shot>_<timestamp>.xlsx")
    print("  - Each run also writes <output>.prompt_manifest.json")
    print(f"{'=' * 60}")


def select_experiment_track(default_track: str) -> str:
    print("\nSelect experiment track:")
    for index, track in enumerate(Config.EXPERIMENT_TRACKS, start=1):
        print(f"{index} = {track}")
    print(f"Current preset: {default_track}")

    while True:
        choice = input("Enter choice (1/2/3/track, press Enter to keep preset): ").strip()
        if not choice:
            return default_track
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(Config.EXPERIMENT_TRACKS):
                return Config.EXPERIMENT_TRACKS[idx]
            print("Invalid number.")
        except ValueError:
            try:
                return normalize_experiment_track(choice)
            except ValueError as exc:
                print(exc)


def _candidate_input_dirs_for_track(experiment_track: str) -> list[Path]:
    track = normalize_experiment_track(experiment_track)
    candidates: list[Path] = []

    if track == "stable_release":
        candidates.extend(
            [
                Config.STABLE_RELEASE_LABELS_DIR,
                Config.STABLE_RELEASE_LEGACY_LABELS_DIR,
                Config.STABLE_RELEASE_TASK_DIR,
            ]
        )
    elif track == "legacy_archive":
        candidates.extend(
            [
                Config.LEGACY_BASELINE_TASK_DIR / "labels",
                Config.LEGACY_BASELINE_TASK_DIR,
                Config.TRAIN_DIR,
            ]
        )
    else:
        candidates.append(Config.TRAIN_DIR)

    seen = set()
    existing: list[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists():
            existing.append(path)
    return existing


def select_input_file(experiment_track: str, default_input: str = None):
    dirs = _candidate_input_dirs_for_track(experiment_track)
    files: list[Path] = []
    for directory in dirs:
        files.extend(sorted(directory.glob("*.xlsx")))

    seen = set()
    unique_files: list[Path] = []
    for file_path in files:
        resolved = file_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_files.append(file_path)

    unique_files.sort(key=lambda item: item.name.lower())

    if not unique_files:
        if default_input:
            print(f"No Excel files found for track={experiment_track}, using preset input: {default_input}")
            return default_input
        print(f"No Excel files found for track={experiment_track}.")
        return None

    print(f"\nFound {len(unique_files)} Excel files for track={experiment_track}:")
    for index, file_path in enumerate(unique_files, start=1):
        print(f"{index}: {file_path}")
    if default_input:
        print(f"Preset input: {default_input}")

    while True:
        choice = input("\nSelect input file number (or press Enter to keep preset/default): ").strip()
        if not choice:
            return default_input
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(unique_files):
                return str(unique_files[idx])
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


def normalize_experiment_track(track: str) -> str:
    value = str(track or "").strip().lower()
    if value not in Config.EXPERIMENT_TRACKS:
        raise ValueError(
            f"Invalid experiment track: {track}. Available tracks: {', '.join(Config.EXPERIMENT_TRACKS)}"
        )
    return value


def normalize_order_id(order_id: str | None) -> str:
    return str(order_id or "canonical_title_order").strip() or "canonical_title_order"


def default_train_input() -> str | None:
    resolved = Config.default_train_input_file()
    if resolved is None:
        return None
    return str(resolved)


def is_train_scoped_input(input_value: str | None) -> bool:
    if not input_value:
        return False
    try:
        input_path = Path(input_value).resolve()
        train_root = Config.TRAIN_DIR.resolve()
    except FileNotFoundError:
        return False
    return input_path == train_root or train_root in input_path.parents


def infer_experiment_track(explicit_track: str | None, input_value: str | None) -> str:
    if explicit_track:
        return normalize_experiment_track(explicit_track)
    if is_train_scoped_input(input_value):
        return "research_matrix"
    return "stable_release"


def default_input_for_track(experiment_track: str) -> str | None:
    if experiment_track == "stable_release" and Config.STABLE_RELEASE_LABEL_FILE.exists():
        return str(Config.STABLE_RELEASE_LABEL_FILE)
    return default_train_input()


def resolve_dataset_id(input_path: Path, dataset_id: str | None, experiment_track: str) -> str:
    if dataset_id:
        return dataset_id
    if experiment_track == "stable_release":
        return Config.STABLE_RELEASE_DATASET_ID
    parents = input_path.resolve().parents
    data_root = Config.DATA_DIR.resolve()
    if data_root in parents:
        for parent in parents:
            if parent.parent == data_root:
                if parent.name.lower() == "train":
                    continue
                return parent.name
    return input_path.stem


def resolve_truth_file(input_path: Path, truth_file: str | None, dataset_id: str, experiment_track: str) -> str:
    if truth_file:
        return str(Path(truth_file).resolve())
    if experiment_track == "stable_release" and Config.STABLE_RELEASE_LABEL_FILE.exists():
        return str(Config.STABLE_RELEASE_LABEL_FILE.resolve())
    if input_path.parent.name == "labels":
        return str(input_path.resolve())
    for candidate in (
        Config.DATA_DIR / dataset_id / "input" / "labels",
        Config.DATA_DIR / dataset_id / "labels",
    ):
        if candidate.exists():
            files = sorted(candidate.glob("*.xlsx"))
            if len(files) == 1:
                return str(files[0].resolve())
    return ""


def determine_session_policy(
    task_mode: TaskType,
    urban_method: UrbanMethod,
    explicit_policy: str | None,
) -> str:
    if explicit_policy:
        return explicit_policy
    if task_mode == TaskType.SPATIAL:
        return "per_paper_isolated"
    if urban_method == UrbanMethod.PURE_LLM_API:
        return "cross_paper_long_context"
    return "per_paper_isolated"


def validate_session_policy(value: str) -> str:
    policy = str(value or "").strip()
    allowed = {"per_paper_isolated", "cross_paper_long_context"}
    if policy not in allowed:
        raise ValueError(f"Invalid session policy: {value}. Available: {', '.join(sorted(allowed))}")
    return policy


def validate_stable_release_contract(
    *,
    input_path: Path,
    dataset_id: str,
    experiment_track: str,
    task_mode: TaskType,
    urban_method: UrbanMethod,
    hybrid_llm_assist_enabled: bool,
):
    if experiment_track != "stable_release":
        return
    if dataset_id != Config.STABLE_RELEASE_DATASET_ID:
        raise ValueError(
            f"stable_release requires dataset_id={Config.STABLE_RELEASE_DATASET_ID}, got {dataset_id}"
        )
    stable_root = Config.STABLE_RELEASE_TASK_DIR.resolve()
    resolved_input = input_path.resolve()
    if stable_root not in resolved_input.parents and resolved_input != Config.STABLE_RELEASE_LABEL_FILE.resolve():
        raise ValueError(f"stable_release input must come from {stable_root}, got {resolved_input}")
    if task_mode != TaskType.URBAN_RENEWAL:
        raise ValueError("stable_release only supports task_mode=urban_renewal. Use research_matrix for spatial or both.")
    if urban_method != UrbanMethod.THREE_STAGE_HYBRID or not hybrid_llm_assist_enabled:
        raise ValueError(
            "stable_release only supports urban_method=three_stage_hybrid with --hybrid-llm-assist on."
        )


def build_run_context(
    *,
    experiment_track: str,
    dataset_id: str,
    truth_file: str,
    task_mode: TaskType,
    urban_method: UrbanMethod,
    hybrid_llm_assist_enabled: bool,
    session_policy: str,
    order_id: str,
    order_seed: int | None,
    max_samples_per_window: int,
    dynamic_topics_enabled: bool,
    dynamic_topics_include_full_corpus: bool,
    dynamic_binary_refinement_enabled: bool,
    dynamic_binary_refinement_unknown_only: bool,
    dynamic_binary_refinement_allow_flip: bool,
) -> dict:
    return {
        "experiment_track": experiment_track,
        "dataset_id": dataset_id,
        "truth_file": truth_file,
        "task_mode": task_mode.value,
        "urban_method": urban_method.value,
        "hybrid_llm_assist_enabled": bool(hybrid_llm_assist_enabled),
        "session_policy": session_policy,
        "order_id": order_id,
        "order_seed": order_seed,
        "max_samples_per_window": int(max_samples_per_window),
        "dynamic_topics_enabled": bool(dynamic_topics_enabled),
        "dynamic_topics_include_full_corpus": bool(dynamic_topics_include_full_corpus),
        "dynamic_binary_refinement_enabled": bool(dynamic_binary_refinement_enabled),
        "dynamic_binary_refinement_unknown_only": bool(dynamic_binary_refinement_unknown_only),
        "dynamic_binary_refinement_allow_flip": bool(dynamic_binary_refinement_allow_flip),
    }


def _manifest_context_for_output(run_context: dict, pred_scope: str) -> dict:
    return {
        **run_context,
        "pred_scope": pred_scope,
    }


def write_manifests_for_results(
    *,
    task_mode: TaskType,
    input_file: str,
    urban_definition: PromptStrategyDefinition,
    spatial_definition: PromptStrategyDefinition,
    results,
    run_context: dict,
):
    registry = load_prompt_strategy_registry()
    urban_snapshot = build_strategy_snapshot(
        registry=registry,
        template_root=TEMPLATE_ROOT,
        theme="urban_renewal",
        strategy_or_alias=urban_definition.name,
    )
    spatial_snapshot = build_strategy_snapshot(
        registry=registry,
        template_root=TEMPLATE_ROOT,
        theme="spatial",
        strategy_or_alias=spatial_definition.name,
    )
    runtime_context = {
        "python_version": f"{sys.version_info[0]}.{sys.version_info[1]}",
        "python_executable": sys.executable,
        "entrypoint": str(Path(__file__).resolve()),
    }

    if task_mode == TaskType.URBAN_RENEWAL:
        manifest = build_run_prompt_manifest(
            task_mode=task_mode.value,
            active_tasks=["urban_renewal"],
            input_file=Path(input_file),
            registry_path=REGISTRY_PATH,
            strategy_snapshots={"urban_renewal": urban_snapshot},
            experiment_context=_manifest_context_for_output(run_context, "urban_renewal"),
            runtime_context=runtime_context,
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
            experiment_context=_manifest_context_for_output(run_context, "spatial"),
            runtime_context=runtime_context,
        )
        write_prompt_manifest(results, manifest)
        return

    urban_manifest = build_run_prompt_manifest(
        task_mode=task_mode.value,
        active_tasks=["urban_renewal"],
        input_file=Path(input_file),
        registry_path=REGISTRY_PATH,
        strategy_snapshots={"urban_renewal": urban_snapshot},
        experiment_context=_manifest_context_for_output(run_context, "urban_renewal"),
        runtime_context=runtime_context,
    )
    spatial_manifest = build_run_prompt_manifest(
        task_mode=task_mode.value,
        active_tasks=["spatial"],
        input_file=Path(input_file),
        registry_path=REGISTRY_PATH,
        strategy_snapshots={"spatial": spatial_snapshot},
        experiment_context=_manifest_context_for_output(run_context, "spatial"),
        runtime_context=runtime_context,
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
        experiment_context=_manifest_context_for_output(run_context, "merged"),
        runtime_context=runtime_context,
    )
    write_prompt_manifest(results["urban_renewal"], urban_manifest)
    write_prompt_manifest(results["spatial"], spatial_manifest)
    if results.get("merged"):
        write_prompt_manifest(results["merged"], merged_manifest)


def resolve_selected_shots(
    args,
    strategy_registry: PromptStrategyRegistry,
    urban_enabled,
    spatial_enabled,
) -> tuple[str, str]:
    urban_default = Config.DEFAULT_SHOT_MODE if Config.DEFAULT_SHOT_MODE in urban_enabled else urban_enabled[0]
    spatial_default = Config.DEFAULT_SHOT_MODE if Config.DEFAULT_SHOT_MODE in spatial_enabled else spatial_enabled[0]

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
        return urban_shot, spatial_default

    if args.task == TaskType.SPATIAL:
        chosen = args.spatial_shot or args.shot
        spatial_shot = (
            normalize_shot_mode(
                chosen,
                strategy_registry,
                spatial_enabled,
                theme="spatial",
                allow_candidate=args.allow_candidate,
            )
            if chosen
            else (
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
        )
        return urban_default, spatial_shot

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

    spatial_shot = (
        normalize_shot_mode(
            spatial_chosen,
            strategy_registry,
            spatial_enabled,
            theme="spatial",
            allow_candidate=args.allow_candidate,
        )
        if spatial_chosen
        else (
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
    )
    return urban_shot, spatial_shot


def print_run_configuration(
    args,
    *,
    dataset_id: str,
    truth_file: str,
    session_policy: str,
    urban_definition: PromptStrategyDefinition,
    spatial_definition: PromptStrategyDefinition,
    urban_shot: str,
    spatial_shot: str,
) -> None:
    print(f"\n{'=' * 60}")
    print("Urban Renovation Literature Auto-Identification System")
    print(f"{'=' * 60}")
    print(f"Input: {args.input}")
    print(f"Experiment Track: {args.experiment_track}")
    print(f"Dataset ID: {dataset_id}")
    print(f"Truth File: {truth_file or 'UNBOUND'}")
    print(f"Session Policy: {session_policy}")
    print(f"Order ID: {args.order_id}")
    print(f"Order Seed: {args.order_seed if args.order_seed is not None else 'N/A'}")
    print(f"Max Samples / Window: {args.max_samples_per_window}")
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


def run_selected_task(router: TaskRouter, args, run_context: dict) -> dict:
    if args.task == TaskType.URBAN_RENEWAL:
        print("Running Urban Renewal Classification only...")
        return router.run_urban_renewal(
            input_file=args.input,
            output_file=args.output,
            limit=args.limit,
            run_context=run_context,
        )
    if args.task == TaskType.SPATIAL:
        print("Running Spatial Attribute Extraction only...")
        return router.run_spatial(
            input_file=args.input,
            output_file=args.output,
            limit=args.limit,
            run_context=run_context,
        )

    print("Running both tasks (Urban Renewal + Spatial)...")
    result = router.run_both(
        input_file=args.input,
        output_file=args.output,
        limit=args.limit,
        run_context=run_context,
    )
    print(f"\n{'=' * 60}")
    print("Results:")
    print(f"  Urban Renewal: {result['urban_renewal']}")
    print(f"  Spatial: {result['spatial']}")
    if result.get("merged"):
        print(f"  Merged: {result['merged']}")
    return result


def build_argument_parser() -> argparse.ArgumentParser:
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
        "--experiment-track",
        choices=list(Config.EXPERIMENT_TRACKS),
        default=None,
        help="Experiment governance track",
    )
    parser.add_argument("--dataset-id", default=None, help="Dataset identifier for manifest/comparability")
    parser.add_argument("--truth-file", default=None, help="Ground-truth workbook path for strict evaluation binding")
    parser.add_argument(
        "--session-policy",
        choices=["per_paper_isolated", "cross_paper_long_context"],
        default=None,
        help="Session reuse policy for LLM-backed urban runs",
    )
    parser.add_argument(
        "--order-id",
        default="canonical_title_order",
        help="Input ordering identifier recorded into the manifest",
    )
    parser.add_argument("--order-seed", type=int, default=None, help="Shuffle seed recorded into the manifest")
    parser.add_argument(
        "--max-samples-per-window",
        type=int,
        default=50,
        help="Maximum papers per long-context window for LLM-backed urban runs",
    )
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
        "--dynamic-topics",
        choices=["on", "off"],
        default=None,
        help="Enable offline dynamic topic discovery post-processing (default: off)",
    )
    parser.add_argument(
        "--dynamic-topics-full-corpus",
        action="store_true",
        help="When enabling dynamic topics, cluster the full corpus as background evidence",
    )
    parser.add_argument(
        "--dynamic-binary-refine",
        choices=["on", "off"],
        default=None,
        help="Enable dynamic-topic-driven binary refinement (default: off; implies --dynamic-topics on)",
    )
    parser.add_argument(
        "--dynamic-binary-unknown-only",
        action="store_true",
        help="Only refine rows whose binary label is unknown/empty",
    )
    parser.add_argument(
        "--dynamic-binary-allow-flip",
        action="store_true",
        help="Allow refining already-0/1 labels when near-threshold or review-triggered",
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
    return parser


def parse_args():
    return build_argument_parser().parse_args()


def prepare_experiment_args(args) -> None:
    preset_input = args.input or default_train_input()
    explicit_track = args.experiment_track
    args.experiment_track = infer_experiment_track(explicit_track, preset_input)
    auto_train_track = explicit_track is None and is_train_scoped_input(preset_input)
    if not args.non_interactive and not auto_train_track:
        args.experiment_track = select_experiment_track(default_track=args.experiment_track)
    args.order_id = normalize_order_id(args.order_id)
    if args.max_samples_per_window <= 0:
        raise ValueError("--max-samples-per-window must be positive")


def load_selectable_modes(strategy_registry: PromptStrategyRegistry, args) -> tuple[list[str], list[str]]:
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
        raise ValueError("no selectable urban_renewal strategies in registry.")
    if not spatial_enabled:
        raise ValueError("no selectable spatial strategies in registry.")
    return urban_enabled, spatial_enabled


def choose_task_mode(args) -> None:
    default_task = TaskType(args.task) if args.task else TaskType.BOTH
    args.task = default_task if args.non_interactive else select_task_mode(default_task=default_task)


def move_invalid_stable_release_task(args) -> None:
    if not args.non_interactive and args.experiment_track == "stable_release" and args.task != TaskType.URBAN_RENEWAL:
        print("[WARN] stable_release only supports task=urban_renewal. Switching experiment track to research_matrix.")
        args.experiment_track = "research_matrix"


def configure_urban_runtime(args) -> None:
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

    if not args.non_interactive and args.experiment_track == "stable_release":
        if args.urban_method != UrbanMethod.THREE_STAGE_HYBRID or not args.hybrid_llm_assist:
            print(
                "[WARN] stable_release requires urban_method=three_stage_hybrid with --hybrid-llm-assist on. "
                "Switching experiment track to research_matrix."
            )
            args.experiment_track = "research_matrix"


def validate_api_access(args) -> None:
    if task_requires_api_key(
        args.task,
        args.urban_method,
        hybrid_llm_assist_enabled=args.hybrid_llm_assist,
    ) and not Config.API_KEY:
        raise ValueError(
            "LLM API key not found, but the selected task configuration requires LLM access.\n"
            "Please create a .env file in the project root with DEEPSEEK_API_KEY=your_key"
        )


def configure_task_runtime(args) -> None:
    choose_task_mode(args)
    move_invalid_stable_release_task(args)
    configure_urban_runtime(args)
    validate_api_access(args)


def resolve_run_strategies(args, strategy_registry, urban_enabled, spatial_enabled):
    urban_shot, spatial_shot = resolve_selected_shots(
        args,
        strategy_registry,
        urban_enabled,
        spatial_enabled,
    )
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
    return urban_shot, spatial_shot, urban_definition, spatial_definition


def select_run_input(args) -> Path | None:
    preset_input = args.input
    if not preset_input:
        preset_input = default_input_for_track(args.experiment_track)
    args.input = preset_input if args.non_interactive else select_input_file(
        args.experiment_track,
        default_input=preset_input,
    )
    if args.input:
        print(f"Selected input file: {args.input}")
    else:
        print("No input file selected or default file not found.")
        return None

    input_path = Path(args.input).resolve()
    if not args.non_interactive and args.experiment_track == "stable_release":
        stable_root = Config.STABLE_RELEASE_TASK_DIR.resolve()
        resolved_input = input_path.resolve()
        if stable_root not in resolved_input.parents and resolved_input != Config.STABLE_RELEASE_LABEL_FILE.resolve():
            print(
                f"[WARN] stable_release input must come from {stable_root}. "
                "Switching experiment track to research_matrix."
            )
            args.experiment_track = "research_matrix"
    return input_path


def build_execution_context(args, input_path: Path):
    dataset_id = resolve_dataset_id(input_path, args.dataset_id, args.experiment_track)
    truth_file = resolve_truth_file(input_path, args.truth_file, dataset_id, args.experiment_track)
    session_policy = validate_session_policy(
        determine_session_policy(
            args.task,
            args.urban_method,
            args.session_policy,
        )
    )
    validate_stable_release_contract(
        input_path=input_path,
        dataset_id=dataset_id,
        experiment_track=args.experiment_track,
        task_mode=args.task,
        urban_method=args.urban_method,
        hybrid_llm_assist_enabled=args.hybrid_llm_assist,
    )

    def _resolve_toggle(raw: object, default: bool = False) -> bool:
        if raw in ("", None):
            return bool(default)
        if isinstance(raw, bool):
            return bool(raw)
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    dynamic_topics_enabled = _resolve_toggle(args.dynamic_topics, default=False)
    dynamic_binary_refinement_enabled = _resolve_toggle(args.dynamic_binary_refine, default=False)
    if dynamic_binary_refinement_enabled:
        dynamic_topics_enabled = True

    dynamic_binary_allow_flip = bool(args.dynamic_binary_allow_flip)
    dynamic_binary_unknown_only = True
    if dynamic_binary_allow_flip:
        dynamic_binary_unknown_only = False
    if bool(args.dynamic_binary_unknown_only):
        dynamic_binary_unknown_only = True

    run_context = build_run_context(
        experiment_track=args.experiment_track,
        dataset_id=dataset_id,
        truth_file=truth_file,
        task_mode=args.task,
        urban_method=args.urban_method,
        hybrid_llm_assist_enabled=args.hybrid_llm_assist,
        session_policy=session_policy,
        order_id=args.order_id,
        order_seed=args.order_seed,
        max_samples_per_window=args.max_samples_per_window,
        dynamic_topics_enabled=dynamic_topics_enabled,
        dynamic_topics_include_full_corpus=bool(args.dynamic_topics_full_corpus),
        dynamic_binary_refinement_enabled=dynamic_binary_refinement_enabled,
        dynamic_binary_refinement_unknown_only=bool(dynamic_binary_unknown_only),
        dynamic_binary_refinement_allow_flip=bool(dynamic_binary_allow_flip),
    )
    return dataset_id, truth_file, session_policy, run_context


def finalize_output_args(args) -> None:
    if not args.non_interactive:
        args.output = select_output_mode(task=args.task, preset_output=args.output)

    if args.strategy is not None:
        print("[WARN] --strategy is deprecated. Use --task instead.")
        print("[WARN] Ignoring --strategy and using --task mode.")


def build_task_router(args, urban_definition, spatial_definition) -> TaskRouter:
    return TaskRouter(
        shot_mode=urban_definition.name,
        urban_shot_mode=urban_definition.name,
        spatial_shot_mode=spatial_definition.name,
        urban_method=args.urban_method,
        hybrid_llm_assist_enabled=args.hybrid_llm_assist,
    )


def main():
    Config.load_env()
    Config.validate_runtime_environment(require_py313=True, warn_on_minor_drift=True)

    try:
        strategy_registry = load_prompt_strategy_registry()
    except Exception as exc:
        print(f"Error: failed to load strategy registry: {exc}")
        return

    args = parse_args()
    prepare_experiment_args(args)
    try:
        urban_enabled, spatial_enabled = load_selectable_modes(strategy_registry, args)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    print_startup_overview(urban_enabled, spatial_enabled)

    try:
        configure_task_runtime(args)
        urban_shot, spatial_shot, urban_definition, spatial_definition = resolve_run_strategies(
            args,
            strategy_registry,
            urban_enabled,
            spatial_enabled,
        )
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    input_path = select_run_input(args)
    if input_path is None:
        return

    dataset_id, truth_file, session_policy, run_context = build_execution_context(args, input_path)
    finalize_output_args(args)
    print_run_configuration(
        args,
        dataset_id=dataset_id,
        truth_file=truth_file,
        session_policy=session_policy,
        urban_definition=urban_definition,
        spatial_definition=spatial_definition,
        urban_shot=urban_shot,
        spatial_shot=spatial_shot,
    )

    try:
        router = build_task_router(args, urban_definition, spatial_definition)
        result = run_selected_task(router, args, run_context)
        write_manifests_for_results(
            task_mode=args.task,
            input_file=args.input,
            urban_definition=urban_definition,
            spatial_definition=spatial_definition,
            results=result,
            run_context=run_context,
        )
    except Exception as exc:
        print(f"Error: run failed: {exc}")
        raise

    print(f"{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    main()
