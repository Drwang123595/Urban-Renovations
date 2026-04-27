import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prompting.strategy_manager import ConsistencyReport, PromptStrategyManager


def parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise ValueError("enabled only accepts true/false")


def print_strategy(item: Dict):
    print(f"Strategy: {item.get('name')}")
    print(f"  Key: {item.get('key')}")
    print(f"  Theme: {item.get('theme')}")
    print(f"  Enabled: {item.get('enabled')}")
    print(f"  Lifecycle: {item.get('lifecycle')}")
    print(f"  Version: {item.get('version')}")
    print(f"  Owner: {item.get('owner')}")
    print(f"  Aliases: {', '.join(item.get('aliases', [])) or '-'}")
    print(f"  Template File: {item.get('template_file')}")
    print(f"  Description: {item.get('description') or '-'}")
    print(f"  Change Summary: {item.get('change_summary') or '-'}")


def print_report(report: ConsistencyReport):
    status = "PASS" if report.ok else "FAIL"
    print(f"Consistency Check: {status}")
    for item in report.diagnostics:
        print(f"- [{item.severity}] {item.code}: {item.message}")


def add_common_strategy_metadata_args(parser: argparse.ArgumentParser, *, require_lifecycle: bool = False):
    parser.add_argument("--version", required=True, help="Prompt version, e.g. 1.0.0")
    parser.add_argument("--owner", required=True, help="Prompt owner")
    parser.add_argument("--change-summary", required=True, help="One-line change summary")
    parser.add_argument(
        "--lifecycle",
        required=require_lifecycle,
        default="candidate",
        choices=["candidate", "stable", "deprecated"],
        help="Prompt lifecycle",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Governed prompt strategy manager")
    parser.add_argument(
        "--template-root",
        default=str(PROJECT_ROOT / "src" / "templates"),
        help="Template root directory, default src/templates",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List strategies")
    list_parser.add_argument("--theme", help="Filter by theme")
    list_parser.add_argument(
        "--enabled-only",
        action="store_true",
        help="Only show enabled strategies",
    )

    get_parser = subparsers.add_parser("get", help="Show strategy details")
    get_parser.add_argument("name", help="Strategy name or alias")
    get_parser.add_argument("--theme", required=True, help="Strategy theme")

    add_parser = subparsers.add_parser("add", help="Add a new strategy")
    add_parser.add_argument("name", help="Strategy name")
    add_parser.add_argument("--theme", required=True, help="Strategy theme")
    add_parser.add_argument("--template-file", help="Template file name, e.g. zero.yaml")
    add_parser.add_argument("--alias", action="append", help="Strategy alias, repeatable")
    add_parser.add_argument("--description", default="", help="Strategy description")
    add_parser.add_argument("--enabled", default="true", help="Whether the strategy is visible")
    add_parser.add_argument("--system-prompt", help="Template system_prompt content")
    add_common_strategy_metadata_args(add_parser)

    update_parser = subparsers.add_parser("update", help="Update an existing strategy")
    update_parser.add_argument("name", help="Strategy name or alias")
    update_parser.add_argument("--theme", required=True, help="Strategy theme")
    update_parser.add_argument("--template-file", help="Template file name")
    update_parser.add_argument("--alias", action="append", help="Replacement alias set, repeatable")
    update_parser.add_argument("--description", help="Updated strategy description")
    update_parser.add_argument("--enabled", help="Updated enabled state, true/false")
    update_parser.add_argument("--system-prompt", help="Updated template system_prompt content")
    update_parser.add_argument("--version", help="Updated prompt version")
    update_parser.add_argument("--owner", required=True, help="Prompt owner")
    update_parser.add_argument("--change-summary", required=True, help="One-line change summary")
    update_parser.add_argument(
        "--lifecycle",
        choices=["candidate", "stable", "deprecated"],
        help="Updated lifecycle",
    )

    promote_parser = subparsers.add_parser("promote", help="Promote a candidate strategy to stable")
    promote_parser.add_argument("name", help="Strategy name or alias")
    promote_parser.add_argument("--theme", required=True, help="Strategy theme")
    promote_parser.add_argument("--owner", required=True, help="Prompt owner")
    promote_parser.add_argument("--change-summary", required=True, help="Promotion summary")

    deprecate_parser = subparsers.add_parser("deprecate", help="Deprecate a strategy")
    deprecate_parser.add_argument("name", help="Strategy name or alias")
    deprecate_parser.add_argument("--theme", required=True, help="Strategy theme")
    deprecate_parser.add_argument("--owner", required=True, help="Prompt owner")
    deprecate_parser.add_argument("--change-summary", required=True, help="Deprecation summary")

    delete_parser = subparsers.add_parser("delete", help="Delete a strategy")
    delete_parser.add_argument("name", help="Strategy name or alias")
    delete_parser.add_argument("--theme", required=True, help="Strategy theme")
    delete_parser.add_argument(
        "--remove-templates",
        action="store_true",
        help="Permanently delete template files instead of archiving them",
    )
    delete_parser.add_argument(
        "--archive-dir",
        help="Archive directory, default src/templates/_archived",
    )

    subparsers.add_parser("check", help="Run consistency check")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    manager = PromptStrategyManager(Path(args.template_root))

    try:
        if args.command == "list":
            items = manager.list_strategies(
                theme=args.theme,
                include_disabled=not args.enabled_only,
            )
            print(f"Strategies: {len(items)}")
            for item in items:
                print_strategy(item)
        elif args.command == "get":
            item = manager.get_strategy(args.name, theme=args.theme)
            print_strategy(item)
        elif args.command == "add":
            report = manager.add_strategy(
                name=args.name,
                theme=args.theme,
                template_file=args.template_file,
                enabled=parse_bool(args.enabled) if args.enabled is not None else True,
                aliases=args.alias,
                description=args.description,
                version=args.version,
                lifecycle=args.lifecycle,
                owner=args.owner,
                change_summary=args.change_summary,
                template_payload={"system_prompt": args.system_prompt} if args.system_prompt else None,
            )
            print("Add complete")
            print_report(report)
        elif args.command == "update":
            report = manager.update_strategy(
                name_or_alias=args.name,
                theme=args.theme,
                template_file=args.template_file,
                enabled=parse_bool(args.enabled) if args.enabled is not None else None,
                aliases=args.alias,
                description=args.description,
                version=args.version,
                lifecycle=args.lifecycle,
                owner=args.owner,
                change_summary=args.change_summary,
                template_payload={"system_prompt": args.system_prompt} if args.system_prompt else None,
            )
            print("Update complete")
            print_report(report)
        elif args.command == "promote":
            report = manager.promote_strategy(
                name_or_alias=args.name,
                theme=args.theme,
                owner=args.owner,
                change_summary=args.change_summary,
            )
            print("Promote complete")
            print_report(report)
        elif args.command == "deprecate":
            report = manager.deprecate_strategy(
                name_or_alias=args.name,
                theme=args.theme,
                owner=args.owner,
                change_summary=args.change_summary,
            )
            print("Deprecate complete")
            print_report(report)
        elif args.command == "delete":
            report = manager.delete_strategy(
                name_or_alias=args.name,
                theme=args.theme,
                remove_templates=args.remove_templates,
                archive_dir=Path(args.archive_dir) if args.archive_dir else None,
            )
            print("Delete complete")
            print_report(report)
        else:
            report = manager.check_consistency()
            print_report(report)
            if not report.ok:
                raise RuntimeError("Consistency check failed")
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
