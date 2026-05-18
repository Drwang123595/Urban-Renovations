from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REQUIRED_MODULES = (
    "openai",
    "pandas",
    "openpyxl",
    "pytest",
)


@dataclass(frozen=True)
class CheckResult:
    command: tuple[str, ...]
    returncode: int
    output: str


def run_command(command: Sequence[str], *, cwd: Path | None = None) -> CheckResult:
    completed = subprocess.run(
        list(command),
        cwd=str(cwd or PROJECT_ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return CheckResult(
        command=tuple(str(part) for part in command),
        returncode=int(completed.returncode),
        output=str(completed.stdout or ""),
    )


def _import_check_code(required_modules: Iterable[str]) -> str:
    modules_literal = repr(tuple(required_modules))
    return (
        "import importlib.util; missing = []\n"
        f"for name in {modules_literal}:\n"
        "    if importlib.util.find_spec(name) is None:\n"
        "        missing.append(name)\n"
        "if missing:\n"
        "    raise SystemExit('missing modules: ' + ', '.join(missing))\n"
    )


def run_environment_checks(
    *,
    python_executable: str = sys.executable,
    required_modules: Iterable[str] = DEFAULT_REQUIRED_MODULES,
) -> list[CheckResult]:
    return [
        run_command(("uv", "lock", "--check"), cwd=PROJECT_ROOT),
        run_command((python_executable, "-m", "pip", "check"), cwd=PROJECT_ROOT),
        run_command((python_executable, "-c", _import_check_code(required_modules)), cwd=PROJECT_ROOT),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the project Python environment contract.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to validate.")
    parser.add_argument(
        "--module",
        action="append",
        dest="modules",
        help="Required import module. Can be supplied multiple times.",
    )
    args = parser.parse_args(argv)

    modules = tuple(args.modules or DEFAULT_REQUIRED_MODULES)
    failed = False
    for result in run_environment_checks(python_executable=args.python, required_modules=modules):
        command_text = " ".join(result.command)
        print(f"$ {command_text}")
        if result.output.strip():
            print(result.output.rstrip())
        print(f"exit_code={result.returncode}")
        if result.returncode != 0:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
