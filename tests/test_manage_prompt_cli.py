import shlex
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.manage_prompt_strategies import build_parser


DOC_PATH = Path(__file__).resolve().parent.parent / "doc" / "提示词策略长期维护与迁移说明.md"

DOC_COMMANDS = [
    "python scripts/manage_prompt_strategies.py list --enabled-only",
    "python scripts/manage_prompt_strategies.py get few --theme urban_renewal",
    (
        "python scripts/manage_prompt_strategies.py add reflection_v2 --theme urban_renewal "
        "--template-file reflection_v2.yaml --alias self_check_v2 --description \"Reflection template\" "
        "--version 0.1.0 --owner prompt_owner --change-summary \"Add candidate reflection prompt\" "
        "--system-prompt \"system text\""
    ),
    (
        "python scripts/manage_prompt_strategies.py update few --theme urban_renewal "
        "--owner prompt_owner --change-summary \"Refresh few-shot examples\" --version 2.2.0"
    ),
    (
        "python scripts/manage_prompt_strategies.py promote few --theme urban_renewal "
        "--owner prompt_owner --change-summary \"Promote tested candidate to stable\""
    ),
    (
        "python scripts/manage_prompt_strategies.py deprecate reflection --theme urban_renewal "
        "--owner prompt_owner --change-summary \"Retire legacy reflection prompt\""
    ),
    "python scripts/manage_prompt_strategies.py delete reflection --theme urban_renewal --archive-dir src/templates/_archived",
    "python scripts/manage_prompt_strategies.py check",
]


def test_doc_examples_exist_and_parse():
    doc_text = DOC_PATH.read_text(encoding="utf-8")
    parser = build_parser()
    for command in DOC_COMMANDS:
        assert command in doc_text
        tokens = shlex.split(command)
        args = parser.parse_args(tokens[2:])
        assert args.command
