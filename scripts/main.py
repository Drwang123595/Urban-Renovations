from __future__ import annotations

import sys
import subprocess
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_PIPELINE_ENTRY = _PROJECT_ROOT / "scripts" / "pipeline" / "main_py313.py"
_PY313_PYTHON = _PROJECT_ROOT / ".venv-bertopic313" / "Scripts" / "python.exe"


def _is_same_file(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def _run_with_project_python(argv: list[str] | None = None) -> int | None:
    if sys.version_info[:2] == (3, 13):
        return None
    if not _PY313_PYTHON.exists():
        return None
    if _is_same_file(Path(sys.executable), _PY313_PYTHON):
        return None

    args = list(sys.argv[1:] if argv is None else argv)
    return subprocess.call([str(_PY313_PYTHON), str(_PIPELINE_ENTRY), *args])


def _run_as_script(argv: list[str] | None = None) -> int:
    project_python_result = _run_with_project_python(argv)
    if project_python_result is not None:
        return int(project_python_result)

    from scripts.pipeline import main_py313

    return int(main_py313.main(argv) or 0)


if __name__ == "__main__":
    raise SystemExit(_run_as_script())


from scripts._compat import load_script_module

_module = load_script_module(__name__, "scripts.pipeline.main_py313", globals())
