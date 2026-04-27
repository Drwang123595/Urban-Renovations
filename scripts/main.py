import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._compat import load_script_module

_module = load_script_module(__name__, "scripts.pipeline.main_py313", globals())

if __name__ == "__main__":
    print("[WARN] scripts/main.py is a legacy compatibility wrapper. Use scripts/pipeline/main_py313.py.")
    _module.main()
