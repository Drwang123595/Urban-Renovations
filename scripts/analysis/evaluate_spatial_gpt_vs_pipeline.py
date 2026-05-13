import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._compat import load_script_module

_module = load_script_module(__name__, "scripts.analysis.spatial.evaluate_gpt_vs_pipeline", globals())

if __name__ == "__main__":
    raise SystemExit(_module.main())
