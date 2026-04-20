import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.main_py313 import *  # noqa: F401,F403


if __name__ == "__main__":
    print("[WARN] scripts/main.py is a legacy compatibility wrapper. Use scripts/main_py313.py.")
    main()
