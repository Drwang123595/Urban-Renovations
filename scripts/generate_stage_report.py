import sys
from importlib import import_module
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_module = import_module("scripts.reporting.generate_stage_report")
for _name, _value in _module.__dict__.items():
    if _name not in {"__name__", "__package__", "__loader__", "__spec__", "__file__", "__cached__"}:
        globals()[_name] = _value

if __name__ != "__main__":
    sys.modules[__name__] = _module

if __name__ == "__main__":
    main()

