from importlib import import_module as _import_module
import sys as _sys

_module = _import_module("src.tasks.merged_output")
for _name, _value in _module.__dict__.items():
    if _name not in {"__name__", "__package__", "__loader__", "__spec__", "__file__", "__cached__"}:
        globals()[_name] = _value
_sys.modules[__name__] = _module
