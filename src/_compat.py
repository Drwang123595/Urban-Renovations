from __future__ import annotations

from importlib import import_module
import sys
from types import ModuleType
from typing import Any, MutableMapping


_MODULE_METADATA = {
    "__name__",
    "__package__",
    "__loader__",
    "__spec__",
    "__file__",
    "__cached__",
}


def alias_module(
    current_name: str,
    target_name: str,
    namespace: MutableMapping[str, Any],
) -> ModuleType:
    module = import_module(target_name)
    for name, value in module.__dict__.items():
        if name not in _MODULE_METADATA:
            namespace[name] = value
    sys.modules[current_name] = module
    return module
