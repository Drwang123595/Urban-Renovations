from .base import ExtractionStrategy
from .single import SingleTurnStrategy
from .stepwise import StepwiseStrategy
from .stepwise_long import StepwiseLongContextStrategy
from .cot import CoTStrategy
from .reflection import ReflectionStrategy

__all__ = [
    "StrategyRegistry",
    "ExtractionStrategy",
    "SingleTurnStrategy",
    "StepwiseStrategy",
    "StepwiseLongContextStrategy",
    "CoTStrategy",
    "ReflectionStrategy",
]


def __getattr__(name):
    if name == "StrategyRegistry":
        from .registry import StrategyRegistry

        return StrategyRegistry
    raise AttributeError(name)
