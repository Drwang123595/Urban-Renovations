from typing import Dict, Type
from .base import ExtractionStrategy
from .single import SingleTurnStrategy
from .stepwise import StepwiseStrategy
from .stepwise_long import StepwiseLongContextStrategy
from .cot import CoTStrategy
from .reflection import ReflectionStrategy

class StrategyRegistry:
    _strategies: Dict[str, Type[ExtractionStrategy]] = {
        "single": SingleTurnStrategy,
        "stepwise": StepwiseStrategy,
        "stepwise_long": StepwiseLongContextStrategy,
        "cot": CoTStrategy,
        "reflection": ReflectionStrategy
    }

    @classmethod
    def get_strategy(cls, name: str) -> Type[ExtractionStrategy]:
        return cls._strategies.get(name)

    @classmethod
    def register(cls, name: str, strategy_cls: Type[ExtractionStrategy]):
        cls._strategies[name] = strategy_cls
        
    @classmethod
    def list_strategies(cls):
        return list(cls._strategies.keys())
