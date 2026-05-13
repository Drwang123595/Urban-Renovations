"""Optional SPECTER2 document-embedding support for urban-renewal A/B tests."""

from .config import Specter2Config
from .encoder import EncodingResult, Specter2Availability, Specter2Encoder, check_availability
from .evaluator import AblationResult, AblationThresholds, evaluate_specter2_ablation

__all__ = [
    "AblationResult",
    "AblationThresholds",
    "EncodingResult",
    "Specter2Availability",
    "Specter2Config",
    "Specter2Encoder",
    "check_availability",
    "evaluate_specter2_ablation",
]
