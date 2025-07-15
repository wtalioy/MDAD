from .base import Baseline
from .aasist.aasist import AASIST

BASELINE_MAP = {
    "aasist": AASIST,
}

__all__ = ["Baseline", "BASELINE_MAP"]