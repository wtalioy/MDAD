from .base import Baseline
from .aasist.aasist import AASIST
from .ardetect.ardetect import ARDetect

BASELINE_MAP = {
    "aasist": AASIST,
    "ardetect": ARDetect,
}

__all__ = ["Baseline", "BASELINE_MAP"]