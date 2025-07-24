from .base import Baseline
from .aasist.aasist import AASIST
from .ardetect.ardetect import ARDetect
from .TSSDNet.tssdnet import TSSDNet

BASELINE_MAP = {
    "aasist": AASIST,
    "ardetect": ARDetect,
    "tssdnet": TSSDNet,
}

__all__ = ["Baseline", "BASELINE_MAP"]