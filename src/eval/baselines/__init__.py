from .base import Baseline
from .aasist.aasist import AASIST, AASIST_L
from .ardetect.ardetect import ARDetect
from .TSSDNet.tssdnet import Res_TSSDNet, Inc_TSSDNet
from .RawNet2.rawnet2 import RawNet2

BASELINE_MAP = {
    "aasist": AASIST,
    "aasist-l": AASIST_L,
    "ardetect": ARDetect,
    "res-tssdnet": Res_TSSDNet,
    "inc-tssdnet": Inc_TSSDNet,
    "rawnet2": RawNet2,
}

__all__ = ["Baseline", "BASELINE_MAP"]