from .base import BaseVC
from .knnvc import KNNVC
from .freevc import FreeVC
from .openvoice import OpenVoice

VC_MODEL_MAP = {
    "knnvc": KNNVC,
    "freevc": FreeVC,
    "openvoice": OpenVoice,
}

__all__ = ["BaseVC", "VC_MODEL_MAP"]