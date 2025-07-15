from .base import BaseTTS
from .vits.vits import VITS
from .XTTSv2 import XTTSv2
from .YourTTS import YourTTS


MODEL_MAP = {
    "vits": VITS,
    "xttsv2": XTTSv2,
    "yourtts": YourTTS,
}

__all__ = ["BaseTTS", "MODEL_MAP"]