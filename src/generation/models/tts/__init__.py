from .base import BaseTTS
from .vits.vits import VITS
from .xttsv2 import XTTSv2
from .yourtts import YourTTS
from .tacotron2 import Tacotron2
from .bark import Bark
from .melotts import MeloTTS

TTS_MODEL_MAP = {
    "vits": VITS,
    "xttsv2": XTTSv2,
    "yourtts": YourTTS,
    "tacotron2": Tacotron2,
    "bark": Bark,
    "melotts": MeloTTS,
}

__all__ = ["BaseTTS", "TTS_MODEL_MAP"]