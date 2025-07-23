from .base import BaseTTS
from .vits.vits import VITS
from .xttsv2 import XTTSv2
from .yourtts import YourTTS
from .tacotron2 import Tacotron2
from .bark import Bark
from .melotts import MeloTTS
from .elevenlabs_tts import ElevenLabsTTS
from .geminitts import GeminiTTS
from .gpt4omini import GPT4oMiniTTS

TTS_MODEL_MAP = {
    "vits": VITS,
    "xttsv2": XTTSv2,
    "yourtts": YourTTS,
    "tacotron2": Tacotron2,
    "bark": Bark,
    "melotts": MeloTTS,
    "elevenlabs": ElevenLabsTTS,
    "geminitts": GeminiTTS,
    "gpt4omini": GPT4oMiniTTS,
}

__all__ = ["BaseTTS", "TTS_MODEL_MAP"]