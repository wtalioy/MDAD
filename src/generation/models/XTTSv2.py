import numpy as np
import TTS
from TTS.api import TTS
from .base import BaseTTS

class XTTSv2(BaseTTS):
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2"):
        super().__init__()
        self.tts = TTS(model_name=model_name).cuda()
        assert self.tts.synthesizer is not None, "TTS synthesizer is not initialized."
        self.sample_rate = int(self.tts.synthesizer.output_sample_rate)

    def infer(self, text: str, ref_audio: str, language="en"):
        wav = self.tts.tts(text=text, speaker_wav=ref_audio, language=language)
        return np.array(wav), self.sample_rate