import numpy as np
from TTS.api import TTS
from .base import BaseTTS

class Bark(BaseTTS):
    def __init__(self, *args, **kwargs):
        self.model_name = "Bark"
        self.require_vc = True
        self.model_id = "tts_models/multilingual/multi-dataset/bark"
        self.model = TTS(model_name=self.model_id).cuda()
        assert self.model.synthesizer is not None, "TTS synthesizer is not initialized."
        self.sample_rate = int(self.model.synthesizer.output_sample_rate)

    def infer(self, text: str, ref_audio: str, **kwargs):
        wav = self.model.tts(text=text, speaker_wav=ref_audio)
        return np.array(wav), self.sample_rate