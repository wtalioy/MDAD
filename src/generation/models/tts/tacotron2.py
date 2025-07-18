import numpy as np
from TTS.api import TTS
from .base import BaseTTS

class Tacotron2(BaseTTS):
    def __init__(self, *args, **kwargs):
        self.model_name = "Tacotron2"
        self.require_vc = True
        self.model_id = "tts_models/zh-CN/baker/tacotron2-DDC-GST"
        self.model = TTS(model_name=self.model_id).cuda()
        assert self.model.synthesizer is not None, "TTS synthesizer is not initialized."
        self.sample_rate = int(self.model.synthesizer.output_sample_rate)

    def infer(self, text: str, **kwargs):
        wav = self.model.tts(text=text)
        return np.array(wav), self.sample_rate