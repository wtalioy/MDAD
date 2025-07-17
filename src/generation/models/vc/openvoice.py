from TTS.api import TTS
from .base import BaseVC

class OpenVoice(BaseVC):
    def __init__(self, *args, **kwargs):
        self.model_name = "OpenVoice"
        self.model_id = "voice_conversion_models/multilingual/multi-dataset/openvoice_v2"
        self.model = TTS(model_name=self.model_id).cuda()
        assert self.model.synthesizer is not None, "TTS synthesizer is not initialized."
        self.sample_rate = int(self.model.synthesizer.output_sample_rate)

    def convert(self, source_path: str, target_path: str, output_path: str, **kwargs):
        self.model.voice_conversion_to_file(source_wav=source_path, target_wav=target_path, file_path=output_path)
