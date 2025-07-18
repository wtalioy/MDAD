from TTS.api import TTS
from .base import BaseVC

class KNNVC(BaseVC):
    def __init__(self, *args, **kwargs):
        self.model_name = "KNNVC"
        self.model_id = "voice_conversion_models/multilingual/multi-dataset/knnvc"
        self.model = TTS(model_name=self.model_id).cuda()
        assert self.model.voice_converter is not None, "VC voice converter is not initialized."
        self.sample_rate = int(self.model.voice_converter.output_sample_rate)

    def convert(self, source_path: str, target_path: str, output_path: str, **kwargs):
        self.model.voice_conversion_to_file(source_wav=source_path, target_wav=target_path, file_path=output_path)
