from typing import List
from models import BaseTTS, BaseVC
from .base import BaseRawDataset

class PublicSpeech(BaseRawDataset):
    def __init__(self, data_dir=None, *args, **kwargs):
        super().__init__(data_dir or "data/PublicSpeech", *args, **kwargs)

    def generate(self, tts_models: List[BaseTTS], vc_models: List[BaseVC] = [], *args, **kwargs):
        super().generate(tts_models, vc_models, language="en", use_case="publicspeech", *args, **kwargs)