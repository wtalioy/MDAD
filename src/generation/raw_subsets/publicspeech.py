from typing import List
import os
from ..models import BaseTTS, BaseVC
from .base import BaseRawSubset

class PublicSpeech(BaseRawSubset):
    def __init__(self, data_dir=None, *args, **kwargs):
        super().__init__(os.path.join(data_dir or "data", "PublicSpeech"), *args, **kwargs)

    def generate(self, tts_models: List[BaseTTS], vc_models: List[BaseVC] = [], *args, **kwargs):
        super().generate(tts_models, vc_models, language="en", use_case="publicspeech", *args, **kwargs)