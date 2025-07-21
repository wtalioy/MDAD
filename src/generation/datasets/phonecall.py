import os
from typing import List
from models import BaseTTS, BaseVC
from .base import BaseRawDataset

class PhoneCall(BaseRawDataset):
    def __init__(self, data_dir=None, split="en", *args, **kwargs):
        super().__init__(os.path.join(data_dir or "data/PhoneCall", split), *args, **kwargs)

    def generate(self, tts_models: List[BaseTTS], vc_models: List[BaseVC] = [], *args, **kwargs):
        super().generate(tts_models, vc_models, language="en", use_case="phonecall", *args, **kwargs)