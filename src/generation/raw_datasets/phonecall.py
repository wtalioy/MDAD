import os
from typing import List
from models import BaseTTS, BaseVC
from .base import BaseRawDataset

class PhoneCall(BaseRawDataset):
    def __init__(self, data_dir=None, subset="en", *args, **kwargs):
        super().__init__(os.path.join(data_dir or "data/PhoneCall", subset), *args, **kwargs)
        self.subset = subset

    def generate(self, tts_models: List[BaseTTS], vc_models: List[BaseVC] = [], *args, **kwargs):
        super().generate(tts_models, vc_models, language=self.subset, use_case="phonecall", *args, **kwargs)