import os
from typing import List
from ..models import BaseTTS, BaseVC
from .base import BaseRawSubset

class PhoneCall(BaseRawSubset):
    def __init__(self, data_dir=None, partition="en", *args, **kwargs):
        super().__init__(os.path.join(os.path.join(data_dir or "data", "PhoneCall"), partition), *args, **kwargs)
        self.partition = partition

    def generate(self, tts_models: List[BaseTTS], vc_models: List[BaseVC] = [], *args, **kwargs):
        super().generate(tts_models, vc_models, language=self.partition, use_case="phonecall", *args, **kwargs)