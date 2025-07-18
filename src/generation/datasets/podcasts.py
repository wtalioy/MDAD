from typing import List, Optional
from models import BaseTTS, BaseVC
from .base import BaseRawDataset

class Podcast(BaseRawDataset):
    def __init__(self, data_dir=None):
        super().__init__(data_dir or "data/Podcasts")

    def generate(self, tts_models: List[BaseTTS], vc_models: Optional[List[BaseVC]] = None, *args, **kwargs):
        super().generate(tts_models, language="en", use_case="podcasts", *args, **kwargs)