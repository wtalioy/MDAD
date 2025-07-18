from typing import List, Optional
from models import BaseTTS, BaseVC
from .base import BaseRawDataset

class News(BaseRawDataset):
    def __init__(self, data_dir=None):
        super().__init__(data_dir or "data/News")

    def generate(self, tts_models: List[BaseTTS], vc_models: Optional[List[BaseVC]] = None, *args, **kwargs):
        super().generate(tts_models, language="zh-cn", use_case="news", *args, **kwargs)