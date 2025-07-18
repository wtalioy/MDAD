from typing import List, Optional
from models import BaseTTS, BaseVC
from .base import BaseRawDataset

class News(BaseRawDataset):
    def __init__(self, data_dir="data/News"):
        super().__init__(data_dir)

    def generate(self, tts_models: List[BaseTTS], vc_models: Optional[List[BaseVC]] = None, *args, **kwargs):
        super().generate(tts_models, language="zh-cn", use_case="news", *args, **kwargs)