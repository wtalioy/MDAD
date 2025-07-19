from typing import List
from models import BaseVC
from .base import BaseRawDataset

class Movie(BaseRawDataset):
    def __init__(self, data_dir=None, *args, **kwargs):
        super().__init__(data_dir or "data/Movie", *args, **kwargs)

    def generate(self, vc_models: List[BaseVC] = [], *args, **kwargs):
        super().generate_vc_only(vc_models, *args, **kwargs)