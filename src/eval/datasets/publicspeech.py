from .base import BaseDataset

class PublicSpeech(BaseDataset):
    def __init__(self, data_dir=None, *args, **kwargs):
        super().__init__(data_dir or "data/PublicSpeech", *args, **kwargs)