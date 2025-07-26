from .base import BaseDataset

class NoisySpeech(BaseDataset):
    def __init__(self, data_dir=None, *args, **kwargs):
        super().__init__(data_dir or "data/NoisySpeech", *args, **kwargs)
        self.name = "NoisySpeech"