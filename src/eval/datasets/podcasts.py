from .base import BaseDataset

class Podcasts(BaseDataset):
    def __init__(self, data_dir=None):
        super().__init__(data_dir or "data/Podcasts")