from .base import BaseDataset

class Podcast(BaseDataset):
    def __init__(self, data_dir=None, *args, **kwargs):
        super().__init__(data_dir or "data/Podcast", *args, **kwargs)