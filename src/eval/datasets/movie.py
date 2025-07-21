from .base import BaseDataset

class Movie(BaseDataset):
    def __init__(self, data_dir=None, *args, **kwargs):
        super().__init__(data_dir or "data/Movie", *args, **kwargs)