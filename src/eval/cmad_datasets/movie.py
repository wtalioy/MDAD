import os
from .base import BaseDataset

class Movie(BaseDataset):
    def __init__(self, data_dir=None, *args, **kwargs):
        super().__init__(os.path.join(data_dir or "data", "Movie"), *args, **kwargs)
        self.name = "Movie"