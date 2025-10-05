import os
from .base import BaseDataset

class Audiobook(BaseDataset):
    def __init__(self, data_dir=None, *args, **kwargs):
        super().__init__(os.path.join(data_dir or "data", "Audiobook"))
        self.name = "Audiobook"