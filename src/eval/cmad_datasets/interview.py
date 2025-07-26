from .base import BaseDataset

class Interview(BaseDataset):
    def __init__(self, data_dir=None, *args, **kwargs):
        super().__init__(data_dir or "data/Interview", *args, **kwargs)
        self.name = "Interview"