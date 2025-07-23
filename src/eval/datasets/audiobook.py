import os
from .base import BaseDataset

class Audiobook(BaseDataset):
    def __init__(self, data_dir=None, subset="en", *args, **kwargs):
        data_dir = os.path.join(data_dir or "data/Audiobook", subset)
        super().__init__(data_dir)