import os
from .base import BaseDataset

class Audiobook(BaseDataset):
    def __init__(self, data_dir=None, split="en"):
        data_dir = os.path.join(data_dir or "data/Audiobooks", split)
        super().__init__(data_dir)