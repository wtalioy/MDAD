import os
from .base import BaseDataset

class PhoneCall(BaseDataset):
    def __init__(self, data_dir=None, split="en", *args, **kwargs):
        data_dir = os.path.join(data_dir or "data/PhoneCall", split)
        super().__init__(data_dir)