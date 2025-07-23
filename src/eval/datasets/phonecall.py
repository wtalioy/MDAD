import os
from .base import BaseDataset

class PhoneCall(BaseDataset):
    def __init__(self, data_dir=None, subset="en", *args, **kwargs):
        data_dir = os.path.join(data_dir or "data/PhoneCall", subset)
        super().__init__(data_dir)