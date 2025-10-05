import os
from .base import BaseDataset

class PhoneCall(BaseDataset):
    def __init__(self, data_dir=None, subset=None, *args, **kwargs):
        data_dir = os.path.join(data_dir or "data", "PhoneCall", subset or "")
        super().__init__(data_dir)
        if subset:
            self.name = f"PhoneCall-{subset}"
        else:
            self.name = "PhoneCall"