import os
from .base import BaseSubset

class PhoneCall(BaseSubset):
    def __init__(self, data_dir=None, *args, **kwargs):
        super().__init__(os.path.join(data_dir or "data", "PhoneCall"), *args, **kwargs)
        self.name = "PhoneCall"