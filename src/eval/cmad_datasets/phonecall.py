import os
from .base import BaseDataset

class PhoneCall(BaseDataset):
    def __init__(self, data_dir=None, subset=None, *args, **kwargs):
        self.subset = subset if subset is not None else "zh-cn"
        data_dir = os.path.join(data_dir or "data/PhoneCall", self.subset)
        super().__init__(data_dir)
        self.name = f"PhoneCall-{self.subset}"