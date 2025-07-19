import os
import numpy as np
from .base import BaseDataset

class PartialFake(BaseDataset):
    def __init__(self, data_dir=None, *args, **kwargs):
        super().__init__(data_dir or "data/PartialFake", *args, **kwargs)