from .base import BaseDataset

class PublicFigure(BaseDataset):
    def __init__(self, data_dir=None, *args, **kwargs):
        super().__init__(data_dir or "data/PublicFigure", *args, **kwargs)
        self.name = "PublicFigure"