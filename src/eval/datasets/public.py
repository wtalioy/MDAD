import csv
import os
import numpy as np
from .base import BaseDataset
from config import Label

class PublicFigures(BaseDataset):
    def __init__(self, data_dir=None):
        super().__init__(data_dir or "data/PublicFigures")

    def _load_meta(self):
        with open(os.path.join(self.data_dir, 'meta.csv'), 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            file_paths = []
            labels = []
            for row in reader:
                file_path = str(os.path.join(self.data_dir, row['file']))
                label = Label.real.value if row['label'] == 'bona-fide' else Label.fake.value
                file_paths.append(file_path)
                labels.append(label)
        return file_paths, np.array(labels)