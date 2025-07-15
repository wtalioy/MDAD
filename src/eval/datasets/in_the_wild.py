import csv
import os
from typing import List
import numpy as np
from .base import BaseDataset
from ..baselines import Baseline

class InTheWild(BaseDataset):
    def __init__(self):
        super().__init__()
        self.data, self.labels = self._load_meta()

    def _load_meta(self):
        with open(os.path.join(self.data_dir, 'meta.csv'), 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            file_paths = []
            labels = []
            for row in reader:
                file_path = str(os.path.join(self.data_dir, row['file']))
                label = 0 if row['label'] == 'bona-fide' else 1
                file_paths.append(file_path)
                labels.append(label)
        return file_paths, np.array(labels)

    def evaluate(self, baseline: Baseline, metric: str | List[str]) -> dict:
        return baseline.evaluate(data=self.data, labels=self.labels, metric=metric)