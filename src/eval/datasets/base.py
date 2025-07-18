import os
import numpy as np
from typing import List
import json
from ..baselines import Baseline
from ..config import Label

class BaseDataset:
    def __init__(self, data_dir: str, *args, **kwargs):
        self.data_dir = data_dir
        self.data, self.labels = self._load_meta()

    def _load_meta(self):
        with open(os.path.join(self.data_dir, 'meta.json'), 'r', encoding='utf-8') as f:
            meta = json.load(f)
        file_paths = []
        labels = []
        for item in meta:
            file_paths.append(os.path.join(self.data_dir, item['audio']['real']))
            labels.append(Label.real.value)
            file_paths.append(os.path.join(self.data_dir, item['audio']['fake']))
            labels.append(Label.fake.value)
        return file_paths, np.array(labels)

    def evaluate(self, baseline: Baseline, metrics: List[str]) -> dict:
        """
        Evaluate the dataset using a baseline model and specified metrics.
        
        Args:
            baseline: The baseline model to use for evaluation
            metrics: Metric(s) to evaluate
            
        Returns:
            Dictionary containing evaluation results
        """
        return baseline.evaluate(data=self.data, labels=self.labels, metrics=metrics)