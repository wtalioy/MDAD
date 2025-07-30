import os
import numpy as np
from typing import List
from tqdm import tqdm
import json
from baselines import Baseline
from config import Label

class BaseDataset:
    def __init__(self, data_dir: str, *args, **kwargs):
        self.name = "BaseDataset"
        self.data_dir = data_dir
        self.splits = ['train', 'dev', 'test']
        self.data, self.labels = self._load_meta()

    def _load_meta(self):
        split_data = {}
        split_labels = {}
        for split in self.splits:
            with open(os.path.join(self.data_dir, f'meta_{split}.json'), 'r', encoding='utf-8') as f:
                meta = json.load(f)
            file_paths = []
            labels = []
            for item in tqdm(meta, desc=f"Loading {split} split"):
                if 'real' in item['audio']:
                    file_paths.append(os.path.join(self.data_dir, item['audio']['real']))
                    labels.append(Label.real.value)
                if 'fake' in item['audio']:
                    for fake_path in item['audio']['fake'].values():
                        file_paths.append(os.path.join(self.data_dir, fake_path))
                        labels.append(Label.fake.value)
            split_data[split] = file_paths
            split_labels[split] = np.array(labels)
        return split_data, split_labels

    def evaluate(self, baseline: Baseline, metrics: List[str], in_domain: bool = False) -> dict:
        """
        Evaluate the dataset using a baseline model and specified metrics.
        
        Args:
            baseline: The baseline model to use for evaluation
            metrics: Metric(s) to evaluate
            in_domain: Whether the evaluation is in-domain
            
        Returns:
            Dictionary containing evaluation results
        """
        return baseline.evaluate(data=self.data['test'], labels=self.labels['test'], metrics=metrics, in_domain=in_domain, dataset_name=self.name)

    def train(self, baseline: Baseline) -> str:
        """
        Train the baseline model on the dataset.

        Args:
            baseline: The baseline model to train

        Returns:
            Path to the checkpoint file
        """
        ckpt_path = baseline.train(train_data=self.data['train'], train_labels=self.labels['train'], eval_data=self.data['dev'], eval_labels=self.labels['dev'], dataset_name=self.name)
        return ckpt_path