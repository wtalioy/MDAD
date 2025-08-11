import os
import librosa
import numpy as np
from typing import List, Optional
from tqdm import tqdm
import json
from baselines import Baseline
from config import Label

class BaseDataset:
    def __init__(self, data_dir: Optional[str] = None, *args, **kwargs):
        self.name = "BaseDataset"
        self.data_dir = data_dir
        self.splits = ['train', 'dev', 'test']
        self.sr = 16000

    def _load_meta(self, split: Optional[str] = None):
        if split not in self.splits:
            raise ValueError(f"Invalid split: {split}")
        meta_path = os.path.join(self.data_dir, f'meta_{split}.json') if split is not None else os.path.join(self.data_dir, 'meta.json')
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        data = []
        labels = []
        for item in tqdm(meta, desc=f"Loading {split if split is not None else 'all'} split"):
            if 'real' in item['audio']:
                audio, _ = librosa.load(os.path.join(self.data_dir, item['audio']['real']), sr=None)
                data.append(audio)
                labels.append(Label.real)
            if 'fake' in item['audio']:
                for fake_path in item['audio']['fake'].values():
                    audio, _ = librosa.load(os.path.join(self.data_dir, fake_path), sr=None)
                    data.append(audio)
                    labels.append(Label.fake)
        return data, np.array(labels)

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
        if in_domain:
            data, labels = self._load_meta(split='test')
            if baseline.name == 'ARDetect':
                ref_data, ref_labels = self._load_meta(split='train')
                return baseline.evaluate(
                    data=data,
                    labels=labels,
                    ref_data=ref_data,
                    ref_labels=ref_labels,
                    metrics=metrics,
                    sr=self.sr,
                    in_domain=True,
                    dataset_name=self.name
                )
            return baseline.evaluate(
                data=data,
                labels=labels,
                metrics=metrics,
                sr=self.sr,
                in_domain=True,
                dataset_name=self.name
            )
        else:
            data, labels = self._load_meta()
            return baseline.evaluate(
                data=data,
                labels=labels,
                metrics=metrics,
                sr=self.sr,
                in_domain=False
            )

    def train(self, baseline: Baseline):
        """
        Train the baseline model on the dataset.

        Args:
            baseline: The baseline model to train

        Returns:
            Path to the checkpoint file
        """
        train_data, train_labels = self._load_meta(split='train')
        eval_data, eval_labels = self._load_meta(split='dev')
        baseline.train(
            train_data=train_data,
            train_labels=train_labels,
            eval_data=eval_data,
            eval_labels=eval_labels,
            dataset_name=self.name
        )