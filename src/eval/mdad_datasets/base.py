import os
import json
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Dict
from functools import reduce

import librosa
import numpy as np
from tqdm import tqdm

from baselines import Baseline
from config import Label

class BaseDataset:
    def __init__(self, data_dir: Optional[str] = None, *args, **kwargs):
        self.name = "BaseDataset"
        self.data_dir = data_dir
        self.splits = ['train', 'dev', 'test']
        self.sr = 16000
        self.data, self.labels = self._load_meta()

    def _load_meta(self) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, List[Label]]]:
        split_data = {}
        split_labels = {}

        max_workers = min(32, (os.cpu_count() or 8))

        def _load_audio(path_and_label: Tuple[str, Label]) -> Tuple[np.ndarray, Label]:
            rel_path, label = path_and_label
            abs_path = os.path.join(self.data_dir, rel_path)
            audio, _ = librosa.load(abs_path, sr=self.sr)
            return audio, label

        for split in self.splits:
            split_data[split] = []
            split_labels[split] = []

            meta_path = os.path.join(self.data_dir, f"meta_{split}.json")
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            tasks = []
            for item in tqdm(meta, desc=f"Scanning {split} metadata"):
                if "real" in item["audio"]:
                    tasks.append((item["audio"]["real"], Label.real))
                if "fake" in item["audio"]:
                    for fake_path in item["audio"]["fake"].values():
                        tasks.append((fake_path, Label.fake))

            if len(tasks) == 0:
                continue

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for audio, label in tqdm(
                    executor.map(_load_audio, tasks),
                    total=len(tasks),
                    desc=f"Loading {split} audio",
                ):
                    split_data[split].append(audio)
                    split_labels[split].append(label)

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
        ref_data = None
        ref_labels = None
        if in_domain:
            data = self.data['test']
            labels = self.labels['test']
            if baseline.name == 'MKRT':
                ref_data = self.data['train'][:baseline.ref_num]
                ref_labels = self.labels['train'][:baseline.ref_num]
        else:
            data = reduce(lambda x, y: x + y, list(self.data.values()))
            labels = reduce(lambda x, y: x + y, list(self.labels.values()))
            
        return baseline.evaluate(
            data=data,
            labels=labels,
            ref_data=ref_data,
            ref_labels=ref_labels,
            metrics=metrics,
            sr=self.sr,
            in_domain=in_domain,
            dataset_name=self.name
        )

    def train(self, baseline: Baseline):
        """
        Train the baseline model on the dataset.

        Args:
            baseline: The baseline model to train

        Returns:
            Path to the checkpoint file
        """
        ref_data = None
        ref_labels = None
        if baseline.name == 'MKRT':
            train_data = self.data['train'][baseline.ref_num:]
            train_labels = self.labels['train'][baseline.ref_num:]
            ref_data = self.data['train'][:baseline.ref_num]
            ref_labels = self.labels['train'][:baseline.ref_num]
        else:
            train_data = self.data['train']
            train_labels = self.labels['train']
        eval_data = self.data['dev']
        eval_labels = self.labels['dev']
        baseline.train(
            train_data=train_data,
            train_labels=train_labels,
            eval_data=eval_data,
            eval_labels=eval_labels,
            ref_data=ref_data,
            ref_labels=ref_labels,
            dataset_name=self.name,
            sr=self.sr
        )