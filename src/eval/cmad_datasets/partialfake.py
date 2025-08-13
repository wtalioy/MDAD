import os
import json
import numpy as np
import librosa
from tqdm import tqdm
from typing import List
from concurrent.futures import ThreadPoolExecutor
from baselines import Baseline
from .base import BaseDataset
from config import Label

class PartialFake(BaseDataset):
    def __init__(self, data_dir=None, *args, **kwargs):
        super().__init__(os.path.join(data_dir or "data", "PartialFake"), *args, **kwargs)
        self.name = "PartialFake"

    def _load_meta(self):
        with open(os.path.join(self.data_dir, f'meta.json'), 'r', encoding='utf-8') as f:
                meta = json.load(f)
        source_data = {}
        source_labels = {}
        for source_name, items in meta.items():
            data = []
            labels = []

            tasks = []
            for item in items:
                if 'real' in item['audio']:
                    tasks.append((os.path.join("data", source_name, item['audio']['real']), Label.real))
                if 'fake' in item['audio']:
                    tasks.append((os.path.join(self.data_dir, item['audio']['fake']), Label.fake))

            if len(tasks) == 0:
                source_data[source_name] = data
                source_labels[source_name] = np.array(labels)
                continue

            max_workers = min(32, (os.cpu_count() or 8))

            def _load_audio(path_and_label):
                path, label = path_and_label
                try:
                    audio, _ = librosa.load(path, sr=None)
                    return audio, label
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    return None

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for result in tqdm(executor.map(_load_audio, tasks), total=len(tasks), desc=f"Loading {source_name} source"):
                    if result is None:
                        continue
                    audio, label = result
                    data.append(audio)
                    labels.append(label)
            source_data[source_name] = data
            source_labels[source_name] = np.array(labels)
        return source_data, source_labels

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
        results = {}
        for source_name, source_data in self.data.items():
            source_labels = self.labels[source_name]
            source_results = baseline.evaluate(data=source_data, labels=source_labels, metrics=metrics, in_domain=in_domain, dataset_name=self.name)
            results[source_name] = source_results
        return results