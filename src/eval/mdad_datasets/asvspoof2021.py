from typing import List
import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from .base import BaseDataset
from baselines import Baseline
from config import Label

class ASVspoof2021(BaseDataset):
    def __init__(self, subset="LA", *args, **kwargs):
        self.subset = subset
        super().__init__()
        self.name = f"ASVspoof2021_{subset}"

    def _load_meta(self):
        hf_dataset = load_dataset(f"MoaazTalab/ASVspoof_2021_{self.subset}_Balanced_Normalized", split="test")
        data = []
        labels = []

        max_workers = min(32, (os.cpu_count() or 8))

        def _process(idx: int):
            try:
                item = hf_dataset[int(idx)]
                audio = item["audio"]["array"]
                label = Label.fake if item["label"] == 0 else Label.real
                return audio, label
            except Exception as e:
                print(f"Error loading item at index {idx}: {e}")
                return None

        indices = list(range(len(hf_dataset)))
        if len(indices) == 0:
            return data, np.array(labels)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for result in tqdm(executor.map(_process, indices), total=len(indices), desc="Loading dataset"):
                if result is None:
                    continue
                audio, label = result
                data.append(audio)
                labels.append(label)

        return data, np.array(labels)

    def train(self, baseline: Baseline) -> str:
        raise NotImplementedError("Training is not supported for ASVspoof2021")