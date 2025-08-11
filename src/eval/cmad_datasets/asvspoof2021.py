from typing import List
import os
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from .base import BaseDataset
from baselines import Baseline
from config import Label

class ASVspoof2021(BaseDataset):
    def __init__(self, subset="LA", *args, **kwargs):
        self.subset = subset
        super().__init__()
        self.name = f"ASVspoof2021_{subset}"

    def _load_meta(self, split: str = "test"):
        hf_dataset = load_dataset(f"MoaazTalab/ASVspoof_2021_{self.subset}_Balanced_Normalized", split=split)
        data = []
        labels = []
        for item in tqdm(hf_dataset, desc="Loading dataset"):
            try:
                audio, _ = item["audio"]["array"], item["audio"]["sampling_rate"]
                data.append(audio)
                labels.append(Label.fake if item["label"] == 0 else Label.real)
            except Exception as e:
                print(f"Error loading item {item}: {e}")
                continue
        return data, np.array(labels)

    def train(self, baseline: Baseline) -> str:
        raise NotImplementedError("Training is not supported for ASVspoof2021")