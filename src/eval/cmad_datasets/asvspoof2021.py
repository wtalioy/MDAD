from typing import List
import os
import json
import soundfile as sf
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from .base import BaseDataset
from baselines import Baseline
from config import Label

class ASVspoof2021(BaseDataset):
    def __init__(self, data_dir=None, subset="LA", *args, **kwargs):
        self.subset = subset
        data_dir = os.path.join(data_dir or "data/ASVspoof2021", f"ASVspoof2021_{subset}_eval")
        super().__init__(data_dir)
        self.name = f"ASVspoof2021_{subset}"

    def _load_meta(self):
        paths_meta = os.path.join(self.data_dir, "audio_paths.json")
        labels_meta = os.path.join(self.data_dir, "labels.json")
        if not os.path.exists(paths_meta) or not os.path.exists(labels_meta):
            os.makedirs(self.data_dir, exist_ok=True)
            file_paths = []
            labels = []
            hf_dataset = load_dataset(f"MoaazTalab/ASVspoof_2021_{self.subset}_Balanced_Normalized", split="test")
            for i, item in enumerate(tqdm(hf_dataset, desc="Loading dataset")):
                try:
                    wav, sr = item["audio"]["array"], item["audio"]["sampling_rate"]
                    audio_path = os.path.join(self.data_dir, f"{i}.wav")
                    sf.write(audio_path, wav, sr)
                    label = Label.fake.value if item["label"] == 0 else Label.real.value
                    file_paths.append(audio_path)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading item {i}: {e}")
                    continue
            with open(paths_meta, "w", encoding="utf-8") as f:
                json.dump(file_paths, f)
            with open(labels_meta, "w", encoding="utf-8") as f:
                json.dump(labels, f)
        else:
            with open(paths_meta, "r", encoding="utf-8") as f:
                file_paths = json.load(f)
            with open(labels_meta, "r", encoding="utf-8") as f:
                labels = json.load(f)
        return file_paths, np.array(labels)

    def train(self, baseline: Baseline) -> str:
        raise NotImplementedError("Training is not supported for ASVspoof2021")

    def evaluate(self, baseline: Baseline, metrics: List[str], in_domain: bool = False) -> dict:
        return baseline.evaluate(data=self.data, labels=self.labels, metrics=metrics, in_domain=False, dataset_name=self.name)