from typing import List
import csv
import os
import librosa
from tqdm import tqdm
from .base import BaseDataset
from baselines import Baseline
from config import Label

class InTheWild(BaseDataset):
    def __init__(self, data_dir=None, *args, **kwargs):
        super().__init__(data_dir or "data/release_in_the_wild", *args, **kwargs)
        self.name = "in-the-wild"

    def _load_meta(self):
        data = []
        labels = []
        with open(os.path.join(self.data_dir, "meta.csv"), "r", encoding="utf-8") as f:
            lines = f.readlines()[1:]
            reader = csv.reader(lines)
            for row in tqdm(reader, total=len(lines), desc="Loading dataset"):
                audio, _ = librosa.load(os.path.join(self.data_dir, row[0]), sr=None)
                data.append(audio)
                labels.append(Label.real if row[-1] == "bona-fide" else Label.fake)
        return data, labels

    def train(self, baseline: Baseline) -> str:
        raise NotImplementedError("Training is not supported for InTheWild")

    def evaluate(self, baseline: Baseline, metrics: List[str], in_domain: bool = False) -> dict:
        return baseline.evaluate(data=self.data, labels=self.labels, metrics=metrics, in_domain=False, dataset_name=self.name)