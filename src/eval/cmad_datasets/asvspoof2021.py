from typing import List
import os

from tqdm import tqdm
from .base import BaseDataset
from baselines import Baseline
from config import Label

class ASVspoof2021(BaseDataset):
    def __init__(self, data_dir=None, subset="LA", *args, **kwargs):
        data_dir = os.path.join(data_dir or "data/ASVspoof2021", f"ASVspoof2021_{subset}_eval")
        super().__init__(data_dir)
        self.name = f"ASVspoof2021_{subset}"

    def _load_meta(self):
        with open(os.path.join(self.data_dir, "trial_metadata.txt"), "r") as f:
            lines = f.readlines()
        file_paths = []
        labels = []
        for line in tqdm(lines, desc="Loading dataset"):
            line = line.strip().split(" ")
            file_paths.append(os.path.join(self.data_dir, "flac", line[1] + ".flac"))
            labels.append(Label.real.value if line[5] == "bonafide" else Label.fake.value)
        return file_paths, labels

    def train(self, baseline: Baseline) -> str:
        raise NotImplementedError("Training is not supported for ASVspoof2021")

    def evaluate(self, baseline: Baseline, metrics: List[str], in_domain: bool = False) -> dict:
        baseline.evaluate(data=self.data, labels=self.labels, metrics=metrics, in_domain=False, dataset_name=self.name)