from typing import List
import csv
import os
import librosa
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from .base import BaseDataset
from baselines import Baseline
from config import Label

class InTheWild(BaseDataset):
    def __init__(self, data_dir=None, *args, **kwargs):
        super().__init__(os.path.join(data_dir or "data", "release_in_the_wild"), *args, **kwargs)
        self.name = "in-the-wild"

    def _load_meta(self):
        data = []
        labels = []
        with open(os.path.join(self.data_dir, "meta.csv"), "r", encoding="utf-8") as f:
            lines = f.readlines()[1:]
            reader = csv.reader(lines)

            tasks = []
            for row in reader:
                path = os.path.join(self.data_dir, row[0])
                label = Label.real if row[-1] == "bona-fide" else Label.fake
                tasks.append((path, label))

        if len(tasks) == 0:
            return data, labels

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
            for result in tqdm(executor.map(_load_audio, tasks), total=len(tasks), desc="Loading dataset"):
                if result is None:
                    continue
                audio, label = result
                data.append(audio)
                labels.append(label)

        return data, labels

    def train(self, baseline: Baseline) -> str:
        raise NotImplementedError("Training is not supported for InTheWild")