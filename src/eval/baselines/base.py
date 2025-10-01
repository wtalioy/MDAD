import shutil
from typing import List, Optional
import librosa
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
from config import Label

class Baseline:
    def __init__(self, device: str = "cuda", **kwargs):
        self.name = None
        self.device = device
        self.supported_metrics = ["eer"]
    
    def _load_model_config(self, model_dir: str, model_name: Optional[str] = None) -> dict:
        if model_name is None:
            config_path = os.path.join(model_dir, "config", "model.yaml")
        else:
            config_path = os.path.join(model_dir, "config", f"{model_name}.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    def _load_train_config(self, model_dir: str, dataset_name: str) -> dict:
        config_path = os.path.join(model_dir, "config", f"train_{dataset_name.lower()}.yaml")
        if not os.path.exists(config_path):
            config_path = os.path.join(model_dir, "config", "train_default.yaml")
            shutil.copy(config_path, os.path.join(model_dir, "config", f"train_{dataset_name.lower()}.yaml"))
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    def _prepare_loader(self, data: List[np.ndarray], labels: List[Label], batch_size: int = 128, shuffle: bool = True, drop_last: bool = True, num_workers: int = 8):
        def pad(x, max_len=64600):
            x_len = x.shape[0]
            if x_len >= max_len:
                return x[:max_len]
            # need to pad
            num_repeats = int(max_len / x_len) + 1
            padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
            return padded_x

        class CustomDataset(Dataset):
            def __init__(self, data, labels):
                self.data = data
                self.labels = labels
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                x = self.data[idx]
                x_pad = pad(x)
                x_inp= torch.from_numpy(x_pad).float()
                y = self.labels[idx]
                return x_inp, y

        dataset = CustomDataset(data, labels)
        if len(data) < batch_size:
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            pin_memory=True if self.device == "cuda" else False,
            persistent_workers=True if num_workers > 0 else False,
            )
        return loader

    def evaluate(self, data: List[np.ndarray], labels: List[Label], metrics: List[str], sr: int = 16000, in_domain: bool = False, dataset_name: Optional[str] = None, ref_data: Optional[List[np.ndarray]] = None, ref_labels: Optional[List[Label]] = None) -> dict:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def train(self, train_data: List[np.ndarray], train_labels: List[Label], eval_data: List[np.ndarray], eval_labels: List[Label], dataset_name: str, ref_data: Optional[List[np.ndarray]] = None, ref_labels: Optional[List[Label]] = None, sr: int = 16000):
        raise NotImplementedError("This method should be overridden by subclasses.")