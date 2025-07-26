from typing import List, Optional
import numpy as np

class Baseline:
    def __init__(self, device: str = "cuda", **kwargs):
        self.device = device
        self.supported_metrics = ["eer"]

    def evaluate(self, data: List[str], labels: np.ndarray, metrics: List[str], ckpt_path: Optional[str] = None) -> dict:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def train(self, train_data: List[str], train_labels: np.ndarray, eval_data: List[str], eval_labels: np.ndarray, dataset_name: str) -> str:
        raise NotImplementedError("This method should be overridden by subclasses.")