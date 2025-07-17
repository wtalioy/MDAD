from typing import List
import numpy as np

class Baseline:

    def evaluate(self, data: List[str], labels: np.ndarray, metrics: List[str]) -> dict:
        raise NotImplementedError("This method should be overridden by subclasses.")