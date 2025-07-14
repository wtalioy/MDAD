from typing import List
import numpy as np

class Baseline:

    def evaluate(self, data: List[str], metric: str | List[str], labels: np.ndarray) -> dict:
        raise NotImplementedError("This method should be overridden by subclasses.")