from typing import List
from torch.utils.data import DataLoader

class Baseline:
    
    def eval_with(self, data: DataLoader, metric: str | List[str]) -> dict:
        raise NotImplementedError("This method should be overridden by subclasses.")