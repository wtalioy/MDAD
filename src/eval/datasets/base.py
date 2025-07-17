from typing import List
from ..baselines import Baseline

class BaseDataset:
    def __init__(self, data_dir: str = "data", *args, **kwargs):
        self.data_dir = data_dir

    def evaluate(self, baseline: Baseline, metrics: List[str]) -> dict:
        """
        Evaluate the dataset using a baseline model and specified metrics.
        
        Args:
            baseline: The baseline model to use for evaluation
            metrics: Metric(s) to evaluate
            
        Returns:
            Dictionary containing evaluation results
        """
        raise NotImplementedError("This method should be overridden by subclasses.")