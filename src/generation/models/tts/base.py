import numpy as np
from typing import Tuple

class BaseTTS:
    def __init__(self, *args, **kwargs):
        self.model_name = None

    def infer(self, text: str, *args, **kwargs) -> Tuple[np.ndarray, int]:
        raise NotImplementedError("This method should be overridden by subclasses.")