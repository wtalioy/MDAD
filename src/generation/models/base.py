import numpy as np
from typing import Tuple

class BaseTTS:
    def __init__(self, *args, **kwargs):
        """
        Base class for Text-to-Speech models.
        This class should be inherited by specific TTS model implementations.
        """
        pass

    def infer(self, text: str, *args, **kwargs) -> Tuple[np.ndarray, int]:
        """
        Generate audio from text input.
        
        Args:
            text (str): Input text to be converted to speech.

        Returns:
            Tuple[np.ndarray, int]: Generated audio waveform and sample_rate.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")