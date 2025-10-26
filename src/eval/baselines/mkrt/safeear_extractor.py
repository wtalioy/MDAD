import torch
import hydra
from omegaconf import OmegaConf
import torchaudio
import numpy as np
import os
from .safeear_decouple.decouple import SpeechTokenizer

class SafeEarExtractor:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"SafeEar config file not found at {config_path}")
        cfg = OmegaConf.load(config_path)
        
        self.model: torch.nn.Module = hydra.utils.instantiate(cfg.decouple_model)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"SafeEar checkpoint file not found at {checkpoint_path}")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
        self.sample_rate = self.model.sample_rate

    @torch.inference_mode()
    def __call__(self, audio_batch: list[np.ndarray], sr: int) -> list[torch.Tensor]:
        processed_batch = []
        for waveform_np in audio_batch:
            waveform = torch.from_numpy(waveform_np).float().to(self.device)
            
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate).to(self.device)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            processed_batch.append(waveform)

        max_len = max(w.shape[1] for w in processed_batch)
        
        padded_batch = []
        for waveform in processed_batch:
            padding_needed = max_len - waveform.shape[1]
            padded_waveform = torch.nn.functional.pad(waveform, (0, padding_needed))
            padded_batch.append(padded_waveform)
            
        batch_tensor = torch.stack(padded_batch)

        _, _, _, acoustic_tokens = self.model(batch_tensor, layers=list(range(8)))
        
        features = torch.cat(acoustic_tokens, dim=1)
        features = features.permute(0, 2, 1).cpu()
        
        feature_list = list(torch.split(features, 1, dim=0))
        
        return feature_list
