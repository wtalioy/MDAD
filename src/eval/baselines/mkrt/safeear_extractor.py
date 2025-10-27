import torch
import hydra
from omegaconf import OmegaConf
import numpy as np
import os
from baselines.utils import download_from_url

class SafeEarExtractor:
    def __init__(self, device: str = "cuda"):
        self.device = device

        config_path = os.path.join(os.path.dirname(__file__), "config", "extractor.yaml")
        cfg = OmegaConf.load(config_path)
        
        self.model: torch.nn.Module = hydra.utils.instantiate(cfg.decouple_model)
        
        speechtokenizer_path = download_from_url(
            url="https://cloud.tsinghua.edu.cn/f/413a0cd2e6f749eea956/?dl=1",
            save_dir=os.path.join(os.path.dirname(__file__), "cache"),
            filename="SpeechTokenizer.pt"
        )
        self.model.load_state_dict(torch.load(speechtokenizer_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        
    @torch.inference_mode()
    def __call__(self, audio_batch: list[np.ndarray]) -> list[torch.Tensor]:
        processed_batch = []
        for waveform_np in audio_batch:
            waveform = torch.from_numpy(waveform_np).float().to(self.device)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
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
