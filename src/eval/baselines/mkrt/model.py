import torch
import torch.nn as nn

from .extractor import Extractor
from .mmd_model import MMDBaseModel

class MKRTModel(nn.Module):
    def __init__(self, config: dict, device: str):
        super().__init__()
        self.device = device
        self.extractor = Extractor(d_args=config["extractor"]).to(self.device)
        self.mmd_model = MMDBaseModel(config=config["mmd_model"]).to(device)

        # MMD-specific parameters
        self.sigma = nn.Parameter(torch.tensor(1.0, dtype=torch.float, device=device))
        self.sigma0_u = nn.Parameter(torch.tensor(1.0, dtype=torch.float, device=device))
        self.raw_ep = nn.Parameter(torch.tensor(0.0, dtype=torch.float, device=device))

        self.coeff_xy = config.get("coeff_xy", 2)
        self.is_yy_zero = config.get("is_yy_zero", False)
        self.is_xx_zero = config.get("is_xx_zero", False)

    def forward(self, x, Freq_aug=False):
        # Feature extraction
        e = self.extractor(x, Freq_aug=Freq_aug)
        features = e.permute(0, 3, 1, 2).contiguous()
        features = features.view(features.size(0), features.size(1), -1)
        
        # MMD model processing
        output = self.mmd_model(features)
        
        return features, output

    @property
    def ep(self):
        return torch.nn.functional.softplus(self.raw_ep) + 1e-6
        
    def load_from_checkpoint(self, ckpt_path: str):
        """Load a trained model from checkpoint."""
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        
        # Load mmd_model state dict
        state_dict = checkpoint["net"]
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_key = key[7:]
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        self.mmd_model.load_state_dict(new_state_dict)

        # Load extractor state dict
        self.extractor.load_state_dict(state_dict)

        with torch.no_grad():
            self.sigma.data.fill_(checkpoint["sigma"])
            self.sigma0_u.data.fill_(checkpoint["sigma0_u"])
            ep_val = torch.tensor(checkpoint["ep"], device=self.device)
            raw_val = torch.log(torch.exp(ep_val - 1e-6) - 1.0 + 1e-12)
            self.raw_ep.data.copy_(raw_val)

    def save_to_checkpoint(self, ckpt_path: str):
        torch.save({
            "net": self.mmd_model.state_dict(),
            "extractor": self.extractor.state_dict(),
            "sigma": self.sigma.detach().item(),
            "sigma0_u": self.sigma0_u.detach().item(),
            "ep": self.ep.detach().item()
        }, ckpt_path)
