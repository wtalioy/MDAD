import os
import torch
import soundfile as sf
from typing import List, Optional
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from .models import SSDNet1D, DilatedNet
from baselines import Baseline

class TSSDNet_Base(Baseline):
    def __init__(self, ckpt: str = "Res-TSSDNet", device: str = "cuda", **kwargs):
        self.device = device
        self.default_ckpt = os.path.join(os.path.dirname(__file__), "pretrained", f"{ckpt}.pth")
        self.model = self._load_model(ckpt)
        self.supported_metrics = ["eer", "acc"]

    def _load_model(self, ckpt: str):
        if ckpt == "Res-TSSDNet":
            model = SSDNet1D()
        elif ckpt == "Inc-TSSDNet":
            model = DilatedNet()
        else:
            raise ValueError(f"Invalid checkpoint: {ckpt}")
        model.to(self.device)
        return model

    def evaluate(self, data: List[str], labels: np.ndarray, metrics: List[str], in_domain: bool = False, dataset_name: Optional[str] = None) -> dict:
        if in_domain:
            self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "ckpts", f"{dataset_name}_best.pt")))
        else:
            self.model.load_state_dict(torch.load(self.default_ckpt)["model_state_dict"])
        self.model.eval()

        def pad(x, max_len=64600):
            x_len = x.shape[0]
            if x_len >= max_len:
                return x[:max_len]
            # need to pad
            num_repeats = int(max_len / x_len) + 1
            padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
            return padded_x

        class CustomDataset(Dataset):
            def __init__(self, data):
                self.paths = data

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                x, _ = sf.read(self.paths[idx])
                x = pad(x)
                x = torch.from_numpy(x).float().unsqueeze(0)
                return x

        dataset = CustomDataset(data)
        data_loader = DataLoader(dataset, batch_size=16, num_workers=4, pin_memory=True)

        results = {}
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}")
            func = getattr(self, f"_evaluate_{metric}")
            metric_rst = func(data_loader, labels)
            results[metric] = metric_rst
        return results

    @torch.no_grad()
    def _evaluate_acc(self, data_loader: DataLoader, labels: np.ndarray) -> float:
        preds = []
        for batch in tqdm(data_loader, desc="Evaluating ACC"):
            batch = batch.to(self.device)
            output = self.model(batch)
            pred = output.argmax(dim=1)
            preds.extend(pred.tolist())
        preds = np.array(preds)
        return np.mean(preds == labels)

    @torch.no_grad()
    def _evaluate_eer(self, data_loader: DataLoader, labels: np.ndarray) -> float:
        probs = torch.empty(0, 2)
        for batch in tqdm(data_loader, desc="Evaluating EER"):
            batch = batch.to(self.device)
            output = self.model(batch)
            prob = F.softmax(output, dim=1)
            probs = torch.cat((probs, prob.cpu()), dim=0)
        labels = torch.tensor(labels).unsqueeze(-1)
        probs = torch.cat((probs, labels), dim=1)
        return self._cal_roc_eer(probs.cpu())

    def _cal_roc_eer(self, probs):
        """
        probs: tensor, number of samples * 3, containing softmax probabilities
        row wise: [genuine prob, fake prob, label]
        TP: True Fake
        FP: False Fake
        """
        all_labels = probs[:, 2]
        zero_index = torch.nonzero((all_labels == 0)).squeeze(-1)
        one_index = torch.nonzero(all_labels).squeeze(-1)
        zero_probs = probs[zero_index, 0]
        one_probs = probs[one_index, 0]

        threshold_index = torch.linspace(-0.1, 1.01, 10000)
        tpr = torch.zeros(len(threshold_index),)
        fpr = torch.zeros(len(threshold_index),)
        cnt = 0
        for i in threshold_index:
            tpr[cnt] = one_probs.le(i).sum().item()/len(one_probs)
            fpr[cnt] = zero_probs.le(i).sum().item()/len(zero_probs)
            cnt += 1

        sum_rate = tpr + fpr
        distance_to_one = torch.abs(sum_rate - 1)
        eer_index = distance_to_one.argmin(dim=0).item()
        out_eer = 0.5*(fpr[eer_index] + 1 - tpr[eer_index]).numpy()

        return out_eer

class Res_TSSDNet(TSSDNet_Base):
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__("Res-TSSDNet", device, **kwargs)

class Inc_TSSDNet(TSSDNet_Base):
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__("Inc-TSSDNet", device, **kwargs)