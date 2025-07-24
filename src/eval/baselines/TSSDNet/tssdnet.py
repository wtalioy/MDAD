import os
import torch
import torchaudio
from typing import List
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
import numpy as np
from .models import SSDNet1D, DilatedNet
from eval.baselines import Baseline

class TSSDNet(Baseline):
    def __init__(self, ckpt: str = "Res-TSSDNet", device: str = "cuda", **kwargs):
        self.name = "TSSDNet"
        self.device = device
        self.model = self._load_model(ckpt, device)
        self.supported_metrics = ["eer", "acc"]

    def _load_model(self, ckpt: str, device: str):
        if ckpt == "Res-TSSDNet":
            model = SSDNet1D()
        elif ckpt == "Inc-TSSDNet":
            model = DilatedNet()
        else:
            raise ValueError(f"Invalid checkpoint: {ckpt}")
        ckpt_path = os.path.join(os.path.dirname(__file__), "pretrained", ckpt + ".pth")
        model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
        model.to(device)
        model.eval()
        return model

    def evaluate(self, data: List[str], labels: np.ndarray, metrics: List[str]) -> dict:
        class CustomDataset(Dataset):
            def __init__(self, data):
                self.paths = data

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                x, _ = torchaudio.load(self.paths[idx])
                return x

        dataset = CustomDataset(data)
        data_loader = DataLoader(dataset, batch_size=1, num_workers=4, pin_memory=True, shuffle=True)

        results = {}
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}")
            func = getattr(self, f"_evaluate_{metric}")
            metric_rst = func(data_loader, labels)
            results[metric] = metric_rst
        return results

    def _evaluate_acc(self, data_loader: DataLoader, labels: np.ndarray) -> float:
        preds = []
        for batch in data_loader:
            batch = batch.to(self.device)
            output = self.model(batch)
            pred = output.argmax(dim=1)
            preds.extend(pred.tolist())
        preds = np.array(preds)
        return np.mean(preds == labels)

    def _evaluate_eer(self, data_loader: DataLoader, labels: np.ndarray) -> float:
        probs = torch.empty(0, 2).to(self.device)
        for batch in data_loader:
            batch = batch.to(self.device)
            output = self.model(batch)
            prob = F.softmax(output, dim=1)
            probs = torch.cat((probs, prob), dim=0)
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