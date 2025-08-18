import os
import torch
from typing import List, Optional
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import roc_auc_score
from baselines.TSSDNet.models import SSDNet1D, DilatedNet
from baselines import Baseline
from config import Label

class TSSDNet_Base(Baseline):
    def __init__(self, ckpt: str = "Res-TSSDNet", device: str = "cuda", **kwargs):
        super().__init__(device, **kwargs)
        self.name = ckpt
        self.default_ckpt = os.path.join(os.path.dirname(__file__), "ckpts", f"{ckpt}.pth")
        self.model = self._load_model(ckpt)
        self.supported_metrics = ["eer", "acc", "auroc"]

    def _load_model(self, ckpt: str):
        if ckpt == "Res-TSSDNet":
            model = SSDNet1D()
        elif ckpt == "Inc-TSSDNet":
            model = DilatedNet()
        else:
            raise ValueError(f"Invalid checkpoint: {ckpt}")
        model.to(self.device)
        return model

    def _init_train(self, args: dict):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args["lr"])
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

    def _train_epoch(self, epoch: int, train_loader: DataLoader, loss_type: str = "WCE"):
        self.model.train()
        with tqdm(total = len(train_loader), desc="Training") as pbar:
            for samples, labels in train_loader:
                samples = samples.to(self.device)
                labels = labels.to(self.device)
                
                if loss_type == 'mixup':
                    alpha = 0.1
                    lam = np.random.beta(alpha, alpha)
                    lam = torch.tensor(lam, requires_grad=False)
                    index = torch.randperm(len(labels))
                    samples = lam*samples + (1-lam)*samples[index, :]
                    preds = self.model(samples)
                    labels_b = labels[index]
                    loss = lam * F.cross_entropy(preds, labels) + (1 - lam) * F.cross_entropy(preds, labels_b)
                else:
                    preds = self.model(samples)
                    loss = F.cross_entropy(preds, labels)

                self.optimizer.zero_grad()
                loss.backward()
                pbar.set_description('epoch: %d, loss:%.3f'%(epoch, loss.item()))
                pbar.update(1)

    def train(self, train_data: List[np.ndarray], train_labels: np.ndarray, eval_data: List[np.ndarray], eval_labels: np.ndarray, dataset_name: str, **kwargs):
        train_config = self._load_train_config(os.path.dirname(__file__), dataset_name)
        train_loader = self._prepare_loader(train_data, train_labels, shuffle=True, drop_last=True, batch_size=train_config['batch_size'])
        eval_loader = self._prepare_loader(eval_data, eval_labels, shuffle=False, drop_last=False, batch_size=32)

        log_id = logger.add("logs/train.log", rotation="100 MB", retention="60 days")
        logger.info(f"Training TSSDNet on {dataset_name}")

        self._init_train(train_config)

        best_eer = 100
        best_epoch = 0
        save_path = os.path.join(os.path.dirname(__file__), "ckpts", f"{dataset_name}_best.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        for epoch in range(train_config["num_epoch"]):
            self._train_epoch(epoch, train_loader, train_config["loss_type"])
            eer = self._evaluate_eer(eval_loader, eval_labels)
            logger.info(f"Epoch {epoch} EER: {100*eer:.2f}%")
            if eer < best_eer:
                best_eer = eer
                best_epoch = epoch
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"New best EER: {100*best_eer:.2f}% at epoch {epoch}")

        logger.info(f"Training complete! Best EER: {100*best_eer:.2f}% at epoch {best_epoch}")
        logger.remove(log_id)

    def evaluate(self, data: List[np.ndarray], labels: List[Label], metrics: List[str], in_domain: bool = False, dataset_name: Optional[str] = None, **kwargs) -> dict:
        if in_domain:
            self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "ckpts", f"{dataset_name}_best.pt")))
        else:
            self.model.load_state_dict(torch.load(self.default_ckpt)["model_state_dict"])
            if Label.real != 0:
                labels = 1 - labels
        eval_loader = self._prepare_loader(data, labels, shuffle=False, drop_last=False, batch_size=32)

        results = {}
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}")
            func = getattr(self, f"_evaluate_{metric}")
            metric_rst = func(eval_loader, labels)
            results[metric] = metric_rst
        return results

    @torch.no_grad()
    def _evaluate_acc(self, data_loader: DataLoader, labels: List[Label]) -> float:
        self.model.eval()
        preds = []
        for batch, _ in tqdm(data_loader, desc="Evaluating ACC"):
            batch = batch.to(self.device)
            output = self.model(batch)
            pred = output.argmax(dim=1)
            preds.extend(pred.tolist())
        preds = np.array(preds)
        return np.mean(preds == labels)

    @torch.no_grad()
    def _evaluate_eer(self, data_loader: DataLoader, labels: List[Label]) -> float:
        self.model.eval()
        probs = torch.empty(0, 2)
        for batch, _ in tqdm(data_loader, desc="Evaluating EER"):
            batch = batch.unsqueeze(1).to(self.device)
            output = self.model(batch)
            prob = F.softmax(output, dim=1)
            probs = torch.cat((probs, prob.cpu()), dim=0)
        labels = torch.tensor(labels).unsqueeze(-1)
        probs = torch.cat((probs, labels), dim=1)
        return self._cal_roc_eer(probs.cpu())

    @torch.no_grad()
    def _evaluate_auroc(self, data_loader: DataLoader, labels: List[Label]) -> float:
        self.model.eval()
        scores = []
        for batch, _ in tqdm(data_loader, desc="Evaluating AUROC"):
            batch = batch.unsqueeze(1).to(self.device)
            output = self.model(batch)
            prob = F.softmax(output, dim=1)
            scores.extend(prob[:, 1].detach().cpu().numpy().tolist())
        return float(roc_auc_score(labels, np.array(scores)))

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