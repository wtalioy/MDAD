import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm
from loguru import logger
from typing import List, Optional
import numpy as np
import os

from baselines.RawNet2.model import RawNet as RawNetModel
from baselines.RawNet2.utils import *

from baselines import Baseline
from config import Label

class RawNet2(Baseline):
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__(device, **kwargs)
        self.name = "RawNet2"
        self.default_ckpt = os.path.join(os.path.dirname(__file__), "ckpts", "asvspoof2019_LA.pth")
        model_args = self._load_model_config(os.path.dirname(__file__))
        self.model = RawNetModel(model_args, device).to(device)
        
        self.supported_metrics = ["eer", "auroc"]

    def _init_train(self, args: dict):
        self.model.apply(init_weights)
        self.criterion = nn.CrossEntropyLoss()
        params = [
            {
                'params': [
                    param for name, param in self.model.named_parameters()
                    if 'bn' not in name
                ]
            },
            {
                'params': [
                    param for name, param in self.model.named_parameters()
                    if 'bn' in name
                ],
                'weight_decay':
                0
            },
        ]
        if args['optimizer'].lower() == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=args['lr'], weight_decay=args['wd'])
        elif args['optimizer'].lower() == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=args['lr'], weight_decay=args['wd'], amsgrad=args['amsgrad'])
        else:
            raise ValueError(f"Optimizer {args['optimizer']} not supported")

    def _train_epoch(self, epoch: int, train_loader: DataLoader):
        self.model.train()
        with tqdm(total = len(train_loader), desc="Training") as pbar:
            for batch, label in train_loader:
                batch, label = batch.to(self.device), label.to(self.device)
                label = label.view(-1).type(torch.int64)
                output = self.model(batch)
                cce_loss = self.criterion(output, label)
                loss = cce_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_description('epoch: %d, cce:%.3f'%(epoch, cce_loss))
                pbar.update(1)

    @torch.no_grad()
    def _evaluate_eer(self, eval_loader: DataLoader) -> float:
        self.model.eval()
        scores = []
        labels = []
        with tqdm(total = len(eval_loader), desc="Evaluating EER") as pbar:
            for batch, label in eval_loader:
                batch = batch.to(self.device)
                code = self.model(batch)
                scores.extend(code[:, 1].cpu().numpy().ravel().tolist())
                labels.extend(label)
                pbar.update(1)
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return float(eer)

    @torch.no_grad()
    def _evaluate_auroc(self, eval_loader: DataLoader) -> float:
        self.model.eval()
        scores = []
        labels = []
        with tqdm(total = len(eval_loader), desc="Evaluating AUROC") as pbar:
            for batch, label in eval_loader:
                batch = batch.to(self.device)
                code = self.model(batch)
                scores.extend(code[:, 1].cpu().numpy().ravel().tolist())
                labels.extend(label)
                pbar.update(1)
        return float(roc_auc_score(labels, np.array(scores)))

    def train(self, train_data: List[np.ndarray], train_labels: np.ndarray, eval_data: List[np.ndarray], eval_labels: np.ndarray, dataset_name: str):
        args = self._load_train_config(os.path.dirname(__file__), dataset_name)
        train_loader = self._prepare_loader(train_data, train_labels, batch_size=args['bs'])
        eval_loader = self._prepare_loader(eval_data, eval_labels, batch_size=128)

        log_id = logger.add("logs/train.log", rotation="100 MB", retention="60 days")
        logger.info(f"Training RawNet2 on {dataset_name}")

        self._init_train(args)

        best_eer = 100
        best_epoch = 0
        save_path = os.path.join(os.path.dirname(__file__), "ckpts", f"{dataset_name}_best.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for epoch in range(args['epoch']):
            self._train_epoch(epoch, train_loader)
            eer = self._evaluate_eer(eval_loader)
            logger.info(f"Epoch {epoch} EER: {100*eer:.2f}%")
            if eer < best_eer:
                best_eer = eer
                best_epoch = epoch
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"New best EER: {100*best_eer:.2f}% at epoch {epoch}")

        logger.info(f"Training complete! Best EER: {100*best_eer:.2f}% at epoch {best_epoch}")
        logger.remove(log_id)

    def evaluate(self, data: List[np.ndarray], labels: np.ndarray, metrics: List[str], in_domain: bool = False, dataset_name: Optional[str] = None, **kwargs) -> dict:
        if in_domain:
            self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "ckpts", f"{dataset_name}_best.pt")))
        else:
            self.model.load_state_dict(torch.load(self.default_ckpt))
            if Label.real != 1:
                labels = 1 - labels
        eval_loader = self._prepare_loader(data, labels, shuffle=False, drop_last=False, batch_size=128)
        
        results = {}
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}")
            func = getattr(self, f"_evaluate_{metric}")
            metric_rst = func(eval_loader)
            results[metric] = metric_rst
        return results