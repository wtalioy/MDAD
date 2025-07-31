import torch
import torch.nn as nn
import os
import numpy as np
from typing import List, Optional
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA
from importlib import import_module
from tqdm import tqdm
from loguru import logger
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from baselines.aasist.utils import create_optimizer

from baselines import Baseline
from config import Label

class AASIST_Base(Baseline):
    def __init__(self, model_name: str = "AASIST", device: str = "cuda", **kwargs):
        super().__init__(device, **kwargs)
        self.default_ckpt = os.path.join(os.path.dirname(__file__), "ckpts", f"{model_name}.pth")
        model_args = self._load_model_config()
        self.model = self._load_model(model_args)
        self.supported_metrics = ['eer', 'tdcf']

    def _load_model(self, config: dict):
        module = import_module("baselines.aasist.models.{}".format(config["architecture"]))
        _model = getattr(module, "Model")
        model = _model(config).to(self.device)
        return model

    @torch.no_grad()
    def _evaluate_eer(self, eval_loader: DataLoader) -> float:
        self.model.eval()
        scores = []
        labels = []
        with tqdm(total = len(eval_loader), desc="Evaluating EER") as pbar:
            for batch, label in eval_loader:
                batch = batch.to(self.device)
                _, batch_out = self.model(batch)
                batch_scores = batch_out[:, 1].data.cpu().numpy().ravel()
                scores.extend(batch_scores.tolist())
                labels.extend(label)
                pbar.update(1)
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return float(eer)

    def _init_train(self, optim_config: dict):
        self.optimizer, self.scheduler = create_optimizer(self.model.parameters(), optim_config)
        self.optimizer_swa = SWA(self.optimizer)
        weight = torch.FloatTensor([0.1, 0.9]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weight)

    def _train_epoch(self, epoch: int, data_loader: DataLoader):
        running_loss = 0
        num_total = 0.0
        ii = 0
        self.model.train()

        with tqdm(total = len(data_loader), desc="Training") as pbar:
            for batch_x, batch_y in data_loader:
                batch_size = batch_x.size(0)
                num_total += batch_size
                ii += 1
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.view(-1).type(torch.int64).to(self.device)
                _, batch_out = self.model(batch_x)
                batch_loss = self.criterion(batch_out, batch_y)
                running_loss += batch_loss.item() * batch_size
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                pbar.set_description('epoch: %d, cce:%.3f'%(epoch, batch_loss))
                pbar.update(1)

    def train(self, train_data: List[str], train_labels: np.ndarray, eval_data: List[str], eval_labels: np.ndarray, dataset_name: str):
        train_config = self._load_train_config(dataset_name)
        train_loader = self._prepare_loader(train_data, train_labels, batch_size=train_config['batch_size'])
        eval_loader = self._prepare_loader(eval_data, eval_labels, shuffle=False, drop_last=False, batch_size=16)
        optim_config = train_config["optim_config"]
        optim_config["steps_per_epoch"] = len(train_loader)
        optim_config["epochs"] = train_config['num_epochs']

        log_id = logger.add("logs/train.log", rotation="100 MB", retention="60 days")
        logger.info(f"Training AASIST on {dataset_name}")

        self._init_train(optim_config)

        best_eer = 100
        best_epoch = 0
        save_path = os.path.join(os.path.dirname(__file__), "ckpts", f"{dataset_name}_best.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for epoch in range(train_config['num_epochs']):
            self._train_epoch(epoch, train_loader)
            eer = self._evaluate_eer(eval_loader)
            logger.info(f"Epoch {epoch} EER: {100*eer:.2f}%")
            if eer < best_eer:
                best_eer = eer
                best_epoch = epoch
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"New best EER: {100*best_eer:.2f}% at epoch {epoch}")
            self.optimizer_swa.update_swa()

        self.optimizer_swa.swap_swa_sgd()
        self.optimizer_swa.bn_update(train_loader, self.model, device=self.device)
        logger.info(f"Training complete! Best EER: {100*best_eer:.2f}% at epoch {best_epoch}")
        logger.remove(log_id)

    def evaluate(self, data: List[str], labels: np.ndarray, metrics: List[str], in_domain: bool = False, dataset_name: Optional[str] = None) -> dict:
        if in_domain:
            self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "ckpts", f"{dataset_name}_best.pt")))
        else:
            self.model.load_state_dict(torch.load(self.default_ckpt))
            if Label.real != 1:
                labels = 1 - labels

        data_loader = self._prepare_loader(data, labels, shuffle=False, drop_last=False, batch_size=16)

        results = {}
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}")
            func = getattr(self, f"_evaluate_{metric}")
            metric_rst = func(data_loader)
            results[metric] = metric_rst
        return results

class AASIST(AASIST_Base):
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__("AASIST", device, **kwargs)

class AASIST_L(AASIST_Base):
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__("AASIST-L", device, **kwargs)