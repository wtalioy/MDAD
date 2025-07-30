import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm
from loguru import logger
from typing import List, Optional
import numpy as np
import librosa
import os
import shutil
import yaml

from baselines.RawNet2.model import RawNet as RawNetModel
from baselines.RawNet2.utils import *

from baselines import Baseline
from config import Label

class RawNet2(Baseline):
    def __init__(self, device: str = "cuda", **kwargs):
        self.device = device
        self.default_ckpt = os.path.join(os.path.dirname(__file__), "ckpts", "asvspoof2019_LA.pth")
        model_args = self._load_model_config()
        self.model = RawNetModel(model_args, device).to(device)
        
        self.supported_metrics = ["eer"]

    def _load_model_config(self) -> dict:
        config_path = os.path.join(os.path.dirname(__file__), "config", "model.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    def _load_train_config(self, dataset_name: str) -> dict:
        config_path = os.path.join(os.path.dirname(__file__), "config", f"train_{dataset_name.lower()}.yaml")
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(__file__), "config", "train_default.yaml")
            shutil.copy(config_path, os.path.join(os.path.dirname(__file__), "config", f"train_{dataset_name.lower()}.yaml"))
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    def _init_train(self, args: dict):
        if args['multi_gpu']:
            self.model_to_save = self.model
            self.model = nn.DataParallel(self.model_to_save).to(self.device)
        else:
            self.model_to_save = self.model
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

    def _prepare_loader(self, data: List[str], labels: np.ndarray, batch_size: int = 128, shuffle: bool = True, drop_last: bool = True, num_workers: int = 8):
        def pad(x, max_len=64600):
            x_len = x.shape[0]
            if x_len >= max_len:
                return x[:max_len]
            # need to pad
            num_repeats = int(max_len / x_len) + 1
            padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
            return padded_x

        class CustomDataset(Dataset):
            def __init__(self, data, labels):
                self.paths = data
                self.labels = labels
            def __len__(self):
                return len(self.paths)
            def __getitem__(self, idx):
                path = self.paths[idx]
                X, _ = librosa.load(path, sr=None)
                X_pad = pad(X)
                x_inp= torch.from_numpy(X_pad).float()
                y = self.labels[idx]
                return x_inp, y

        dataset = CustomDataset(data, labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
        return loader

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

    def train(self, train_data: List[str], train_labels: np.ndarray, eval_data: List[str], eval_labels: np.ndarray, dataset_name: str):
        args = self._load_train_config(dataset_name)
        train_loader = self._prepare_loader(train_data, train_labels)
        eval_loader = self._prepare_loader(eval_data, eval_labels)

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
                torch.save(self.model_to_save.state_dict(), save_path)
                logger.info(f"New best EER: {100*best_eer:.2f}% at epoch {epoch}")

        logger.info(f"Training complete! Best EER: {100*best_eer:.2f}% at epoch {best_epoch}")
        logger.remove(log_id)
        self.model = self.model_to_save

    def evaluate(self, data: List[str], labels: np.ndarray, metrics: List[str], in_domain: bool = False, dataset_name: Optional[str] = None) -> dict:
        if in_domain:
            self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "ckpts", f"{dataset_name}_best.pt")))
        else:
            self.model.load_state_dict(torch.load(self.default_ckpt))
            if Label.real.value == 0:
                labels = 1 - labels
        eval_loader = self._prepare_loader(data, labels, shuffle=False, drop_last=False)
        
        results = {}
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}")
            func = getattr(self, f"_evaluate_{metric}")
            metric_rst = func(eval_loader)
            results[metric] = metric_rst
        return results