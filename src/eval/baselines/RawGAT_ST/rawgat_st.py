import os
import torch
import shutil
import librosa
from typing import List, Optional, Tuple
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from loguru import logger
import yaml
from baselines.RawGAT_ST.model import RawGAT_ST as RawGAT_ST_Model
from baselines import Baseline
from config import Label

class RawGAT_ST(Baseline):
    def __init__(self, device: str = "cuda", **kwargs):
        self.device = device
        self.default_ckpt = os.path.join(os.path.dirname(__file__), "ckpts", "RawGAT_ST_mul.pth")
        model_args = self._load_model_config()
        self.model = RawGAT_ST_Model(model_args, device).to(device)
        
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
        weight = torch.FloatTensor([0.1, 0.9]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        
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
                output = self.model(batch, Freq_aug=True)
                loss = self.criterion(output, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_description('epoch: %d, loss:%.3f'%(epoch, loss.item()))
                pbar.update(1)

    def _compute_det_curve(self, target_scores: np.ndarray, nontarget_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Detection Error Tradeoff (DET) curve."""
        n_scores = target_scores.size + nontarget_scores.size
        all_scores = np.concatenate((target_scores, nontarget_scores))
        labels = np.concatenate(
            (np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

        # Sort labels based on scores
        indices = np.argsort(all_scores, kind='mergesort')
        labels = labels[indices]

        # Compute false rejection and false acceptance rates
        tar_trial_sums = np.cumsum(labels)
        nontarget_trial_sums = nontarget_scores.size - \
            (np.arange(1, n_scores + 1) - tar_trial_sums)

        # false rejection rates
        frr = np.concatenate(
            (np.atleast_1d(0), tar_trial_sums / target_scores.size))
        far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums /
                              nontarget_scores.size))  # false acceptance rates
        # Thresholds are the sorted scores
        thresholds = np.concatenate(
            (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))

        return frr, far, thresholds

    def _evaluate_eer(self, target_scores: np.ndarray, nontarget_scores: np.ndarray) -> Tuple[float, float]:
        """
        Compute Equal Error Rate (EER) and the corresponding threshold.
        
        Args:
            target_scores: Scores for bonafide (target) samples
            nontarget_scores: Scores for spoof (nontarget) samples
            
        Returns:
            Tuple of (EER, threshold)
        """
        frr, far, _ = self._compute_det_curve(target_scores, nontarget_scores)
        abs_diffs = np.abs(frr - far)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((frr[min_index], far[min_index]))
        return float(eer)

    def _run_inference(self, data_loader: DataLoader) -> np.ndarray:
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for batch_data, _ in tqdm(data_loader, desc="Running inference"):
                # Handle different data loader formats
                batch_x = batch_data
                batch_x = batch_x.to(self.device)

                # Run model inference
                batch_out = self.model(batch_x, Freq_aug=False)
                batch_scores = batch_out[:, 1].data.cpu().numpy().ravel()
                scores.extend(batch_scores.tolist())     

        return np.array(scores)

    def train(self, train_data: List[str], train_labels: np.ndarray, eval_data: List[str], eval_labels: np.ndarray, dataset_name: str):
        args = self._load_train_config(dataset_name)
        train_loader = self._prepare_loader(train_data, train_labels, batch_size=args['batch_size'])
        eval_loader = self._prepare_loader(eval_data, eval_labels, shuffle=False, drop_last=False, batch_size=128)

        log_id = logger.add("logs/train.log", rotation="100 MB", retention="60 days")
        logger.info(f"Training RawGAT-ST on {dataset_name}")

        self._init_train(args)

        best_eer = 100
        best_epoch = 0
        save_path = os.path.join(os.path.dirname(__file__), "ckpts", f"{dataset_name}_best.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for epoch in range(args['num_epoch']):
            self._train_epoch(epoch, train_loader)
            scores = self._run_inference(eval_loader)
            bonafide_scores = scores[eval_labels == Label.real.value]
            spoof_scores = scores[eval_labels == Label.fake.value]
            eer = self._evaluate_eer(bonafide_scores, spoof_scores)
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
        eval_loader = self._prepare_loader(data, labels, shuffle=False, drop_last=False, batch_size=128)
        
        results = {}
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}")
            func = getattr(self, f"_evaluate_{metric}")
            metric_rst = func(eval_loader)
            results[metric] = metric_rst
        return results