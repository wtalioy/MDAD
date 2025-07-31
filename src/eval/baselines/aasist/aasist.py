import torch
import torch.nn as nn
import shutil
import os
import numpy as np
import yaml
import librosa
from typing import List, Tuple, Optional
from torch.utils.data import DataLoader, Dataset
from torchcontrib.optim import SWA
from importlib import import_module
from tqdm import tqdm
from baselines.aasist.utils import create_optimizer
from baselines import Baseline
from config import Label
from loguru import logger

class AASIST_Base(Baseline):
    def __init__(self, ckpt: str = "AASIST", device: str = "cuda", **kwargs):
        self.device = device
        self.default_ckpt = os.path.join(os.path.dirname(__file__), "ckpts", f"{ckpt}.pth")
        self.model = self._load_model_config(ckpt)
        self.supported_metrics = ['eer', 'tdcf']

    def _load_model_config(self, ckpt: str):
        config_path = os.path.join(os.path.dirname(__file__), "config", f"{ckpt}.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        module = import_module("baselines.aasist.models.{}".format(config["architecture"]))
        _model = getattr(module, "Model")
        model = _model(config).to(self.device)
        return model

    def _load_train_config(self, dataset_name: str):
        config_path = os.path.join(os.path.dirname(__file__), "config", f"train_{dataset_name.lower()}.yaml")
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(__file__), "config", "train_default.yaml")
            shutil.copy(config_path, os.path.join(os.path.dirname(__file__), "config", f"train_{dataset_name.lower()}.yaml"))
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    def _prepare_loader(self, data: List[str], labels: np.ndarray, batch_size: int = 16, shuffle: bool = True, drop_last: bool = True, num_workers: int = 8):
        def pad_random(x: np.ndarray, max_len: int = 64600):
            x_len = x.shape[0]
            # if duration is already long enough
            if x_len >= max_len:
                stt = np.random.randint(x_len - max_len)
                return x[stt:stt + max_len]

            # if too short
            num_repeats = int(max_len / x_len) + 1
            padded_x = np.tile(x, (num_repeats))[:max_len]
            return padded_x

        class CustomDataset(Dataset):
            def __init__(self, data):
                self.paths = data
                self.labels = labels

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                x, _ = librosa.load(self.paths[idx], sr=None)
                x_pad = pad_random(x)
                x_inp = torch.tensor(x_pad, dtype=torch.float32)
                y = self.labels[idx]
                return x_inp, y

        dataset = CustomDataset(data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, pin_memory=True)
        return loader

    def _run_inference(self, data_loader: DataLoader) -> np.ndarray:
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for batch_data, _ in tqdm(data_loader, desc="Running inference"):
                # Handle different data loader formats
                batch_x = batch_data
                batch_x = batch_x.to(self.device)

                # Run model inference
                _, batch_out = self.model(batch_x)
                batch_scores = batch_out[:, 1].data.cpu().numpy().ravel()
                scores.extend(batch_scores.tolist())     

        return np.array(scores)

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
        eval_loader = self._prepare_loader(eval_data, eval_labels, batch_size=16)
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
            scores = self._run_inference(eval_loader)
            bonafide_scores = scores[eval_labels == Label.real.value]
            spoof_scores = scores[eval_labels == Label.fake.value]
            eer = self._evaluate_eer(bonafide_scores, spoof_scores)
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


    def evaluate(self, data: List[str], labels: np.ndarray, metrics: List[str], in_domain: bool = False, dataset_name: Optional[str] = None) -> dict:
        if in_domain:
            self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "ckpts", f"{dataset_name}_best.pt")))
        else:
            self.model.load_state_dict(torch.load(self.default_ckpt))

        data_loader = self._prepare_loader(data, labels, shuffle=False, drop_last=False, batch_size=16)

        # Run inference to get predictions
        scores = self._run_inference(data_loader)
            
        # Separate bonafide and spoof scores based on labels
        bonafide_mask = labels == Label.real.value
        spoof_mask = labels == Label.fake.value

        bonafide_scores = scores[bonafide_mask]
        spoof_scores = scores[spoof_mask]
        
        if len(bonafide_scores) == 0 or len(spoof_scores) == 0:
            raise ValueError("Need both bonafide and spoof samples for evaluation")
        
        results = {}
        
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}")
            
            func = getattr(self, f"_evaluate_{metric}")
            metric_rst = func(bonafide_scores, spoof_scores)
            results[metric] = metric_rst
        
        return results

class AASIST(AASIST_Base):
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__("AASIST", device, **kwargs)

class AASIST_L(AASIST_Base):
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__("AASIST-L", device, **kwargs)