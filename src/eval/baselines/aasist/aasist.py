import torch
import json
import os
import numpy as np
import librosa
from typing import List, Tuple, Optional
from torch.utils.data import DataLoader, Dataset
from torchcontrib.optim import SWA
from importlib import import_module
from tqdm import tqdm
from baselines.aasist.utils import create_optimizer, seed_worker, set_seed, str_to_bool
from baselines import Baseline
from config import Label

class AASIST_Base(Baseline):
    def __init__(self, ckpt: str = "AASIST", device: str = "cuda", **kwargs):
        self.device = device
        self.default_ckpt = os.path.join(os.path.dirname(__file__), "ckpts", f"{ckpt}.pth")
        self.model, self.optim_config = self._load_config(os.path.join(os.path.dirname(__file__), "config", f"{ckpt}.conf"), device)
        self.supported_metrics = ['eer', 'tdcf']

    def _load_config(self, config_path: str, device: str):
        with open(config_path, "r") as f_json:
            config = json.loads(f_json.read())
        model_config = config.get("model_config", {})
        optim_config = config.get("optim_config", {})
        optim_config["epochs"] = config.get("num_epochs", 100)
        module = import_module("baselines.aasist.models.{}".format(model_config["architecture"]))
        _model = getattr(module, "Model")
        model = _model(model_config).to(device)
        return model, optim_config

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
        """
        Run inference on the data loader and return scores and labels.
        
        Returns:
            Tuple of (scores, labels) where:
            - scores: CM scores for each sample
        """
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

    def _init_train(self):
        self.optimizer, self.scheduler = create_optimizer(self.model.parameters(), self.optim_config)
        self.optimizer_swa = SWA(self.optimizer)

    def _train_epoch(self, data_loader: DataLoader):


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
        frr, far, thresholds = self._compute_det_curve(target_scores, nontarget_scores)
        abs_diffs = np.abs(frr - far)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((frr[min_index], far[min_index]))
        return float(eer), float(thresholds[min_index])


    def evaluate(self, data: List[str], labels: np.ndarray, metrics: List[str], in_domain: bool = False, dataset_name: Optional[str] = None) -> dict:
        if in_domain:
            self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "ckpts", f"{dataset_name}_best.pt")))
        else:
            self.model.load_state_dict(torch.load(self.default_ckpt))

        data_loader = self._prepare_loader(data, labels)

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