import torch
import json
import os
import numpy as np
import soundfile as sf
from typing import List, Tuple, Optional
from torch.utils.data import DataLoader, Dataset
from importlib import import_module
from tqdm import tqdm
from baselines import Baseline
from config import Label

class AASIST_Base(Baseline):
    def __init__(self, config: str = "AASIST.conf", device: str = "cuda", **kwargs):
        self.device = device
        self.model, self.default_ckpt = self._load_model(os.path.join(os.path.dirname(__file__), "config", config), device)
        self.supported_metrics = ['eer', 'tdcf']

    def _load_model(self, config_path: str, device: str):
        with open(config_path, "r") as f_json:
            config = json.loads(f_json.read())
        model_config = config.get("model_config", {})
        module = import_module("baselines.aasist.models.{}".format(model_config["architecture"]))
        _model = getattr(module, "Model")
        model = _model(model_config).to(device)
        default_ckpt = os.path.join(os.path.dirname(__file__), config["model_path"])
        return model, default_ckpt

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
            for batch_data in tqdm(data_loader, desc="Running inference"):
                # Handle different data loader formats
                batch_x = batch_data
                batch_x = batch_x.to(self.device)

                # Run model inference
                _, batch_out = self.model(batch_x)
                batch_scores = batch_out[:, 1].data.cpu().numpy().ravel()
                scores.extend(batch_scores.tolist())     

        return np.array(scores)

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

    def compute_eer(self, target_scores: np.ndarray, nontarget_scores: np.ndarray) -> Tuple[float, float]:
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

    def compute_tDCF(self, bonafide_scores: np.ndarray, spoof_scores: np.ndarray, 
                     Pfa_asv: float = 0.01, Pmiss_asv: float = 0.01, 
                     Pmiss_spoof_asv: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Tandem Detection Cost Function (t-DCF).
        
        Args:
            bonafide_scores: CM scores for bonafide samples
            spoof_scores: CM scores for spoof samples
            Pfa_asv: False alarm rate of ASV system (default: 0.01)
            Pmiss_asv: Miss rate of ASV system (default: 0.01)
            Pmiss_spoof_asv: Miss rate of spoof samples in ASV (default: None, uses Pmiss_asv)
            
        Returns:
            Tuple of (normalized t-DCF curve, CM thresholds)
        """
        # Default cost model parameters
        cost_model = {
            'Pspoof': 0.05,
            'Ptar': 0.95 * 0.99,
            'Pnon': 0.95 * 0.01,
            'Cmiss': 1,
            'Cfa': 10,
            'Cmiss_asv': 1,
            'Cfa_asv': 10,
            'Cmiss_cm': 1,
            'Cfa_cm': 10,
        }
        
        if Pmiss_spoof_asv is None:
            Pmiss_spoof_asv = Pmiss_asv
            
        # Sanity checks
        combined_scores = np.concatenate((bonafide_scores, spoof_scores))
        if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
            raise ValueError('Scores contain nan or inf.')
            
        n_uniq = np.unique(combined_scores).size
        if n_uniq < 3:
            raise ValueError('You should provide soft CM scores - not binary decisions')

        # Obtain miss and false alarm rates of CM
        Pmiss_cm, Pfa_cm, CM_thresholds = self._compute_det_curve(bonafide_scores, spoof_scores)

        # Constants for t-DCF computation
        C1 = cost_model['Ptar'] * (cost_model['Cmiss_cm'] - cost_model['Cmiss_asv'] * Pmiss_asv) - \
            cost_model['Pnon'] * cost_model['Cfa_asv'] * Pfa_asv
        C2 = cost_model['Cfa_cm'] * cost_model['Pspoof'] * (1 - Pmiss_spoof_asv)

        # Sanity check of the weights
        if C1 < 0 or C2 < 0:
            raise ValueError('Cannot evaluate tDCF with negative weights')

        # Obtain t-DCF curve for all thresholds
        tDCF = C1 * Pmiss_cm + C2 * Pfa_cm

        # Normalized t-DCF
        tDCF_norm = tDCF / np.minimum(C1, C2)

        return tDCF_norm, CM_thresholds

    def evaluate(self, data: List[str], metrics: List[str], 
                  labels: np.ndarray, 
                  asv_scores: Optional[dict] = None,
                  ckpt_path: Optional[str] = None) -> dict:
        self.model.load_state_dict(torch.load(ckpt_path or self.default_ckpt))

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

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                # Assuming data is a list of file paths or similar
                x, _ = sf.read(self.paths[idx])
                x_pad = pad_random(x)
                x_inp = torch.tensor(x_pad, dtype=torch.float32)
                return x_inp

        # Create DataLoader
        dataset = CustomDataset(data)
        data_loader = DataLoader(dataset, batch_size=16, pin_memory=True)

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
        
        for m in metrics:
            if m not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {m}")
            
            if m == 'eer':
                eer, threshold = self.compute_eer(bonafide_scores, spoof_scores)
                results['eer'] = float(eer)
                
            elif m == 'tdcf':
                # Use default ASV parameters if not provided
                Pfa_asv = 0.01
                Pmiss_asv = 0.01
                Pmiss_spoof_asv = None
                
                if asv_scores is not None:
                    Pfa_asv = asv_scores.get('Pfa_asv', Pfa_asv)
                    Pmiss_asv = asv_scores.get('Pmiss_asv', Pmiss_asv)
                    Pmiss_spoof_asv = asv_scores.get('Pmiss_spoof_asv', Pmiss_spoof_asv)
                
                try:
                    tDCF_curve, thresholds = self.compute_tDCF(
                        bonafide_scores, spoof_scores, 
                        Pfa_asv, Pmiss_asv, Pmiss_spoof_asv
                    )
                    min_tDCF_index = np.argmin(tDCF_curve)
                    min_tDCF = tDCF_curve[min_tDCF_index]
                    
                    results['tdcf'] = float(min_tDCF)
                except Exception as e:
                    print(f"Warning: Could not compute t-DCF: {e}")
                    results['tdcf'] = None
        
        return results

class AASIST(AASIST_Base):
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__("AASIST.conf", device, **kwargs)

class AASIST_L(AASIST_Base):
    def __init__(self, device: str = "cuda", **kwargs):
        super().__init__("AASIST-L.conf", device, **kwargs)