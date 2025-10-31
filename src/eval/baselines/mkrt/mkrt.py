from typing import Any, List, Tuple, Optional
import os
import librosa
import numpy as np
import math
import torch
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset
from transformers import Wav2Vec2FeatureExtractor, HubertModel, get_cosine_schedule_with_warmup
from datasets import load_dataset
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import combine_pvalues

from ..base import Baseline
from .mmd_model import MMDModel
from .mmd_utils import MMD_3_Sample_Test, MMDu
from .aasist_extractor import AASISTExtractor
from ...config import Label

class MKRT(Baseline):
    def __init__(self, device: str = "cuda", **kwargs):
        self.name = "MKRT"
        self.device = device
        self.sample_rate = 16000
        self.ref_num = 200
        self.seed = 34
        self.supported_metrics = ['eer', 'auroc']

        self.extractor = AASISTExtractor(device=device)
        self.net = MMDModel(config=self._load_model_config(os.path.dirname(__file__)), device=device)

    def _init_train(self, args: dict):
        param_groups = [
            {
                'params': self.net.basemodel.parameters(),
                'lr': args['basemodel_lr'],
                'weight_decay': args['basemodel_wd'],
            },
            {
                'params': [self.net.sigma, self.net.sigma0_u],
                'lr': args['mmd_lr'],
                'weight_decay': 0,
            },
            {
                'params': [self.net.raw_ep],
                'lr': args['mmd_lr'] * 0.1,
                'weight_decay': 0,
            },
        ]
        self.optimizer = torch.optim.Adam(param_groups)
        
        total_steps = args['num_epoch'] * args['steps_per_epoch']
        warmup_steps = max(1, int(0.01 * total_steps))

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        torch.manual_seed(args['seed'])
        torch.cuda.manual_seed(args['seed'])

    def _split_features(self, fea: torch.Tensor, labels: List[Label]) -> Tuple[torch.Tensor, torch.Tensor]:
        labels_np = np.array(labels)
        mask_real = labels_np == Label.real
        mask_fake = ~mask_real
        fea_real = fea[mask_real]
        fea_fake = fea[mask_fake]
        return fea_real, fea_fake

    def _train_epoch(self, epoch: int, fea_real: torch.Tensor, fea_fake: torch.Tensor, batch_size: int):
        self.net.basemodel.train()

        min_len = min(fea_real.size(0), fea_fake.size(0))
        fea_real = fea_real[:min_len]
        fea_fake = fea_fake[:min_len]

        dataset = TensorDataset(fea_real, fea_fake)
        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, min_len),
            shuffle=False,
            drop_last=False,
        )

        with tqdm(total=len(loader), desc="Training") as pbar:
            for fea_real_sample, fea_fake_sample in loader:
                if fea_real_sample.size(0) == 1 or fea_fake_sample.size(0) == 1:
                    # only happens when the size of the last batch is 1
                    continue

                inputs = torch.cat([fea_real_sample, fea_fake_sample], dim=0)
                outputs = self.net(inputs)

                temp = MMDu(
                    outputs,
                    inputs.view(inputs.shape[0], -1),
                    fea_real_sample.shape[0],
                    self.net.sigma,
                    self.net.sigma0_u,
                    self.net.ep,
                    coeff_xy=self.net.coeff_xy,
                    is_yy_zero=self.net.is_yy_zero,
                    is_xx_zero=self.net.is_xx_zero,
                )
                mmd_value_temp = -1 * (temp[0])
                if temp[1] is not None:
                    mmd_std_temp = torch.sqrt(temp[1] + 10 ** (-8))
                else:
                    mmd_std_temp = torch.sqrt(torch.tensor(10 ** (-8), device=mmd_value_temp.device))

                loss = torch.div(mmd_value_temp, mmd_std_temp)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.basemodel.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_([self.net.sigma, self.net.sigma0_u, self.net.raw_ep], max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()
                pbar.set_description('epoch:%d, sigma:%.3f, sigma0_u:%.3f, ep:%.3f, loss:%.3f'%(epoch, self.net.sigma.item(), self.net.sigma0_u.item(), self.net.ep.item(), loss.item()))
                pbar.update(1)

    def train(
        self,
        train_data: List[np.ndarray],
        train_labels: List[Label],
        eval_data: List[np.ndarray],
        eval_labels: List[Label],
        dataset_name: str,
        ref_data: Optional[List[np.ndarray]] = None,
        ref_labels: Optional[List[Label]] = None,
        sr: int = 16000,
    ):
        if ref_data is None or ref_labels is None:
            raise ValueError(f"{self.name} requires ref_data and ref_labels to be provided")
        
        # cache ref_data & ref_labels
        self.ref_data = ref_data
        self.ref_labels = ref_labels
        
        args = self._load_train_config(os.path.dirname(__file__), dataset_name)
        train_fea = self._load_features(train_data, cache_name=f"train_{dataset_name}")
        train_real, train_fake = self._split_features(train_fea, train_labels)

        effective_len = min(train_real.size(0), train_fake.size(0))
        args['steps_per_epoch'] = math.ceil(effective_len / args['batch_size'])

        ref_fea = self._load_features(ref_data, cache_name=f"ref_{dataset_name}")
        ref_real, ref_fake = self._split_features(ref_fea, ref_labels)

        self._auto_tune_bandwidths(train_real, train_fake)

        eval_labels = np.array(eval_labels)

        log_id = logger.add("logs/train.log", rotation="10 MB", retention="60 days")
        logger.info(f"Training MKRT on {dataset_name}")
        
        self._init_train(args)
        
        best_eer = 1.0
        best_epoch = 0
        save_path = os.path.join(os.path.dirname(__file__), "ckpts", f"{dataset_name}_best.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        worse_epochs = 0
        patience = args.get("patience", 5) 

        for epoch in range(args['num_epoch']):
            self._train_epoch(epoch, train_real, train_fake, batch_size=args['batch_size'])
            if epoch % args['eval_interval'] == 0 and epoch > 0:
                self._precompute_ref_cache(ref_real, ref_fake)
                eer = self._evaluate_eer(data=eval_data, labels=eval_labels, sr=sr)
                logger.info(f"Epoch {epoch} EER: {100*eer:.2f}%")
                if eer < best_eer:
                    best_eer = eer
                    best_epoch = epoch
                    worse_epochs = 0
                    self.net.save_state_dict(save_path)
                    logger.info(f"New best EER: {100*best_eer:.2f}% at epoch {epoch}")
                else:
                    worse_epochs += 1
                
                if worse_epochs >= patience:
                    logger.info(f"Early stopping at epoch {epoch} due to no improvement in EER for {patience * args['eval_interval']} epochs.")
                    break
        logger.info(f"Training complete! Best EER: {100*best_eer:.2f}% at epoch {best_epoch}")
        logger.remove(log_id)
    
    def _load_features(
        self,
        audio_data: List[np.ndarray],
        cache_name: Optional[str] = None,
        batch_size: int = 16,
    ) -> torch.Tensor:
        if cache_name is not None:
            cache_path = os.path.join(os.path.dirname(__file__), "cache", f"{cache_name}.pt")
            if cache_path and os.path.exists(cache_path):
                logger.info(f"Feature cache found at {cache_path}")
                cached = torch.load(cache_path, map_location=self.device)
                return cached.to(self.device, non_blocking=True)
            
        chunks = []
        for i in range(0, len(audio_data), batch_size):
            batch_audio = audio_data[i : i + batch_size]
            features = self.extractor(
                batch_audio,
                padding=True,
                truncation=True,
                max_length=10000,
            )
            chunks.extend(features)
            del batch_audio

        features = torch.cat(chunks, dim=0)  # [N, T, H]

        if cache_name is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(features.cpu(), cache_path)
            logger.info(f"Feature cache saved to {cache_path}")
            
        return features

    def _load_default(self, split: str = "train", limit: Optional[int] = 512, shuffle: bool = False, seed: Optional[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        real_data = []
        fake_data = []
        logger.info(f"Loading default ASVspoof2019 LA {split} ...")
        dataset = load_dataset("./ASVspoof_2019_LA", split=split)
        if shuffle:
            dataset = dataset.shuffle(seed=seed if seed is not None else self.seed)
        real_count = 0
        fake_count = 0
        for item in dataset:
            if item["key"] == 0 and real_count < limit:
                real_data.append(item["audio"]["array"])
                real_count += 1
            elif item["key"] == 1 and fake_count < limit:
                fake_data.append(item["audio"]["array"])
                fake_count += 1
            if real_count >= limit and fake_count >= limit:
                break

        return real_data, fake_data

    def _aggregate_data(self, real_data: List[np.ndarray], fake_data: List[np.ndarray]) -> Tuple[List[np.ndarray], List[Label]]:
        data = real_data + fake_data
        labels = [Label.real] * len(real_data) + [Label.fake] * len(fake_data)
        return data, labels

    @torch.inference_mode()
    def _precompute_ref_cache(self, fea_real: torch.Tensor, fea_fake: torch.Tensor):
        self.net.basemodel.eval()
        self._ref_cache = {
            "fea_real": fea_real,
            "fea_fake": fea_fake,
        }
        self._ref_cache["net_real"] = self.net(self._ref_cache["fea_real"])
        self._ref_cache["net_fake"] = self.net(self._ref_cache["fea_fake"])

    def evaluate(
        self,
        data: List[np.ndarray],
        labels: List[Label],
        metrics: List[str],
        sr: int,
        in_domain: bool = False,
        dataset_name: Optional[str] = None,
        ref_data: Optional[List[np.ndarray]] = None,
        ref_labels: Optional[List[Label]] = None
    ) -> dict:
        torch.manual_seed(self.seed)
        if not in_domain:
            dataset_name = "default"
            ref_data, ref_labels = self._aggregate_data(*self._load_default(split="train", limit=self.ref_num))
            default_ckpt = os.path.join(os.path.dirname(__file__), "ckpts", f"default_best.pt")
            if not os.path.exists(default_ckpt):
                logger.info(f"Default model not found at {default_ckpt}, training from scratch")
                train_real_data, train_fake_data = self._load_default(split="train", limit=8192+self.ref_num)
                train_real_data, train_fake_data = train_real_data[self.ref_num:], train_fake_data[self.ref_num:]
                train_data, train_labels = self._aggregate_data(train_real_data, train_fake_data)
                eval_data, eval_labels = self._aggregate_data(*self._load_default(split="validation", limit=768))
                self.train(train_data, train_labels, eval_data, eval_labels, dataset_name="default", ref_data=ref_data, ref_labels=ref_labels, sr=sr)
        else:
            default_ckpt = os.path.join(os.path.dirname(__file__), "ckpts", f"{dataset_name}_best.pt")

        # cached during training
        if ref_data is None or ref_labels is None:
            ref_data, ref_labels = self.ref_data, self.ref_labels
        
        self.net.load_state_dict(default_ckpt)
        ref_fea = self._load_features(ref_data, cache_name=f"ref_{dataset_name}")
        ref_real, ref_fake = self._split_features(ref_fea, ref_labels)
        self._precompute_ref_cache(ref_real, ref_fake)

        labels = np.array(labels)
        if Label.real != 1:
            labels = 1 - labels

        results = {}
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}")
            func = getattr(self, f"_evaluate_{metric}")
            metric_rst = func(data, labels, sr=sr)
            results[metric] = metric_rst

        return results

    @torch.inference_mode()
    def _evaluate_eer(self, data: List[np.ndarray], labels: np.ndarray, sr: int) -> float:
        logger.info("Evaluating EER")
        self.net.basemodel.eval()

        if sr != self.sample_rate:
            data = [librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate) for audio in data]

        fea_data = self._load_features(data)
        scores = self._three_sample_test(fea_data, round=8)
        scores = np.array(scores)
        eer, _ = self._compute_eer(scores, labels)
        
        return float(eer)

    @torch.inference_mode()
    def _evaluate_auroc(self, data: List[np.ndarray], labels: np.ndarray, sr: int) -> float:
        logger.info("Evaluating AUROC")
        self.net.basemodel.eval()
        if sr != self.sample_rate:
            data = [librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate) for audio in data]
        
        fea_data = self._load_features(data)
        scores = self._three_sample_test(fea_data, round=8)
        scores = np.array(scores)
        
        return float(roc_auc_score(labels, scores))

    def _three_sample_test(
        self,
        fea_tests: torch.Tensor,
        round: int = 1,
        batch_size: int = 64
    ) -> List[float]:
        scores = []
        for i in tqdm(range(0, fea_tests.size(0), batch_size), desc="Running three sample test"):
            fea_tests_batch = fea_tests[i:i+batch_size]
            net_tests_batch = self.net(fea_tests_batch)
            
            for fea_test, net_test in zip(fea_tests_batch, net_tests_batch):
                fea_real = self._ref_cache["fea_real"]
                net_real = self._ref_cache["net_real"]
                fea_fake = self._ref_cache["fea_fake"]
                net_fake = self._ref_cache["net_fake"]

                idx_real = torch.randperm(len(fea_real))
                idx_fake = torch.randperm(len(fea_fake))
                fea_real = fea_real[idx_real]
                fea_fake = fea_fake[idx_fake]
                net_real = net_real[idx_real]
                net_fake = net_fake[idx_fake]
                fea_test = fea_test.unsqueeze(0)
                net_test = net_test.unsqueeze(0)

                p_value_list = []
                for j in range(max(round, 1)):
                    p_value = MMD_3_Sample_Test(
                        net_test,
                        net_real[[j]],
                        net_fake[[j]],
                        fea_test.reshape(fea_test.shape[0], -1),
                        fea_real[[j]].reshape(fea_real[[j]].shape[0], -1),
                        fea_fake[[j]].reshape(fea_fake[[j]].shape[0], -1),
                        self.net.sigma,
                        self.net.sigma0_u,
                        self.net.ep,
                    )
                    p_value_list.append(p_value)        
                _, p_value = combine_pvalues(p_value_list, method='stouffer')
                if np.isnan(p_value):
                    logger.warning("NaN score encountered")
                    p_value = 1.0
                scores.append(float(p_value))

        return scores

    def _compute_eer(self, scores: np.ndarray, labels: np.ndarray) -> Tuple[Any, Any]:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1 - tpr

        eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
        eer = (fpr[np.nanargmin(np.abs(fnr - fpr))] + fnr[np.nanargmin(np.abs(fnr - fpr))]) / 2

        return eer, eer_threshold

    def _auto_tune_bandwidths(self, feas_real: torch.Tensor, feas_fake: torch.Tensor, sample_size: int = 512):
        """Estimate reasonable initial values for sigma (original space) and sigma0_u (hidden space).

        The method samples up to ``sample_size`` utterances from the reference real & fake pools,
        computes pair-wise squared Euclidean distances in both the original feature space and the
        network hidden space, and sets ``self.net.sigma`` to the median original-space distance and
        ``self.net.sigma0_u`` to the median hidden-space distance.
        """
        with torch.inference_mode():
            # Sample from real and fake features and concatenate
            num_real = feas_real.size(0)
            num_fake = feas_fake.size(0)
            total = num_real + num_fake

            if total > sample_size:
                # Proportional sampling
                real_sample_size = int(sample_size * num_real / total)
                fake_sample_size = sample_size - real_sample_size

                real_indices = torch.randperm(num_real, device=feas_real.device)[:real_sample_size]
                fake_indices = torch.randperm(num_fake, device=feas_fake.device)[:fake_sample_size]
                
                feas = torch.cat([feas_real[real_indices], feas_fake[fake_indices]], dim=0)
            else:
                feas = torch.cat([feas_real, feas_fake], dim=0)

            # Original-space distances
            feas_flat = feas.view(feas.size(0), -1)
            D_org = torch.cdist(feas_flat, feas_flat, p=2).pow(2)
            sigma_org = torch.median(D_org)  # 0.5 quantile

            # Hidden-space distances (one forward pass)
            hidden = self.net(feas)
            D_hid = torch.cdist(hidden, hidden, p=2).pow(2)
            sigma_hid = torch.quantile(D_hid, 0.9)

            eps = 1e-6
            sigma_org = torch.clamp(sigma_org, min=eps)
            sigma_hid = torch.clamp(sigma_hid, min=eps)

            with torch.no_grad():
                self.net.sigma.data.fill_(sigma_org.item())
                self.net.sigma0_u.data.fill_(sigma_hid.item())

            logger.info(
                f"Auto-tuned sigma={sigma_org.item():.3f}, "
                f"sigma0_u={sigma_hid.item():.3f}"
            )