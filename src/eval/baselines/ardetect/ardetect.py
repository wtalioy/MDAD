from typing import Any, List, Tuple, Optional
import os
import librosa
import numpy as np
from loguru import logger
from tqdm import tqdm
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from datasets import load_dataset
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import combine_pvalues
from baselines import Baseline
from baselines.ardetect.mmd_model import MMDModel
from baselines.ardetect.mmd_utils import MMD_3_Sample_Test, MMDu
from config import Label

class ARDetect(Baseline):
    def __init__(self, device: str = "cuda", **kwargs):
        self.name = "ARDetect"
        self.device = device
        self.sample_rate = 16000
        self.ref_num = 200
        self.seed = 34
        self.supported_metrics = ['eer', 'auroc']

        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained("./wav2vec2-xls-r-2b")
        self.model = Wav2Vec2Model.from_pretrained("./wav2vec2-xls-r-2b").to(device)
        self.model.eval()

        self.net = MMDModel(config=self._load_model_config(os.path.dirname(__file__)), device=device)

    def _init_train(self, args: dict):
        self.net.sigma.requires_grad = True
        self.net.sigma0_u.requires_grad = True
        self.net.ep.requires_grad = True
        
        self.optimizer = torch.optim.Adam(
            list(self.net.basemodel.parameters()) + [self.net.sigma, self.net.sigma0_u, self.net.ep],
            lr=args['lr'],
            weight_decay=args['wd']
        )

        torch.manual_seed(args['seed'])
        torch.cuda.manual_seed(args['seed'])

    def _split_features(self, fea: torch.Tensor, labels: List[Label]) -> Tuple[torch.Tensor, torch.Tensor]:
        fea_real = np.array(fea)[np.array(labels) == Label.real]
        fea_fake = np.array(fea)[np.array(labels) == Label.fake]
        fea_real = torch.from_numpy(fea_real).squeeze(1).to(self.device)
        fea_fake = torch.from_numpy(fea_fake).squeeze(1).to(self.device)
        return fea_real, fea_fake

    def _train_epoch(self, epoch: int, fea_real: torch.Tensor, fea_fake: torch.Tensor, batch_size: int):
        self.net.basemodel.train()

        fea_real = fea_real[torch.randperm(fea_real.shape[0])]
        fea_fake = fea_fake[torch.randperm(fea_fake.shape[0])]
        min_len = min(len(fea_real), len(fea_fake))
        fea_real = fea_real[:min_len]
        fea_fake = fea_fake[:min_len]

        batch_size = min(batch_size, len(fea_fake))
        
        with tqdm(total = len(fea_real) // batch_size + 1, desc="Training") as pbar:
            for start_idx in range(0, len(fea_real), batch_size):
                end_idx = min(start_idx + batch_size, len(fea_real))
                fea_real_sample = fea_real[start_idx:end_idx].to(self.device, non_blocking=True)
                fea_fake_sample = fea_fake[start_idx:end_idx].to(self.device, non_blocking=True)
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
                loss.backward()
                self.optimizer.step()
                pbar.set_description('epoch: %d, loss:%.3f'%(epoch, loss.item()))
                pbar.update(1)

    def train(
        self,
        train_data: List[np.ndarray],
        train_labels: List[Label],
        eval_data: List[np.ndarray],
        eval_labels: List[Label],
        ref_data: List[np.ndarray],
        ref_labels: List[Label],
        dataset_name: str,
        sr: int,
    ):
        args = self._load_train_config(os.path.dirname(__file__), dataset_name)
        train_fea = self._load_features(train_data, cache_name=f"train_{dataset_name}")
        train_real, train_fake = self._split_features(train_fea, train_labels)
        ref_fea = self._load_features(ref_data, cache_name=f"ref_{dataset_name}")
        ref_real, ref_fake = self._split_features(ref_fea, ref_labels)

        log_id = logger.add("logs/train.log", rotation="10 MB", retention="60 days")
        logger.info(f"Training ARDetect on {dataset_name}")
        
        self._init_train(args)
        
        best_eer = 100
        best_epoch = 0
        save_path = os.path.join(os.path.dirname(__file__), "ckpts", f"{dataset_name}_best.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for epoch in range(args['num_epoch']):
            self._train_epoch(epoch, train_real, train_fake, batch_size=args['batch_size'])
            if epoch % 4 == 0:
                self._precompute_ref_cache(ref_real, ref_fake)
                eer = self._evaluate_eer(data=eval_data, labels=eval_labels, ref_real=ref_real, ref_fake=ref_fake, sr=sr, epoch=epoch)
                logger.info(f"Epoch {epoch} EER: {100*eer:.2f}%")
                if eer < best_eer:
                    best_eer = eer
                    best_epoch = epoch
                    self.net.save_state_dict(save_path)
                    logger.info(f"New best EER: {100*best_eer:.2f}% at epoch {epoch}")
        logger.info(f"Training complete! Best EER: {100*best_eer:.2f}% at epoch {best_epoch}")
        logger.remove(log_id)
    
    def _load_features(
        self,
        audio_data: List[np.ndarray],
        cache_name: Optional[str] = None,
        batch_size: int = 8,
    ) -> List[torch.Tensor]:
        if cache_name is not None:
            cache_path = os.path.join(os.path.dirname(__file__), "cache", f"{cache_name}.pt")
            if cache_path and os.path.exists(cache_path):
                logger.info(f"Feature cache found at {cache_path}")
                return torch.load(cache_path)
            
        features = []
        for i in range(0, len(audio_data), batch_size):
            batch_audio = audio_data[i : i + batch_size]
            inputs = self.extractor(
                batch_audio,
                sampling_rate=self.sample_rate,
                padding="max_length",
                max_length=10000,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

            with torch.inference_mode():
                last_hidden = self.model(**inputs).last_hidden_state.detach().cpu()

            features.extend(torch.split(last_hidden, 1, dim=0))

            del inputs, last_hidden, batch_audio

        if cache_name is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(features, cache_path)
            logger.info(f"Feature cache saved to {cache_path}")
            
        return features

    def _load_default(self, split: str = "train", limit: Optional[int] = 512, shuffle: bool = False, seed: Optional[int] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        real_data = []
        fake_data = []
        logger.info(f"Loading default ASVspoof2019 LA {split} ...")
        dataset = load_dataset("Bisher/ASVspoof_2019_LA", split=split)
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
            default_ckpt = os.path.join(os.path.dirname(__file__), "ckpts", f"{dataset_name}_best.pt")
            if not os.path.exists(default_ckpt):
                logger.info(f"Default model not found at {default_ckpt}, training from scratch")
                train_real_data, train_fake_data = self._load_default(split="train", limit=2048+self.ref_num)
                train_real_data, train_fake_data = train_real_data[self.ref_num:], train_fake_data[self.ref_num:]
                train_data, train_labels = self._aggregate_data(train_real_data, train_fake_data)
                eval_data, eval_labels = self._aggregate_data(*self._load_default(split="validation", limit=256))
                self.train(train_data, train_labels, eval_data, eval_labels, ref_data, ref_labels, dataset_name=dataset_name, sr=sr)
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
            metric_rst = func(data, labels, ref_real, ref_fake, sr=sr)
            results[metric] = metric_rst

        return results

    @torch.inference_mode()
    def _evaluate_eer(self, data: List[np.ndarray], labels: np.ndarray, ref_real: torch.Tensor, ref_fake: torch.Tensor, sr: int, epoch: Optional[int] = None) -> float:
        self.net.basemodel.eval()
        scores = []
        for i, audio in enumerate(tqdm(data, desc="Evaluating EER")):
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            fea_test = self._load_features([audio])[0]
            score, _ = self._three_sample_test(fea_test, ref_real, ref_fake, round=8)
            scores.append(score)
        scores = np.array(scores)
        eer, _ = self._compute_eer(scores, labels)
        
        return float(eer)

    @torch.inference_mode()
    def _evaluate_auroc(self, data: List[np.ndarray], labels: np.ndarray, ref_real: torch.Tensor, ref_fake: torch.Tensor, sr: int) -> float:
        self.net.basemodel.eval()
        scores = []
        for audio in tqdm(data, desc="Evaluating AUROC"):
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            fea_test = self._load_features([audio])[0]
            score, _ = self._three_sample_test(fea_test, ref_real, ref_fake, round=8)
            scores.append(score)
        scores = np.array(scores)
        return float(roc_auc_score(labels, scores))

    def _three_sample_test(
        self,
        fea_test: torch.Tensor,
        fea_real: torch.Tensor,
        fea_fake: torch.Tensor,
        round: int = 1,
        alpha: float = 0.498
    ) -> Tuple[float, List[float]]:
        fea_test = fea_test.to(self.device)
        net_test = self.net(fea_test)
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

        p_value_list = []

        for i in range(max(round, 1)):
            p_value = MMD_3_Sample_Test(
                net_test,
                net_real[[i]],
                net_fake[[i]],
                fea_test.view(fea_test.shape[0], -1),
                fea_real[[i]].view(fea_real[[i]].shape[0], -1),
                fea_fake[[i]].view(fea_fake[[i]].shape[0], -1),
                self.net.sigma,
                self.net.sigma0_u,
                self.net.ep,
                alpha
            )
        
            p_value_list.append(p_value)

        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        _, p_value = combine_pvalues(p_value_list, method='stouffer')
        
        return p_value, p_value_list

    def _compute_eer(self, scores: np.ndarray, labels: np.ndarray) -> Tuple[Any, Any]:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1 - tpr

        eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
        eer = (fpr[np.nanargmin(np.abs(fnr - fpr))] + fnr[np.nanargmin(np.abs(fnr - fpr))]) / 2

        return eer, eer_threshold