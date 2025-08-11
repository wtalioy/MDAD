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
from baselines import Baseline
from baselines.ardetect.mmd_model import MMDModel
from baselines.ardetect.mmd_utils import MMD_3_Sample_Test, MMDu
from config import Label

class ARDetect(Baseline):
    def __init__(self,
                 mmd_model_path: str = "src/eval/baselines/ardetect/mmd.pth",
                 device: str = "cuda",
                 **kwargs):
        self.name = "ARDetect"
        self.device = device
        self.sample_rate = 16000
        self.segment_sec = 0.625
        self.supported_metrics = ['eer', 'auroc']

        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained("/home/ruimingwang/AR-Detect/cache/extractors/wav2vec2-xls-r-2b")
        self.model = Wav2Vec2Model.from_pretrained("/home/ruimingwang/AR-Detect/cache/extractors/wav2vec2-xls-r-2b").to(device)
        self.model.eval()

        self.net = MMDModel(config=self._load_model_config(os.path.dirname(__file__)), device=device)
        self.net.load_state_dict(mmd_model_path)

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

    def _train_epoch(self, epoch: int, fea_real: torch.Tensor, fea_fake: torch.Tensor, batch_size: int):
        self.net.basemodel.train()

        fea_real = fea_real[torch.randperm(fea_real.shape[0])]
        fea_fake = fea_fake[torch.randperm(fea_fake.shape[0])]
        min_len = min(len(fea_real), len(fea_fake))
        fea_real = fea_real[:min_len]
        fea_fake = fea_fake[:min_len]

        batch_size = min(batch_size, len(fea_fake))
        
        with tqdm(total = len(fea_real), desc="Training") as pbar:
            for start_idx in range(0, len(fea_real), batch_size):
                end_idx = min(start_idx + batch_size, len(fea_real))
                fea_real_sample = fea_real[start_idx:end_idx].to(self.device, non_blocking=True)
                fea_fake_sample = fea_fake[start_idx:end_idx].to(self.device, non_blocking=True)
                inputs = torch.cat([fea_real_sample, fea_fake_sample], dim=0)
                outputs = self.net(inputs)

                temp = MMDu(
                    outputs,
                    inputs.shape[0],
                    inputs.view(inputs.shape[0], -1),
                    self.net.sigma,
                    self.net.sigma0_u,
                    self.net.ep,
                    coeff_xy=self.args.coeff_xy,
                    is_yy_zero=self.args.is_yy_zero,
                    is_xx_zero=self.args.is_xx_zero,
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

    def train(self, train_data: List[np.ndarray], train_labels: np.ndarray, eval_data: List[np.ndarray], eval_labels: np.ndarray, dataset_name: str):
        args = self._load_train_config(os.path.dirname(__file__), dataset_name)
        train_real = self._load_features(train_data[train_labels == Label.real])
        train_fake = self._load_features(train_data[train_labels == Label.fake])

        log_id = logger.add("logs/train.log", rotation="100 MB", retention="60 days")
        logger.info(f"Training ARDetect on {dataset_name}")
        
        best_eer = 100
        best_epoch = 0
        save_path = os.path.join(os.path.dirname(__file__), "ckpts", f"{dataset_name}_best.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        for epoch in range(args['num_epoch']):
            self._train_epoch(epoch, train_real, train_fake)
            eer = self._evaluate_eer(eval_data, eval_labels)
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
        dataset_name: str = "default",
        batch_size: int = 8,
    ) -> List[torch.Tensor]:
        cache_path = os.path.join(os.path.dirname(__file__), "cache", f"{dataset_name}_features.pt")
        if cache_path and os.path.exists(cache_path):
            return torch.load(cache_path)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
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

        torch.save(features, cache_path, weights_only=True)
        return features

    def _load_default(self, split: str = "train", limit: Optional[int] = 512, shuffle: bool = False):
        data = []
        labels = []
        dataset = load_dataset("Bisher/ASVspoof_2019_LA", split=split)
        if shuffle:
            dataset = dataset.shuffle(seed=42)
        for i, item in enumerate(tqdm(dataset, desc=f"Loading ASVspoof2019_LA {split} split")):
            if limit is not None and i >= limit:
                break
            data.append(item["audio"]["array"])
            labels.append(Label.real if item["key"] == 0 else Label.fake)

        return data, np.array(labels)

    def _segment_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        segment_length = round(self.segment_sec * self.sample_rate)
        
        total_frames = audio.shape[0]
        num_full_segments = total_frames // segment_length
        audio_segments = []

        for i in range(num_full_segments):
            start_frame = i * segment_length
            end_frame = start_frame + segment_length
            audio_segments.append(audio[start_frame:end_frame])

        remaining_frames = total_frames % segment_length

        if remaining_frames > 0:
            if num_full_segments == 0:
                audio_segments.append(audio)
            else:
                if len(audio_segments) >= 1:
                    last_segment = audio_segments.pop()
                    combined_segment = np.concatenate((last_segment, audio[num_full_segments * segment_length:]), axis=0)
                    audio_segments.append(combined_segment)
                else:
                    audio_segments.append(audio[num_full_segments * segment_length:])

        return audio_segments

    def evaluate(
        self,
        data: List[np.ndarray],
        labels: np.ndarray,
        metrics: List[str],
        sr: int,
        in_domain: bool = False,
        dataset_name: Optional[str] = None,
        ref_data: Optional[List[np.ndarray]] = None,
        ref_labels: Optional[np.ndarray] = None
    ) -> dict:
        if not in_domain:
            dataset_name = "default"
            default_ckpt = os.path.join(os.path.dirname(__file__), "ckpts", f"{dataset_name}_best.pt")
            if not os.path.exists(default_ckpt):
                logger.info(f"Default model not found at {default_ckpt}, training from scratch")
                train_data, train_labels = self._load_default(split="train", limit=8192)
                eval_data, eval_labels = self._load_default(split="validation", limit=512)
                self.train(train_data, train_labels, eval_data, eval_labels, dataset_name=dataset_name)
            self.net.load_state_dict(default_ckpt)
            ref_data, ref_labels = self._load_default(split="train", limit=512)

        ref_fea = self._load_features(ref_data, dataset_name=dataset_name)
        fea_real = ref_fea[ref_labels == Label.real]
        fea_fake = ref_fea[ref_labels == Label.fake]

        if Label.real != 1:
            labels = 1 - labels

        results = {}
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}")
            func = getattr(self, f"_evaluate_{metric}")
            metric_rst = func(data, labels, fea_real, fea_fake, sr=sr)
            results[metric] = metric_rst

        return results

    @torch.inference_mode()
    def _evaluate_eer(self, data: List[np.ndarray], labels: np.ndarray, fea_real: List[torch.Tensor], fea_fake: List[torch.Tensor], sr: int) -> float:
        scores = []
        for audio in tqdm(data, desc="Evaluating EER"):
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            segments = self._segment_audio(audio)
            if len(segments) > 1:
                fea_test = self._load_features(segments, batch_size=8)
                score = self._three_sample_test(fea_test, fea_real, fea_fake, round=4)
                scores.append(score)
            else:
                logger.warning(f"Audio too short to test")
                scores.append(1)

        scores = np.array(scores)
        eer, _ = self._compute_eer(scores, labels)
        
        return float(eer)

    @torch.inference_mode()
    def _evaluate_auroc(self, data: List[np.ndarray], labels: np.ndarray, fea_real: List[torch.Tensor], fea_fake: List[torch.Tensor], sr: int) -> float:
        scores = []
        for audio in tqdm(data, desc="Evaluating AUROC"):
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            segments = self._segment_audio(audio)
            if len(segments) > 1:
                fea_test = self._load_features(segments, batch_size=8)
                score = self._three_sample_test(fea_test, fea_real, fea_fake, round=4)
                scores.append(score)
            else:
                logger.warning(f"Audio too short to test")
                scores.append(1)
        scores = np.array(scores)
        return float(roc_auc_score(labels, scores))

    def _three_sample_test(
        self,
        fea_test_single: List[torch.Tensor],
        fea_real: List[torch.Tensor],
        fea_fake: List[torch.Tensor],
        round: int = 10,
    ) -> float:
        self.net.eval()
        fea_real = torch.cat(fea_real, dim=0).to(self.device)
        fea_generated = torch.cat(fea_fake, dim=0).to(self.device)
        fea_test = torch.cat(fea_test_single, dim=0).to(self.device)
        min_len = min(len(fea_real), len(fea_generated), len(fea_test))

        h_u_list = []
        p_value_list = []
        t_list = []

        for _ in range(max(round, 1)):
            fea_real_sample = fea_real[torch.randperm(len(fea_real))[:min_len]]
            fea_generated_sample = fea_generated[torch.randperm(len(fea_generated))[:min_len]]
            fea_test_sample = fea_test[torch.randperm(len(fea_test))[:min_len]]

            with torch.inference_mode():
                net_test = self.net(fea_test_sample)
                net_real = self.net(fea_real_sample)
                net_generated = self.net(fea_generated_sample)

            h_u, p_value, t = MMD_3_Sample_Test(
                net_test,
                net_real,
                net_generated,
                fea_test_sample.view(fea_test_sample.shape[0], -1),
                fea_real_sample.view(fea_real_sample.shape[0], -1),
                fea_generated_sample.view(fea_generated_sample.shape[0], -1),
                self.net.sigma,
                self.net.sigma0_u,
                self.net.ep,
                0.05,
            )
        
            h_u_list.append(h_u)
            p_value_list.append(p_value)
            t_list.append(t)
            
            del (fea_real_sample, fea_generated_sample, fea_test_sample,
                 net_test, net_real, net_generated)

        del fea_real, fea_generated, fea_test
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        power = sum(h_u_list) / len(h_u_list) if h_u_list else 0.0
        return power

    def _compute_eer(self, scores: np.ndarray, labels: np.ndarray) -> Tuple[Any, Any]:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1 - tpr

        eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
        eer = (fpr[np.nanargmin(np.abs(fnr - fpr))] + fnr[np.nanargmin(np.abs(fnr - fpr))]) / 2

        return eer, eer_threshold