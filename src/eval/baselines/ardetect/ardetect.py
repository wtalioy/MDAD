from typing import Dict, Any, cast, List, Tuple, Optional
import os
import soundfile as sf
import numpy as np
from loguru import logger
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from datasets import load_dataset
from sklearn.metrics import roc_curve
from baselines import Baseline
from baselines.ardetect.mmd_model import ModelLoader
from baselines.ardetect.mmd_utils import MMD_3_Sample_Test

class ARDetect(Baseline):
    def __init__(self,
                 mmd_model_path: str = "src/eval/baselines/ardetect/mmd.pth",
                 device: str = "cuda",
                 **kwargs):
        self.cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.device = device
        self.sample_rate = 16000
        self.segment_sec = 0.625
        self.supported_metrics = ['eer']

        # Check for multiple GPUs
        self.num_gpus = torch.cuda.device_count() if device == "cuda" else 1
        logger.info(f"Using {self.num_gpus} GPU(s)")

        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-2b")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-2b")
        
        # Enable multi-GPU support for Wav2Vec2 model
        if self.num_gpus > 1:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(device).eval()

        self.net = ModelLoader.from_pretrained(model_path=mmd_model_path, device=device)
        # Enable multi-GPU support for MMD model
        if self.num_gpus > 1:
            self.net.model = nn.DataParallel(self.net.model)

        self.real_data, self.fake_data = self._load_ref_data()
        self.real_features, self.fake_features = self._load_ref_features()

    def _load_ref_features(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        batch_size = 8 * self.num_gpus if self.num_gpus > 1 else 1
        logger.info(f"Loading real features with batch size {batch_size}")
        if not os.path.exists(os.path.join(self.cache_dir, "real_ref_features.pt")):
            real_features = self._load_features(self.real_data, batch_size=batch_size)
            torch.save(real_features, os.path.join(self.cache_dir, "real_ref_features.pt"))
        else:
            real_features = torch.load(os.path.join(self.cache_dir, "real_ref_features.pt"))
        logger.info(f"Loading fake features with batch size {batch_size}")
        if not os.path.exists(os.path.join(self.cache_dir, "fake_ref_features.pt")):
            fake_features = self._load_features(self.fake_data, batch_size=batch_size)
            torch.save(fake_features, os.path.join(self.cache_dir, "fake_ref_features.pt"))
        else:
            fake_features = torch.load(os.path.join(self.cache_dir, "fake_ref_features.pt"))
        return real_features, fake_features

    def _load_ref_data(self):
        return self._load_asvspoof()
    
    def _load_features(self, audio_data: List[np.ndarray], batch_size: int = 8) -> List[torch.Tensor]:
        features = []
        effective_batch_size = batch_size * self.num_gpus if self.num_gpus > 1 else batch_size
        
        for i in range(0, len(audio_data), effective_batch_size):
            batch_audio = audio_data[i:i + effective_batch_size]

            batch_inputs = []
            for audio_array in batch_audio:
                inputs = self.extractor(
                    audio_array,
                    sampling_rate=self.sample_rate,
                    padding="max_length",
                    max_length=12000,
                    truncation=True,
                    return_tensors="pt"
                )
                batch_inputs.append(inputs)
            
            if batch_inputs:
                # Stack inputs for batch processing
                if len(batch_inputs) > 1:
                    stacked_inputs = {}
                    for key in batch_inputs[0].keys():
                        stacked_inputs[key] = torch.cat([inp[key] for inp in batch_inputs], dim=0)
                    stacked_inputs = {k: v.to(self.device) for k, v in stacked_inputs.items()}
                    
                    # Process entire batch at once
                    with torch.no_grad():
                        outputs = self.model(**stacked_inputs)
                        batch_features = torch.split(outputs.last_hidden_state.cpu(), 1, dim=0)
                        features.extend([feat.squeeze(0) for feat in batch_features])
                else:
                    # Single item processing
                    inputs = {k: v.to(self.device) for k, v in batch_inputs[0].items()}
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        features.append(outputs.last_hidden_state.cpu().squeeze(0))
        
        return features


    def _load_asvspoof(self, limit: int = 1024) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        real_data = []
        fake_data = []
        data = load_dataset("Bisher/ASVspoof_2019_LA", split="train")
        for item in data:
            item = cast(Dict[str, Any], item)
            if item["key"] == 0 and len(real_data) < limit:
                real_data.append(item["audio"]["array"])
            elif item["key"] == 1 and len(fake_data) < limit:
                fake_data.append(item["audio"]["array"])
            else:
                continue
        return real_data, fake_data


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


    @torch.no_grad()
    def evaluate(self, data: List[str], labels: np.ndarray, metrics: List[str], in_domain: bool = False, dataset_name: Optional[str] = None) -> dict:
        feature_list = []
        # Use larger batch size for multi-GPU evaluation
        batch_size = 8 * self.num_gpus if self.num_gpus > 1 else 8
        
        for audio_path in tqdm(data, desc="Loading test data"):
            audio, sr = sf.read(audio_path)
            if sr != self.sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            segments = self._segment_audio(audio)
            feature_list.append(self._load_features(segments, batch_size=batch_size))
        
        results = {}
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}")
            func = getattr(self, f"_evaluate_{metric}")
            metric_rst = func(feature_list, labels)
            results[metric] = metric_rst

        return results


    def _evaluate_eer(self, feature_list: List[List[torch.Tensor]], labels: np.ndarray) -> float:
        scores = []
        for fea_test in tqdm(feature_list, desc="Evaluating EER"):
            score = self._three_sample_test(fea_test, round=10)
            scores.append(score)

        scores = np.array(scores)
        eer, _ = self._compute_eer(scores, labels)
        
        return float(eer)


    def _three_sample_test(
        self,
        fea_test_single: List[torch.Tensor],
        round: int = 10,
    ) -> float:
        ## Concatenate the hidden states of the real and generated data
        fea_real = torch.cat(self.real_features, dim=0).to(self.device)
        fea_generated = torch.cat(self.fake_features, dim=0).to(self.device)
        fea_test = torch.cat(fea_test_single, dim=0).to(self.device)
        ## Cut the real and generated data to the same length (according to the minimum length of the real and generated data)
        min_len = min(len(fea_real), len(fea_generated), len(fea_test))


        h_u_list = []
        p_value_list = []
        t_list = []

        for _ in range(max(round, 1)):
            fea_real_sample = fea_real[torch.randperm(len(fea_real))[:min_len]]
            fea_generated_sample = fea_generated[torch.randperm(len(fea_generated))[:min_len]]
            fea_test_sample = fea_test[torch.randperm(len(fea_test))[:min_len]]

            # Use the MMD network (potentially with DataParallel)
            with torch.no_grad():
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
                self.net.module.sigma if hasattr(self.net, 'module') else self.net.sigma,
                self.net.module.sigma0_u if hasattr(self.net, 'module') else self.net.sigma0_u,
                self.net.module.ep if hasattr(self.net, 'module') else self.net.ep,
                0.05,
            )
        
            h_u_list.append(h_u)
            p_value_list.append(p_value)
            t_list.append(t)

        power = sum(h_u_list) / len(h_u_list) if h_u_list else 0.0
        return power


    def _compute_eer(self, scores: np.ndarray, labels: np.ndarray) -> Tuple[Any, Any]:
        """
        Compute Equal Error Rate (EER) from target and nontarget scores.
        Args:
            target_scores: Scores for target samples (e.g., real audio)
            nontarget_scores: Scores for nontarget samples (e.g., generated audio)
        Returns:
            eer: Equal Error Rate (EER) value
            eer_threshold: Threshold at which EER occurs
        """
        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1 - tpr

        eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
        eer = (fpr[np.nanargmin(np.abs(fnr - fpr))] + fnr[np.nanargmin(np.abs(fnr - fpr))]) / 2

        return eer, eer_threshold