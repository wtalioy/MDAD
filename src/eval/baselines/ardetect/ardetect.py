from typing import Dict, Any, cast, List, Tuple
import soundfile as sf
import numpy as np
from loguru import logger
from tqdm import tqdm
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sklearn.metrics import roc_curve
from eval.baselines.base import Baseline
from eval.baselines.ardetect.mmd_model import ModelLoader
from eval.baselines.ardetect.mmd_utils import MMD_3_Sample_Test
from eval.config import Label

# Fix datasets import conflict by temporarily modifying sys.path
import sys
_original_path = sys.path.copy()
sys.path = [p for p in sys.path if not (p.endswith('src') or p.endswith('src/eval'))]
try:
    from datasets import load_dataset as hf_load_dataset
finally:
    sys.path = _original_path

class ARDetect(Baseline):
    def __init__(self,
                 wav2vec_model_path: str | None = None,
                 mmd_model_path: str = "src/eval/baselines/ardetect/mmd.pth",
                 device: str = "cuda",
                 **kwargs):
        self.name = "ARDetect"
        self.device = device
        self.sample_rate = 16000
        self.segment_sec = 0.625
        self.supported_metrics = ['eer']

        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model_path or "facebook/wav2vec2-xls-r-2b")
        self.model = Wav2Vec2Model.from_pretrained(wav2vec_model_path or "facebook/wav2vec2-xls-r-2b")
        self.model = self.model.to(device).eval()

        self.net = ModelLoader.from_pretrained(model_path=mmd_model_path, device=device)

        self.real_data, self.fake_data = self._load_ref_data()
        self.real_features = self._load_features(self.real_data)
        self.fake_features = self._load_features(self.fake_data)

    def _load_ref_data(self):
        return self._load_asvspoof()
    
    def _load_features(self, audio_data: List[np.ndarray], batch_size: int = 8) -> List[torch.Tensor]:
        features = []
        for i in tqdm(range(0, len(audio_data), batch_size), desc="Extracting features"):
            batch_audio = audio_data[i:i + batch_size]

            batch_inputs = []
            for audio_array in batch_audio:
                inputs = self.extractor(
                    audio_array,
                    sampling_rate=self.sample_rate,
                    padding="max_length",
                    max_length=25000,
                    truncation=True,
                    return_tensors="pt"
                )
                batch_inputs.append(inputs)
            
            if batch_inputs:
                batch_features = []
                for inputs in batch_inputs:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model(**inputs)
                    batch_features.append(outputs.last_hidden_state.cpu())
                features.extend(batch_features)
        
        return features


    def _load_asvspoof(self, limit: int = 1024) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        real_data = []
        fake_data = []
        data = hf_load_dataset("Bisher/ASVspoof_2019_LA", split="train")
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
            audio_segments.append(audio[:, start_frame:end_frame])

        remaining_frames = total_frames % segment_length

        if remaining_frames > 0:
            if num_full_segments == 0:
                audio_segments.append(audio)
            else:
                if len(audio_segments) >= 1:
                    last_segment = audio_segments.pop()
                    combined_segment = np.concatenate((last_segment, audio[:, num_full_segments * segment_length:]), axis=1)
                    audio_segments.append(combined_segment)
                else:
                    audio_segments.append(audio[:, num_full_segments * segment_length:])

        return audio_segments


    def evaluate(self, data: List[str], labels: np.ndarray, metrics: List[str]) -> dict:
        feature_list = []
        for audio_path in data:
            audio, sr = sf.read(audio_path)
            if sr != self.sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            segments = self._segment_audio(audio)
            feature_list.append(self._load_features(segments))
        
        results = {}
        for metric in metrics:
            if metric not in self.supported_metrics:
                raise ValueError(f"Unsupported metric: {metric}")
            func = getattr(self, f"_evaluate_{metric}")
            metric_rst = func(feature_list, labels)
            results[metric] = metric_rst

        return results


    def _evaluate_eer(self, feature_list: List[List[torch.Tensor]], labels: np.ndarray) -> float:
        fea_real = feature_list[labels == Label.real.value]
        fea_fake = feature_list[labels == Label.fake.value]

        real_test_stats = []
        generated_test_stats = []
        
        # For real test samples: compute test statistics against training distributions
        for fea_test_real in fea_real:
            stats = self._three_sample_test(fea_test_real, round=10)
            real_test_stats.append(stats)
        
        # For generated test samples: compute test statistics against training distributions  
        for fea_test_generated in fea_fake:
            stats = self._three_sample_test(fea_test_generated, round=10)
            generated_test_stats.append(stats)
        
        # Convert to numpy arrays for EER calculation
        real_test_stats = np.array(real_test_stats)
        generated_test_stats = np.array(generated_test_stats)
        
        # Remove any NaN values
        real_test_stats = real_test_stats[~np.isnan(real_test_stats)]
        generated_test_stats = generated_test_stats[~np.isnan(generated_test_stats)]
        
        if len(real_test_stats) == 0 or len(generated_test_stats) == 0:
            logger.warning("No valid test statistics computed for EER calculation")
            return 0.5
        
        # Compute EER using the test statistics
        eer, eer_threshold = self._compute_eer(real_test_stats, generated_test_stats)
        
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
            fea_real = fea_real[torch.randperm(len(fea_real))[:min_len]]
            fea_generated = fea_generated[torch.randperm(len(fea_generated))[:min_len]]
            fea_test = fea_test[torch.randperm(len(fea_test))[:min_len]]

            h_u, p_value, t = MMD_3_Sample_Test(
                self.net(fea_test),
                self.net(fea_real),
                self.net(fea_generated),
                fea_test.view(fea_test.shape[0], -1),
                fea_real.view(fea_real.shape[0], -1),
                fea_generated.view(fea_generated.shape[0], -1),
                self.net.sigma,
                self.net.sigma0_u,
                self.net.ep,
                0.05,
            )
        
            h_u_list.append(h_u)
            p_value_list.append(p_value)
            t_list.append(t)

        power = sum(h_u_list) / len(h_u_list) if h_u_list else 0.0
        return power


    def _compute_eer(self, target_scores: np.ndarray, nontarget_scores: np.ndarray) -> Tuple[Any, Any]:
        """
        Compute Equal Error Rate (EER) from target and nontarget scores.
        Args:
            target_scores: Scores for target samples (e.g., real audio)
            nontarget_scores: Scores for nontarget samples (e.g., generated audio)
        Returns:
            eer: Equal Error Rate (EER) value
            eer_threshold: Threshold at which EER occurs
        """
        scores = np.concatenate([target_scores, nontarget_scores])
        labels = np.concatenate([np.zeros(len(target_scores)), np.ones(len(nontarget_scores))])

        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1 - tpr

        eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
        eer = (fpr[np.nanargmin(np.abs(fnr - fpr))] + fnr[np.nanargmin(np.abs(fnr - fpr))]) / 2

        return eer, eer_threshold