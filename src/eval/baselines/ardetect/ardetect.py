from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from eval.baselines.base import Baseline
from eval.baselines.ardetect.mmd_model import ModelLoader

class ARDetect(Baseline):
    def __init__(self,
                 wav2vec_model_path: str | None = None,
                 mmd_model_path: str = "src/eval/baselines/ardetect/mmd.pth",
                 device: str = "cuda",
                 **kwargs):
        self.name = "ARDetect"
        self.device = device

        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model_path or "facebook/wav2vec2-xls-r-2b")
        self.model = Wav2Vec2Model.from_pretrained(wav2vec_model_path or "facebook/wav2vec2-xls-r-2b")
        self.model = self.model.to(device).eval()

        self.net = ModelLoader.from_pretrained(model_path=mmd_model_path, device=device)
