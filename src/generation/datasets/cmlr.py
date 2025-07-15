import os
import json
import logging
import soundfile as sf
from tqdm import tqdm

from models.base import BaseTTS
from .base import BaseRawDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class CMLR(BaseRawDataset):
    def __init__(self, data_dir="data/CMLR"):
        super().__init__(data_dir)
        self.meta_path = os.path.join(self.data_dir, "meta.json")

    def generate(self, tts_model: BaseTTS, *args, **kwargs):
        output_dir = os.path.join(self.data_dir, "audio/fake")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(self.meta_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)

        for item in tqdm(meta_data, desc="Generating audio"):
            text = item['text']
            audio_path = os.path.join(self.data_dir, item['audio']['real'])
            output_path = audio_path.replace("audio/real", "audio/fake")

            try:
                # Create directory structure for the output file
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                fake_audio, sample_rate = tts_model.infer(text, ref_audio=audio_path, language="zh-cn")
                sf.write(output_path, fake_audio, sample_rate)
                logger.info(f"Generated audio at {output_path}")
                item['audio']['fake'] = output_path
            except Exception as e:
                logger.error(f"While generating audio for text: {text}")
                logger.error(e)
                continue

        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Meta data updated and saved to {self.meta_path}")