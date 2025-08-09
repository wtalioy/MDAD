import json
from typing import List
import nemo.collections.asr as nemo_asr

class Parakeet:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")

    def transcribe(self, audio_paths: List[str], language: str = "en") -> List[str]:
        return [(output.text, output.timestamp['word']) for output in self.model.transcribe(audio_paths, timestamps=True)]

    def get_word_timestamps(self, audio_paths: List[str]) -> List[str]:
        outputs = self.model.transcribe(audio_paths, timestamps=True)
        return [output.timestamp['word'] for output in outputs]

if __name__ == "__main__":
    from argparse import ArgumentParser
    from tqdm import tqdm
    from loguru import logger
    import os
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/CallFriend")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    model = Parakeet()
    with open(os.path.join(args.data_dir, "meta.json"), "r") as f:
        meta_data = json.load(f)
    new_meta_data = []
    for i in tqdm(range(0, len(meta_data), args.batch_size)):
        batch = meta_data[i:i+args.batch_size]
        audio_paths = [os.path.join(args.data_dir, item["audio"]["real"]) for item in batch]
        try:
            texts = model.transcribe(audio_paths, args.language)
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            continue
        for j, item in enumerate(batch):
            item["text"] = texts[j][0]
            word_timestamps = texts[j][1]
            if len(word_timestamps) < 3:
                logger.warning(f"Word timestamps are less than 3: {word_timestamps}")
                logger.warning(f"Audio path: {audio_paths[j]}")
                if os.path.exists(audio_paths[j]):
                    os.remove(audio_paths[j])
                if os.path.exists(audio_paths[j].replace("real", "fake")):
                    os.remove(audio_paths[j].replace("real", "fake"))
            else:
                new_meta_data.append(item)
    with open(os.path.join(args.data_dir, "meta.json"), "w") as f:
        json.dump(new_meta_data, f, indent=2, ensure_ascii=False)