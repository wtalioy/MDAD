from .base import BaseRawDataset
import os
import json
from pydub import AudioSegment
import random
from tqdm import tqdm

class NoisySpeech(BaseRawDataset):
    def __init__(self, data_dir=None, *args, **kwargs):
        super().__init__(os.path.join(data_dir or "data", "NoisySpeech"), *args, **kwargs)
        data_sources = ["News", "Interview", "Podcast", "PublicSpeech", "Audiobook", "Movie", "PhoneCall/en"]
        self.data_sources = [os.path.join(data_dir, data_source) for data_source in data_sources]
        self.noise_files = {
            "concert": "src/generation/noise/concert.wav",
            "gusts": "src/generation/noise/gusts.wav",
            "traffic": "src/generation/noise/traffic.wav",
            "typing": "src/generation/noise/typing.wav"
        }
        self.ratio = 0.4
        self.noise_audios = self._load_noise_audios()

    def _load_noise_audios(self):
        noise_audios = {}
        for noise_type in self.noise_files:
            audio = AudioSegment.from_file(self.noise_files[noise_type])
            # Ensure sample rate is 16000 Hz
            audio = audio.set_frame_rate(16000)
            noise_audios[noise_type] = audio
        return noise_audios

    def _add_noise(self, audio: AudioSegment, noise: AudioSegment, noise_level_db=-12) -> AudioSegment:
        quiet_noise = noise.apply_gain(noise_level_db)
        if len(quiet_noise) >= len(audio):
            overlay_start = random.randint(0, len(quiet_noise) - len(audio))
            noise_to_overlay = quiet_noise[overlay_start:overlay_start+len(audio)]
            noisy_audio = audio.overlay(noise_to_overlay)
        else:
            noisy_audio = audio.overlay(quiet_noise, loop=True)
        return noisy_audio

    def generate(self, *args, **kwargs):
        os.makedirs(os.path.join(self.data_dir, "audio"), exist_ok=True)
        meta_data = {}
        for data_source in self.data_sources:
            with open(os.path.join(data_source, "meta_test.json"), "r") as f:
                domain_meta_data = json.load(f)

            random.shuffle(domain_meta_data)
            domain_meta_data = domain_meta_data[:int(len(domain_meta_data) * self.ratio)]
            if data_source.endswith("PhoneCall/en"):
                source_name = "PhoneCall/en"
            else:
                source_name = data_source.split("/")[-1]

            for i, item in enumerate(tqdm(domain_meta_data, desc=f"Processing {source_name}")):
                os.makedirs(os.path.join(self.data_dir, "audio", f"{source_name.lower()}"), exist_ok=True)
                for key in list(item.keys()):
                    if key not in ["text", "audio"]:
                        del item[key]
                if "fake" not in item["audio"]:
                    raise ValueError(f"No fake audio found for {item}")
                audio_path = list(item["audio"]["fake"].values())[0]
                audio = AudioSegment.from_file(os.path.join(data_source, audio_path))
                # Ensure sample rate is 16000 Hz
                audio = audio.set_frame_rate(16000)
                noise_type = list(self.noise_audios.keys())[i % len(self.noise_audios)]
                noisy_audio = self._add_noise(audio, self.noise_audios[noise_type])
                output_path = os.path.join("audio", f"{source_name.lower()}/{i}.wav")
                noisy_audio.export(os.path.join(self.data_dir, output_path), format="wav")
                item["audio"]["fake"] = output_path
                item["noise_type"] = noise_type

            meta_data[source_name] = domain_meta_data

        with open(os.path.join(self.data_dir, "meta.json"), "w") as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)