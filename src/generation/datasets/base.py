from typing import List, Optional
from models import BaseTTS, BaseVC
import os
import json
import logging
from tqdm import tqdm
import soundfile as sf
import librosa

logger = logging.getLogger(__name__)

class BaseRawDataset:
    def __init__(self, data_dir: str, *args, **kwargs):
        self.data_dir = data_dir
        self.sample_rate = kwargs.get('sample_rate', 16000)
        self.meta_path = os.path.join(self.data_dir, "meta.json")

    def generate(self, tts_models: List[BaseTTS], vc_models: Optional[List[BaseVC]] = None, *args, **kwargs):
        output_dir = os.path.join(self.data_dir, "audio/fake")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(self.meta_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)

        total_items = len(meta_data)
        
        # If no VC models provided, only use TTS models with even distribution
        if vc_models is None or len(vc_models) == 0:
            num_models = len(tts_models)
            items_per_model = total_items // num_models
            remainder = total_items % num_models

            logger.info(f"TTS-only mode: Total items: {total_items}, TTS Models: {num_models}, Items per model: {items_per_model}")

            start_idx = 0
            for model_idx, tts_model in enumerate(tts_models):
                if model_idx < remainder:
                    end_idx = start_idx + items_per_model + 1
                else:
                    end_idx = start_idx + items_per_model
                
                model_items = meta_data[start_idx:end_idx]
                logger.info(f"TTS Model {model_idx + 1} processing items {start_idx + 1}-{end_idx} ({len(model_items)} items)")

                for item in tqdm(model_items, desc=f"Generating audio with TTS model {model_idx + 1}"):
                    text = item['text']
                    audio_path = os.path.join(self.data_dir, item['audio']['real'])
                    output_path = audio_path.replace("audio/real", "audio/fake")

                    try:
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        fake_audio, sample_rate = tts_model.infer(text, ref_audio=audio_path, language=kwargs.get('language', 'en'))
                        fake_audio = librosa.resample(fake_audio, orig_sr=sample_rate, target_sr=self.sample_rate)
                        sf.write(output_path, fake_audio, self.sample_rate)
                        logger.info(f"Generated audio at {output_path}")
                        item['audio']['fake'] = output_path
                        item['audio']['model'] = tts_model.model_name
                    except Exception as e:
                        logger.error(f"While generating audio for text: {text}")
                        logger.error(e)
                        continue

                start_idx = end_idx
        else:
            # TTS and VC permutation combination mode
            num_tts = len(tts_models)
            num_vc = len(vc_models)
            total_combinations = num_tts * num_vc
            
            logger.info(f"TTS+VC mode: Total items: {total_items}, TTS Models: {num_tts}, VC Models: {num_vc}, Total combinations: {total_combinations}")
            
            # Calculate data items per combination
            items_per_combination = total_items // total_combinations
            remainder = total_items % total_combinations
            
            combination_idx = 0
            start_idx = 0
            
            for tts_idx, tts_model in enumerate(tts_models):
                for vc_idx, vc_model in enumerate(vc_models):
                    # Calculate data range for current combination
                    if combination_idx < remainder:
                        end_idx = start_idx + items_per_combination + 1
                    else:
                        end_idx = start_idx + items_per_combination
                    
                    combination_items = meta_data[start_idx:end_idx]
                    logger.info(f"Combination {combination_idx + 1} (TTS {tts_idx + 1} + VC {vc_idx + 1}) processing items {start_idx + 1}-{end_idx} ({len(combination_items)} items)")

                    for item in tqdm(combination_items, desc=f"Generating audio with TTS {tts_idx + 1} + VC {vc_idx + 1}"):
                        text = item['text']
                        audio_path = os.path.join(self.data_dir, item['audio']['real'])
                        output_path = audio_path.replace("audio/real", "audio/fake")

                        try:
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            
                            # First generate audio with TTS
                            fake_audio, sample_rate = tts_model.infer(text, ref_audio=audio_path, language=kwargs.get('language', 'en'))
                            fake_audio = librosa.resample(fake_audio, orig_sr=sample_rate, target_sr=self.sample_rate)
                            
                            # Save TTS audio as temporary file
                            temp_tts_path = output_path.replace(".wav", "_temp_tts.wav")
                            sf.write(temp_tts_path, fake_audio, self.sample_rate)
                            
                            # Then perform voice conversion with VC
                            vc_model.convert(temp_tts_path, audio_path, output_path)
                            
                            # Clean up temporary file
                            if os.path.exists(temp_tts_path):
                                os.remove(temp_tts_path)
                            logger.info(f"Generated audio at {output_path}")
                            item['audio']['fake'] = output_path
                            item['audio']['model'] = f"{tts_model.model_name} + {vc_model.model_name}"
                        except Exception as e:
                            logger.error(f"While generating audio for text: {text}")
                            logger.error(e)
                            continue

                    start_idx = end_idx
                    combination_idx += 1

        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Meta data updated and saved to {self.meta_path}")