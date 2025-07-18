from typing import List, Optional
from models import BaseTTS, BaseVC
import os
import json
from loguru import logger
from tqdm import tqdm
import soundfile as sf
import librosa

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
        
        # Separate TTS models by require_vc attribute
        tts_only_models = [model for model in tts_models if not model.require_vc]
        tts_vc_models = [model for model in tts_models if model.require_vc]
        
        logger.info(f"TTS-only models: {[model.model_name for model in tts_only_models]}")
        logger.info(f"TTS+VC models: {[model.model_name for model in tts_vc_models]}")
        
        # Calculate total combinations for data distribution
        total_combinations = 0
        
        # TTS-only combinations
        if len(tts_only_models) > 0:
            total_combinations += len(tts_only_models)
        
        # TTS+VC combinations
        if len(tts_vc_models) > 0 and vc_models and len(vc_models) > 0:
            total_combinations += len(tts_vc_models) * len(vc_models)
        elif len(tts_vc_models) > 0:
            # If VC models not provided but TTS models require VC, treat as TTS-only
            total_combinations += len(tts_vc_models)
            tts_only_models.extend(tts_vc_models)
            tts_vc_models = []
        
        if total_combinations == 0:
            logger.warning("No valid model combinations found")
            return
            
        # Calculate data items per combination
        items_per_combination = total_items // total_combinations
        remainder = total_items % total_combinations
        
        logger.info(f"Total items: {total_items}, Total combinations: {total_combinations}, Items per combination: {items_per_combination}")
        
        combination_idx = 0
        start_idx = 0
        
        # Process TTS-only models
        for tts_model in tts_only_models:
            # Calculate data range for current combination
            if combination_idx < remainder:
                end_idx = start_idx + items_per_combination + 1
            else:
                end_idx = start_idx + items_per_combination
            
            model_items = meta_data[start_idx:end_idx]
            logger.info(f"TTS-only Model {tts_model.model_name} processing items {start_idx + 1}-{end_idx} ({len(model_items)} items)")

            for item in tqdm(model_items, desc=f"Generating audio with TTS model {tts_model.model_name}"):
                text = item['text']
                audio_path = os.path.join(self.data_dir, item['audio']['real'])
                output_path = audio_path.replace("audio/real", "audio/fake")

                try:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    fake_audio, sample_rate = tts_model.infer(text, ref_audio=audio_path, **kwargs)
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
            combination_idx += 1
        
        # Process TTS+VC model combinations
        for tts_model in tts_vc_models:
            for vc_model in (vc_models or []):
                # Calculate data range for current combination
                if combination_idx < remainder:
                    end_idx = start_idx + items_per_combination + 1
                else:
                    end_idx = start_idx + items_per_combination
                
                combination_items = meta_data[start_idx:end_idx]
                logger.info(f"Combination {combination_idx + 1} (TTS {tts_model.model_name} + VC {vc_model.model_name}) processing items {start_idx + 1}-{end_idx} ({len(combination_items)} items)")

                for item in tqdm(combination_items, desc=f"Generating audio with TTS {tts_model.model_name} + VC {vc_model.model_name}"):
                    text = item['text']
                    audio_path = os.path.join(self.data_dir, item['audio']['real'])
                    output_path = audio_path.replace("audio/real", "audio/fake")

                    try:
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        
                        # First generate audio with TTS
                        fake_audio, sample_rate = tts_model.infer(text, ref_audio=audio_path, **kwargs)
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