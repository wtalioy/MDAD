from typing import List
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

    def _try_generate_audio(self, item, tts_model, vc_model=None, **kwargs):
        """Try to generate audio with given models. Returns True if successful, False otherwise."""
        text = item['text']
        audio_path = os.path.join(self.data_dir, item['audio']['real'])
        output_path = audio_path.replace("audio/real", "audio/fake")
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if vc_model is None:
                # TTS only
                fake_audio, sample_rate = tts_model.infer(text, ref_audio=audio_path, **kwargs)
                fake_audio = librosa.resample(fake_audio, orig_sr=sample_rate, target_sr=self.sample_rate)
                sf.write(output_path, fake_audio, self.sample_rate)
                model_name = tts_model.model_name
            else:
                # TTS + VC
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
                model_name = f"{tts_model.model_name} + {vc_model.model_name}"
            
            logger.info(f"Generated audio at {output_path}")
            item['audio']['fake'] = output_path
            item['audio']['model'] = model_name
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate audio for text: {text} with {tts_model.model_name}" + 
                        (f" + {vc_model.model_name}" if vc_model else ""))
            logger.error(e)
            return False

    def generate(self, tts_models: List[BaseTTS], vc_models: List[BaseVC] = [], *args, **kwargs):
        output_dir = os.path.join(self.data_dir, "audio/fake")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(self.meta_path, 'r', encoding='utf-8') as f:
            meta_data = json.load(f)

        # Create all possible model combinations
        all_combinations = []
        
        # TTS-only combinations
        tts_only_models = [model for model in tts_models if not model.require_vc]
        for tts_model in tts_only_models:
            all_combinations.append((tts_model, None))
        
        # TTS+VC combinations
        tts_vc_models = [model for model in tts_models if model.require_vc]
        if vc_models and len(vc_models) > 0:
            for tts_model in tts_vc_models:
                for vc_model in vc_models:
                    all_combinations.append((tts_model, vc_model))
        else:
            # If no VC models provided, treat TTS+VC models as TTS-only
            for tts_model in tts_vc_models:
                all_combinations.append((tts_model, None))
        
        if not all_combinations:
            logger.warning("No valid model combinations found")
            return
        
        logger.info(f"Available combinations: {len(all_combinations)}")
        for i, (tts, vc) in enumerate(all_combinations):
            combo_name = tts.model_name + (f" + {vc.model_name}" if vc else "")
            logger.info(f"  {i+1}. {combo_name}")
        
        # Process each item with round-robin distribution and fallback logic
        failed_items = []
        for idx, item in enumerate(tqdm(meta_data, desc="Generating audio")):
            success = False
            
            # Start with the assigned combination (round-robin distribution)
            start_combo_idx = idx % len(all_combinations)
            
            # Try combinations starting from the assigned one, then cycle through others
            for i in range(len(all_combinations)):
                combo_idx = (start_combo_idx + i) % len(all_combinations)
                tts_model, vc_model = all_combinations[combo_idx]
                
                combo_name = tts_model.model_name + (f" + {vc_model.model_name}" if vc_model else "")
                if i == 0:
                    logger.info(f"Processing item {idx+1} with assigned combination: {combo_name}")
                else:
                    logger.info(f"Trying fallback combination {i+1}: {combo_name}")
                
                if self._try_generate_audio(item, tts_model, vc_model, **kwargs):
                    success = True
                    logger.info(f"Successfully generated audio for item {idx+1}")
                    break
            
            if not success:
                logger.warning(f"Failed to generate audio for item {idx+1} with all combinations: {item['text'][:50]}...")
                failed_items.append(item)

        # Save updated metadata
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Meta data updated and saved to {self.meta_path}")
        logger.info(f"Successfully processed: {len(meta_data) - len(failed_items)}/{len(meta_data)} items")
        if failed_items:
            logger.warning(f"Failed items: {len(failed_items)}")