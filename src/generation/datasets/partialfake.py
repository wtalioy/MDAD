from .base import BaseRawDataset
import os
import random
import soundfile as sf
from loguru import logger
import numpy as np

class PartialFake(BaseRawDataset):
    def __init__(self, data_dir=None, *args, **kwargs):
        super().__init__(data_dir or "data/PartialFake", *args, **kwargs)

    def _select_partial_fake(self, item, min_word_num=1, max_word_num=3, sample_rate=16000) -> tuple[str, np.ndarray, np.ndarray]:
        word_timestamps = item["word_timestamps"]
        num_words = random.randint(min_word_num, max_word_num)
        
        # Choose a random starting index that allows for words consecutive words
        max_start_index = len(word_timestamps) - num_words
        start_index = random.randint(0, max_start_index)

        text = ' '.join([word_item['word'] for word_item in word_timestamps[start_index:start_index + num_words]])    
        start_time = word_timestamps[start_index] * sample_rate
        end_time = word_timestamps[start_index + num_words - 1] * sample_rate

        array = sf.read(os.path.join(self.data_dir, item['audio']['real']))[0]
        array = array[int(start_time):int(end_time)]
        org_head = array[:int(start_time)]
        org_tail = array[int(end_time):]
        
        return text, org_head, org_tail

    def _try_generate_audio(self, item, tts_model, vc_model=None, language: str = "en", **kwargs):
        """Try to generate audio with given models. Returns True if successful, False otherwise."""
        text = item['text']
        audio_rel_path = item['audio']['real']
        output_rel_path = audio_rel_path.replace("audio/real", "audio/fake")
        audio_path = os.path.join(self.data_dir, audio_rel_path)
        output_path = os.path.join(self.data_dir, output_rel_path)
        item['audio']['fake'] = {}
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            text, org_head, org_tail = self._select_partial_fake(item)
            
            if vc_model is None:
                # TTS only
                fake_audio, sample_rate = tts_model.infer(text, ref_audio=audio_path, language=language, **kwargs)
                model_name = tts_model.model_name
            else:
                # TTS + VC
                fake_audio, sample_rate = tts_model.infer(text, language=language, **kwargs)

                # Convert the voice of the corresponding sample to that of the real audio
                sample_path = f"src/generation/samples/{language}.wav"
                temp_vc_path = output_path.replace(".wav", "_temp_vc.wav")
                vc_model.convert(sample_path, audio_path, temp_vc_path)
                
                # Save TTS audio as temporary file
                temp_tts_path = output_path.replace(".wav", "_temp_tts.wav")
                sf.write(temp_tts_path, fake_audio, sample_rate)
                
                # Then perform voice conversion with VC
                vc_model.convert(temp_tts_path, temp_vc_path, output_path)
                
                # Clean up temporary file
                if os.path.exists(temp_tts_path):
                    os.remove(temp_tts_path)
                if os.path.exists(temp_vc_path):
                    os.remove(temp_vc_path)
                    
                model_name = f"{tts_model.model_name} + {vc_model.model_name}"

            fake_audio = np.concatenate([org_head, fake_audio, org_tail])
            sf.write(output_path, fake_audio, sample_rate)
            
            logger.info(f"Generated audio at {output_path}")
            item['audio']['fake'][model_name] = output_rel_path
            return True
            
        except Exception as e:
            logger.error(f"Failed to generate audio for text: {text} with {tts_model.model_name}" + 
                        (f" + {vc_model.model_name}" if vc_model else ""))
            logger.error(e)
            return False