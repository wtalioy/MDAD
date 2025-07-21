import numpy as np
from elevenlabs import ElevenLabs
from .base import BaseTTS

USE_CASE = {
    "news": "5Q0t7uMcjvnagumLfvZi", # Paul
    "podcast": "9BWtsMINqrJLrRacOk9x", # Aria
    "phonecall": "2EiwWnXFnvU5JabPnv8n", # Clyde
    "publicspeech": "9BWtsMINqrJLrRacOk9x", # Aria
    "movie": "EXAVITQu4vr4xnSDxMaL", # Sarah
    "interview": "21m00Tcm4TlvDq8ikWAM", # Rachel
    "phonecall": "CYw3kZ02Hs0563khs1Fj", # Dave
}

class ElevenLabsTTS(BaseTTS):
    def __init__(self, api_key: str, *args, **kwargs):
        self.model_name = "ElevenLabs"
        self.require_vc = True
        self.client = ElevenLabs(api_key=api_key)

    def get_voices(self):
        response = self.client.voices.search(include_total_count=True)
        return response.voices
    
    def infer(self, text: str, use_case: str, sample_rate: int = 16000, **kwargs):
        voice_id = USE_CASE[use_case]
        audio = self.client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_multilingual_v2",
            output_format=f"pcm_{sample_rate}"
        )
        
        # Convert PCM bytes to numpy array
        # First convert iterator to bytes, then to numpy array
        audio_bytes = b''.join(audio)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Convert to float32 and normalize to [-1, 1] range
        audio_array = audio_array.astype(np.float32) / 32768.0
        
        return audio_array, sample_rate

if __name__ == "__main__":
    import soundfile as sf
    import os
    model = ElevenLabsTTS(api_key=os.getenv("ELEVENLABS_API_KEY"))
    audio, sample_rate = model.infer(text="Hello", use_case="news")
    sf.write("test.wav", audio, sample_rate)