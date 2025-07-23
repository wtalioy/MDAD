import os
import numpy as np
import soundfile as sf
from google import genai
from .base import BaseTTS

USE_CASE = {
    "news": "Erinome",
    "podcast": "lapetus",
    "phonecall": "Zubenelgenubi",
    "publicspeech": "Charon",
    "movie": "Algieba",
    "interview": "Sadalbager",
}

class GeminiTTS(BaseTTS):
    def __init__(self, *args, **kwargs):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_name = "gemini-2.5-pro-preview-tts"

    def infer(self, text: str, use_case: str, language: str = "en", **kwargs):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=f"Say this in {language}:\n\n{text}",
            config={
                "response_modalities": ['Audio'],
                "speech_config": {
                    "voice_config": {
                        "prebuilt_voice_config": {
                            "voice_name": USE_CASE[use_case]
                        }
                    }
                }
            },
        )
        audio_bytes = response.candidates[0].content.parts[0].inline_data.data
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_array = audio_array.astype(np.float32) / 32768.0
        
        return audio_array, 24000

if __name__ == "__main__":
    tts = GeminiTTS()
    audio_array, sample_rate = tts.infer("Hello", "news")
    sf.write("output.wav", audio_array, sample_rate)