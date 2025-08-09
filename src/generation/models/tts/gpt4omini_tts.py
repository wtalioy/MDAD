import os
import io
import soundfile as sf
from openai import OpenAI
from .base import BaseTTS

class GPT4oMiniTTS(BaseTTS):
    def __init__(self, *args, **kwargs):
        self.model_name = "gpt-4o-mini-tts"
        self.require_vc = True
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def infer(self, text: str, **kwargs):
        binary_data = self.client.audio.speech.create(
            model=self.model_name,
            input=text,
            voice="onyx",
            response_format="wav"
        ).content
        audio_stream = io.BytesIO(binary_data)
        audio_array, sample_rate = sf.read(audio_stream)
        return audio_array, sample_rate

if __name__ == "__main__":
    tts = GPT4oMiniTTS()
    audio_array, sample_rate = tts.infer("Hello")
    sf.write("test.wav", audio_array, sample_rate)