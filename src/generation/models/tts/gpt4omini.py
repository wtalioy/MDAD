import os
import io
import soundfile as sf
from openai.types.audio import SpeechModel
from .base import BaseTTS

class GPT4oMiniTTS(BaseTTS):
    def __init__(self, *args, **kwargs):
        self.model_name = "gpt-4o-mini-tts"
        self.client = SpeechModel(api_key=os.getenv("OPENAI_API_KEY"))

    def infer(self, text: str) -> str:
        binary_data = self.client.create(
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
    audio_array, sample_rate = tts.infer("Hello, how are you?")
    sf.write("test.wav", audio_array, sample_rate)