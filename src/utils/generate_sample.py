from generation.models.tts.melotts import MeloTTS
import soundfile
import os

text_en = "This is an audio sample for English. It should be at least 5 seconds long. Let's say whatever we want to say to make it not too short."
text_zh = "这是一个中文音频样本。它至少应该持续5秒。让我们随便说点什么，可不要让它太短了。"

if __name__ == "__main__":
    if not os.path.exists("src/generation/samples"):
        os.makedirs("src/generation/samples")
    tts_model = MeloTTS()
    wav, sr = tts_model.infer(text_en, language="en")
    soundfile.write("src/generation/samples/en.wav", wav, sr)

    wav, sr = tts_model.infer(text_zh, language="zh-cn")
    soundfile.write("src/generation/samples/zh-cn.wav", wav, sr)