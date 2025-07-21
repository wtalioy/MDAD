from melo.api import TTS
from .base import BaseTTS

class MeloTTS(BaseTTS):
    def __init__(self, *args, **kwargs):
        self.model_name = "MeloTTS"
        self.require_vc = True

    def infer(self, text: str, language="en", **kwargs):
        speed = 1.0
        if language == "en":
            model = TTS(language="EN", device="auto")
            speaker_id = model.hps.data.spk2id['EN-Default']
        elif language == "zh-cn":
            model = TTS(language="ZH", device="auto")
            speaker_id = model.hps.data.spk2id['ZH']
        else:
            model = TTS(language=language.upper(), device="auto")
            speaker_id = model.hps.data.spk2id[language.upper()]
        wav = model.tts_to_file(text, speaker_id, speed=speed)
        return wav, model.hps.data.sampling_rate