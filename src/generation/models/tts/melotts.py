from melo.api import TTS
from .base import BaseTTS

class MeloTTS(BaseTTS):
    def __init__(self, *args, **kwargs):
        self.model_name = "MeloTTS"
        self.require_vc = True

    def infer(self, text: str, language="en", **kwargs):
        speed = 1.0
        model = TTS(language=language, device="auto")
        speaker_ids = model.hps.data.spk2id
        if language == "en":
            speaker_id = speaker_ids['EN-Default']
        elif language == "zh-cn":
            speaker_id = speaker_ids['ZH']
        else:
            speaker_id = speaker_ids[language.upper()]
        wav = model.tts_to_file(text, speaker_id, speed=speed)
        return wav, model.hps.data.sampling_rate