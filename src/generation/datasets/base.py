from models import BaseTTS

class BaseRawDataset:
    def __init__(self, data_dir: str, *args, **kwargs):
        self.data_dir = data_dir

    def generate(self, tts_model: BaseTTS, *args, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses.")