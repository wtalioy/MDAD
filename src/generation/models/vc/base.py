class BaseVC:
    def __init__(self, *args, **kwargs):
        self.model_name = None

    def convert(self, source_path: str, target_path: str, output_path: str, **kwargs):
        raise NotImplementedError("This method should be overridden by the subclass")