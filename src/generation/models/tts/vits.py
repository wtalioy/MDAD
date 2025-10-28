import torch
from .src_vits import commons

from .src_vits.utils import get_hparams_from_file, load_checkpoint
from .src_vits.models import SynthesizerTrn
from .src_vits.text.symbols import symbols
from .src_vits.text import text_to_sequence
from models.tts.base import BaseTTS

class VITS(BaseTTS):
    def __init__(self, config="src/generation/models/vits/configs/vctk_base.json", ckpt="src/generation/models/vits/pretrained_vctk.pth", *args, **kwargs):
        self.model_name = "VITS"
        self.hps = get_hparams_from_file(config)
        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model).cuda()
        self.net_g.eval()
        load_checkpoint(ckpt, self.net_g, None)

    def infer(self, text: str, sid=4, **kwargs):
        text_norm = text_to_sequence(text, self.hps.data.text_cleaners)
        if self.hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        stn_tst = torch.LongTensor(text_norm)
        with torch.no_grad():
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            sid = torch.LongTensor([sid]).cuda()
            wav = self.net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
        return wav, self.hps.data.sampling_rate