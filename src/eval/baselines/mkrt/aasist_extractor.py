import random
from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import os


class CONV(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    def __init__(self,
                 out_channels,
                 kernel_size,
                 sample_rate=16000,
                 in_channels=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 groups=1,
                 mask=False):
        super().__init__()
        if in_channels != 1:

            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (
                in_channels)
            raise ValueError(msg)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2,
                                  (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2*fmax/self.sample_rate) * \
                np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow = (2*fmin/self.sample_rate) * \
                np.sinc(2*fmin*self.hsupp/self.sample_rate)
            hideal = hHigh - hLow

            self.band_pass[i, :] = Tensor(np.hamming(
                self.kernel_size)) * Tensor(hideal)

    def forward(self, x, mask=False):
        band_pass_filter = self.band_pass.clone().to(x.device)
        if mask:
            A = np.random.uniform(0, 20)
            A = int(A)
            A0 = random.randint(0, band_pass_filter.shape[0] - A)
            band_pass_filter[A0:A0 + A, :] = 0
        else:
            band_pass_filter = band_pass_filter

        self.filters = (band_pass_filter).view(self.out_channels, 1,
                                               self.kernel_size)

        return F.conv1d(x,
                        self.filters,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        bias=None,
                        groups=1)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(1, 1),
                               stride=1)
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               kernel_size=(2, 3),
                               padding=(0, 1),
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=(0, 1),
                                             kernel_size=(1, 3),
                                             stride=1)

        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
        out = self.conv1(x)

        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


import yaml
from importlib import import_module


class AASISTExtractor(nn.Module):
    def __init__(self, device: str = "cuda"):
        super().__init__()
        self.device = device

        aasist_config_path = os.path.join(os.path.dirname(__file__), "..", "aasist", "config", "AASIST.yaml")
        with open(aasist_config_path, "r") as f:
            aasist_config = yaml.load(f, Loader=yaml.FullLoader)
        
        module = import_module("..models.{}".format(aasist_config["architecture"]), __name__)
        _model = getattr(module, "Model")
        aasist_model = _model(aasist_config).to(self.device)
        
        aasist_ckpt_path = os.path.join(os.path.dirname(__file__), "..", "aasist", "ckpts", "AASIST.pth")
        aasist_model.load_state_dict(torch.load(aasist_ckpt_path, map_location=device))
        aasist_model.eval()

        self.conv_time = aasist_model.conv_time
        self.first_bn = aasist_model.first_bn
        self.selu = aasist_model.selu
        self.encoder = aasist_model.encoder

    @torch.inference_mode()
    def __call__(
        self,
        audio_batch: list[np.ndarray],
        padding: bool = True,
        truncation: bool = False,
        max_length: Optional[int] = None,
    ) -> list[torch.Tensor]:
        processed_batch = []
        for waveform_np in audio_batch:
            waveform = torch.from_numpy(waveform_np).float().to(self.device)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            if truncation and max_length is not None:
                waveform = waveform[:, :max_length]

            processed_batch.append(waveform)

        if not padding:
            if len(processed_batch) > 0:
                first_len = processed_batch[0].shape[1]
                if not all(w.shape[1] == first_len for w in processed_batch):
                    raise ValueError(
                        "All samples must have the same length when padding is disabled."
                    )
            padded_batch = processed_batch
        else:
            if max_length is not None:
                target_len = max_length
                if not truncation and any(
                    w.shape[1] > target_len for w in processed_batch
                ):
                    raise ValueError(
                        f"Found sample with length > max_length ({target_len}), but truncation is disabled."
                    )
            else:
                target_len = (
                    max(w.shape[1] for w in processed_batch) if processed_batch else 0
                )

            padded_batch = []
            for waveform in processed_batch:
                padding_needed = target_len - waveform.shape[1]
                padded_waveform = torch.nn.functional.pad(
                    waveform, (0, padding_needed)
                )
                padded_batch.append(padded_waveform)

        if not padded_batch:
            return []
            
        batch_tensor = torch.stack(padded_batch)
        
        features = self.forward(batch_tensor)
        
        feature_list = list(torch.split(features, 1, dim=0))
        
        return feature_list

    def forward(self, x, Freq_aug=False):
        x = x.unsqueeze(1)
        x = self.conv_time(x, mask=Freq_aug)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)
        e = self.encoder(x)
        
        e = e.permute(0, 3, 1, 2).contiguous()
        e = e.view(e.size(0), e.size(1), -1)
        
        return e
