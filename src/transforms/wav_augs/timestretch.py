import torch_audiomentations
import numpy as np
import torchaudio
from torch import Tensor, nn


class TimeStretch(nn.Module):
    def __init__(self, prob=0.1, low=0.8, high=1.25, n_freq=128, *args, **kwargs):
        super().__init__()
        print("rand", np.random.uniform())
        self.is_stretching = False
        self.value = None
        self._aug = torchaudio.transforms.TimeStretch(n_freq=n_freq)
        if 0 <= prob:
            self.value = np.random.uniform(low, high)
            self.is_stretching = True

    def __call__(self, data: Tensor):
        if self.is_stretching:
            x = data.unsqueeze(1)
            return self._aug(x, self.value).abs().squeeze(1)

        return x
