from typing import List

import torch
from torch import Tensor
from torchaudio.models.decoder import ctc_decoder
from string import ascii_lowercase

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer

# TODO add beam search/lm versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxCERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        cers_sum = 0
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for index, (log_prob_vec, length, target_text) in enumerate(zip(predictions, lengths, text)):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_probs[index][:length])
            cers_sum += calc_cer(target_text, pred_text)
        return cers_sum / len(predictions)
