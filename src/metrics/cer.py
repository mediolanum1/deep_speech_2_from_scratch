from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer


# debugging 
import inspect

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
        cers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        caller = inspect.stack()[1]
        print(f"Called from function: {caller.function}")

        print(type(log_probs_length))
        log_probs_length = torch.tensor(log_probs_length)
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            print(log_prob_vec[:length])
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
                        
            print("-------",pred_text,"----------------")

            cers.append(calc_cer(target_text, pred_text))
        return sum(cers) / len(cers)
