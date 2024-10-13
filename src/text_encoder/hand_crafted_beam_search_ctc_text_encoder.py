import re
from string import ascii_lowercase
from src.text_encoder.ctc_text_encoder import CTCTextEncoder
from torchaudio.models.decoder import ctc_decoder
from collections import defaultdict

from src.metrics.utils import calc_cer

import torch

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCBeamSearchTextEncoderHandsCrafted(CTCTextEncoder):
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, beam_size=3, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        print(beam_size)
        super().__init__()
        self.beam_size = beam_size
        self.beam_search_decoder = ctc_decoder(
            lexicon=None,
            tokens=self.vocab,
            blank_token=self.EMPTY_TOK,
            sil_token="", nbest=1, beam_size=beam_size
        )

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()
    
    def add_char(self, state, log_probs):
        output = defaultdict(float)

        for ind, prob in enumerate(log_probs):
            for (context, last), value in state.items():
                cur_char = self.ind2char[ind]

                if cur_char == last:
                    context, last, value = context, last, value + prob
                else:
                    if cur_char == self.EMPTY_TOK:
                        context, last, value = context, cur_char, value + prob
                    else:
                        context, last, value = context + cur_char, cur_char, value + prob
                
                output[(context, last)] += value

        return output


    def clean_state(self, state):
        return sorted(list(state.items()), key=lambda x: -x[1])[:self.beam_size]
    
    def ctc_decode_debug(self, log_probs) -> str:
        output = []
        prev = -1

        for ind in self.beam_search_decoder(log_probs.unsqueeze(0).cpu())[0][0].tokens:
            current_token = self.ind2char[ind.item()]

            if current_token != self.EMPTY_TOK:
                output.append(current_token)
        return "".join(output)

    def ctc_decode(self, log_probs) -> str:
        state = {
            ("", self.EMPTY_TOK): 1.0
        }

        for prob in log_probs:
            state = self.add_char(state, prob)
            state = dict(self.clean_state(state))

        output = [(context, prob) for (context, last), prob in sorted(state.items(), key=lambda x: -x[1])][0][0]
        print(calc_cer(self.ctc_decode_debug(log_probs), output))
        print(output)
        print(self.ctc_decode_debug(log_probs))
        assert calc_cer(self.ctc_decode_debug(log_probs), output) < 0.1, "FAILED BEAM SEARCH"
        print("Test - OK")
        return output

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
