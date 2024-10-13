import re
from string import ascii_lowercase
from src.text_encoder.ctc_text_encoder import CTCTextEncoder
from torchaudio.models.decoder import ctc_decoder

import torch

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCBeamSearchTextEncoder(CTCTextEncoder):
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, beam_size=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        super().__init__()
        self.beam_search_decoder = ctc_decoder(
            lexicon=None,
            tokens=self.vocab,
            blank_token=self.EMPTY_TOK,
            sil_token="", nbest=1, beam_size=10
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

    def ctc_decode(self, log_probs) -> str:
        output = []
        prev = -1

        for ind in self.beam_search_decoder(log_probs.unsqueeze(0).cpu())[0][0].tokens:
            current_token = self.ind2char[ind.item()]

            if current_token != self.EMPTY_TOK:
                output.append(current_token)
        return "".join(output)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
