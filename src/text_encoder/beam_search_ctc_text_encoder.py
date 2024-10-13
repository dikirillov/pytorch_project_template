import re
from string import ascii_lowercase
from src.text_encoder.ctc_text_encoder import CTCTextEncoder
from torchaudio.models.decoder import ctc_decoder
from torchaudio.models.decoder import download_pretrained_files

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
        files = download_pretrained_files("librispeech-4-gram")
        LM_WEIGHT = 0
        WORD_SCORE = -0.26
        self.beam_search_decoder = ctc_decoder(
            lexicon=self.fix_lexicon(files.lexicon),
            tokens=self.vocab,
            lm=files.lm,
            lm_weight=LM_WEIGHT,
            word_score=WORD_SCORE,
            blank_token=self.EMPTY_TOK,
            sil_token=" ", nbest=1, beam_size=10
        )
    
    def fix_lexicon(self, lexicon):
        new_path = "custom_lexicon.txt"
        new_lexicon = []

        with open(lexicon, "r") as fin:
            for line in fin.readlines():
                new_lexicon.append(self.normalize_text(line))
        
        with open(new_path, "w") as fout:
            for line in new_lexicon:
                print(line, file=fout)

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
        return "".join(output).strip()

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
