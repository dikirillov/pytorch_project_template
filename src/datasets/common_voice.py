import json
import re
from pathlib import Path

import torchaudio
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class CommonVoiceDataset(BaseDataset):
    def __init__(self, split, *args, **kwargs):
        pass

    def _get_or_load_index(self, split):
        pass
