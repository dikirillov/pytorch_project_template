from pathlib import Path

import torchaudio
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

class CustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        data = []
        for path in Path(ROOT_PATH / audio_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                t_info = torchaudio.info(str(path))
                length = t_info.num_frames / t_info.sample_rate
                entry["audio_len"] = length
                entry["path"] = str(path)
                if transcription_dir and Path(ROOT_PATH / transcription_dir).exists():
                    transc_path = Path(transcription_dir) / (path.stem + ".txt")
                    if transc_path.exists():
                        with transc_path.open() as f:
                            entry["text"] = f.read().strip()
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
