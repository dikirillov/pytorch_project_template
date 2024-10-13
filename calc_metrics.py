import warnings

import hydra
import torch
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.metrics.utils import calc_cer, calc_wer

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    """
    Script for calculating CER and WER of given files.

    Args:
        config (DictConfig): hydra config with paths to predictions and ground trooth directories.
    """
    project_config = OmegaConf.to_container(config)

    predictions_dir = config.predictions_dir
    ground_truth_dir = config.ground_truth_dir

    cer_avg, wer_avg, counter = 0, 0, 0

    for prediction_path in Path(predictions_dir).iterdir():
        if ground_truth_dir and Path(ground_truth_dir).exists():
            gt_path = Path(ground_truth_dir) / (prediction_path.stem + ".txt")

            predicted_text, gt_text = "", ""
            with prediction_path.open() as f:
                predicted_text = f.read().strip()
            with gt_path.open() as f:
                gt_text = f.read().strip()
            
            cur_cer = calc_cer(gt_text, predicted_text)
            cur_wer = calc_wer(gt_text, predicted_text)
            cer_avg += cur_cer
            wer_avg += cur_wer

    print("AVG CER: {cer}".format(cer=cer_avg))
    print("AVG WER: {wer}".format(wer=wer_avg))


if __name__ == "__main__":
    main()
