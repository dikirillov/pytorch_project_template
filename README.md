# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## Результаты

| Test | CER | WER |
| ------------- | ------------- | ------------- |
| Clean | 0.065 | 0.214 | 
| Noisy | 0.170 | 0.432 |

[Отчет](https://www.comet.com/dikirillov/pytorch-template-asr-example/notes)

## Команды

Запуск лучшего сетапа обучения
`HYDRA_FULL_ERROR=1 python3 train.py -cn=baseline trainer.override=True writer=wandb datasets.train.part=train_all model=deep_speech writer.run_name="DEEPSPEECH_PRETRAIN_augs_gain_freq_20_lr_1e_3_n_feats_128_out_4_hidden_512_layers_4_100k_32_batches" lr_scheduler.max_lr=1e-3 trainer.epoch_len=2000 trainer.log_step=5000 dataloader.batch_size=32 trainer.save_period=10 writer.log_checkpoints=True`

Запуск инференса:

`HYDRA_FULL_ERROR=1 python3 inference.py -cn=inference model=deep_speech datasets=example  inferencer.from_pretrained="/home/dikirillov/study/pytorch_project_template/saved/DEEPSPEECH_PRETRAIN_augs_gain_freq_20_lr_1e_3_n_feats_128_out_4_hidden_512_layers_4_100k_32_batches/checkpoint-epoch50.pth" dataloader.batch_size=32 text_encoder._target_=src.text_encoder.CTCTextEncoder`

Рассчитать метрики Wer и Cer

`python calc_metrics.py -cn=calc_metrics`

управляется опциями `predictions_dir` и `/home/dikirillov/study/pytorch_project_template/data/test_metrics/ground_truth`

Сравнить работу нашего бим серча и работу бим серча из торча

`HYDRA_FULL_ERROR=1 python3 inference.py -cn=inference model=deep_speech datasets=customdir  inferencer.from_pretrained="//home/dikirillov/study/pytorch_project_template/saved/Gain_only/model_best.pth" dataloader.batch_size=3 text_encoder._target_=src.text_encoder.CTCBeamSearchTextEncoderHandsCrafted`
