import torch
import torch.nn.functional as F


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    keys_to_pad = ["audio", "spectrogram", "text_encoded", "augmented_spectrogram"]
    keys_to_merge = ["text", "audio_path"]
    seq_lens = [0 for i in range(len(keys_to_pad))]

    finalized_batch = {}
    for key in keys_to_pad:
        finalized_batch[key] = []
        finalized_batch["{key}_length".format(key=key)] = []
    for key in keys_to_merge:
        finalized_batch[key] = []

    for single_item in dataset_items:
        for index, key in enumerate(keys_to_pad):
            seq_lens[index] = max(seq_lens[index], single_item[key].shape[-1])
            finalized_batch["{key}_length".format(key=key)].append(
                single_item[key].shape[-1]
            )

    for single_item in dataset_items:
        for index, key in enumerate(keys_to_pad):
            finalized_batch[key].append(
                F.pad(
                    single_item[key], (0, seq_lens[index] - single_item[key].shape[-1])
                )
            )
        for key in keys_to_merge:
            finalized_batch[key].append(single_item[key])

    for key in keys_to_pad:
        finalized_batch[key] = torch.cat(finalized_batch[key])
        finalized_batch["{key}_length".format(key=key)] = torch.tensor(
            finalized_batch["{key}_length".format(key=key)]
        )

    return finalized_batch
