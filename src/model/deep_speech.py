import torch
from torch import nn
from torch.nn import Sequential


class DeepSpeech2(nn.Module):
    """
    Deep Speech 2 http://proceedings.mlr.press/v48/amodei16.pdf
    """

    def __init__(
        self,
        n_feats,
        n_tokens,
        input_channels=1,
        output_channels=4,
        rnn_hidden=512,
        rnn_layers=10,
    ):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.n_feats = n_feats
        self.n_tokens = n_tokens
        self.output_channels = output_channels
        print(
            n_feats, n_tokens, input_channels, output_channels, rnn_hidden, rnn_layers
        )

        self.convolutional_part = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )

        self.rnn = nn.GRU(
            input_size=n_feats * output_channels,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
        )

        # self.look_ahead = nn.Conv1d(
        #     in
        # )

        # self.fully_connected = nn.Linear(
        #     in_features=n_feats*output_channels,
        #     out_features=n_tokens
        # )

        self.fully_connected = Sequential(
            # people say it can approximate any function...
            nn.Linear(in_features=rnn_hidden, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=n_tokens),
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """

        output = self.convolutional_part(torch.unsqueeze(spectrogram, dim=1))
        batch_size, seq_len = output.shape[0], output.shape[-1]
        output = self.rnn(output.permute(0, 3, 1, 2).view(batch_size, seq_len, -1))[0]
        # output = output.permute(0, 3, 1, 2).view(batch_size, seq_len, -1)
        output = self.fully_connected(output)

        log_probs = nn.functional.log_softmax(output, dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return input_lengths  # we don't reduce time dimension here

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
