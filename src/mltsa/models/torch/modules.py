"""Neural network modules used by the mltsa torch wrappers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Simple multilayer perceptron over flattened time-series features."""

    def __init__(
        self,
        input_shape: tuple[int, int],
        n_classes: int,
        *,
        hidden_sizes: Sequence[int] = (64, 32),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        input_dim = input_shape[0] * input_shape[1]
        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(previous_dim, int(hidden_size)))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            previous_dim = int(hidden_size)

        layers.append(nn.Linear(previous_dim, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of time-series samples."""

        flattened = torch.flatten(X, start_dim=1)
        return self.network(flattened)


class LSTMClassifier(nn.Module):
    """Single-head LSTM classifier using the final hidden state."""

    def __init__(
        self,
        input_shape: tuple[int, int],
        n_classes: int,
        *,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        _, n_features = input_shape
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.bidirectional = bidirectional
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        output_width = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(output_width, n_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of time-series samples."""

        _, (hidden_state, _) = self.encoder(X)
        if self.bidirectional:
            last_hidden = torch.cat((hidden_state[-2], hidden_state[-1]), dim=1)
        else:
            last_hidden = hidden_state[-1]
        return self.classifier(last_hidden)


class CNN1DClassifier(nn.Module):
    """Compact 1D convolutional classifier over sequence steps."""

    def __init__(
        self,
        input_shape: tuple[int, int],
        n_classes: int,
        *,
        channels: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _, n_features = input_shape
        padding = kernel_size // 2
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.classifier = nn.Linear(channels, n_classes)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Return class logits for a batch of time-series samples."""

        channels_first = X.transpose(1, 2)
        encoded = self.encoder(channels_first)
        flattened = torch.flatten(encoded, start_dim=1)
        return self.classifier(self.dropout(flattened))


__all__ = ["CNN1DClassifier", "LSTMClassifier", "MLPClassifier"]
