"""Training utilities for mltsa torch classifiers."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .datasets import TimeSeriesDataset, ensure_feature_array


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible tiny experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(slots=True)
class TrainerConfig:
    """Configuration for the simple supervised training loop."""

    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    random_state: int = 0
    device: str = "cpu"


class TorchTrainer:
    """Small trainer for CPU-friendly sequence classification experiments."""

    def __init__(self, config: TrainerConfig | None = None) -> None:
        self.config = config or TrainerConfig()

    def fit(self, module: nn.Module, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Fit a module on encoded targets and return per-epoch mean losses."""

        set_random_seed(self.config.random_state)
        device = self._resolve_device()
        module.to(device)
        module.train()

        dataset = TimeSeriesDataset(X, y)
        batch_size = min(self.config.batch_size, len(dataset))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(
            module.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()
        history: list[float] = []

        for _ in range(self.config.epochs):
            running_loss = 0.0
            sample_count = 0
            for features, targets in loader:
                features = features.to(device)
                targets = targets.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = module(features)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                batch_size = int(targets.shape[0])
                running_loss += float(loss.detach().cpu()) * batch_size
                sample_count += batch_size

            history.append(running_loss / max(sample_count, 1))

        module.to("cpu")
        module.eval()
        return history

    @torch.no_grad()
    def predict_proba(self, module: nn.Module, X: object) -> np.ndarray:
        """Return class probabilities for a batch of feature arrays."""

        features = torch.from_numpy(ensure_feature_array(X))
        device = self._resolve_device()
        module.to(device)
        module.eval()
        logits = module(features.to(device))
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        module.to("cpu")
        return probabilities

    def _resolve_device(self) -> torch.device:
        """Return the configured torch device."""

        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device(self.config.device)


__all__ = ["TorchTrainer", "TrainerConfig", "set_random_seed"]
