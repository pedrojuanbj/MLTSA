"""High-level PyTorch wrappers for time-series classification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
from torch import nn

from .datasets import encode_targets, ensure_feature_array
from .modules import CNN1DClassifier, LSTMClassifier, MLPClassifier
from .trainer import TorchTrainer, TrainerConfig

TorchModelT = TypeVar("TorchModelT", bound="TorchClassifierBase")


class TorchClassifierBase(ABC):
    """Base wrapper exposing a small sklearn-like API for torch models."""

    canonical_name = "torch_model"

    def __init__(
        self,
        *,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        random_state: int = 0,
        device: str = "cpu",
    ) -> None:
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.random_state = int(random_state)
        self.device = str(device)

        self.model_name = self.canonical_name
        self.module_: nn.Module | None = None
        self.classes_: np.ndarray | None = None
        self.input_shape_: tuple[int, int] | None = None
        self.training_loss_: list[float] = []

    def fit(self, X: object, y: object) -> "TorchClassifierBase":
        """Fit the wrapped torch classifier and return ``self``."""

        features = ensure_feature_array(X)
        classes, encoded_targets = encode_targets(y, features.shape[0])
        input_shape = (int(features.shape[1]), int(features.shape[2]))

        self.module_ = self._build_module(input_shape, int(classes.shape[0]))
        self.classes_ = classes
        self.input_shape_ = input_shape
        self.training_loss_ = self._trainer().fit(self.module_, features, encoded_targets)
        return self

    def predict(self, X: object) -> np.ndarray:
        """Predict class labels for the provided samples."""

        probabilities = self.predict_proba(X)
        indices = probabilities.argmax(axis=1)
        classes = self._require_classes()
        return classes[indices]

    def predict_proba(self, X: object) -> np.ndarray:
        """Predict class probabilities for the provided samples."""

        module = self._require_module()
        features = ensure_feature_array(X)
        self._validate_input_shape(features)
        return self._trainer().predict_proba(module, features)

    def score(self, X: object, y: object) -> float:
        """Return simple classification accuracy on the provided samples."""

        predictions = self.predict(X)
        targets = np.asarray(y)
        if targets.ndim != 1 or targets.shape[0] != predictions.shape[0]:
            raise ValueError("y must be one-dimensional and match the number of samples in X.")
        return float(np.mean(predictions == targets))

    def save(self, path: str | Path) -> Path:
        """Persist model weights and wrapper configuration to disk."""

        module = self._require_module()
        classes = self._require_classes()
        input_shape = self._require_input_shape()
        target_path = Path(path)
        payload = {
            "class_name": type(self).__name__,
            "canonical_name": self.canonical_name,
            "init_params": self._get_init_params(),
            "classes": classes.tolist(),
            "input_shape": input_shape,
            "state_dict": module.state_dict(),
        }
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as buffer:
            torch.save(payload, buffer)
        return target_path

    @classmethod
    def load(cls: type[TorchModelT], path: str | Path) -> TorchModelT:
        """Load a saved wrapper instance from disk."""

        source_path = Path(path)
        with source_path.open("rb") as buffer:
            payload = torch.load(buffer, map_location="cpu", weights_only=False)
        model = cls(**payload["init_params"])
        input_shape = tuple(int(value) for value in payload["input_shape"])
        classes = np.asarray(payload["classes"])

        model.module_ = model._build_module(input_shape, int(classes.shape[0]))
        model.module_.load_state_dict(payload["state_dict"])
        model.module_.eval()
        model.classes_ = classes
        model.input_shape_ = input_shape
        return model

    def _trainer(self) -> TorchTrainer:
        """Construct a trainer for the current wrapper settings."""

        return TorchTrainer(
            TrainerConfig(
                epochs=self.epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                random_state=self.random_state,
                device=self.device,
            )
        )

    def _validate_input_shape(self, features: np.ndarray) -> None:
        """Ensure inference data matches the training-time feature shape."""

        expected = self._require_input_shape()
        actual = (int(features.shape[1]), int(features.shape[2]))
        if actual != expected:
            raise ValueError(f"Expected input shape {expected}, received {actual}.")

    def _require_module(self) -> nn.Module:
        """Return the fitted torch module or raise a clear error."""

        if self.module_ is None:
            raise RuntimeError("This model has not been fitted yet.")
        return self.module_

    def _require_classes(self) -> np.ndarray:
        """Return the fitted class labels or raise a clear error."""

        if self.classes_ is None:
            raise RuntimeError("This model has not been fitted yet.")
        return self.classes_

    def _require_input_shape(self) -> tuple[int, int]:
        """Return the fitted input shape or raise a clear error."""

        if self.input_shape_ is None:
            raise RuntimeError("This model has not been fitted yet.")
        return self.input_shape_

    @abstractmethod
    def _build_module(self, input_shape: tuple[int, int], n_classes: int) -> nn.Module:
        """Build a fresh torch module for fitting or loading."""

    @abstractmethod
    def _get_init_params(self) -> dict[str, Any]:
        """Return constructor arguments needed to recreate the wrapper."""


class MLP(TorchClassifierBase):
    """Lightweight multilayer perceptron classifier."""

    canonical_name = "mlp"

    def __init__(
        self,
        *,
        hidden_sizes: tuple[int, ...] = (64, 32),
        dropout: float = 0.0,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        random_state: int = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            random_state=random_state,
            device=device,
        )
        self.hidden_sizes = tuple(int(value) for value in hidden_sizes)
        self.dropout = float(dropout)

    def _build_module(self, input_shape: tuple[int, int], n_classes: int) -> nn.Module:
        return MLPClassifier(
            input_shape=input_shape,
            n_classes=n_classes,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout,
        )

    def _get_init_params(self) -> dict[str, Any]:
        return {
            "hidden_sizes": self.hidden_sizes,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "random_state": self.random_state,
            "device": self.device,
        }


class LSTM(TorchClassifierBase):
    """LSTM-based sequence classifier."""

    canonical_name = "lstm"

    def __init__(
        self,
        *,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        random_state: int = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            random_state=random_state,
            device=device,
        )
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.bidirectional = bool(bidirectional)

    def _build_module(self, input_shape: tuple[int, int], n_classes: int) -> nn.Module:
        return LSTMClassifier(
            input_shape=input_shape,
            n_classes=n_classes,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

    def _get_init_params(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "bidirectional": self.bidirectional,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "random_state": self.random_state,
            "device": self.device,
        }


class CNN1D(TorchClassifierBase):
    """Compact 1D convolutional sequence classifier."""

    canonical_name = "cnn1d"

    def __init__(
        self,
        *,
        channels: int = 32,
        kernel_size: int = 3,
        dropout: float = 0.0,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        random_state: int = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            random_state=random_state,
            device=device,
        )
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.dropout = float(dropout)

    def _build_module(self, input_shape: tuple[int, int], n_classes: int) -> nn.Module:
        return CNN1DClassifier(
            input_shape=input_shape,
            n_classes=n_classes,
            channels=self.channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
        )

    def _get_init_params(self) -> dict[str, Any]:
        return {
            "channels": self.channels,
            "kernel_size": self.kernel_size,
            "dropout": self.dropout,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "random_state": self.random_state,
            "device": self.device,
        }


__all__ = ["CNN1D", "LSTM", "MLP", "TorchClassifierBase"]
