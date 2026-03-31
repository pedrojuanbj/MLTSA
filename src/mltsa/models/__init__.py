"""Model abstractions and training entry points for mltsa."""

from __future__ import annotations

from typing import Any, Callable

from .sklearn_wrappers import ExtraTrees, GradientBoosting, HistGradientBoosting, RandomForest
from .torch import CNN1D, LSTM, MLP


def get_model(name: str, **kwargs: Any) -> object:
    """Build a supported sklearn or PyTorch model wrapper by name."""

    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    factories: dict[str, Callable[..., object]] = {
        "random_forest": RandomForest,
        "rf": RandomForest,
        "gradient_boosting": GradientBoosting,
        "gb": GradientBoosting,
        "gbdt": GradientBoosting,
        "hist_gradient_boosting": HistGradientBoosting,
        "hist_gbdt": HistGradientBoosting,
        "hgb": HistGradientBoosting,
        "extra_trees": ExtraTrees,
        "et": ExtraTrees,
        "mlp": MLP,
        "torch_mlp": MLP,
        "lstm": LSTM,
        "torch_lstm": LSTM,
        "cnn1d": CNN1D,
        "cnn_1d": CNN1D,
        "cnn": CNN1D,
        "torch_cnn1d": CNN1D,
    }
    try:
        return factories[normalized](**kwargs)
    except KeyError as exc:
        supported = ", ".join(sorted(factories))
        raise ValueError(f"Unsupported model name {name!r}. Supported names: {supported}.") from exc


__all__ = [
    "CNN1D",
    "ExtraTrees",
    "GradientBoosting",
    "HistGradientBoosting",
    "LSTM",
    "MLP",
    "RandomForest",
    "get_model",
]
