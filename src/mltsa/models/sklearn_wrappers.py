"""Small sklearn classifier wrappers used by mltsa."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)


@dataclass(slots=True)
class _SklearnModelWrapper:
    """Thin wrapper around a sklearn estimator with a stable mltsa-facing API."""

    estimator: Any
    model_name: str

    def fit(self, X: Any, y: Any) -> _SklearnModelWrapper:
        """Fit the wrapped estimator and return ``self``."""

        self.estimator.fit(X, y)
        return self

    def predict(self, X: Any) -> Any:
        """Delegate prediction to the wrapped estimator."""

        return self.estimator.predict(X)

    def predict_proba(self, X: Any) -> Any:
        """Delegate probability prediction when supported."""

        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(f"{self.model_name} does not support predict_proba().")
        return self.estimator.predict_proba(X)

    def score(self, X: Any, y: Any) -> float:
        """Delegate scoring to the wrapped estimator."""

        return float(self.estimator.score(X, y))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Expose wrapped estimator parameters."""

        return dict(self.estimator.get_params(deep=deep))

    def set_params(self, **params: Any) -> _SklearnModelWrapper:
        """Update wrapped estimator parameters and return ``self``."""

        self.estimator.set_params(**params)
        return self

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped estimator."""

        return getattr(self.estimator, name)


class RandomForest(_SklearnModelWrapper):
    """Random forest classifier wrapper."""

    canonical_name: ClassVar[str] = "random_forest"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("random_state", 0)
        super().__init__(RandomForestClassifier(**kwargs), self.canonical_name)


class GradientBoosting(_SklearnModelWrapper):
    """Gradient boosting classifier wrapper."""

    canonical_name: ClassVar[str] = "gradient_boosting"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("random_state", 0)
        super().__init__(GradientBoostingClassifier(**kwargs), self.canonical_name)


class HistGradientBoosting(_SklearnModelWrapper):
    """Histogram-based gradient boosting classifier wrapper."""

    canonical_name: ClassVar[str] = "hist_gradient_boosting"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("random_state", 0)
        super().__init__(HistGradientBoostingClassifier(**kwargs), self.canonical_name)


class ExtraTrees(_SklearnModelWrapper):
    """Extra trees classifier wrapper."""

    canonical_name: ClassVar[str] = "extra_trees"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("random_state", 0)
        super().__init__(ExtraTreesClassifier(**kwargs), self.canonical_name)


def get_model(name: str, **kwargs: Any) -> _SklearnModelWrapper:
    """Build a supported sklearn wrapper from a short model name."""

    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    factories = {
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
    }
    try:
        return factories[normalized](**kwargs)
    except KeyError as exc:
        supported = ", ".join(sorted(factories))
        raise ValueError(f"Unsupported model name {name!r}. Supported names: {supported}.") from exc


__all__ = [
    "ExtraTrees",
    "GradientBoosting",
    "HistGradientBoosting",
    "RandomForest",
    "get_model",
]
