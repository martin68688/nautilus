"""Managed sklearn adapter."""

from __future__ import annotations

from typing import Any, Sequence

from ..session import ProtocolSession
from ..views import DataViewHandle


def _validate(estimator: Any) -> None:
    if type(estimator).__module__.split(".", 1)[0] != "sklearn":
        raise TypeError("sklearn adapter only supports sklearn estimators")


def fit(
    session: ProtocolSession,
    estimator: Any,
    view: DataViewHandle,
    *,
    feature_keys: Sequence[str],
    label_key: str,
    **kwargs: Any,
) -> Any:
    _validate(estimator)
    return session.fit(
        estimator,
        view,
        feature_keys=feature_keys,
        label_key=label_key,
        component="sklearn_model",
        **kwargs,
    )


def predict(
    session: ProtocolSession,
    estimator: Any,
    view: DataViewHandle,
    *,
    feature_keys: Sequence[str],
    method: str = "predict",
    **kwargs: Any,
) -> Any:
    _validate(estimator)
    return session.predict(
        estimator,
        view,
        feature_keys=feature_keys,
        component="sklearn_model",
        method=method,
        **kwargs,
    )


__all__ = ["fit", "predict"]
