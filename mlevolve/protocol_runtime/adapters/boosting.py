"""Managed XGBoost/LightGBM adapter."""

from __future__ import annotations

from typing import Any, Sequence

from ..session import ProtocolSession
from ..views import DataViewHandle


def _validate(estimator: Any) -> str:
    root = type(estimator).__module__.split(".", 1)[0]
    if root not in {"xgboost", "lightgbm"}:
        raise TypeError("Boosting adapter only supports xgboost/lightgbm estimators")
    return root


def fit(
    session: ProtocolSession,
    estimator: Any,
    view: DataViewHandle,
    *,
    feature_keys: Sequence[str],
    label_key: str,
    **kwargs: Any,
) -> Any:
    root = _validate(estimator)
    return session.fit(
        estimator,
        view,
        feature_keys=feature_keys,
        label_key=label_key,
        component=f"{root}_model",
        **kwargs,
    )


def predict(
    session: ProtocolSession,
    estimator: Any,
    view: DataViewHandle,
    *,
    feature_keys: Sequence[str],
    **kwargs: Any,
) -> Any:
    root = _validate(estimator)
    return session.predict(
        estimator,
        view,
        feature_keys=feature_keys,
        component=f"{root}_model",
        **kwargs,
    )


__all__ = ["fit", "predict"]
