"""Custom-loop PyTorch scope helpers."""

from __future__ import annotations

from contextlib import AbstractContextManager

from ..session import ProtocolSession
from ..views import DataViewHandle


def fit_scope(
    session: ProtocolSession,
    *,
    component: str,
    data_view: DataViewHandle,
) -> AbstractContextManager:
    return session.fit_scope(component=component, data_view=data_view)


def prediction_scope(
    session: ProtocolSession,
    *,
    component: str,
    data_view: DataViewHandle,
) -> AbstractContextManager:
    return session.prediction_scope(component=component, data_view=data_view)


__all__ = ["fit_scope", "prediction_scope"]
