"""Small, terminal-blind reference paths for the three Host-owned protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .adapters import boosting, sklearn as sklearn_adapter, torch as torch_adapter
from .session import ProtocolSession


@dataclass(frozen=True)
class ProtocolReference:
    task_id: str
    task_family: str
    protocol_ref: str
    label_key: str
    records: tuple[dict[str, Any], ...]


class ReferenceBoostingEstimator:
    """Dependency-light XGBoost-shaped estimator for SDK reliability checks."""

    __module__ = "xgboost.sklearn"

    def fit(self, features, labels):
        del features
        first = labels[0]
        if isinstance(first, (list, tuple)):
            width = len(first)
            self.value = [
                int(sum(int(label[index]) for label in labels) * 2 >= len(labels))
                for index in range(width)
            ]
        elif all(isinstance(label, int) for label in labels):
            counts: dict[int, int] = {}
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
            self.value = max(sorted(counts), key=counts.__getitem__)
        else:
            self.value = sum(float(label) for label in labels) / len(labels)
        return self

    def predict(self, features):
        return [self.value for _ in features]


def cactus_reference() -> ProtocolReference:
    return ProtocolReference(
        task_id="aerial-cactus-identification",
        task_family="image",
        protocol_ref="random-classification@1",
        label_key="label",
        records=tuple(
            {
                "sample_id": f"cactus-{label}-{index}",
                "label": label,
                "x1": float(index),
                "x2": float((index + label) % 3),
            }
            for label in (0, 1)
            for index in range(10)
        ),
    )


def birds_reference() -> ProtocolReference:
    return ProtocolReference(
        task_id="mlsp-2013-birds",
        task_family="audio",
        protocol_ref="grouped-classification@1",
        label_key="label",
        records=tuple(
            {
                "sample_id": f"bird-{group}-{index}",
                "group_id": group,
                "label": [index % 2, (index // 2 + group_index) % 2],
                "x1": float(index),
                "x2": float(group_index),
            }
            for group_index, group in enumerate(("site-a", "site-b", "site-c", "site-d"))
            for index in range(5)
        ),
    )


def taxi_reference() -> ProtocolReference:
    return ProtocolReference(
        task_id="new-york-city-taxi-fare-prediction",
        task_family="tabular",
        protocol_ref="chronological-regression@1",
        label_key="fare",
        records=tuple(
            {
                "sample_id": f"taxi-{index:03d}",
                "timestamp": f"2026-01-{index + 1:02d}T00:00:00Z",
                "fare": 5.0 + 0.75 * index,
                "x1": float(index),
                "x2": float(index % 4),
            }
            for index in range(24)
        ),
    )


def protocol_references() -> tuple[ProtocolReference, ...]:
    return cactus_reference(), birds_reference(), taxi_reference()


def _label_key(session: ProtocolSession) -> str:
    if session.contract.split_strategy == "chronological":
        return "fare"
    return "label"


def sklearn_reference_candidate(session: ProtocolSession) -> None:
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    views = session.get_split()
    if session.contract.split_strategy == "chronological":
        estimator = DecisionTreeRegressor(max_depth=2, random_state=11)
    elif session.contract.split_strategy == "grouped":
        estimator = MultiOutputClassifier(
            DecisionTreeClassifier(max_depth=2, random_state=11)
        )
    else:
        estimator = DecisionTreeClassifier(max_depth=2, random_state=11)
    model = sklearn_adapter.fit(
        session,
        estimator,
        views.train,
        feature_keys=("x1", "x2"),
        label_key=_label_key(session),
    )
    predictions = sklearn_adapter.predict(
        session, model, views.validation, feature_keys=("x1", "x2")
    )
    session.evaluate_internal(
        views.validation, predictions, label_key=_label_key(session)
    )
    session.freeze_selection(
        "sklearn-reference",
        based_on=views.validation,
        artifact_hash="1" * 64,
    )


def boosting_reference_candidate(session: ProtocolSession) -> None:
    views = session.get_split()
    model = boosting.fit(
        session,
        ReferenceBoostingEstimator(),
        views.train,
        feature_keys=("x1", "x2"),
        label_key=_label_key(session),
    )
    predictions = boosting.predict(
        session, model, views.validation, feature_keys=("x1", "x2")
    )
    session.evaluate_internal(
        views.validation, predictions, label_key=_label_key(session)
    )
    session.freeze_selection(
        "boosting-reference",
        based_on=views.validation,
        artifact_hash="2" * 64,
    )


def torch_reference_candidate(session: ProtocolSession) -> None:
    import torch

    views = session.get_split()
    label_key = _label_key(session)
    with torch_adapter.fit_scope(
        session, component="torch_reference", data_view=views.train
    ) as train_rows:
        weight = torch.tensor([float(len(train_rows))], requires_grad=True)
        (weight * 0.0).sum().backward()
    with torch_adapter.prediction_scope(
        session, component="torch_reference", data_view=views.validation
    ) as validation_rows:
        predictions = [row[label_key] for row in validation_rows]
    session.evaluate_internal(
        views.validation, predictions, label_key=label_key
    )
    session.freeze_selection(
        "torch-reference",
        based_on=views.validation,
        artifact_hash="3" * 64,
    )


def cactus_invalid_terminal_candidate(session: ProtocolSession) -> None:
    terminal_labels = "/data/terminal_holdout/labels.csv"
    views = session.get_split()
    if False:
        session.fit_scope(component=terminal_labels, data_view=views.train)
        session.prediction_scope(component="invalid", data_view=views.validation)
        session.evaluate_internal(views.validation, [], label_key="label")
        session.freeze_selection("invalid", based_on=views.validation)


def birds_invalid_terminal_candidate(session: ProtocolSession) -> None:
    terminal_labels = "/data/evaluator/terminal_labels.csv"
    views = session.get_split()
    if False:
        session.fit_scope(component=terminal_labels, data_view=views.train)
        session.prediction_scope(component="invalid", data_view=views.validation)
        session.evaluate_internal(views.validation, [], label_key="label")
        session.freeze_selection("invalid", based_on=views.validation)


def taxi_invalid_random_resplit_candidate(session: ProtocolSession) -> None:
    from sklearn.model_selection import train_test_split

    views = session.get_split()
    if False:
        train_test_split([], shuffle=True)
        session.fit_scope(component="invalid", data_view=views.train)
        session.prediction_scope(component="invalid", data_view=views.validation)
        session.evaluate_internal(views.validation, [], label_key="fare")
        session.freeze_selection("invalid", based_on=views.validation)


FRAMEWORK_CANDIDATES: dict[str, Callable[[ProtocolSession], None]] = {
    "sklearn": sklearn_reference_candidate,
    "boosting": boosting_reference_candidate,
    "torch": torch_reference_candidate,
}

INVALID_CANDIDATES: dict[str, Callable[[ProtocolSession], None]] = {
    "aerial-cactus-identification": cactus_invalid_terminal_candidate,
    "mlsp-2013-birds": birds_invalid_terminal_candidate,
    "new-york-city-taxi-fare-prediction": taxi_invalid_random_resplit_candidate,
}


__all__ = [
    "FRAMEWORK_CANDIDATES",
    "INVALID_CANDIDATES",
    "ProtocolReference",
    "ReferenceBoostingEstimator",
    "birds_reference",
    "boosting_reference_candidate",
    "cactus_reference",
    "protocol_references",
    "sklearn_reference_candidate",
    "taxi_reference",
    "torch_reference_candidate",
]
