"""Host-owned managed and custom-loop Protocol Session SDK."""

from __future__ import annotations

from contextlib import contextmanager
import contextvars
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

from authority.protocol_execution_contract import ProtocolExecutionContract

from .collector_client import CollectorClient
from .errors import InvalidViewHandle, ProtocolStateError
from .events import canonical_json
from .views import DataViewHandle, ProtocolSplit


_CURRENT_SESSION: contextvars.ContextVar["ProtocolSession | None"] = (
    contextvars.ContextVar("mlevolve_protocol_session", default=None)
)
SHADOW_OBSERVATION_SCHEMA = "mlevolve_protocol_session_shadow_observations_v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hash(artifact: Any) -> str:
    if isinstance(artifact, bytes):
        return hashlib.sha256(artifact).hexdigest()
    if isinstance(artifact, (str, Path)):
        path = Path(artifact)
        if path.is_file() and not path.is_symlink():
            return _file_sha256(path)
    value = {
        "type": f"{type(artifact).__module__}.{type(artifact).__qualname__}",
        "repr": repr(artifact),
    }
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _matrix(records: Iterable[dict[str, Any]], feature_keys: Sequence[str]):
    keys = tuple(map(str, feature_keys))
    if not keys:
        raise ValueError("Managed adapter requires at least one feature key")
    return [[row[key] for key in keys] for row in records]


class ProtocolSession:
    def __init__(
        self,
        contract: ProtocolExecutionContract,
        split: ProtocolSplit,
        collector_client: CollectorClient,
        *,
        runtime_mode: str = "host_sdk_enforce",
        shadow_observation_path: str | Path | None = None,
    ):
        mode = str(runtime_mode or "host_sdk_enforce").lower()
        if mode not in {"host_sdk_shadow", "host_sdk_enforce"}:
            raise ValueError(f"Unsupported ProtocolSession runtime mode: {mode}")
        self.contract = contract
        self._split = split
        self._client = collector_client
        self.runtime_mode = mode
        self._shadow_observation_path = (
            Path(shadow_observation_path).resolve()
            if shadow_observation_path is not None
            else None
        )
        self._shadow_observations: list[dict[str, Any]] = []
        self._split_observed = False
        self._fit_observed = False
        self._prediction_observed = False
        self._evaluator_observed = False
        self._selection_frozen = False

    def _validate_view(
        self,
        view: DataViewHandle,
        expected_role: str,
        *,
        operation: str,
    ) -> None:
        expected = {
            "train": self._split.train,
            "internal_validation": self._split.validation,
            "inference": self._split.inference,
        }.get(expected_role)
        if expected is None:
            raise InvalidViewHandle(f"Host did not issue a {expected_role} view")
        issued = tuple(
            item
            for item in (
                self._split.train,
                self._split.validation,
                self._split.inference,
            )
            if item is not None
        )
        if not any(view is item for item in issued):
            raise InvalidViewHandle("View handle was not issued to this ProtocolSession")
        if view.contract_hash != self.contract.contract_hash:
            raise InvalidViewHandle("View handle role or Contract binding mismatch")
        if view is not expected or view.role != expected_role:
            message = (
                f"{operation} expected the {expected_role} DataViewHandle but "
                f"received the Host-issued {view.role} handle"
            )
            if not self._shadow:
                raise InvalidViewHandle(message)
            self._record_shadow_observation(
                "host_issued_view_role_mismatch",
                message,
                operation=operation,
            )

    @property
    def _shadow(self) -> bool:
        return self.runtime_mode == "host_sdk_shadow"

    def _record_shadow_observation(
        self,
        code: str,
        message: str,
        *,
        operation: str,
    ) -> None:
        observation = {
            "sequence": len(self._shadow_observations),
            "code": str(code),
            "message": str(message),
            "operation": str(operation),
        }
        self._shadow_observations.append(observation)
        path = self._shadow_observation_path
        if path is None:
            return
        payload = {
            "schema": SHADOW_OBSERVATION_SCHEMA,
            "runtime_mode": self.runtime_mode,
            "observations": list(self._shadow_observations),
        }
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, path)
        except Exception as error:
            # Observation persistence is intentionally not an admission gate in
            # shadow mode.  Keep the failure attached to the in-memory trace so
            # callers can still diagnose it without killing Candidate work.
            observation["persistence_error"] = (
                f"{type(error).__name__}: {error}"
            )
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _shadow_or_raise(
        self,
        code: str,
        message: str,
        *,
        operation: str,
    ) -> None:
        if not self._shadow:
            raise ProtocolStateError(message)
        self._record_shadow_observation(
            code,
            message,
            operation=operation,
        )

    def _emit(
        self,
        kind: str,
        *,
        capabilities: tuple[str, ...],
        component: str,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        try:
            self._client.emit(
                kind,
                capabilities=capabilities,
                component=component,
                payload=payload,
            )
        except Exception as error:
            if not self._shadow:
                raise
            self._record_shadow_observation(
                "collector_event_rejected",
                f"{kind}: {type(error).__name__}: {error}",
                operation=kind,
            )
            return False
        return True

    def _ensure_mutable_selection(self, operation: str) -> None:
        if self._selection_frozen:
            self._shadow_or_raise(
                "operation_after_selection_freeze",
                "Selection is frozen; training/search cannot continue",
                operation=operation,
            )

    def get_split(self) -> ProtocolSplit:
        if not self._split_observed:
            self._emit(
                "split_lineage",
                capabilities=(
                    self._split.train._capability,
                    self._split.validation._capability,
                ),
                component="protocol_session.get_split",
            )
            self._split_observed = True
        return self._split

    @contextmanager
    def fit_scope(self, *, component: str, data_view: DataViewHandle):
        self._ensure_mutable_selection("fit_scope")
        self.get_split()
        self._validate_view(
            data_view,
            "train",
            operation="fit_scope",
        )
        records = data_view.records()
        try:
            yield records
        except Exception:
            raise
        else:
            self._emit(
                "fit_scope",
                capabilities=(data_view._capability,),
                component=component,
                payload={"component": component},
            )
            self._fit_observed = True

    @contextmanager
    def prediction_scope(self, *, component: str, data_view: DataViewHandle):
        if not self._fit_observed:
            self._shadow_or_raise(
                "prediction_before_fit_scope",
                "Prediction requires a completed fit scope",
                operation="prediction_scope",
            )
        self._validate_view(
            data_view,
            "internal_validation",
            operation="prediction_scope",
        )
        final_inference = self._selection_frozen
        records = data_view.records()
        try:
            yield records
        except Exception:
            raise
        else:
            # A frozen model may still perform final/submission inference.  Its
            # internal-validation prediction receipt was already sealed before
            # selection freeze; emitting another event would violate the
            # collector's append-only terminal state.  Evaluation and fitting
            # remain prohibited after freeze, so this cannot alter selection.
            if not final_inference and not self._prediction_observed:
                self._emit(
                    "prediction_scope",
                    capabilities=(data_view._capability,),
                    component=component,
                )
                self._prediction_observed = True

    @contextmanager
    def inference_scope(self, *, component: str, data_view: DataViewHandle):
        """Expose immutable unlabeled test rows after model selection freezes."""

        if not self._fit_observed:
            self._shadow_or_raise(
                "inference_before_fit_scope",
                "Inference requires a completed fit scope",
                operation="inference_scope",
            )
        if not self._selection_frozen:
            self._shadow_or_raise(
                "inference_before_selection_freeze",
                "Submission inference requires selection freeze",
                operation="inference_scope",
            )
        self._validate_view(
            data_view,
            "inference",
            operation="inference_scope",
        )
        records = data_view.records()
        yield records

    def fit(
        self,
        estimator: Any,
        data_view: DataViewHandle,
        *,
        feature_keys: Sequence[str],
        label_key: str,
        component: str = "model",
        **fit_kwargs: Any,
    ) -> Any:
        with self.fit_scope(component=component, data_view=data_view) as records:
            estimator.fit(
                _matrix(records, feature_keys),
                [row[label_key] for row in records],
                **fit_kwargs,
            )
        return estimator

    def fit_preprocessor(
        self,
        preprocessor: Any,
        data_view: DataViewHandle,
        *,
        feature_keys: Sequence[str],
        **fit_kwargs: Any,
    ) -> Any:
        with self.fit_scope(
            component="preprocessor", data_view=data_view
        ) as records:
            preprocessor.fit(_matrix(records, feature_keys), **fit_kwargs)
        return preprocessor

    def predict(
        self,
        estimator: Any,
        data_view: DataViewHandle,
        *,
        feature_keys: Sequence[str],
        component: str = "model",
        method: str = "predict",
        **predict_kwargs: Any,
    ) -> Any:
        with self.prediction_scope(
            component=component, data_view=data_view
        ) as records:
            function = getattr(estimator, method)
            predictions = function(_matrix(records, feature_keys), **predict_kwargs)
        return predictions

    def evaluate_internal(
        self,
        data_view: DataViewHandle,
        predictions: Any,
        *,
        label_key: str,
    ) -> Any:
        self._ensure_mutable_selection("evaluate_internal")
        self._validate_view(
            data_view,
            "internal_validation",
            operation="evaluate_internal",
        )
        if not self._prediction_observed:
            if self._shadow:
                self._record_shadow_observation(
                    "evaluate_before_prediction_scope_exit",
                    "Internal evaluation was called before prediction evidence was sealed",
                    operation="evaluate_internal",
                )
                self._emit(
                    "prediction_scope",
                    capabilities=(data_view._capability,),
                    component="protocol_session.shadow_recovery",
                    payload={
                        "shadow_recovery": "evaluate_before_prediction_scope_exit"
                    },
                )
                self._prediction_observed = True
            else:
                raise ProtocolStateError(
                    "Internal evaluation requires prediction evidence"
                )
        records = data_view.records()
        labels = [
            row[label_key]
            if label_key in row
            else (row.get("assets") or {})[label_key]
            for row in records
        ]
        metric_spec = dict(self.contract.evaluator_spec.get("metric") or {})
        metric_name = str(metric_spec.get("name") or "")
        if metric_name == "macro_f1":
            from sklearn.metrics import f1_score

            score = f1_score(labels, predictions, average="macro")
        elif metric_name == "roc_auc":
            import numpy as np
            from sklearn.metrics import roc_auc_score

            values = np.asarray(predictions)
            if values.ndim == 2:
                if values.shape[1] != 2:
                    raise ProtocolStateError(
                        "Binary ROC-AUC requires scalar or two-class probabilities"
                    )
                values = values[:, 1]
            elif values.ndim != 1:
                raise ProtocolStateError(
                    "Binary ROC-AUC predictions have invalid dimensions"
                )
            score = roc_auc_score(labels, values)
        elif metric_name == "log_loss":
            from sklearn.metrics import log_loss

            score = log_loss(labels, predictions)
        elif metric_name == "rmse":
            if labels and isinstance(labels[0], (str, Path)):
                import numpy as np
                from PIL import Image

                squared_error = 0.0
                element_count = 0
                for record, label_path, raw_prediction in zip(
                    records, labels, predictions, strict=True
                ):
                    prediction = raw_prediction
                    # Image candidates commonly retain their sample ID next to
                    # each reconstructed array.  Accept that representation
                    # only when its ID exactly matches the Host-ordered record;
                    # any third, candidate-supplied target is deliberately
                    # ignored because the trusted target comes from DataView.
                    if isinstance(raw_prediction, (tuple, list)) and len(
                        raw_prediction
                    ) in {2, 3}:
                        sample_id = str(record.get("sample_id") or "")
                        if str(raw_prediction[0]) != sample_id:
                            raise ProtocolStateError(
                                "Image RMSE prediction sample_id does not match Host order"
                            )
                        prediction = raw_prediction[1]
                    target = np.asarray(Image.open(label_path), dtype=np.float64) / 255.0
                    if isinstance(prediction, (str, Path)):
                        predicted = (
                            np.asarray(Image.open(prediction), dtype=np.float64) / 255.0
                        )
                    else:
                        predicted = np.asarray(prediction, dtype=np.float64)
                    if target.shape != predicted.shape:
                        raise ProtocolStateError(
                            "Image RMSE prediction shape does not match Host target"
                        )
                    squared_error += float(np.square(target - predicted).sum())
                    element_count += int(target.size)
                if element_count <= 0:
                    raise ProtocolStateError("Image RMSE requires non-empty predictions")
                score = (squared_error / element_count) ** 0.5
            else:
                paired = [
                    (float(label), float(prediction))
                    for label, prediction in zip(labels, predictions, strict=True)
                ]
                if not paired:
                    raise ProtocolStateError("RMSE requires non-empty predictions")
                score = (
                    sum((label - prediction) ** 2 for label, prediction in paired)
                    / len(paired)
                ) ** 0.5
        else:
            raise ProtocolStateError(
                f"No Host-owned internal evaluator adapter for {metric_name!r}"
            )
        self._emit(
            "evaluator",
            capabilities=(data_view._capability,),
            component="protocol_session.evaluate_internal",
            payload={"metric_name": metric_name, "metric_value": float(score)},
        )
        self._evaluator_observed = True
        return score

    def freeze_selection(
        self,
        artifact: Any,
        *,
        based_on: DataViewHandle,
        artifact_hash: str | None = None,
    ) -> str:
        self._ensure_mutable_selection("freeze_selection")
        if not self._evaluator_observed:
            self._shadow_or_raise(
                "selection_freeze_before_internal_evaluation",
                "Selection freeze requires internal evaluation",
                operation="freeze_selection",
            )
        self._validate_view(
            based_on,
            "internal_validation",
            operation="freeze_selection",
        )
        if artifact_hash is None:
            digest = _artifact_hash(artifact)
        else:
            requested = str(artifact_hash)
            if len(requested) == 64 and all(
                character in "0123456789abcdef" for character in requested
            ):
                digest = requested
            else:
                # LLM candidates frequently pass the checkpoint path in the
                # ``artifact_hash`` slot.  Treat it as a path only when it is a
                # real, non-symlink regular file and let the Host compute the
                # digest; never forward an unverified candidate string as a
                # trusted selection hash.
                candidate_path = Path(requested)
                if not candidate_path.is_file() or candidate_path.is_symlink():
                    if not self._shadow:
                        raise ProtocolStateError(
                            "Selection artifact_hash must be SHA-256 or a regular checkpoint path"
                        )
                    self._record_shadow_observation(
                        "unverifiable_selection_artifact_hash",
                        "Selection artifact_hash was neither SHA-256 nor a regular checkpoint path",
                        operation="freeze_selection",
                    )
                    digest = _artifact_hash(artifact)
                else:
                    digest = _file_sha256(candidate_path)
        self._emit(
            "selection_freeze",
            capabilities=(based_on._capability,),
            component="protocol_session.freeze_selection",
            payload={"artifact_hash": digest},
        )
        self._selection_frozen = True
        return digest

    @property
    def selection_frozen(self) -> bool:
        return self._selection_frozen

    @property
    def shadow_observations(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(row) for row in self._shadow_observations)


def current_session() -> ProtocolSession:
    session = _CURRENT_SESSION.get()
    if session is None:
        raise ProtocolStateError("No Host ProtocolSession is active")
    return session


@contextmanager
def activate_session(session: ProtocolSession):
    token = _CURRENT_SESSION.set(session)
    try:
        yield session
    finally:
        _CURRENT_SESSION.reset(token)


__all__ = [
    "SHADOW_OBSERVATION_SCHEMA",
    "ProtocolSession",
    "activate_session",
    "current_session",
]
