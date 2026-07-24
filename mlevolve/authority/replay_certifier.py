from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


REPLAY_VERIFICATION_SCHEMA = "clean_replay_verification_v1"
PROTOCOL_REPAIR_SURFACE_SCHEMA = "protocol_repair_surface_v1"


class ReplayIdentity(str, Enum):
    METHOD_PRESERVED = "method_preserved"
    SUCCESSOR_METHOD = "successor_method"
    REQUIRE_HUMAN_REVIEW = "require_human_review"


_CHANGE_ALIASES = {
    "cross_fit": "split_api",
    "data_split": "split_api",
    "split": "split_api",
    "split_api": "split_api",
    "fit_scope": "preprocessing_scope",
    "fold_local_fit": "preprocessing_scope",
    "preprocessing_scope": "preprocessing_scope",
    "prediction_scope": "holdout_access",
    "evaluator": "evaluator",
    "evaluator_integrity": "evaluator",
    "metric_direction": "evaluator",
    "selection_freeze": "selection_freeze",
    "seed_aggregation": "seed_aggregation",
    "holdout": "holdout_access",
    "holdout_access": "holdout_access",
    "final_holdout": "holdout_access",
    "instrumentation": "instrumentation",
    "protocol_logging": "instrumentation",
}
_CHANGE_KINDS = frozenset(_CHANGE_ALIASES.values())


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, set):
        return sorted((_jsonable(item) for item in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProtocolRepairSurface:
    """Explicit protocol-only edit surface; an empty surface is fail-closed."""

    allowed_change_kinds: tuple[str, ...]
    protocol_ref: str = ""
    source: str = "explicit"
    surface_hash: str = ""
    schema: str = PROTOCOL_REPAIR_SURFACE_SCHEMA

    def __post_init__(self) -> None:
        normalized = tuple(
            sorted(
                {
                    _CHANGE_ALIASES.get(str(value).strip(), str(value).strip())
                    for value in self.allowed_change_kinds
                    if str(value).strip()
                }
            )
        )
        unknown = sorted(set(normalized) - _CHANGE_KINDS)
        if unknown:
            raise ValueError(f"Unknown protocol repair surface: {unknown}")
        object.__setattr__(self, "allowed_change_kinds", normalized)
        payload = self.as_dict()
        payload.pop("surface_hash", None)
        expected = _sha256_json(payload)
        if self.surface_hash and self.surface_hash != expected:
            raise ValueError("Protocol repair surface hash mismatch")
        object.__setattr__(self, "surface_hash", expected)

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    @classmethod
    def from_allowed_changes(
        cls,
        values: Iterable[str] | None,
        *,
        protocol_ref: str = "",
        source: str = "explicit",
    ) -> "ProtocolRepairSurface":
        return cls(tuple(str(value) for value in values or ()), protocol_ref, source)

    @classmethod
    def from_protocol_spec(cls, spec: Any) -> "ProtocolRepairSurface":
        policy = getattr(spec, "promotion_policy", {}) or {}
        replay = policy.get("clean_replay") or {}
        values = replay.get("allowed_protocol_changes") or ()
        return cls.from_allowed_changes(
            values,
            protocol_ref=spec.ref().key(),
            source="ProtocolSpec.promotion_policy.clean_replay",
        )


@dataclass(frozen=True)
class MethodFingerprint:
    ast_hash: str
    model_families: tuple[str, ...]
    feature_symbols: tuple[str, ...]
    training_calls: tuple[str, ...]
    inference_calls: tuple[str, ...]
    model_signatures: tuple[str, ...]
    feature_logic_hash: str
    feature_signatures: tuple[str, ...] = ()
    loss_objective_signatures: tuple[str, ...] = ()
    search_space_signatures: tuple[str, ...] = ()
    compute_budget_signatures: tuple[str, ...] = ()
    inference_signatures: tuple[str, ...] = ()
    ensemble_signatures: tuple[str, ...] = ()
    residual_method_logic_hash: str = ""
    protected_surface_hash: str = ""

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    def digest(self) -> str:
        return _sha256_json(self.as_dict())

    def protected_payload(self) -> dict[str, Any]:
        return {
            "model_families": self.model_families,
            "model_signatures": self.model_signatures,
            "training_calls": self.training_calls,
            "feature_signatures": self.feature_signatures,
            "feature_logic_hash": self.feature_logic_hash,
            "loss_objective_signatures": self.loss_objective_signatures,
            "search_space_signatures": self.search_space_signatures,
            "compute_budget_signatures": self.compute_budget_signatures,
            "inference_calls": self.inference_calls,
            "inference_signatures": self.inference_signatures,
            "ensemble_signatures": self.ensemble_signatures,
            "residual_method_logic_hash": self.residual_method_logic_hash,
        }


def _call_name(node: ast.Call) -> str:
    target = node.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        parts = [target.attr]
        while isinstance(target.value, ast.Attribute):
            target = target.value
            parts.append(target.attr)
        if isinstance(target.value, ast.Name):
            parts.append(target.value.id)
        return ".".join(reversed(parts))
    return ""


class _Canonicalizer(ast.NodeTransformer):
    """Erase local variable spelling while retaining operators and literals."""

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = "value"
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = "value"
        return node


def _canonical_dump(node: ast.AST) -> str:
    normalized = _Canonicalizer().visit(copy.deepcopy(node))
    ast.fix_missing_locations(normalized)
    return ast.dump(normalized, annotate_fields=True, include_attributes=False)


def _call_signature(node: ast.Call) -> str:
    return _canonical_dump(node)


def _target_names(node: ast.AST) -> tuple[str, ...]:
    output: list[str] = []
    for value in ast.walk(node):
        if isinstance(value, ast.Name) and isinstance(value.ctx, ast.Store):
            output.append(value.id)
        elif isinstance(value, ast.Attribute) and isinstance(value.ctx, ast.Store):
            output.append(value.attr)
    return tuple(sorted(set(output)))


_FEATURE_TERMS = (
    "feature",
    "vector",
    "embed",
    "tfidf",
    "pca",
    "scaler",
    "encoder",
    "imputer",
    "polynomial",
    "selector",
)
_FEATURE_CALL_TERMS = _FEATURE_TERMS + (
    "onehot",
    "countvectorizer",
    "standardscale",
    "minmaxscale",
)
_MODEL_TERMS = (
    "classifier",
    "regressor",
    "xgb",
    "lightgbm",
    "lgbm",
    "catboost",
    "bert",
    "resnet",
    "svm",
    "forest",
    "network",
    "model",
)
_NON_MODEL_CONSTRUCTORS = {
    "Path",
    "DataFrame",
    "Series",
    "Dataset",
    "DataLoader",
    "Sampler",
    "ProtocolProvenanceGuard",
    "KFold",
    "StratifiedKFold",
    "GroupKFold",
    "TimeSeriesSplit",
    "GridSearchCV",
    "RandomizedSearchCV",
}
_TRAINING_LEAVES = {"fit", "fit_transform", "partial_fit", "train"}
_INFERENCE_LEAVES = {"predict", "predict_proba", "decision_function", "transform", "forward"}
_LOSS_TERMS = ("loss", "criterion", "objective", "crossentropy", "mse", "mae", "nll")
_SEARCH_TERMS = (
    "gridsearch",
    "randomizedsearch",
    "bayessearch",
    "optuna",
    "hyperopt",
    "suggest_",
    "search_space",
    "param_grid",
)
_BUDGET_NAMES = {
    "epochs",
    "num_epochs",
    "n_epochs",
    "max_epochs",
    "n_estimators",
    "max_iter",
    "num_boost_round",
    "n_trials",
    "timeout",
    "training_budget",
    "compute_budget",
    "batch_size",
}
_ENSEMBLE_TERMS = ("ensemble", "voting", "stack", "blend", "averag", "weighted")
_PROTOCOL_TARGET_TERMS = (
    "train",
    "valid",
    "val",
    "test",
    "holdout",
    "fold",
    "split",
    "index",
    "indices",
    "idx",
    "oof",
    "seed",
    "metric",
    "score",
    "guard",
    "lineage",
)


def _is_feature_call(name: str) -> bool:
    lower = name.lower()
    return any(term in lower for term in _FEATURE_CALL_TERMS)


def _is_search_call(name: str) -> bool:
    lower = name.lower()
    return any(term in lower for term in _SEARCH_TERMS)


def _is_loss_call(name: str) -> bool:
    lower = name.lower()
    return any(term in lower for term in _LOSS_TERMS)


def _is_model_call(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    if (
        not leaf
        or leaf in _NON_MODEL_CONSTRUCTORS
        or leaf in _TRAINING_LEAVES
        or leaf in _INFERENCE_LEAVES
        or leaf.endswith("Error")
        or leaf.endswith("Exception")
        or leaf.endswith("Dataset")
        or leaf.endswith("DataLoader")
        or leaf.endswith("Sampler")
        or _is_feature_call(name)
        or _is_search_call(name)
        or _is_loss_call(name)
    ):
        return False
    return bool(
        any(term in name.lower() for term in _MODEL_TERMS)
        or (leaf[0].isupper() and leaf not in {"Path", "DataFrame", "Series"})
    )


def _assignment_signatures(
    tree: ast.AST,
    predicate,
) -> tuple[str, ...]:
    output = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        names = _target_names(node)
        dump = ast.dump(node, annotate_fields=False, include_attributes=False).lower()
        calls = [_call_name(value) for value in ast.walk(node) if isinstance(value, ast.Call)]
        if predicate(names, dump, calls):
            output.append(_canonical_dump(node))
    return tuple(sorted(set(output)))


def fingerprint_method(code: str) -> MethodFingerprint:
    tree = ast.parse(code)
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    call_names = tuple(sorted({_call_name(node) for node in calls if _call_name(node)}))
    models = tuple(sorted({name for name in call_names if _is_model_call(name)}))
    model_signatures = tuple(
        sorted(
            {
                _call_signature(node)
                for node in calls
                if _is_model_call(_call_name(node))
            }
        )
    )
    features = tuple(
        sorted(
            {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name)
                and any(term in node.id.lower() for term in _FEATURE_TERMS)
            }
        )
    )
    training = tuple(
        sorted(
            {
                name
                for name in call_names
                if name.rsplit(".", 1)[-1] in _TRAINING_LEAVES
            }
        )
    )
    inference = tuple(
        sorted(
            {
                name
                for name in call_names
                if name.rsplit(".", 1)[-1] in _INFERENCE_LEAVES
                and not _is_feature_call(name)
            }
        )
    )
    feature_assignments = _assignment_signatures(
        tree,
        lambda names, dump, statement_calls: bool(
            (
                any(any(term in name.lower() for term in _FEATURE_TERMS) for name in names)
                or any(term in dump for term in _FEATURE_TERMS)
                or any(_is_feature_call(name) for name in statement_calls)
            )
            and not (
                statement_calls
                and all(
                    name.rsplit(".", 1)[-1]
                    in {"fit", "fit_transform", "transform"}
                    for name in statement_calls
                )
            )
        ),
    )
    feature_call_signatures = {
        _call_signature(node)
        for node in calls
        if _is_feature_call(_call_name(node))
        and _call_name(node).rsplit(".", 1)[-1]
        not in {"fit", "fit_transform", "transform"}
    }
    feature_signatures = tuple(sorted(set(feature_assignments) | feature_call_signatures))

    loss_assignments = _assignment_signatures(
        tree,
        lambda names, dump, statement_calls: bool(
            any(any(term in name.lower() for term in _LOSS_TERMS) for name in names)
            or any(term in dump for term in _LOSS_TERMS)
            or any(_is_loss_call(name) for name in statement_calls)
        ),
    )
    loss_signatures = tuple(
        sorted(
            set(loss_assignments)
            | {_call_signature(node) for node in calls if _is_loss_call(_call_name(node))}
            | {
                _canonical_dump(keyword)
                for node in calls
                for keyword in node.keywords
                if keyword.arg and keyword.arg.lower() in {"loss", "objective", "criterion"}
            }
        )
    )
    search_assignments = _assignment_signatures(
        tree,
        lambda names, dump, statement_calls: bool(
            any(any(term in name.lower() for term in _SEARCH_TERMS) for name in names)
            or any(term in dump for term in _SEARCH_TERMS)
            or any(_is_search_call(name) for name in statement_calls)
        ),
    )
    search_signatures = tuple(
        sorted(
            set(search_assignments)
            | {_call_signature(node) for node in calls if _is_search_call(_call_name(node))}
        )
    )
    budget_signatures = tuple(
        sorted(
            {
                _canonical_dump(keyword)
                for node in calls
                for keyword in node.keywords
                if keyword.arg and keyword.arg.lower() in _BUDGET_NAMES
            }
            | set(
                _assignment_signatures(
                    tree,
                    lambda names, _dump, _calls: bool(
                        {name.lower() for name in names} & _BUDGET_NAMES
                    ),
                )
            )
        )
    )
    inference_signatures = tuple(
        sorted(
            {
                _call_name(node).rsplit(".", 1)[-1]
                for node in calls
                if _call_name(node).rsplit(".", 1)[-1] in _INFERENCE_LEAVES
                and not _is_feature_call(_call_name(node))
            }
        )
    )
    ensemble_assignments = _assignment_signatures(
        tree,
        lambda names, dump, statement_calls: bool(
            any(any(term in name.lower() for term in _ENSEMBLE_TERMS) for name in names)
            or any(term in dump for term in _ENSEMBLE_TERMS)
            or any(any(term in name.lower() for term in _ENSEMBLE_TERMS) for name in statement_calls)
        ),
    )
    ensemble_signatures = tuple(
        sorted(
            set(ensemble_assignments)
            | {
                _call_signature(node)
                for node in calls
                if any(term in _call_name(node).lower() for term in _ENSEMBLE_TERMS)
            }
        )
    )

    protected_assignment_nodes = set(
        feature_assignments
        + loss_assignments
        + search_assignments
        + ensemble_assignments
    )
    residual = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return)):
            continue
        signature = _canonical_dump(node)
        if signature in protected_assignment_nodes:
            continue
        names = _target_names(node)
        lower_names = " ".join(names).lower()
        statement_dump = ast.dump(
            node, annotate_fields=False, include_attributes=False
        ).lower()
        statement_calls = [
            _call_name(value) for value in ast.walk(node) if isinstance(value, ast.Call)
        ]
        if any(
            term in lower_names or term in statement_dump
            for term in _PROTOCOL_TARGET_TERMS
        ):
            continue
        if statement_calls and all(
            _is_model_call(name)
            or _is_feature_call(name)
            or _is_loss_call(name)
            or _is_search_call(name)
            or name.rsplit(".", 1)[-1] in (_TRAINING_LEAVES | _INFERENCE_LEAVES)
            for name in statement_calls
        ):
            continue
        residual.append(signature)
    residual_hash = _sha256_json(sorted(set(residual)))
    feature_logic_hash = _sha256_json(feature_signatures)
    payload = {
        "model_families": models,
        "model_signatures": model_signatures,
        "training_calls": training,
        "feature_signatures": feature_signatures,
        "feature_logic_hash": feature_logic_hash,
        "loss_objective_signatures": loss_signatures,
        "search_space_signatures": search_signatures,
        "compute_budget_signatures": budget_signatures,
        "inference_calls": inference,
        "inference_signatures": inference_signatures,
        "ensemble_signatures": ensemble_signatures,
        "residual_method_logic_hash": residual_hash,
    }
    return MethodFingerprint(
        ast_hash=_sha256_text(normalized),
        model_families=models,
        feature_symbols=features,
        training_calls=training,
        inference_calls=inference,
        model_signatures=model_signatures,
        feature_logic_hash=feature_logic_hash,
        feature_signatures=feature_signatures,
        loss_objective_signatures=loss_signatures,
        search_space_signatures=search_signatures,
        compute_budget_signatures=budget_signatures,
        inference_signatures=inference_signatures,
        ensemble_signatures=ensemble_signatures,
        residual_method_logic_hash=residual_hash,
        protected_surface_hash=_sha256_json(payload),
    )


def _protected_changes(
    source: MethodFingerprint,
    replay: MethodFingerprint,
) -> dict[str, dict[str, Any]]:
    changes: dict[str, dict[str, Any]] = {}
    for field_name in source.protected_payload():
        before = getattr(source, field_name)
        after = getattr(replay, field_name)
        if before != after:
            changes[field_name] = {"source": _jsonable(before), "replay": _jsonable(after)}
    return changes


_ALLOWED_CALL_TERMS = {
    "split_api": (
        "split",
        "kfold",
        "cross_val",
        "groupkfold",
        "timeseriessplit",
        "next",
    ),
    "preprocessing_scope": (
        "fit",
        "fit_transform",
        "transform",
        "concat",
        "concatenate",
    ),
    "evaluator": ("metric", "score", "accuracy", "f1", "auc", "rmse", "mae"),
    "selection_freeze": ("freeze", "candidate", "selection", "select"),
    "seed_aggregation": ("seed", "aggregate", "mean", "median"),
    "holdout_access": ("holdout", "final_evaluation", "evaluate"),
    "instrumentation": (
        "protocolprovenanceguard",
        "register_",
        "record_",
        "assert",
        "log",
        "error",
        "exception",
    ),
}


def _call_deltas(source_code: str, replay_code: str) -> tuple[list[str], list[str]]:
    source_tree = ast.parse(source_code)
    replay_tree = ast.parse(replay_code)
    source = {_call_name(node) for node in ast.walk(source_tree) if isinstance(node, ast.Call)}
    replay = {_call_name(node) for node in ast.walk(replay_tree) if isinstance(node, ast.Call)}
    return sorted(source - replay), sorted(replay - source)


def _allowed_protocol_call(name: str, surface: ProtocolRepairSurface) -> bool:
    lower = str(name).lower()
    leaf = lower.rsplit(".", 1)[-1]
    if leaf == "protocolprovenanceguard" and surface.allowed_change_kinds:
        return True
    if leaf.endswith("error") or leaf.endswith("exception"):
        return "instrumentation" in surface.allowed_change_kinds or bool(
            surface.allowed_change_kinds
        )
    return any(
        any(term in lower for term in _ALLOWED_CALL_TERMS[kind])
        for kind in surface.allowed_change_kinds
    )


@dataclass
class ReplayVerificationReport:
    source_artifact_id: str
    replay_artifact_id: str
    source_code_sha256: str
    replay_code_sha256: str
    source_method_fingerprint: str
    replay_method_fingerprint: str
    source_protected_surface_hash: str
    replay_protected_surface_hash: str
    repair_surface_hash: str
    repair_protocol_ref: str
    repair_surface_source: str
    allowed_protocol_changes: tuple[str, ...]
    identity: ReplayIdentity
    protected_changes: dict[str, Any] = field(default_factory=dict)
    removed_calls: tuple[str, ...] = ()
    added_calls: tuple[str, ...] = ()
    unclassified_call_deltas: tuple[str, ...] = ()
    reason: str = ""
    report_hash: str = ""
    schema: str = REPLAY_VERIFICATION_SCHEMA

    def finalize(self) -> "ReplayVerificationReport":
        payload = self.as_dict()
        payload.pop("report_hash", None)
        self.report_hash = _sha256_json(payload)
        return self

    def verify(self) -> None:
        if self.schema != REPLAY_VERIFICATION_SCHEMA:
            raise ValueError("Unsupported replay verification schema")
        payload = self.as_dict()
        payload.pop("report_hash", None)
        if self.report_hash != _sha256_json(payload):
            raise ValueError("Replay verification report hash mismatch")

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(self)


def verify_protocol_only_patch(
    source_code: str,
    replay_code: str,
    repair_surface: ProtocolRepairSurface,
    *,
    source_artifact_id: str = "source",
    replay_artifact_id: str = "replay",
) -> ReplayVerificationReport:
    try:
        source = fingerprint_method(source_code)
        replay = fingerprint_method(replay_code)
    except SyntaxError as error:
        return ReplayVerificationReport(
            source_artifact_id=str(source_artifact_id),
            replay_artifact_id=str(replay_artifact_id),
            source_code_sha256=_sha256_text(source_code),
            replay_code_sha256=_sha256_text(replay_code),
            source_method_fingerprint="",
            replay_method_fingerprint="",
            source_protected_surface_hash="",
            replay_protected_surface_hash="",
            repair_surface_hash=repair_surface.surface_hash,
            repair_protocol_ref=repair_surface.protocol_ref,
            repair_surface_source=repair_surface.source,
            allowed_protocol_changes=repair_surface.allowed_change_kinds,
            identity=ReplayIdentity.REQUIRE_HUMAN_REVIEW,
            reason=f"syntax_error:{type(error).__name__}",
        ).finalize()

    changes = _protected_changes(source, replay)
    removed, added = _call_deltas(source_code, replay_code)
    if changes:
        identity = ReplayIdentity.SUCCESSOR_METHOD
        unclassified: tuple[str, ...] = ()
        reason = "protected_method_surface_changed"
    elif source.ast_hash == replay.ast_hash:
        identity = ReplayIdentity.METHOD_PRESERVED
        unclassified = ()
        reason = "canonical_ast_identical"
    elif not repair_surface.allowed_change_kinds:
        identity = ReplayIdentity.REQUIRE_HUMAN_REVIEW
        unclassified = tuple(sorted(set(removed + added)))
        reason = "code_changed_without_declared_protocol_repair_surface"
    else:
        unclassified = tuple(
            sorted(
                name
                for name in set(removed + added)
                if name
                and not _allowed_protocol_call(name, repair_surface)
                and not _is_model_call(name)
                and not _is_feature_call(name)
                and not _is_loss_call(name)
                and not _is_search_call(name)
                and name.rsplit(".", 1)[-1]
                not in (_TRAINING_LEAVES | _INFERENCE_LEAVES)
            )
        )
        if unclassified:
            identity = ReplayIdentity.REQUIRE_HUMAN_REVIEW
            reason = "unclassified_non_protocol_call_delta"
        else:
            identity = ReplayIdentity.METHOD_PRESERVED
            reason = "protected_surface_equal_and_protocol_delta_declared"
    return ReplayVerificationReport(
        source_artifact_id=str(source_artifact_id),
        replay_artifact_id=str(replay_artifact_id),
        source_code_sha256=_sha256_text(source_code),
        replay_code_sha256=_sha256_text(replay_code),
        source_method_fingerprint=source.digest(),
        replay_method_fingerprint=replay.digest(),
        source_protected_surface_hash=source.protected_surface_hash,
        replay_protected_surface_hash=replay.protected_surface_hash,
        repair_surface_hash=repair_surface.surface_hash,
        repair_protocol_ref=repair_surface.protocol_ref,
        repair_surface_source=repair_surface.source,
        allowed_protocol_changes=repair_surface.allowed_change_kinds,
        identity=identity,
        protected_changes=changes,
        removed_calls=tuple(removed),
        added_calls=tuple(added),
        unclassified_call_deltas=unclassified,
        reason=reason,
    ).finalize()


def certify_replay(
    source_code: str,
    replay_code: str,
    allowed_protocol_changes: list[str] | None = None,
) -> ReplayIdentity:
    """Compatibility wrapper returning only the replay identity."""

    return verify_protocol_only_patch(
        source_code,
        replay_code,
        ProtocolRepairSurface.from_allowed_changes(allowed_protocol_changes),
    ).identity


__all__ = [
    "MethodFingerprint",
    "PROTOCOL_REPAIR_SURFACE_SCHEMA",
    "ProtocolRepairSurface",
    "REPLAY_VERIFICATION_SCHEMA",
    "ReplayIdentity",
    "ReplayVerificationReport",
    "certify_replay",
    "fingerprint_method",
    "verify_protocol_only_patch",
]
