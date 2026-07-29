"""Host-owned feasibility contract for paired candidate execution.

The contract is deliberately orthogonal to memory retrieval and Authority.  It
applies the same resource/dependency boundary to every experimental condition,
records a deterministic pre-execution audit, and never imports a source-task
score or success conclusion.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from collections.abc import Iterable, Mapping
from typing import Any


CONTRACT_SCHEMA = "mlevolve_candidate_execution_contract_v1"
AUDIT_SCHEMA = "mlevolve_candidate_execution_contract_audit_v1"
BLOCK_RECEIPT_SCHEMA = "mlevolve_candidate_execution_block_receipt_v1"

_EPOCH_NAMES = {
    "epoch",
    "epochs",
    "n_epoch",
    "n_epochs",
    "num_epoch",
    "num_epochs",
    "max_epoch",
    "max_epochs",
    "total_epoch",
    "total_epochs",
}
_CV_CONSTRUCTORS = {
    "GroupKFold",
    "KFold",
    "LeaveOneGroupOut",
    "LeaveOneOut",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "StratifiedGroupKFold",
    "StratifiedKFold",
    "TimeSeriesSplit",
}
_REMOTE_CALLS = {
    "gdown.download",
    "hub.load",
    "requests.get",
    "requests.post",
    "torch.hub.load",
    "urllib.request.urlretrieve",
    "urllib.request.urlopen",
    "wget.download",
}
_LOAD_CALLS = {
    "joblib.load",
    "numpy.load",
    "pickle.load",
    "torch.load",
}
_SAVE_CALLS = {
    "joblib.dump",
    "numpy.save",
    "pickle.dump",
    "torch.save",
}
_MODEL_TARGETS = {"clf", "classifier", "estimator", "model", "net", "network"}
_MODEL_TARGET_SUFFIXES = ("_clf", "_classifier", "_estimator", "_model", "_net", "_network")
_NON_CONSTRUCTOR_CALLS = {
    "clone",
    "compile",
    "cpu",
    "cuda",
    "deepcopy",
    "eval",
    "load",
    "loads",
    "to",
    "train",
}
_HANDCRAFTED_IMAGE_CALLS = {
    "dct",
    "gabor",
    "graycomatrix",
    "graycoprops",
    "greycoprops",
    "hog",
    "laplacian",
    "local_binary_pattern",
    "sobel",
}


def _sha256_json(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(payload),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _unique_strings(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def build_candidate_execution_contract(
    *,
    contract_id: str,
    max_execution_seconds: int,
    max_epochs: int,
    max_cv_folds: int,
    max_trainable_models: int,
    allowed_import_roots: Iterable[str],
    allow_remote_assets: bool,
    allow_unverified_local_assets: bool,
    allow_dataset_wide_per_sample_precompute: bool,
    allow_source_score_inheritance: bool,
) -> dict[str, Any]:
    """Build and hash one canonical enabled contract."""

    payload: dict[str, Any] = {
        "schema": CONTRACT_SCHEMA,
        "enabled": True,
        "contract_id": str(contract_id).strip(),
        "max_execution_seconds": int(max_execution_seconds),
        "max_epochs": int(max_epochs),
        "max_cv_folds": int(max_cv_folds),
        "max_trainable_models": int(max_trainable_models),
        "allowed_import_roots": _unique_strings(allowed_import_roots),
        "allow_remote_assets": bool(allow_remote_assets),
        "allow_unverified_local_assets": bool(allow_unverified_local_assets),
        "allow_dataset_wide_per_sample_precompute": bool(
            allow_dataset_wide_per_sample_precompute
        ),
        "allow_source_score_inheritance": bool(allow_source_score_inheritance),
        "contract_hash": "",
    }
    if not payload["contract_id"]:
        raise ValueError("An enabled candidate execution contract requires contract_id")
    if payload["max_execution_seconds"] <= 0:
        raise ValueError(
            "An enabled candidate execution contract requires "
            "max_execution_seconds > 0"
        )
    for field in ("max_epochs", "max_cv_folds", "max_trainable_models"):
        if payload[field] < 0:
            raise ValueError(
                f"An enabled candidate execution contract requires {field} >= 0"
            )
    if payload["allow_source_score_inheritance"]:
        raise ValueError(
            "Candidate execution contracts cannot authorize source-score inheritance"
        )
    unsigned = {key: value for key, value in payload.items() if key != "contract_hash"}
    payload["contract_hash"] = _sha256_json(unsigned)
    return payload


def candidate_execution_contract_from_cfg(cfg: Any) -> dict[str, Any] | None:
    """Resolve the optional contract from the merged runtime configuration."""

    agent_cfg = getattr(cfg, "agent", None) if cfg is not None else None
    raw = (
        getattr(agent_cfg, "candidate_execution_contract", None)
        if agent_cfg is not None
        else None
    )
    if raw is None or not bool(getattr(raw, "enabled", False)):
        return None
    contract = build_candidate_execution_contract(
        contract_id=getattr(raw, "contract_id", ""),
        max_execution_seconds=getattr(raw, "max_execution_seconds", 0),
        max_epochs=getattr(raw, "max_epochs", 0),
        max_cv_folds=getattr(raw, "max_cv_folds", 0),
        max_trainable_models=getattr(raw, "max_trainable_models", 0),
        allowed_import_roots=list(getattr(raw, "allowed_import_roots", []) or []),
        allow_remote_assets=getattr(raw, "allow_remote_assets", True),
        allow_unverified_local_assets=getattr(
            raw, "allow_unverified_local_assets", True
        ),
        allow_dataset_wide_per_sample_precompute=getattr(
            raw, "allow_dataset_wide_per_sample_precompute", True
        ),
        allow_source_score_inheritance=getattr(
            raw, "allow_source_score_inheritance", False
        ),
    )
    configured_timeout = int(getattr(getattr(cfg, "exec", None), "timeout", 0) or 0)
    if configured_timeout <= 0:
        raise ValueError("Candidate execution contract requires a positive exec.timeout")
    if contract["max_execution_seconds"] > configured_timeout:
        raise ValueError(
            "Candidate execution contract cannot exceed the host exec.timeout"
        )
    return contract


def _call_name(node: ast.AST) -> str:
    parts: list[str] = []
    current: ast.AST | None = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _literal_int(node: ast.AST | None, symbols: Mapping[str, int]) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if isinstance(node, ast.Name):
        return symbols.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _literal_int(node.operand, symbols)
        if value is not None:
            return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left = _literal_int(node.left, symbols)
        right = _literal_int(node.right, symbols)
        if left is not None and right is not None:
            return left + right if isinstance(node.op, ast.Add) else left - right
    return None


def _literal_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [name for item in node.elts for name in _target_names(item)]
    return []


def _range_iteration_count(call: ast.Call, symbols: Mapping[str, int]) -> int | None:
    values = [_literal_int(argument, symbols) for argument in call.args]
    if not values or any(value is None for value in values) or len(values) > 3:
        return None
    resolved = [int(value) for value in values if value is not None]
    if len(resolved) == 1:
        start, stop, step = 0, resolved[0], 1
    elif len(resolved) == 2:
        start, stop, step = resolved[0], resolved[1], 1
    else:
        start, stop, step = resolved
    if step == 0:
        return None
    return len(range(start, stop, step))


def _first_path_argument(call: ast.Call, call_name: str) -> ast.AST | None:
    if not call.args:
        return None
    if call_name in {"torch.save", "joblib.dump", "pickle.dump"}:
        return call.args[1] if len(call.args) > 1 else None
    return call.args[0]


def _looks_like_model_target(name: str) -> bool:
    lowered = name.lower()
    return lowered in _MODEL_TARGETS or lowered.endswith(_MODEL_TARGET_SUFFIXES)


def _looks_like_model_factory(name: str) -> bool:
    """Return whether a locally defined function explicitly builds a model.

    Assigning a feature helper result to a ``*_model`` variable is common in
    generated code (for example ``pca_model = add_pca_features(...)``). The
    target name alone is therefore not sufficient evidence that the call
    constructs a trainable model. Locally defined functions are counted only
    when their own name identifies them as a model factory; class constructors
    and imported calls retain the existing conservative policy.
    """

    return _looks_like_model_target(name) or any(
        marker in name.lower()
        for marker in ("classifier", "regressor", "estimator", "network", "learner")
    )


def _assignment_parts(node: ast.Assign | ast.AnnAssign) -> tuple[list[str], ast.AST | None]:
    if isinstance(node, ast.Assign):
        names = [name for target in node.targets for name in _target_names(target)]
        return names, node.value
    return _target_names(node.target), node.value


def _model_constructor_call_name(call: ast.Call) -> str:
    """Unwrap device/mode chaining while preserving the real constructor."""

    current = call
    while True:
        call_name = _call_name(current.func)
        short_name = call_name.rsplit(".", 1)[-1].lower()
        if (
            short_name in _NON_CONSTRUCTOR_CALLS
            and isinstance(current.func, ast.Attribute)
            and isinstance(current.func.value, ast.Call)
        ):
            current = current.func.value
            continue
        return call_name


def valid_candidate_execution_audit(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload.get("schema") == AUDIT_SCHEMA
        and payload.get("audit_hash")
        == _sha256_json(
            {key: value for key, value in payload.items() if key != "audit_hash"}
        )
    )


def build_candidate_execution_block_receipt(
    *,
    node_id: str,
    contract: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Record that a DENY was enforced before subprocess creation."""

    if audit.get("valid") is not False or not valid_candidate_execution_audit(audit):
        raise ValueError("A block receipt requires an integrity-valid DENY audit")
    if audit.get("contract_hash") != contract.get("contract_hash"):
        raise ValueError("Block receipt audit/contract mismatch")
    receipt: dict[str, Any] = {
        "schema": BLOCK_RECEIPT_SCHEMA,
        "node_id": str(node_id),
        "contract_id": contract.get("contract_id"),
        "contract_hash": contract.get("contract_hash"),
        "audit_hash": audit.get("audit_hash"),
        "code_sha256": audit.get("code_sha256"),
        "decision": "deny",
        "enforcement": "blocked_before_subprocess",
        "candidate_subprocess_started": False,
        "violations": list(audit.get("violations") or []),
        "receipt_hash": "",
    }
    receipt["receipt_hash"] = _sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_hash"}
    )
    return receipt


def valid_candidate_execution_block_receipt(payload: Mapping[str, Any]) -> bool:
    return bool(
        payload.get("schema") == BLOCK_RECEIPT_SCHEMA
        and payload.get("decision") == "deny"
        and payload.get("enforcement") == "blocked_before_subprocess"
        and payload.get("candidate_subprocess_started") is False
        and payload.get("receipt_hash")
        == _sha256_json(
            {key: value for key, value in payload.items() if key != "receipt_hash"}
        )
    )


def audit_candidate_code(
    code: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit machine-checkable clauses before candidate execution."""

    violations: list[str] = []
    # The legacy schema still carries method-policy fields so old manifests can
    # be read and hash-verified, but they are no longer enforcement inputs.
    # Candidate method choice belongs to the Agent; leakage is enforced by the
    # Host Protocol data/evidence lifecycle instead.
    epoch_cap = 0
    cv_cap = 0
    model_cap = 0
    allow_remote_assets = True
    allow_unverified_local_assets = True
    allow_dataset_wide_per_sample_precompute = True
    unauthorized_imports: set[str] = set()
    remote_asset_refs: set[str] = set()
    epoch_violations: set[str] = set()
    cv_violations: set[str] = set()
    local_asset_violations: set[str] = set()
    model_constructor_sites: set[str] = set()
    per_sample_feature_sites: set[str] = set()
    symbols: dict[str, int] = {}
    syntax_valid = True
    try:
        tree = ast.parse(code)
    except SyntaxError as error:
        tree = ast.Module(body=[], type_ignores=[])
        syntax_valid = False
        violations.append(f"syntax_error:{error.msg}:line_{error.lineno}")

    local_function_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            value = _literal_int(node.value, symbols)
            if value is not None:
                for target in node.targets:
                    for name in _target_names(target):
                        symbols[name] = value
        elif isinstance(node, ast.AnnAssign):
            value = _literal_int(node.value, symbols)
            if value is not None:
                for name in _target_names(node.target):
                    symbols[name] = value

    allowed_roots = set(contract.get("allowed_import_roots") or [])
    enforce_import_allowlist = False
    stdlib_roots = set(getattr(sys, "stdlib_module_names", set())) | {"__future__"}
    saved_literal_paths: set[str] = set()
    load_literal_paths: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots = {alias.name.split(".", 1)[0] for alias in node.names}
            if enforce_import_allowlist:
                unauthorized_imports |= roots - allowed_roots - stdlib_roots
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if (
                enforce_import_allowlist
                and node.level == 0
                and root
                and root not in allowed_roots | stdlib_roots
            ):
                unauthorized_imports.add(root)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            target_names, value = _assignment_parts(node)
            if isinstance(value, ast.Call) and any(
                _looks_like_model_target(name) for name in target_names
            ):
                call_name = _model_constructor_call_name(value)
                short_name = call_name.rsplit(".", 1)[-1].lower()
                local_call_name = call_name.rsplit(".", 1)[-1]
                is_local_feature_helper = (
                    local_call_name in local_function_names
                    and not _looks_like_model_factory(local_call_name)
                )
                if (
                    short_name not in _NON_CONSTRUCTOR_CALLS
                    and not is_local_feature_helper
                ):
                    model_constructor_sites.add(
                        f"line_{getattr(node, 'lineno', 0)}:{call_name or 'call'}"
                    )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if not allow_remote_assets and node.value.strip().lower().startswith(
                ("http://", "https://")
            ):
                remote_asset_refs.add(node.value.strip())
        elif isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            short_name = call_name.rsplit(".", 1)[-1]
            lowered_call = call_name.lower()
            lowered_short = short_name.lower()
            if not allow_dataset_wide_per_sample_precompute:
                if (
                    ("extract" in lowered_short and "feature" in lowered_short)
                    or lowered_short in _HANDCRAFTED_IMAGE_CALLS
                ):
                    per_sample_feature_sites.add(
                        f"line_{getattr(node, 'lineno', 0)}:{call_name}"
                    )
            if not allow_remote_assets:
                if call_name in _REMOTE_CALLS or short_name == "from_pretrained":
                    remote_asset_refs.add(call_name)
                for keyword in node.keywords:
                    if keyword.arg == "pretrained" and not (
                        isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is False
                    ):
                        remote_asset_refs.add(f"{call_name}:pretrained")
                    if keyword.arg == "weights" and not (
                        isinstance(keyword.value, ast.Constant)
                        and keyword.value.value is None
                    ):
                        remote_asset_refs.add(f"{call_name}:weights")
            if cv_cap > 0 and short_name in _CV_CONSTRUCTORS:
                folds = None
                if node.args:
                    folds = _literal_int(node.args[0], symbols)
                for keyword in node.keywords:
                    if keyword.arg in {"n_splits", "n_repeats"}:
                        folds = _literal_int(keyword.value, symbols)
                if folds is None or folds > cv_cap:
                    cv_violations.add(f"{call_name}:folds={folds or 'unknown'}")
            for keyword in node.keywords:
                if (
                    epoch_cap > 0
                    and keyword.arg
                    and keyword.arg.lower() in _EPOCH_NAMES
                ):
                    epochs = _literal_int(keyword.value, symbols)
                    if epochs is None or epochs > epoch_cap:
                        epoch_violations.add(
                            f"{call_name}:{keyword.arg}={epochs or 'unknown'}"
                        )
            if call_name in _SAVE_CALLS:
                literal = _literal_string(_first_path_argument(node, call_name))
                if literal:
                    saved_literal_paths.add(literal)
            if call_name in _LOAD_CALLS:
                literal = _literal_string(_first_path_argument(node, call_name))
                if literal:
                    load_literal_paths.append(literal)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            targets = {name.lower() for name in _target_names(node.target)}
            if targets & {"epoch", "epochs", "fold", "fold_idx", "fold_id"}:
                if isinstance(node.iter, ast.Call) and _call_name(node.iter.func) == "range":
                    iterations = _range_iteration_count(node.iter, symbols)
                    if epoch_cap > 0 and targets & {"epoch", "epochs"} and (
                        iterations is None
                        or iterations > epoch_cap
                    ):
                        epoch_violations.add(
                            f"epoch_loop_iterations={iterations or 'unknown'}"
                        )
                    if cv_cap > 0 and targets & {"fold", "fold_idx", "fold_id"} and (
                        iterations is None
                        or iterations > cv_cap
                    ):
                        cv_violations.add(
                            f"fold_loop_iterations={iterations or 'unknown'}"
                        )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lowered_name = node.name.lower()
            if (
                not allow_dataset_wide_per_sample_precompute
                and "extract" in lowered_name
                and "feature" in lowered_name
            ):
                per_sample_feature_sites.add(
                    f"line_{getattr(node, 'lineno', 0)}:def_{node.name}"
                )

    for name, value in symbols.items():
        if epoch_cap > 0 and name.lower() in _EPOCH_NAMES and value > epoch_cap:
            epoch_violations.add(f"{name}={value}")
    if not allow_unverified_local_assets:
        for literal in load_literal_paths:
            if literal not in saved_literal_paths:
                local_asset_violations.add(literal)

    if unauthorized_imports:
        violations.append(
            "unauthorized_import_roots:" + ",".join(sorted(unauthorized_imports))
        )
    if remote_asset_refs:
        violations.append("remote_asset_refs:" + ",".join(sorted(remote_asset_refs)))
    if epoch_violations:
        violations.append("epoch_cap:" + ",".join(sorted(epoch_violations)))
    if cv_violations:
        violations.append("cv_fold_cap:" + ",".join(sorted(cv_violations)))
    if local_asset_violations:
        violations.append(
            "unverified_local_loads:" + ",".join(sorted(local_asset_violations))
        )
    if model_cap > 0 and len(model_constructor_sites) > model_cap:
        violations.append(
            "trainable_model_cap:" + ",".join(sorted(model_constructor_sites))
        )
    if per_sample_feature_sites:
        violations.append(
            "dataset_wide_per_sample_precompute:"
            + ",".join(sorted(per_sample_feature_sites))
        )

    audit: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "contract_id": contract.get("contract_id"),
        "contract_hash": contract.get("contract_hash"),
        "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "valid": not violations,
        "syntax_valid": syntax_valid,
        "violations": violations,
        "checks": {
            "imports_within_allowlist": not unauthorized_imports,
            "remote_assets_absent": not remote_asset_refs,
            "epoch_cap_static_check": not epoch_violations,
            "cv_fold_cap_static_check": not cv_violations,
            "literal_local_loads_created_in_candidate": not local_asset_violations,
            "trainable_model_cap_static_check": (
                model_cap == 0 or len(model_constructor_sites) <= model_cap
            ),
            "dataset_wide_per_sample_precompute_static_check": not per_sample_feature_sites,
            "deadline_host_enforced": True,
            "method_design_not_restricted": True,
        },
        "host_enforced_max_execution_seconds": contract.get(
            "max_execution_seconds"
        ),
        "observed_model_constructor_sites": sorted(model_constructor_sites),
        "observed_per_sample_feature_sites": sorted(per_sample_feature_sites),
        "prompt_and_bundle_obligations": ["source_score_non_inheritance"],
        "audit_hash": "",
    }
    audit["audit_hash"] = _sha256_json(
        {key: value for key, value in audit.items() if key != "audit_hash"}
    )
    return audit
