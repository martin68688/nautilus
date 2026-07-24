from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import struct
import sys
import threading
import uuid
from collections.abc import Mapping, Sequence, Set
from typing import Any, Callable

from .protocol_registry import canonical_json


TRACE_MARKER = "MLEVOLVE_HOST_PROTOCOL_TRACE="
PLAN_SCHEMA = "mlevolve_host_protocol_trace_plan_v2"
TRACE_SCHEMA = "mlevolve_host_protocol_trace_v2"
OBSERVATION_SCHEMA = "mlevolve_host_protocol_observation_v2"
PROTOCOL_EVIDENCE_SCHEMA = "mlevolve_host_protocol_evidence_binding_v2"
PROTOCOL_EVIDENCE_LEVEL = (
    "host_runtime_argument_result_scope_trace_plus_deterministic_static_audit"
)
SCOPE_RECORD_SCHEMA = "mlevolve_runtime_protocol_scope_record_v1"
_VALUE_SCOPE_SCHEMA = "mlevolve_runtime_value_scope_v1"
_RESERVED_PREFIX = "__mlevolve_protocol_"
_SCOPE_HASH_CHUNK_SIZE = 1024 * 1024
_SCOPE_MAX_DEPTH = 24
_REQUIRED_KINDS = (
    "split_lineage",
    "fit_scope",
    "prediction_scope",
    "evaluator",
    "selection_freeze",
)
_SPLITTER_CONSTRUCTORS = {
    "GroupKFold",
    "GroupShuffleSplit",
    "KFold",
    "LeaveOneGroupOut",
    "ShuffleSplit",
    "StratifiedGroupKFold",
    "StratifiedKFold",
    "StratifiedShuffleSplit",
    "TimeSeriesSplit",
}
_EVALUATORS = {
    "accuracy_score",
    "average_precision_score",
    "balanced_accuracy_score",
    "cohen_kappa_score",
    "f1_score",
    "log_loss",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "mean_squared_error",
    "ndcg_score",
    "r2_score",
    "roc_auc_score",
    "root_mean_squared_error",
}
_PREDICT_CALLS = {
    "decision_function",
    "predict",
    "predict_log_proba",
    "predict_proba",
}
_FIT_CALLS = {"fit", "fit_transform", "partial_fit"}
_MODULE_PREFIXES = {
    "split": ("sklearn.model_selection",),
    "fit": (
        "catboost",
        "keras",
        "lightgbm",
        "sklearn",
        "tensorflow",
        "torch",
        "xgboost",
    ),
    "predict": (
        "catboost",
        "keras",
        "lightgbm",
        "sklearn",
        "tensorflow",
        "torch",
        "xgboost",
    ),
    "evaluator": ("scipy.stats", "sklearn.metrics", "torchmetrics"),
}


_REGISTRY_LOCK = threading.RLock()
_HOST_ATTESTATIONS: dict[str, str] = {}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _scope_digest(
    value: Any,
    *,
    _seen: set[int] | None = None,
    _depth: int = 0,
) -> tuple[str, str]:
    """Return a deterministic content digest and its coverage strength.

    Runtime values never leave the child process.  Standard containers and the
    numerical objects used by MLEvolve are hashed over their complete logical
    contents; unsupported custom objects are reduced to explicit structural
    metadata and marked ``structural`` rather than silently pretending that a
    full-content digest was obtained.
    """

    seen = _seen if _seen is not None else set()

    def digest_record(tag: str, fields: dict[str, Any]) -> str:
        return _sha256_text(
            canonical_json(
                {
                    "schema": _VALUE_SCOPE_SCHEMA,
                    "tag": tag,
                    **fields,
                }
            )
        )

    if _depth > _SCOPE_MAX_DEPTH:
        return digest_record("depth_limit", {"type": _type_ref(value)}), "structural"
    if value is None:
        return digest_record("none", {}), "content"
    if isinstance(value, bool):
        return digest_record("bool", {"value": value}), "content"
    if isinstance(value, int) and not isinstance(value, bool):
        return digest_record("int", {"value": str(value)}), "content"
    if isinstance(value, float):
        if math.isnan(value):
            encoded = "nan"
        elif math.isinf(value):
            encoded = "+inf" if value > 0 else "-inf"
        else:
            encoded = struct.pack(">d", value).hex()
        return digest_record("float64", {"bits": encoded}), "content"
    if isinstance(value, complex):
        real_hash, _ = _scope_digest(value.real, _seen=seen, _depth=_depth + 1)
        imag_hash, _ = _scope_digest(value.imag, _seen=seen, _depth=_depth + 1)
        return digest_record(
            "complex", {"real_sha256": real_hash, "imag_sha256": imag_hash}
        ), "content"
    if isinstance(value, str):
        return digest_record(
            "str",
            {
                "length": len(value),
                "utf8_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            },
        ), "content"
    if isinstance(value, (bytes, bytearray, memoryview)):
        data = memoryview(value).cast("B")
        content = hashlib.sha256()
        for offset in range(0, len(data), _SCOPE_HASH_CHUNK_SIZE):
            content.update(data[offset : offset + _SCOPE_HASH_CHUNK_SIZE])
        return digest_record(
            "bytes",
            {"length": len(data), "content_sha256": content.hexdigest()},
        ), "content"

    numpy = sys.modules.get("numpy")
    if numpy is not None:
        generic_type = getattr(numpy, "generic", ())
        if generic_type and isinstance(value, generic_type):
            scalar_hash, strength = _scope_digest(
                value.item(), _seen=seen, _depth=_depth + 1
            )
            return digest_record(
                "numpy_scalar",
                {"dtype": str(value.dtype), "value_sha256": scalar_hash},
            ), strength
        ndarray_type = getattr(numpy, "ndarray", ())
        if ndarray_type and isinstance(value, ndarray_type):
            return _numpy_scope_digest(value, seen=seen, depth=_depth)

    pandas = sys.modules.get("pandas")
    if pandas is not None:
        dataframe_type = getattr(pandas, "DataFrame", ())
        series_type = getattr(pandas, "Series", ())
        index_type = getattr(pandas, "Index", ())
        if dataframe_type and isinstance(value, dataframe_type):
            return _pandas_scope_digest(value, "dataframe", seen=seen, depth=_depth)
        if series_type and isinstance(value, series_type):
            return _pandas_scope_digest(value, "series", seen=seen, depth=_depth)
        if index_type and isinstance(value, index_type):
            return _pandas_scope_digest(value, "index", seen=seen, depth=_depth)

    scipy_sparse = sys.modules.get("scipy.sparse")
    if scipy_sparse is not None:
        is_sparse = getattr(scipy_sparse, "issparse", None)
        if callable(is_sparse) and is_sparse(value):
            return _sparse_scope_digest(value, seen=seen, depth=_depth)

    torch = sys.modules.get("torch")
    tensor_type = getattr(torch, "Tensor", ()) if torch is not None else ()
    if tensor_type and isinstance(value, tensor_type):
        return _torch_scope_digest(value, seen=seen, depth=_depth)

    if isinstance(value, Mapping):
        object_id = id(value)
        if object_id in seen:
            return digest_record("cycle", {"type": _type_ref(value)}), "structural"
        seen.add(object_id)
        try:
            entries: list[tuple[str, str, str]] = []
            strengths: list[str] = []
            for key, item in value.items():
                key_hash, key_strength = _scope_digest(
                    key, _seen=seen, _depth=_depth + 1
                )
                item_hash, item_strength = _scope_digest(
                    item, _seen=seen, _depth=_depth + 1
                )
                entries.append((key_hash, item_hash, _type_ref(key)))
                strengths.extend((key_strength, item_strength))
            entries.sort()
            return digest_record(
                "mapping",
                {
                    "type": _type_ref(value),
                    "length": len(entries),
                    "entries": entries,
                },
            ), _combined_scope_strength(strengths)
        finally:
            seen.discard(object_id)

    if isinstance(value, Set) and not isinstance(value, (str, bytes, bytearray)):
        object_id = id(value)
        if object_id in seen:
            return digest_record("cycle", {"type": _type_ref(value)}), "structural"
        seen.add(object_id)
        try:
            values: list[str] = []
            strengths: list[str] = []
            for item in value:
                item_hash, item_strength = _scope_digest(
                    item, _seen=seen, _depth=_depth + 1
                )
                values.append(item_hash)
                strengths.append(item_strength)
            return digest_record(
                "set",
                {
                    "type": _type_ref(value),
                    "length": len(values),
                    "items": sorted(values),
                },
            ), _combined_scope_strength(strengths)
        finally:
            seen.discard(object_id)

    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        object_id = id(value)
        if object_id in seen:
            return digest_record("cycle", {"type": _type_ref(value)}), "structural"
        seen.add(object_id)
        try:
            values: list[str] = []
            strengths: list[str] = []
            for item in value:
                item_hash, item_strength = _scope_digest(
                    item, _seen=seen, _depth=_depth + 1
                )
                values.append(item_hash)
                strengths.append(item_strength)
            return digest_record(
                "sequence",
                {
                    "type": _type_ref(value),
                    "length": len(values),
                    "items": values,
                },
            ), _combined_scope_strength(strengths)
        finally:
            seen.discard(object_id)

    structural_fields: dict[str, Any] = {"type": _type_ref(value)}
    for attribute in ("shape", "dtype", "ndim"):
        try:
            candidate = getattr(value, attribute)
        except Exception:
            continue
        if isinstance(candidate, (str, int, float, bool, tuple, list)):
            structural_fields[attribute] = str(candidate)
    return digest_record("unsupported_object", structural_fields), "structural"


def _type_ref(value: Any) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _combined_scope_strength(strengths: list[str]) -> str:
    if not strengths or all(strength == "content" for strength in strengths):
        return "content"
    return "structural"


def _numpy_scope_digest(
    array: Any,
    *,
    seen: set[int],
    depth: int,
) -> tuple[str, str]:
    object_id = id(array)
    header = {
        "schema": _VALUE_SCOPE_SCHEMA,
        "tag": "numpy_array",
        "dtype": str(array.dtype),
        "shape": [int(item) for item in array.shape],
    }
    if object_id in seen:
        return _sha256_text(canonical_json({**header, "cycle": True})), "structural"
    seen.add(object_id)
    try:
        if bool(getattr(array.dtype, "hasobject", False)):
            item_hashes: list[str] = []
            strengths: list[str] = []
            for item in array.reshape(-1):
                item_hash, strength = _scope_digest(
                    item.item() if hasattr(item, "item") else item,
                    _seen=seen,
                    _depth=depth + 1,
                )
                item_hashes.append(item_hash)
                strengths.append(strength)
            return _sha256_text(
                canonical_json({**header, "item_hashes": item_hashes})
            ), _combined_scope_strength(strengths)
        contiguous = array if bool(array.flags.c_contiguous) else array.copy(order="C")
        data = memoryview(contiguous).cast("B")
        content = hashlib.sha256()
        for offset in range(0, len(data), _SCOPE_HASH_CHUNK_SIZE):
            content.update(data[offset : offset + _SCOPE_HASH_CHUNK_SIZE])
        return _sha256_text(
            canonical_json(
                {
                    **header,
                    "byte_length": len(data),
                    "content_sha256": content.hexdigest(),
                }
            )
        ), "content"
    except Exception as error:
        return _sha256_text(
            canonical_json(
                {
                    **header,
                    "fallback": "structural",
                    "error_type": type(error).__name__,
                }
            )
        ), "structural"
    finally:
        seen.discard(object_id)


def _pandas_scope_digest(
    value: Any,
    kind: str,
    *,
    seen: set[int],
    depth: int,
) -> tuple[str, str]:
    object_id = id(value)
    if object_id in seen:
        return _sha256_text(
            canonical_json(
                {"schema": _VALUE_SCOPE_SCHEMA, "tag": f"pandas_{kind}", "cycle": True}
            )
        ), "structural"
    seen.add(object_id)
    try:
        components: dict[str, str] = {}
        strengths: list[str] = []
        if kind == "dataframe":
            for name, item in (
                ("index", value.index),
                ("columns", value.columns),
                ("values", value.to_numpy(copy=False)),
            ):
                item_hash, strength = _scope_digest(
                    item, _seen=seen, _depth=depth + 1
                )
                components[name] = item_hash
                strengths.append(strength)
        elif kind == "series":
            for name, item in (
                ("index", value.index),
                ("name", value.name),
                ("values", value.to_numpy(copy=False)),
            ):
                item_hash, strength = _scope_digest(
                    item, _seen=seen, _depth=depth + 1
                )
                components[name] = item_hash
                strengths.append(strength)
        else:
            item_hash, strength = _scope_digest(
                value.to_numpy(copy=False), _seen=seen, _depth=depth + 1
            )
            components["values"] = item_hash
            strengths.append(strength)
            components["name"] = _scope_digest(
                value.name, _seen=seen, _depth=depth + 1
            )[0]
        return _sha256_text(
            canonical_json(
                {
                    "schema": _VALUE_SCOPE_SCHEMA,
                    "tag": f"pandas_{kind}",
                    "type": _type_ref(value),
                    "shape": [int(item) for item in getattr(value, "shape", ())],
                    "components": components,
                }
            )
        ), _combined_scope_strength(strengths)
    except Exception as error:
        return _sha256_text(
            canonical_json(
                {
                    "schema": _VALUE_SCOPE_SCHEMA,
                    "tag": f"pandas_{kind}",
                    "type": _type_ref(value),
                    "fallback": "structural",
                    "error_type": type(error).__name__,
                }
            )
        ), "structural"
    finally:
        seen.discard(object_id)


def _sparse_scope_digest(
    value: Any,
    *,
    seen: set[int],
    depth: int,
) -> tuple[str, str]:
    try:
        matrix = value.tocsr(copy=False)
        component_hashes: dict[str, str] = {}
        strengths: list[str] = []
        for name in ("data", "indices", "indptr"):
            item_hash, strength = _scope_digest(
                getattr(matrix, name), _seen=seen, _depth=depth + 1
            )
            component_hashes[name] = item_hash
            strengths.append(strength)
        return _sha256_text(
            canonical_json(
                {
                    "schema": _VALUE_SCOPE_SCHEMA,
                    "tag": "scipy_sparse_csr",
                    "type": _type_ref(value),
                    "shape": [int(item) for item in matrix.shape],
                    "components": component_hashes,
                }
            )
        ), _combined_scope_strength(strengths)
    except Exception as error:
        return _sha256_text(
            canonical_json(
                {
                    "schema": _VALUE_SCOPE_SCHEMA,
                    "tag": "scipy_sparse",
                    "type": _type_ref(value),
                    "fallback": "structural",
                    "error_type": type(error).__name__,
                }
            )
        ), "structural"


def _torch_scope_digest(
    value: Any,
    *,
    seen: set[int],
    depth: int,
) -> tuple[str, str]:
    try:
        detached = value.detach().cpu().contiguous()
        try:
            array = detached.numpy()
        except Exception:
            torch = sys.modules.get("torch")
            array = detached.view(getattr(torch, "uint8")).numpy()
        content_hash, strength = _scope_digest(
            array, _seen=seen, _depth=depth + 1
        )
        return _sha256_text(
            canonical_json(
                {
                    "schema": _VALUE_SCOPE_SCHEMA,
                    "tag": "torch_tensor",
                    "dtype": str(value.dtype),
                    "shape": [int(item) for item in value.shape],
                    "requires_grad": bool(value.requires_grad),
                    "content_sha256": content_hash,
                }
            )
        ), strength
    except Exception as error:
        return _sha256_text(
            canonical_json(
                {
                    "schema": _VALUE_SCOPE_SCHEMA,
                    "tag": "torch_tensor",
                    "type": _type_ref(value),
                    "shape": [int(item) for item in getattr(value, "shape", ())],
                    "fallback": "structural",
                    "error_type": type(error).__name__,
                }
            )
        ), "structural"


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def _receiver_name(node: ast.Call) -> str:
    if not isinstance(node.func, ast.Attribute):
        return ""
    value = node.func.value
    while isinstance(value, (ast.Attribute, ast.Subscript)):
        value = value.value
    return value.id if isinstance(value, ast.Name) else ""


def _splitter_receivers(tree: ast.AST) -> set[str]:
    receivers: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call) or _call_name(value) not in _SPLITTER_CONSTRUCTORS:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        receivers.update(target.id for target in targets if isinstance(target, ast.Name))
    return receivers


def _reserved_namespace_used(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id.startswith(_RESERVED_PREFIX):
            return True
        if isinstance(node, ast.Attribute) and node.attr.startswith(_RESERVED_PREFIX):
            return True
    return False


def _observer_introspection_used(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in {
            "globals",
            "locals",
            "setprofile",
            "settrace",
            "vars",
        }:
            return True
        if isinstance(node, ast.Attribute) and node.attr in {
            "__dict__",
            "_HOST_ATTESTATIONS",
            "setprofile",
            "settrace",
        }:
            return True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [str(node.module or "")]
            )
            if any(module.split(".", 1)[0] in {"dis", "inspect"} for module in modules):
                return True
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _RESERVED_PREFIX in node.value or TRACE_MARKER in node.value:
                return True
    return False


def _classify_call(node: ast.Call, splitter_receivers: set[str]) -> tuple[list[str], str, tuple[str, ...]]:
    name = _call_name(node)
    receiver = _receiver_name(node)
    lower_receiver = receiver.lower()
    if name == "train_test_split" or (
        name == "split"
        and (
            receiver in splitter_receivers
            or any(token in lower_receiver for token in ("kfold", "splitter"))
            or lower_receiver in {"cv", "kf", "skf", "sss"}
        )
    ):
        return ["split_lineage"], "split", _MODULE_PREFIXES["split"]
    if name in _FIT_CALLS or (
        name == "step"
        and any(
            token in lower_receiver for token in ("optim", "optimizer", "scaler")
        )
    ):
        return ["fit_scope"], "fit", _MODULE_PREFIXES["fit"]
    if name in _EVALUATORS:
        return (
            ["prediction_scope", "evaluator", "selection_freeze"],
            "evaluator",
            _MODULE_PREFIXES["evaluator"],
        )
    if name in _PREDICT_CALLS:
        return ["prediction_scope"], "predict", _MODULE_PREFIXES["predict"]
    return [], "", ()


def _event_for_call(
    node: ast.Call,
    splitter_receivers: set[str],
) -> dict[str, Any] | None:
    kinds, call_class, allowed_modules = _classify_call(node, splitter_receivers)
    if not kinds:
        return None
    material = {
        "line": int(getattr(node, "lineno", 0) or 0),
        "column": int(getattr(node, "col_offset", 0) or 0),
        "end_line": int(getattr(node, "end_lineno", 0) or 0),
        "end_column": int(getattr(node, "end_col_offset", 0) or 0),
        "call_name": _call_name(node),
        "call_class": call_class,
        "kinds": sorted(kinds),
        "allowed_modules": list(allowed_modules),
        "ast_sha256": _sha256_text(ast.dump(node, include_attributes=False)),
    }
    material["event_id"] = _sha256_text(canonical_json(material))
    return material


def build_runtime_protocol_plan(
    executed_source: str,
    *,
    source_code_sha256: str | None = None,
) -> dict[str, Any]:
    executed_source = str(executed_source or "")
    executed_sha256 = _sha256_text(executed_source)
    source_sha256 = str(source_code_sha256 or executed_sha256)
    try:
        tree = ast.parse(executed_source)
    except SyntaxError as error:
        unsigned = {
            "schema": PLAN_SCHEMA,
            "status": "blocked",
            "reason": f"syntax_error:{type(error).__name__}",
            "source_code_sha256": source_sha256,
            "executed_source_sha256": executed_sha256,
            "required_kinds": list(_REQUIRED_KINDS),
            "missing_plan_kinds": list(_REQUIRED_KINDS),
            "events": [],
        }
        return {**unsigned, "plan_sha256": _sha256_text(canonical_json(unsigned))}
    events = [
        event
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for event in [_event_for_call(node, _splitter_receivers(tree))]
        if event is not None
    ]
    events.sort(key=lambda item: (item["line"], item["column"], item["event_id"]))
    present = {kind for event in events for kind in event["kinds"]}
    missing = sorted(set(_REQUIRED_KINDS) - present)
    reason = ""
    if _reserved_namespace_used(tree):
        reason = "reserved_runtime_observer_namespace_used"
    elif _observer_introspection_used(tree):
        reason = "runtime_observer_introspection_detected"
    elif missing:
        reason = "missing_protocol_event_plan:" + ",".join(missing)
    unsigned = {
        "schema": PLAN_SCHEMA,
        "status": "ready" if not reason else "blocked",
        "reason": reason,
        "source_code_sha256": source_sha256,
        "executed_source_sha256": executed_sha256,
        "required_kinds": list(_REQUIRED_KINDS),
        "missing_plan_kinds": missing,
        "events": events,
    }
    return {**unsigned, "plan_sha256": _sha256_text(canonical_json(unsigned))}


class _InstrumentCalls(ast.NodeTransformer):
    def __init__(self, events: dict[tuple[int, int, int, int], dict[str, Any]]):
        self.events = events

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        key = (
            int(getattr(node, "lineno", 0) or 0),
            int(getattr(node, "col_offset", 0) or 0),
            int(getattr(node, "end_lineno", 0) or 0),
            int(getattr(node, "end_col_offset", 0) or 0),
        )
        event = self.events.get(key)
        if event is None:
            return node
        wrapped = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=f"{_RESERVED_PREFIX}observer", ctx=ast.Load()),
                attr="call",
                ctx=ast.Load(),
            ),
            args=[
                ast.List(
                    elts=[ast.Constant(kind) for kind in event["kinds"]],
                    ctx=ast.Load(),
                ),
                ast.Constant(event["event_id"]),
                node.func,
                *node.args,
            ],
            keywords=node.keywords,
        )
        return ast.copy_location(wrapped, node)


def instrument_runtime_protocol_source(
    executed_source: str,
    plan: dict[str, Any],
    nonce: str,
    *,
    filename: str,
) -> str:
    tree = ast.parse(executed_source)
    event_map = {
        (
            int(event["line"]),
            int(event["column"]),
            int(event["end_line"]),
            int(event["end_column"]),
        ): event
        for event in plan.get("events") or []
    }
    transformed = _InstrumentCalls(event_map).visit(tree)
    ast.fix_missing_locations(transformed)
    transformed_source = ast.unparse(transformed) + "\n"
    plan_json = canonical_json(plan)
    return (
        "from authority.runtime_protocol import RuntimeProtocolObserver as "
        f"{_RESERVED_PREFIX}Observer\n"
        f"{_RESERVED_PREFIX}observer = {_RESERVED_PREFIX}Observer("
        f"json.loads({plan_json!r}), {str(nonce)!r})\n"
        "try:\n"
        f"    exec(compile({transformed_source!r}, {str(filename)!r}, 'exec'), "
        "globals(), globals())\n"
        "except BaseException:\n"
        "    raise\n"
        "else:\n"
        f"    {_RESERVED_PREFIX}observer.mark_completed()\n"
        "finally:\n"
        f"    {_RESERVED_PREFIX}observer.emit()\n"
    ).replace(
        f"from authority.runtime_protocol import RuntimeProtocolObserver as {_RESERVED_PREFIX}Observer\n",
        "import json\nfrom authority.runtime_protocol import RuntimeProtocolObserver as "
        f"{_RESERVED_PREFIX}Observer\n",
        1,
    )


class RuntimeProtocolObserver:
    """Child-process observer injected by the host executor.

    It can describe executed calls, but it cannot mint trusted Receipts.  Only
    the parent process can validate the unpredictable nonce and register a
    reusable host attestation.
    """

    def __init__(self, plan: dict[str, Any], nonce: str):
        self.plan = copy.deepcopy(plan)
        self.nonce = str(nonce)
        self.events = {
            str(event["event_id"]): event for event in self.plan.get("events") or []
        }
        self.completed_event_ids: set[str] = set()
        self.failed_event_ids: set[str] = set()
        self.event_calls: dict[str, set[tuple[str, str, str, str, str]]] = {}
        self.event_scopes: dict[str, list[dict[str, Any]]] = {}
        self.violations: list[str] = []
        self.completed = False
        self.emitted = False

    @staticmethod
    def _callable_ref(
        function: Callable[..., Any],
    ) -> tuple[str, str, str, str, str]:
        module = str(
            getattr(function, "__module__", "")
            or getattr(type(function), "__module__", "")
        )
        name = str(
            getattr(function, "__qualname__", "")
            or getattr(function, "__name__", "")
            or getattr(type(function), "__qualname__", type(function).__name__)
        )
        implementation = getattr(function, "__func__", function)
        code = getattr(implementation, "__code__", None)
        code_filename = str(getattr(code, "co_filename", "") or "")
        code_sha256 = (
            hashlib.sha256(code.co_code).hexdigest() if code is not None else ""
        )
        module_file = str(
            getattr(sys.modules.get(module), "__file__", "") or ""
        )
        return module, name, code_filename, module_file, code_sha256

    def call(
        self,
        kinds: list[str],
        event_id: str,
        function: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        event_id = str(event_id)
        expected = self.events.get(event_id)
        if expected is None or sorted(kinds) != sorted(expected.get("kinds") or []):
            self.violations.append(f"unplanned_protocol_event:{event_id}")
        callable_ref = self._callable_ref(function)
        input_sha256, input_strength = _scope_digest(
            {"args": args, "kwargs": kwargs}
        )
        try:
            result = function(*args, **kwargs)
        except BaseException:
            self.failed_event_ids.add(event_id)
            raise
        if expected and expected.get("call_class") == "split" and hasattr(
            result, "__next__"
        ):
            return self._observe_split_iterator(
                result,
                event_id,
                callable_ref,
                input_sha256,
                input_strength,
            )
        call_class = str((expected or {}).get("call_class") or "")
        call_name = str((expected or {}).get("call_name") or "")
        if call_class == "fit":
            output_sha256 = ""
            output_strength = "not_recorded"
            scope_basis = (
                "runtime_event_only"
                if call_name == "step"
                else "actual_arguments"
            )
        else:
            output_sha256, output_strength = _scope_digest(result)
            scope_basis = "actual_arguments_and_result"
        self._record_completed(
            event_id,
            callable_ref,
            input_sha256=input_sha256,
            input_strength=input_strength,
            output_sha256=output_sha256,
            output_strength=output_strength,
            scope_basis=scope_basis,
        )
        return result

    def _record_completed(
        self,
        event_id: str,
        callable_ref: tuple[str, str, str, str, str],
        *,
        input_sha256: str,
        input_strength: str,
        output_sha256: str,
        output_strength: str,
        scope_basis: str,
    ) -> None:
        self.completed_event_ids.add(event_id)
        self.event_calls.setdefault(event_id, set()).add(callable_ref)
        expected = self.events.get(event_id) or {}
        scopes = self.event_scopes.setdefault(event_id, [])
        scopes.append(
            {
                "schema": SCOPE_RECORD_SCHEMA,
                "event_id": event_id,
                "call_index": len(scopes) + 1,
                "call_name": str(expected.get("call_name") or ""),
                "call_class": str(expected.get("call_class") or ""),
                "scope_basis": scope_basis,
                "input_sha256": input_sha256,
                "input_strength": input_strength,
                "output_sha256": output_sha256,
                "output_strength": output_strength,
            }
        )

    def _observe_split_iterator(
        self,
        iterator: Any,
        event_id: str,
        callable_ref: tuple[str, str, str, str, str],
        input_sha256: str,
        input_strength: str,
    ):
        observed = False
        try:
            for value in iterator:
                output_sha256, output_strength = _scope_digest(value)
                self._record_completed(
                    event_id,
                    callable_ref,
                    input_sha256=input_sha256,
                    input_strength=input_strength,
                    output_sha256=output_sha256,
                    output_strength=output_strength,
                    scope_basis="actual_arguments_and_yielded_partition",
                )
                observed = True
                yield value
        except GeneratorExit:
            return
        except BaseException:
            self.failed_event_ids.add(event_id)
            raise
        if not observed:
            self.failed_event_ids.add(event_id)

    def mark_completed(self) -> None:
        self.completed = True

    def emit(self) -> None:
        if self.emitted:
            return
        self.emitted = True
        payload = {
            "schema": TRACE_SCHEMA,
            "nonce": self.nonce,
            "plan_sha256": self.plan.get("plan_sha256"),
            "source_code_sha256": self.plan.get("source_code_sha256"),
            "executed_source_sha256": self.plan.get("executed_source_sha256"),
            "completed": self.completed,
            "completed_event_ids": sorted(self.completed_event_ids),
            "failed_event_ids": sorted(self.failed_event_ids),
            "event_calls": {
                event_id: [
                    {
                        "module": module,
                        "qualname": name,
                        "code_filename": code_filename,
                        "module_file": module_file,
                        "code_sha256": code_sha256,
                    }
                    for (
                        module,
                        name,
                        code_filename,
                        module_file,
                        code_sha256,
                    ) in sorted(values)
                ]
                for event_id, values in sorted(self.event_calls.items())
            },
            "event_scopes": {
                event_id: copy.deepcopy(records)
                for event_id, records in sorted(self.event_scopes.items())
            },
            "violations": list(dict.fromkeys(self.violations)),
        }
        stream = getattr(sys, "__stdout__", None) or sys.stdout
        stream.write(TRACE_MARKER + canonical_json(payload) + "\n")
        stream.flush()


def _blocked_observation(
    plan: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "schema": OBSERVATION_SCHEMA,
        "status": "blocked",
        "reason": str(reason),
        "source_code_sha256": plan.get("source_code_sha256", ""),
        "executed_source_sha256": plan.get("executed_source_sha256", ""),
        "plan_sha256": plan.get("plan_sha256", ""),
        "event_hashes": {},
        "scope_hashes": {},
        "scope_input_hashes": {},
        "scope_output_hashes": {},
        "callable_refs": {},
    }


def _module_allowed(module: str, prefixes: list[str]) -> bool:
    return bool(module) and any(
        module == prefix or module.startswith(prefix + ".") for prefix in prefixes
    )


def _callable_origin_allowed(
    call: dict[str, Any],
    prefixes: list[str],
) -> bool:
    module = str(call.get("module") or "")
    if not _module_allowed(module, prefixes):
        return False
    matching_roots = {
        prefix.split(".", 1)[0]
        for prefix in prefixes
        if module == prefix or module.startswith(prefix + ".")
    }
    code_filename = str(call.get("code_filename") or "")
    origins = [
        code_filename
        or str(call.get("module_file") or "")
    ]
    for origin in origins:
        normalized = origin.replace("\\", "/").lower()
        if any(f"/{root.lower()}/" in normalized for root in matching_roots):
            return True
    return False


def _validated_scope_records(
    event: dict[str, Any],
    records: Any,
) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(records, list) or not records:
        return [], "missing_runtime_scope_record"
    call_class = str(event.get("call_class") or "")
    call_name = str(event.get("call_name") or "")
    validated: list[dict[str, Any]] = []
    for expected_index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            return [], "runtime_scope_record_not_mapping"
        if record.get("schema") != SCOPE_RECORD_SCHEMA:
            return [], "runtime_scope_record_schema_mismatch"
        if str(record.get("event_id") or "") != str(event.get("event_id") or ""):
            return [], "runtime_scope_record_event_mismatch"
        if int(record.get("call_index", 0) or 0) != expected_index:
            return [], "runtime_scope_record_sequence_mismatch"
        if str(record.get("call_class") or "") != call_class:
            return [], "runtime_scope_record_class_mismatch"
        if str(record.get("call_name") or "") != call_name:
            return [], "runtime_scope_record_name_mismatch"
        input_sha256 = str(record.get("input_sha256") or "")
        output_sha256 = str(record.get("output_sha256") or "")
        input_strength = str(record.get("input_strength") or "")
        output_strength = str(record.get("output_strength") or "")
        scope_basis = str(record.get("scope_basis") or "")
        if not _is_sha256(input_sha256):
            return [], "runtime_scope_input_hash_invalid"
        if input_strength not in {"content", "structural"}:
            return [], "runtime_scope_input_strength_invalid"

        if call_class == "fit" and call_name == "step" and scope_basis == (
            "runtime_event_only"
        ):
            if output_sha256 or output_strength != "not_recorded":
                return [], "runtime_event_only_scope_has_output"
        elif call_class == "fit":
            if scope_basis != "actual_arguments":
                return [], "runtime_fit_scope_basis_invalid"
            if input_strength != "content":
                return [], "runtime_fit_scope_not_content_bound"
            if output_sha256 or output_strength != "not_recorded":
                return [], "runtime_fit_scope_has_untrusted_output"
        elif call_class == "split":
            if scope_basis not in {
                "actual_arguments_and_result",
                "actual_arguments_and_yielded_partition",
            }:
                return [], "runtime_split_scope_basis_invalid"
            if input_strength != "content" or output_strength != "content":
                return [], "runtime_split_scope_not_content_bound"
            if not _is_sha256(output_sha256):
                return [], "runtime_split_output_hash_invalid"
        elif call_class in {"predict", "evaluator"}:
            if scope_basis != "actual_arguments_and_result":
                return [], "runtime_result_scope_basis_invalid"
            if input_strength != "content" or output_strength != "content":
                return [], "runtime_result_scope_not_content_bound"
            if not _is_sha256(output_sha256):
                return [], "runtime_result_output_hash_invalid"
        else:
            return [], "runtime_scope_unknown_call_class"
        validated.append(copy.deepcopy(record))
    return validated, ""


def _register_host_attestation(observation: dict[str, Any]) -> dict[str, Any]:
    attested = copy.deepcopy(observation)
    attested["attestation_id"] = uuid.uuid4().hex
    unsigned = dict(attested)
    fingerprint = _sha256_text(canonical_json(unsigned))
    attested["attestation_sha256"] = fingerprint
    with _REGISTRY_LOCK:
        _HOST_ATTESTATIONS[attested["attestation_id"]] = fingerprint
    return attested


def verify_persisted_runtime_protocol_observation(
    observation: dict[str, Any] | None,
) -> bool:
    """Verify serialized observation integrity without claiming nonce ownership.

    The active verifier below additionally requires the process-local host
    attestation registry.  That registry intentionally cannot be serialized.
    This stateless verifier is therefore suitable for later publication and
    audit processes, provided they separately trust the hash-bound execution
    record that states active verification succeeded before Receipts were
    minted.
    """

    if (
        not isinstance(observation, dict)
        or observation.get("schema") != OBSERVATION_SCHEMA
        or observation.get("status") != "clean"
        or observation.get("evidence_level") != PROTOCOL_EVIDENCE_LEVEL
        or observation.get("code_snapshot_frozen_before_execution") is not True
    ):
        return False
    attestation_id = str(observation.get("attestation_id") or "")
    claimed = str(observation.get("attestation_sha256") or "")
    if not (
        len(attestation_id) == 32
        and all(character in "0123456789abcdef" for character in attestation_id)
        and _is_sha256(claimed)
        and all(
            _is_sha256(observation.get(field))
            for field in (
                "source_code_sha256",
                "executed_source_sha256",
                "plan_sha256",
                "trace_sha256",
            )
        )
    ):
        return False
    event_hashes = observation.get("event_hashes")
    scope_hashes = observation.get("scope_hashes")
    if not isinstance(event_hashes, dict) or not isinstance(scope_hashes, dict):
        return False
    for kind in _REQUIRED_KINDS:
        for mapping in (event_hashes, scope_hashes):
            values = mapping.get(kind)
            if not isinstance(values, list) or not values or not all(
                _is_sha256(value) for value in values
            ):
                return False
    unsigned = dict(observation)
    unsigned.pop("attestation_sha256", None)
    actual = _sha256_text(canonical_json(unsigned))
    return claimed == actual


def verify_runtime_protocol_observation(observation: dict[str, Any] | None) -> bool:
    if not verify_persisted_runtime_protocol_observation(observation):
        return False
    assert isinstance(observation, dict)
    attestation_id = str(observation["attestation_id"])
    claimed = str(observation["attestation_sha256"])
    with _REGISTRY_LOCK:
        expected = _HOST_ATTESTATIONS.get(attestation_id)
    return bool(expected and claimed == expected)


def parse_runtime_protocol_trace(
    stdout: str,
    plan: dict[str, Any],
    nonce: str,
    *,
    execution_succeeded: bool,
) -> dict[str, Any]:
    if plan.get("status") != "ready":
        return _blocked_observation(plan, str(plan.get("reason") or "protocol_plan_blocked"))
    candidates: list[dict[str, Any]] = []
    for line in str(stdout or "").splitlines():
        if not line.startswith(TRACE_MARKER):
            continue
        try:
            payload = json.loads(line[len(TRACE_MARKER):])
        except Exception:
            continue
        if payload.get("nonce") == nonce:
            candidates.append(payload)
    if len(candidates) != 1:
        return _blocked_observation(plan, "missing_or_ambiguous_host_protocol_trace")
    trace = candidates[0]
    if trace.get("schema") != TRACE_SCHEMA:
        return _blocked_observation(plan, "host_protocol_trace_schema_mismatch")
    for key in (
        "plan_sha256",
        "source_code_sha256",
        "executed_source_sha256",
    ):
        if trace.get(key) != plan.get(key):
            return _blocked_observation(plan, f"host_protocol_trace_{key}_mismatch")
    if not execution_succeeded or trace.get("completed") is not True:
        return _blocked_observation(plan, "host_protocol_execution_incomplete")
    if trace.get("violations"):
        return _blocked_observation(plan, "host_protocol_trace_reported_violation")
    completed = set(trace.get("completed_event_ids") or [])
    failed = set(trace.get("failed_event_ids") or [])
    event_calls = trace.get("event_calls") or {}
    event_scopes = trace.get("event_scopes") or {}
    known_event_ids = {
        str(event.get("event_id") or "") for event in plan.get("events") or []
    }
    if (
        not isinstance(event_calls, dict)
        or not isinstance(event_scopes, dict)
        or completed - known_event_ids
        or failed - known_event_ids
        or set(event_calls) - known_event_ids
        or set(event_scopes) - known_event_ids
        or completed & failed
    ):
        return _blocked_observation(plan, "host_protocol_trace_event_set_mismatch")
    event_hashes: dict[str, list[str]] = {kind: [] for kind in _REQUIRED_KINDS}
    scope_hashes: dict[str, list[str]] = {kind: [] for kind in _REQUIRED_KINDS}
    scope_input_hashes: dict[str, list[str]] = {
        kind: [] for kind in _REQUIRED_KINDS
    }
    scope_output_hashes: dict[str, list[str]] = {
        kind: [] for kind in _REQUIRED_KINDS
    }
    callable_refs: dict[str, list[dict[str, str]]] = {kind: [] for kind in _REQUIRED_KINDS}
    for event in plan.get("events") or []:
        event_id = str(event["event_id"])
        if event_id not in completed:
            continue
        calls = event_calls.get(event_id) or []
        if not calls or any(
            not _callable_origin_allowed(call, event["allowed_modules"])
            for call in calls
        ):
            return _blocked_observation(plan, f"untrusted_runtime_callable:{event_id}")
        scopes, scope_error = _validated_scope_records(
            event, event_scopes.get(event_id)
        )
        if scope_error:
            return _blocked_observation(
                plan, f"{scope_error}:{event_id}"
            )
        event_scope_hashes = [
            _sha256_text(canonical_json(record)) for record in scopes
        ]
        event_hash = _sha256_text(
            canonical_json(
                {
                    "event": event,
                    "calls": calls,
                    "scope_hashes": event_scope_hashes,
                    "source_code_sha256": plan["source_code_sha256"],
                    "executed_source_sha256": plan["executed_source_sha256"],
                }
            )
        )
        for kind in event["kinds"]:
            event_hashes[kind].append(event_hash)
            scope_hashes[kind].extend(event_scope_hashes)
            scope_input_hashes[kind].extend(
                str(record["input_sha256"]) for record in scopes
            )
            scope_output_hashes[kind].extend(
                str(record["output_sha256"])
                for record in scopes
                if record.get("output_sha256")
            )
            callable_refs[kind].extend(copy.deepcopy(calls))
    missing = [kind for kind, hashes in event_hashes.items() if not hashes]
    if missing:
        return _blocked_observation(
            plan, "unexecuted_protocol_event_kinds:" + ",".join(sorted(missing))
        )
    event_hashes = {
        kind: sorted(set(hashes)) for kind, hashes in sorted(event_hashes.items())
    }
    scope_hashes = {
        kind: sorted(set(hashes)) for kind, hashes in sorted(scope_hashes.items())
    }
    scope_input_hashes = {
        kind: sorted(set(hashes))
        for kind, hashes in sorted(scope_input_hashes.items())
    }
    scope_output_hashes = {
        kind: sorted(set(hashes))
        for kind, hashes in sorted(scope_output_hashes.items())
    }
    if any(not hashes for hashes in scope_hashes.values()):
        return _blocked_observation(plan, "missing_runtime_scope_hashes")
    callable_refs = {
        kind: sorted(
            {canonical_json(item): item for item in values}.values(),
            key=canonical_json,
        )
        for kind, values in sorted(callable_refs.items())
    }
    trace_unsigned = dict(trace)
    trace_unsigned.pop("nonce", None)
    observation = {
        "schema": OBSERVATION_SCHEMA,
        "status": "clean",
        "reason": "",
        "evidence_level": PROTOCOL_EVIDENCE_LEVEL,
        "source_code_sha256": plan["source_code_sha256"],
        "executed_source_sha256": plan["executed_source_sha256"],
        "plan_sha256": plan["plan_sha256"],
        "trace_sha256": _sha256_text(canonical_json(trace_unsigned)),
        "event_hashes": event_hashes,
        "scope_hashes": scope_hashes,
        "scope_input_hashes": scope_input_hashes,
        "scope_output_hashes": scope_output_hashes,
        "callable_refs": callable_refs,
        "code_snapshot_frozen_before_execution": True,
    }
    return _register_host_attestation(observation)


def strip_runtime_protocol_markers(output: str) -> str:
    return "".join(
        line
        for line in str(output or "").splitlines(keepends=True)
        if not line.startswith(TRACE_MARKER)
    )
