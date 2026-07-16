from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from enum import Enum


class ReplayIdentity(str, Enum):
    METHOD_PRESERVED = "method_preserved"
    SUCCESSOR_METHOD = "successor_method"
    REQUIRE_HUMAN_REVIEW = "require_human_review"


@dataclass
class MethodFingerprint:
    ast_hash: str
    model_families: tuple[str, ...]
    feature_symbols: tuple[str, ...]
    training_calls: tuple[str, ...]
    inference_calls: tuple[str, ...]
    model_signatures: tuple[str, ...]
    feature_logic_hash: str

    def digest(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def fingerprint_method(code: str) -> MethodFingerprint:
    tree = ast.parse(code)
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    calls = sorted({_call_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)})
    model_terms = ("classifier", "regressor", "xgb", "lightgbm", "catboost", "bert", "resnet", "svm")
    def is_model_call(call_name: str) -> bool:
        leaf = call_name.rsplit(".", 1)[-1]
        return bool(
            any(term in call_name.lower() for term in model_terms)
            or (leaf and leaf[0].isupper() and leaf not in {"Path", "DataFrame", "Series"})
        )

    models = tuple(call for call in calls if is_model_call(call))
    model_signatures = tuple(
        sorted(
            ast.dump(node, annotate_fields=True, include_attributes=False)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and is_model_call(_call_name(node))
        )
    )
    features = tuple(
        sorted(
            {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name) and any(term in node.id.lower() for term in ("feature", "vector", "embed", "tfidf"))
            }
        )
    )
    training = tuple(call for call in calls if call.endswith((".fit", ".fit_transform", ".train")))
    inference = tuple(call for call in calls if call.endswith((".predict", ".predict_proba", ".transform")))
    feature_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            term in ast.dump(node, annotate_fields=False, include_attributes=False).lower()
            for term in ("feature", "vector", "embed", "tfidf")
        )
    ]
    feature_logic = "\n".join(
        sorted(ast.dump(node, annotate_fields=True, include_attributes=False) for node in feature_nodes)
    )
    return MethodFingerprint(
        ast_hash=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        model_families=models,
        feature_symbols=features,
        training_calls=training,
        inference_calls=inference,
        model_signatures=model_signatures,
        feature_logic_hash=hashlib.sha256(feature_logic.encode("utf-8")).hexdigest(),
    )


def certify_replay(source_code: str, replay_code: str, allowed_protocol_changes: list[str] | None = None) -> ReplayIdentity:
    try:
        source = fingerprint_method(source_code)
        replay = fingerprint_method(replay_code)
    except SyntaxError:
        return ReplayIdentity.REQUIRE_HUMAN_REVIEW
    if source.digest() == replay.digest():
        return ReplayIdentity.METHOD_PRESERVED
    if allowed_protocol_changes:
        # Protocol repair may add fold-local fit/transform/predict calls and
        # rename scope variables. Learned component constructors and their
        # hyperparameters remain frozen; the host preservation audit checks
        # protected feature/model calls in greater detail.
        structural_fields = ("model_families", "model_signatures")
    else:
        structural_fields = (
            "model_families",
            "feature_symbols",
            "training_calls",
            "inference_calls",
            "model_signatures",
        )
    if all(getattr(source, field) == getattr(replay, field) for field in structural_fields):
        return ReplayIdentity.METHOD_PRESERVED
    return ReplayIdentity.SUCCESSOR_METHOD
