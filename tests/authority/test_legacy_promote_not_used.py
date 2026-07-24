from __future__ import annotations

import ast
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
PRODUCTION_ROOT = REPO / "mlevolve"
PRODUCTION_SOURCE_DIRS = (
    "agents",
    "authority",
    "engine",
    "fixed_holdout",
)


def _is_legacy_promote(node: ast.AST) -> bool:
    return bool(
        isinstance(node, ast.Attribute)
        and node.attr == "PROMOTE"
        and isinstance(node.value, ast.Name)
        and node.value.id == "Operation"
    )


def test_production_call_sites_never_invoke_legacy_promote() -> None:
    violations: list[str] = []
    paths = [
        path
        for directory in PRODUCTION_SOURCE_DIRS
        for path in (PRODUCTION_ROOT / directory).rglob("*.py")
    ]
    for path in sorted(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            arguments = [*call.args, *(item.value for item in call.keywords)]
            if any(_is_legacy_promote(argument) for argument in arguments):
                violations.append(
                    f"{path.relative_to(REPO)}:{getattr(call, 'lineno', 0)}"
                )
    assert violations == []


def test_positive_memory_gate_invokes_promote_result() -> None:
    path = (
        PRODUCTION_ROOT
        / "authority"
        / "adapters"
        / "mlevolve"
        / "promotion_gate.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    operations = [
        argument.attr
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        for argument in [*call.args, *(item.value for item in call.keywords)]
        if (
            isinstance(argument, ast.Attribute)
            and isinstance(argument.value, ast.Name)
            and argument.value.id == "Operation"
        )
    ]
    assert "PROMOTE_RESULT" in operations
    assert "PROMOTE" not in operations
