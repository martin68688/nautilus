from __future__ import annotations

from pathlib import Path
from typing import Any

from ...models import ProtocolSpec
from ...protocol_registry import ProtocolRegistry


def registry_path(cfg: Any) -> Path:
    configured = str(getattr(getattr(cfg, "evaluation_authority", None), "protocol_registry", "") or "")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        return path.resolve()
    return (Path(__file__).resolve().parents[3] / "config" / "protocols").resolve()


def build_registry(cfg: Any) -> tuple[ProtocolRegistry, ProtocolSpec]:
    authority_cfg = getattr(cfg, "evaluation_authority", None)
    registry = ProtocolRegistry(registry_path(cfg))
    protocol_id = str(getattr(authority_cfg, "active_protocol_id", "mlevolve-default"))
    version = str(getattr(authority_cfg, "active_protocol_version", "1"))
    # A protocol-scoped authority cannot invent its protocol after the run
    # starts. Missing or typoed ids are configuration failures in every mode.
    return registry, registry.get(protocol_id, version)
