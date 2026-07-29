from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .models import ProtocolRef, ProtocolSpec


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_protocol_payload(spec: ProtocolSpec | dict[str, Any]) -> dict[str, Any]:
    payload = spec.as_dict() if isinstance(spec, ProtocolSpec) else dict(spec)
    payload.pop("canonical_hash", None)
    return payload


def protocol_hash(spec: ProtocolSpec | dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(canonical_protocol_payload(spec)).encode("utf-8")).hexdigest()


class ProtocolRegistry:
    def __init__(self, registry_dir: str | Path | None = None):
        self.registry_dir = Path(registry_dir).resolve() if registry_dir else None
        self._specs: dict[tuple[str, str], ProtocolSpec] = {}
        if self.registry_dir and self.registry_dir.exists():
            self.load_directory(self.registry_dir)

    def register(self, spec: ProtocolSpec, *, verify_hash: bool = True) -> ProtocolSpec:
        digest = protocol_hash(spec)
        if verify_hash and spec.canonical_hash and spec.canonical_hash != digest:
            raise ValueError(
                f"Protocol hash mismatch for {spec.protocol_id}@{spec.version}: "
                f"declared={spec.canonical_hash} computed={digest}"
            )
        normalized = replace(spec, canonical_hash=digest)
        key = (normalized.protocol_id, normalized.version)
        existing = self._specs.get(key)
        if existing and existing.canonical_hash != digest:
            raise ValueError(f"Protocol version is immutable once registered: {key}")
        self._specs[key] = normalized
        return normalized

    def load_directory(self, directory: Path) -> None:
        for path in sorted(directory.glob("*.json")):
            # A macOS-created archive can materialize hidden AppleDouble
            # sidecars (``._protocol.json``) when extracted on Linux. They are
            # metadata, not Protocol specs. Visible malformed JSON still fails
            # closed on the read/parse below.
            if path.name.startswith(".") or not path.is_file() or path.is_symlink():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.register(ProtocolSpec(**payload))

    def get(self, protocol_id: str, version: str) -> ProtocolSpec:
        try:
            return self._specs[(protocol_id, version)]
        except KeyError as exc:
            raise KeyError(f"Unknown protocol: {protocol_id}@{version}") from exc

    def resolve(self, value: str | ProtocolRef) -> ProtocolSpec:
        if isinstance(value, ProtocolRef):
            spec = self.get(value.protocol_id, value.version)
            if spec.canonical_hash != value.canonical_hash:
                raise ValueError("ProtocolRef hash does not match registered immutable version")
            return spec
        protocol_id, version = value.split("@", 1)
        return self.get(protocol_id, version)

    def compatible(self, evidence: ProtocolRef, active: ProtocolRef) -> bool:
        if evidence.canonical_hash == active.canonical_hash:
            return True
        active_spec = self.resolve(active)
        accepted = active_spec.compatibility_rules.get("accepted_protocol_hashes", [])
        return evidence.canonical_hash in set(map(str, accepted))

    def refs(self) -> list[ProtocolRef]:
        return [spec.ref() for spec in self._specs.values()]

    def compile_execution_contract(self, protocol: str | ProtocolRef, **kwargs):
        """Compile via a lazy import so registry/model compatibility stays acyclic."""

        from .protocol_execution_contract import compile_protocol_execution_contract

        return compile_protocol_execution_contract(self.resolve(protocol), **kwargs)
