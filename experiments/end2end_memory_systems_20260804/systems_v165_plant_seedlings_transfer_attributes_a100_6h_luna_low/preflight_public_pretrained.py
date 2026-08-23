#!/usr/bin/env python3
"""Make the selected public EfficientNet-B3 initialization offline-loadable.

This preflight is intentionally executable rather than an existence check: it
does not pass until the exact timm build in the experiment container can run
``create_model(..., pretrained=True)`` with Hub access disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any


MODEL_NAME = "efficientnet_b3"
HF_REPOSITORY = "timm/efficientnet_b3.ra2_in1k"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_cache_dir(hub_cache: Path, repository: str) -> Path:
    return hub_cache / ("models--" + repository.replace("/", "--"))


def current_snapshot(repository_dir: Path) -> Path:
    revision = (repository_dir / "refs" / "main").read_text().strip()
    if not revision or "/" in revision or ".." in revision:
        raise RuntimeError(f"unsafe or empty Hub revision: {revision!r}")
    snapshot = repository_dir / "snapshots" / revision
    if not snapshot.is_dir():
        raise RuntimeError(f"missing Hub snapshot directory: {snapshot}")
    return snapshot


def construct_offline() -> tuple[Any, str, str]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import timm

    model = timm.create_model(MODEL_NAME, pretrained=True)
    return model, timm.__version__, model.__class__.__name__


def materialize_timm_bin(safetensors_path: Path, bin_path: Path) -> None:
    """Atomically convert an already-cached public safetensors state dict."""

    import torch
    from safetensors.torch import load_file

    state_dict = load_file(str(safetensors_path), device="cpu")
    if not state_dict or not all(isinstance(key, str) for key in state_dict):
        raise RuntimeError("cached safetensors file is not a model state dict")
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{bin_path.name}.", suffix=".tmp", dir=bin_path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
    try:
        torch.save(state_dict, temporary_path)
        os.replace(temporary_path, bin_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_receipt(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def optional_file_receipt(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub-cache", type=Path)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--repair-cache", action="store_true")
    args = parser.parse_args()

    if args.hub_cache is None:
        args.hub_cache = Path(
            os.environ.get("HUGGINGFACE_HUB_CACHE")
            or os.environ.get("HF_HUB_CACHE")
            or Path.home() / ".cache" / "huggingface" / "hub"
        )
    repository_dir = repository_cache_dir(args.hub_cache, HF_REPOSITORY)
    snapshot = current_snapshot(repository_dir)
    safetensors_path = snapshot / "model.safetensors"
    bin_path = snapshot / "pytorch_model.bin"
    if not safetensors_path.is_file() and not bin_path.is_file():
        raise RuntimeError(
            f"neither model.safetensors nor pytorch_model.bin exists in {snapshot}"
        )

    converted = False
    first_error = ""
    try:
        model, timm_version, model_class = construct_offline()
    except Exception as error:
        first_error = f"{type(error).__name__}: {error}"
        if not args.repair_cache:
            raise
        if not safetensors_path.is_file():
            raise RuntimeError(
                "offline timm construction failed and no cached safetensors source exists"
            ) from error
        materialize_timm_bin(safetensors_path, bin_path)
        converted = True
        model, timm_version, model_class = construct_offline()

    state = model.state_dict()
    if not state:
        raise RuntimeError("offline timm construction returned an empty state dict")
    payload = {
        "schema": "mlevolve_public_pretrained_preflight_v1",
        "status": "passed",
        "network_mode": "offline",
        "model_name": MODEL_NAME,
        "hf_repository": HF_REPOSITORY,
        "timm_version": timm_version,
        "model_class": model_class,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "state_tensor_count": len(state),
        "cache_repaired": converted,
        "initial_construction_error": first_error or None,
        "safetensors": (
            {
                "path": str(safetensors_path),
                "sha256": sha256_file(safetensors_path),
                "bytes": safetensors_path.stat().st_size,
            }
            if safetensors_path.is_file()
            else None
        ),
        # timm/huggingface_hub may legitimately choose the cached safetensors
        # artifact even when an older process requested pytorch_model.bin.  The
        # executable construction above is authoritative; record whichever
        # cache forms remain after that exact call instead of requiring a
        # filename that the installed loader did not use.
        "pytorch_model_bin": optional_file_receipt(bin_path),
    }
    write_receipt(args.receipt, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
