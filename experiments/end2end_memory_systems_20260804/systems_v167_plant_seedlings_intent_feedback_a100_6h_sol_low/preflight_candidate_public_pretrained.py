#!/usr/bin/env python3
"""Verify public pretrained construction in the real Candidate cache sandbox."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


MODEL_REPOSITORY = "timm/efficientnet_b3.ra2_in1k"
MODEL_NAME = "efficientnet_b3"
EXPECTED_DEFAULT_PARAMETERS = 12_233_232
EXPECTED_FEATURE_PARAMETERS = 10_696_232


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-cache-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--candidate-uid", type=int, default=65534)
    args = parser.parse_args()

    if args.runtime_cache_root.exists():
        raise FileExistsError(
            f"refusing to reuse Candidate preflight cache: {args.runtime_cache_root}"
        )
    if args.receipt.exists():
        raise FileExistsError(
            f"refusing to overwrite Candidate preflight receipt: {args.receipt}"
        )

    repo_root = Path(__file__).resolve().parents[3]
    mlevolve_root = repo_root / "mlevolve"
    sys.path.insert(0, str(mlevolve_root))
    from engine.executor import _execution_environment

    env = _execution_environment(args.runtime_cache_root)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    probe = r'''
import json
import os
import timm
from huggingface_hub import constants

default_model = timm.create_model("efficientnet_b3", pretrained=True)
default_parameters = sum(p.numel() for p in default_model.parameters())
del default_model
feature_model = timm.create_model(
    "efficientnet_b3",
    pretrained=True,
    num_classes=0,
    global_pool="avg",
)
feature_parameters = sum(p.numel() for p in feature_model.parameters())
print(json.dumps({
    "timm_version": timm.__version__,
    "default_parameters": default_parameters,
    "feature_parameters": feature_parameters,
    "hf_hub_cache": constants.HF_HUB_CACHE,
    "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
}, sort_keys=True))
'''
    isolation = {}
    if os.geteuid() == 0:
        isolation = {
            "user": args.candidate_uid,
            "group": args.candidate_uid,
            "extra_groups": (),
        }
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(args.runtime_cache_root),
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
        **isolation,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Candidate-isolated offline timm construction failed:\n"
            + completed.stderr
        )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    if result["default_parameters"] != EXPECTED_DEFAULT_PARAMETERS:
        raise RuntimeError(
            "unexpected default EfficientNet-B3 parameter count: "
            f"{result['default_parameters']}"
        )
    if result["feature_parameters"] != EXPECTED_FEATURE_PARAMETERS:
        raise RuntimeError(
            "unexpected feature EfficientNet-B3 parameter count: "
            f"{result['feature_parameters']}"
        )

    destination_hub = Path(env["HF_HUB_CACHE"])
    repository_dir = destination_hub / (
        "models--" + MODEL_REPOSITORY.replace("/", "--")
    )
    revision = (repository_dir / "refs" / "main").read_text().strip()
    weight_path = repository_dir / "snapshots" / revision / "model.safetensors"
    seed_receipt = json.loads(
        (
            args.runtime_cache_root / "host_hf_cache_seed_receipt.json"
        ).read_text()
    )
    receipt = {
        "schema": "mlevolve_candidate_public_pretrained_preflight_v1",
        "status": "passed",
        "network_mode": "offline",
        "candidate_uid": (
            args.candidate_uid if os.geteuid() == 0 else os.geteuid()
        ),
        "model_name": MODEL_NAME,
        "hf_repository": MODEL_REPOSITORY,
        "timm_version": result["timm_version"],
        "default_parameter_count": result["default_parameters"],
        "feature_parameter_count": result["feature_parameters"],
        "runtime_cache_root": str(args.runtime_cache_root.resolve()),
        "hf_hub_cache": result["hf_hub_cache"],
        "weight": {
            "path": str(weight_path),
            "sha256": sha256_file(weight_path),
            "bytes": weight_path.stat().st_size,
        },
        "host_seed_receipt": seed_receipt,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
