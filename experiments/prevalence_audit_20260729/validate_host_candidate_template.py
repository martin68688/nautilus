#!/usr/bin/env python3
"""Run the task-specific Prompt Host candidate template against a frozen bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents.prompts.impl_guideline import _host_full_runtime_validation_source
from authority.protocol_execution_contract import read_contract_artifact
from authority.protocol_registry import ProtocolRegistry
from protocol_runtime.activation import load_host_protocol_binding
from protocol_runtime.collector import HostCollectorIdentity
from protocol_runtime.preflight import PreflightStatus, ProtocolPreflightRunner


LABEL_KEYS = {
    "aerial-cactus-identification": "label",
    "denoising-dirty-documents": "target",
    "leaf-classification": "label",
    "new-york-city-taxi-fare-prediction": "fare",
    "spooky-author-identification": "author",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True, type=Path)
    parser.add_argument("--collector-key", required=True, type=Path)
    parser.add_argument("--registry-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    binding = load_host_protocol_binding(args.binding)
    contract = read_contract_artifact(binding["contract_path"])
    metric_name = str((contract.evaluator_spec.get("metric") or {}).get("name") or "")
    source = _host_full_runtime_validation_source(
        {
            "task_id": contract.task_id,
            "label_key": LABEL_KEYS[contract.task_id],
            "metric_name": metric_name,
        }
    )
    report = ProtocolPreflightRunner(ProtocolRegistry(args.registry_root)).run_source(
        source=source,
        contract=contract,
        identity=HostCollectorIdentity.from_private_key_file(args.collector_key),
        data_view_manifest_path=binding["data_view_manifest_path"],
        output_root=args.output_root,
        image_digest=binding["image_digest"],
        sdk_hash=binding["sdk_hash"],
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "contract_hash": report["contract_hash"],
                "code_sha256": report["code_sha256"],
                "closure_hash": report["closure_hash"],
                "missing_receipts": report["missing_receipts"],
                "runtime_error": report["runtime_error"],
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == PreflightStatus.PASS.value else 1


if __name__ == "__main__":
    raise SystemExit(main())
