"""Freeze and validate one Host Protocol activation bundle.

The binding is the single launcher input for an enforcing Host run.  It binds
the task, Contract, terminal-blind DataView manifest, runtime/report roots,
container image and SDK tree.  Launchers must fail closed when any binding is
missing or has drifted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from authority.protocol_execution_contract import (
    compile_protocol_execution_contract,
    read_contract_artifact,
    write_contract_artifacts,
)
from authority.protocol_registry import ProtocolRegistry
from protocol_runtime.collector import HostCollectorIdentity
from protocol_runtime.data_views import (
    materialize_data_views,
    read_data_view_manifest,
    verify_data_view_manifest,
)
from protocol_runtime.task_staging import TASK_SPECS, stage_task


HOST_PROTOCOL_BINDING_SCHEMA = "mlevolve_host_protocol_binding_v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _hash_payload(value: Mapping[str, Any], hash_field: str) -> str:
    return hashlib.sha256(
        _canonical_json(
            {key: item for key, item in value.items() if key != hash_field}
        ).encode("utf-8")
    ).hexdigest()


def _leakage_only_execution_budget(value: Mapping[str, Any]) -> dict[str, int]:
    """Keep operational timeout metadata, never bind model-method limits.

    Host Protocol is a leakage/evidence boundary.  Fold count, epochs and model
    count must not become Claim validity criteria, even when an older caller
    still supplies those legacy keys.
    """

    timeout_seconds = int(value.get("timeout_seconds", 60) or 0)
    if timeout_seconds <= 0:
        raise ValueError("Host Protocol timeout_seconds must be positive")
    return {"timeout_seconds": timeout_seconds}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_symlink_chain(path: Path, *, label: str) -> None:
    if ".." in path.parts:
        raise ValueError(f"Refusing path traversal in {label}: {path}")
    for component in (path, *path.parents):
        if component.exists() and component.is_symlink():
            raise ValueError(f"Refusing symlink {label}: {component}")


def _regular_file(path: str | Path, *, label: str) -> Path:
    requested = Path(path).expanduser()
    _reject_symlink_chain(requested, label=label)
    resolved = requested.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file")
    return resolved


def _directory(path: str | Path, *, label: str, create: bool) -> Path:
    requested = Path(path).expanduser()
    _reject_symlink_chain(requested, label=label)
    if create:
        requested.mkdir(parents=True, exist_ok=True)
    resolved = requested.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a directory")
    return resolved


def _write_exclusive(path: Path, content: bytes, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def hash_sdk_tree(root: str | Path) -> str:
    """Hash the importable Host SDK source tree, independent of mtimes."""

    sdk_root = _directory(root, label="Host SDK root", create=False)
    rows: list[str] = []
    for path in sorted(sdk_root.rglob("*.py")):
        if (
            "__pycache__" in path.parts
            or path.name.startswith("._")
            or path.is_symlink()
            or not path.is_file()
        ):
            continue
        rows.append(f"{path.relative_to(sdk_root).as_posix()}\0{_sha256_file(path)}")
    if not rows:
        raise ValueError("Host SDK root contains no Python sources")
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def freeze_host_protocol_binding(
    binding_path: str | Path,
    *,
    task_id: str,
    contract_path: str | Path,
    data_view_manifest_path: str | Path,
    description_path: str | Path,
    report_root: str | Path,
    runtime_artifact_root: str | Path,
    image_digest: str,
    sdk_hash: str,
) -> dict[str, Any]:
    """Write an immutable, hash-bound launcher binding."""

    if not task_id.strip():
        raise ValueError("Host Protocol binding requires task_id")
    if not _IMAGE_DIGEST.fullmatch(str(image_digest)):
        raise ValueError("Host Protocol binding requires a sha256 image digest")
    if not _SHA256.fullmatch(str(sdk_hash)):
        raise ValueError("Host Protocol binding requires a SHA-256 SDK hash")

    contract_file = _regular_file(contract_path, label="Execution Contract")
    manifest_file = _regular_file(
        data_view_manifest_path, label="DataView manifest"
    )
    description_file = _regular_file(description_path, label="task description")
    contract = read_contract_artifact(contract_file)
    manifest = read_data_view_manifest(manifest_file)
    if contract.task_id != task_id or manifest.task_id != task_id:
        raise ValueError("Host Protocol task ID does not match Contract/DataView")
    manifest_label_key = str(
        (manifest.strategy_verification or {}).get("label_key") or "label"
    )
    verification = verify_data_view_manifest(
        manifest_file,
        contract=contract,
        label_key=manifest_label_key,
    )
    reports = _directory(report_root, label="Preflight report root", create=True)
    runtime = _directory(
        runtime_artifact_root, label="protocol runtime artifact root", create=True
    )
    data_root = manifest_file.parent.resolve(strict=True)
    for writable, label in (
        (reports, "Preflight report root"),
        (runtime, "protocol runtime artifact root"),
    ):
        try:
            writable.relative_to(data_root)
        except ValueError:
            pass
        else:
            raise ValueError(f"{label} cannot be inside the Candidate DataView root")
    if reports == runtime:
        raise ValueError("Preflight and runtime artifact roots must be distinct")
    payload: dict[str, Any] = {
        "schema": HOST_PROTOCOL_BINDING_SCHEMA,
        "task_id": task_id,
        "contract_path": str(contract_file),
        "contract_file_sha256": _sha256_file(contract_file),
        "contract_hash": contract.contract_hash,
        "data_view_manifest_path": str(manifest_file),
        "data_view_manifest_file_sha256": _sha256_file(manifest_file),
        "data_view_manifest_hash": manifest.manifest_hash,
        "data_view_root": str(data_root),
        "description_path": str(description_file),
        "description_sha256": _sha256_file(description_file),
        "report_root": str(reports),
        "runtime_artifact_root": str(runtime),
        "image_digest": image_digest,
        "sdk_hash": sdk_hash,
        "terminal_exposure_count": int(verification["terminal_exposure_count"]),
        "binding_hash": "",
    }
    payload["binding_hash"] = _hash_payload(payload, "binding_hash")
    destination = Path(binding_path).expanduser().resolve()
    _write_exclusive(
        destination,
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return payload


def load_host_protocol_binding(
    binding_path: str | Path,
    *,
    expected_task_id: str = "",
    expected_image_digest: str = "",
    expected_sdk_hash: str = "",
) -> dict[str, Any]:
    """Load and re-verify every artifact named by a frozen binding."""

    if not str(binding_path or "").strip():
        raise ValueError("Host Protocol Preflight requires binding_path")
    path = _regular_file(binding_path, label="Host Protocol binding")
    if path.stat().st_mode & 0o222:
        raise ValueError("Host Protocol binding must be immutable (read-only)")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema",
        "task_id",
        "contract_path",
        "contract_file_sha256",
        "contract_hash",
        "data_view_manifest_path",
        "data_view_manifest_file_sha256",
        "data_view_manifest_hash",
        "data_view_root",
        "description_path",
        "description_sha256",
        "report_root",
        "runtime_artifact_root",
        "image_digest",
        "sdk_hash",
        "terminal_exposure_count",
        "binding_hash",
    }
    if set(payload) != expected_fields:
        raise ValueError("Host Protocol binding fields do not match schema")
    if payload.get("schema") != HOST_PROTOCOL_BINDING_SCHEMA:
        raise ValueError("Unsupported Host Protocol binding schema")
    if payload.get("binding_hash") != _hash_payload(payload, "binding_hash"):
        raise ValueError("Host Protocol binding hash mismatch")
    if expected_task_id and payload.get("task_id") != expected_task_id:
        raise ValueError(
            "Host Protocol binding task mismatch: "
            f"expected={expected_task_id} actual={payload.get('task_id')}"
        )
    if not _IMAGE_DIGEST.fullmatch(str(payload.get("image_digest") or "")):
        raise ValueError("Host Protocol image digest is not frozen")
    if not _SHA256.fullmatch(str(payload.get("sdk_hash") or "")):
        raise ValueError("Host Protocol SDK hash is not frozen")
    if expected_image_digest and payload["image_digest"] != expected_image_digest:
        raise ValueError("Host Protocol runtime image digest mismatch")
    if expected_sdk_hash and payload["sdk_hash"] != expected_sdk_hash:
        raise ValueError("Host Protocol runtime SDK hash mismatch")
    if payload.get("terminal_exposure_count") != 0:
        raise ValueError("Host Protocol binding exposes terminal data")

    contract_file = _regular_file(payload["contract_path"], label="Execution Contract")
    manifest_file = _regular_file(
        payload["data_view_manifest_path"], label="DataView manifest"
    )
    description_file = _regular_file(
        payload["description_path"], label="task description"
    )
    for actual, expected, label in (
        (_sha256_file(contract_file), payload["contract_file_sha256"], "Contract"),
        (
            _sha256_file(manifest_file),
            payload["data_view_manifest_file_sha256"],
            "DataView manifest",
        ),
        (
            _sha256_file(description_file),
            payload["description_sha256"],
            "task description",
        ),
    ):
        if actual != expected:
            raise ValueError(f"Host Protocol {label} file hash mismatch")
    contract = read_contract_artifact(contract_file)
    manifest = read_data_view_manifest(manifest_file)
    if contract.contract_hash != payload["contract_hash"]:
        raise ValueError("Host Protocol Contract hash mismatch")
    if manifest.manifest_hash != payload["data_view_manifest_hash"]:
        raise ValueError("Host Protocol DataView hash mismatch")
    if contract.task_id != payload["task_id"] or manifest.task_id != payload["task_id"]:
        raise ValueError("Host Protocol task binding mismatch")
    if manifest_file.parent.resolve(strict=True) != Path(
        payload["data_view_root"]
    ).resolve(strict=True):
        raise ValueError("Host Protocol DataView root mismatch")
    manifest_label_key = str(
        (manifest.strategy_verification or {}).get("label_key") or "label"
    )
    verify_data_view_manifest(
        manifest_file,
        contract=contract,
        label_key=manifest_label_key,
        # The immutable binding was created only after a full content check.
        # Runtime activation revalidates the signed manifests and split
        # invariants, but must not reread every image before each experiment.
        verify_asset_contents=False,
    )
    reports = _directory(
        payload["report_root"], label="Preflight report root", create=False
    )
    runtime = _directory(
        payload["runtime_artifact_root"],
        label="protocol runtime artifact root",
        create=False,
    )
    data_root = manifest_file.parent.resolve(strict=True)
    for writable, label in (
        (reports, "Preflight report root"),
        (runtime, "protocol runtime artifact root"),
    ):
        try:
            writable.relative_to(data_root)
        except ValueError:
            pass
        else:
            raise ValueError(f"{label} cannot be inside the Candidate DataView root")
    if reports == runtime:
        raise ValueError("Preflight and runtime artifact roots must be distinct")
    return dict(payload)


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Record line {index} is not a JSON object")
        records.append(value)
    return records


def _host_description_appendix(
    *, task_id: str, label_key: str, inference_enabled: bool
) -> str:
    """Return the authoritative Candidate-visible Host row schema note."""

    lines = [
        "# Host Runtime Schema (authoritative)",
        "",
        "This section overrides any earlier raw-dataset field spelling. "
        "Candidate code receives normalized rows from `ProtocolSession`, not "
        "the original CSV objects.",
        f"The supervised target is the direct training/validation row field "
        f"`{label_key}`.",
    ]
    if task_id == "leaf-classification":
        lines.extend(
            [
                "Leaf feature fields are exactly `margin1`…`margin64`, "
                "`shape1`…`shape64`, and `texture1`…`texture64`.",
                "The numeric suffix has no underscore: `margin_1`, `shape_1`, "
                "and `texture_1` do not exist in Host rows.",
                "Build frames directly with `pd.DataFrame(rows)`; features are "
                "not nested or JSON-encoded.",
            ]
        )
    if inference_enabled:
        lines.append(
            "The inference view is unlabeled and becomes accessible only after "
            "`session.freeze_selection(...)`."
        )
    return "\n".join(lines) + "\n"


def build_host_protocol_bundle(
    *,
    output_root: str | Path,
    records_path: str | Path,
    inference_records_path: str | Path | None = None,
    description_path: str | Path,
    registry_root: str | Path,
    protocol_ref: str,
    task_id: str,
    task_family: str,
    image_digest: str,
    sdk_root: str | Path,
    collector_private_key_output: str | Path,
    split_id: str,
    execution_budget: Mapping[str, Any],
    report_root: str | Path | None = None,
    runtime_artifact_root: str | Path | None = None,
    sample_id_key: str = "sample_id",
    label_key: str = "label",
    group_id_key: str = "group_id",
    time_key: str = "timestamp",
    validation_fraction: float = 0.2,
    seed: str = "0",
) -> dict[str, Any]:
    """Build one immutable Contract/DataView/binding bundle from normalized JSONL."""

    requested_root = Path(output_root).expanduser()
    if requested_root.is_symlink():
        raise ValueError("Refusing symlink Host Protocol output root")
    root = requested_root.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError("Host Protocol output root must be empty")
    root.mkdir(parents=True, exist_ok=True)
    private_key_output = Path(collector_private_key_output).expanduser().resolve()
    try:
        private_key_output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "Host Collector private key output must be outside the Candidate bundle"
        )
    records_file = _regular_file(records_path, label="normalized task records")
    source_description = _regular_file(description_path, label="task description")
    description = root / "TASK_DESCRIPTION.md"
    source_text = source_description.read_text(encoding="utf-8")
    appendix = _host_description_appendix(
        task_id=task_id,
        label_key=label_key,
        inference_enabled=inference_records_path is not None,
    )
    _write_exclusive(
        description,
        (source_text.rstrip() + "\n\n" + appendix).encode("utf-8"),
    )

    if private_key_output.exists():
        identity = HostCollectorIdentity.from_private_key_file(private_key_output)
    else:
        identity = HostCollectorIdentity.generate()
        identity.write_private_key_file(private_key_output)
    registry = ProtocolRegistry(registry_root)
    inference_view_ref = (
        f"view://{task_id}/{split_id}/inference"
        if inference_records_path is not None
        else ""
    )
    adapter_spec = {
        "managed": ["boosting", "sklearn"],
        "scope": ["torch"],
        "legacy_ast_positive_proof": False,
        "full_runtime_sdk_required": True,
        "inference_view_required": bool(inference_records_path is not None),
        "inference_view_ref": inference_view_ref,
    }
    contract = compile_protocol_execution_contract(
        registry.resolve(protocol_ref),
        task_id=task_id,
        task_family=task_family,
        train_view_ref=f"view://{task_id}/{split_id}/train",
        validation_view_ref=f"view://{task_id}/{split_id}/internal-validation",
        terminal_view_ref=f"evaluator-only://{task_id}/{split_id}/terminal",
        execution_budget=_leakage_only_execution_budget(execution_budget),
        allowed_import_roots=(),
        collector_spec=identity.collector_spec(),
        adapter_spec=adapter_spec,
    )
    contract_path, _sidecar = write_contract_artifacts(contract, root / "contract")
    _manifest, manifest_path = materialize_data_views(
        _read_records(records_file),
        root / "data_views",
        contract,
        inference_records=(
            _read_records(
                _regular_file(
                    inference_records_path,
                    label="normalized inference records",
                )
            )
            if inference_records_path is not None
            else None
        ),
        inference_view_ref=inference_view_ref,
        split_id=split_id,
        sample_id_key=sample_id_key,
        label_key=label_key,
        group_id_key=group_id_key,
        time_key=time_key,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    reports = Path(report_root).expanduser() if report_root else root / "reports"
    runtime = (
        Path(runtime_artifact_root).expanduser()
        if runtime_artifact_root
        else root / "runtime"
    )
    reports.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    binding = freeze_host_protocol_binding(
        root / "HOST_PROTOCOL_BINDING.json",
        task_id=task_id,
        contract_path=contract_path,
        data_view_manifest_path=manifest_path,
        description_path=description,
        report_root=reports,
        runtime_artifact_root=runtime,
        image_digest=image_digest,
        sdk_hash=hash_sdk_tree(sdk_root),
    )
    return binding


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify", help="verify a frozen Host binding")
    verify.add_argument("--binding", required=True)
    verify.add_argument("--task-id", default="")
    verify.add_argument("--expected-image-digest", default="")
    verify.add_argument("--expected-sdk-hash", default="")

    build = commands.add_parser("build", help="build a Host activation bundle")
    build.add_argument("--output-root", required=True)
    build.add_argument("--records", required=True)
    build.add_argument("--inference-records")
    build.add_argument("--description", required=True)
    build.add_argument("--registry-root", required=True)
    build.add_argument("--protocol-ref", required=True)
    build.add_argument("--task-id", required=True)
    build.add_argument("--task-family", required=True)
    build.add_argument("--image-digest", required=True)
    build.add_argument("--sdk-root", required=True)
    build.add_argument(
        "--collector-private-key-output",
        required=True,
        help="Host-only path outside output-root for the raw Ed25519 key",
    )
    build.add_argument("--split-id", required=True)
    build.add_argument("--timeout-seconds", type=int, default=60)
    build.add_argument("--report-root")
    build.add_argument("--runtime-artifact-root")
    build.add_argument("--sample-id-key", default="sample_id")
    build.add_argument("--label-key", default="label")
    build.add_argument("--group-id-key", default="group_id")
    build.add_argument("--time-key", default="timestamp")
    build.add_argument("--validation-fraction", type=float, default=0.2)
    build.add_argument("--seed", default="0")

    build_task = commands.add_parser(
        "build-task", help="stage and freeze one supported online task"
    )
    build_task.add_argument("--task-id", required=True, choices=sorted(TASK_SPECS))
    build_task.add_argument("--public-root", required=True)
    build_task.add_argument("--staging-root", required=True)
    build_task.add_argument("--output-root", required=True)
    build_task.add_argument("--description", required=True)
    build_task.add_argument("--registry-root", required=True)
    build_task.add_argument("--image-digest", required=True)
    build_task.add_argument("--sdk-root", required=True)
    build_task.add_argument("--collector-private-key-output", required=True)
    build_task.add_argument("--split-id", required=True)
    build_task.add_argument("--report-root", required=True)
    build_task.add_argument("--runtime-artifact-root", required=True)
    build_task.add_argument("--timeout-seconds", type=int, default=60)
    build_task.add_argument("--validation-fraction", type=float, default=0.2)
    build_task.add_argument("--seed", default="0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "verify":
        payload = load_host_protocol_binding(
            args.binding,
            expected_task_id=args.task_id,
            expected_image_digest=args.expected_image_digest,
            expected_sdk_hash=args.expected_sdk_hash,
        )
    elif args.command == "build":
        payload = build_host_protocol_bundle(
            output_root=args.output_root,
            records_path=args.records,
            inference_records_path=args.inference_records,
            description_path=args.description,
            registry_root=args.registry_root,
            protocol_ref=args.protocol_ref,
            task_id=args.task_id,
            task_family=args.task_family,
            image_digest=args.image_digest,
            sdk_root=args.sdk_root,
            collector_private_key_output=args.collector_private_key_output,
            split_id=args.split_id,
            execution_budget={
                "timeout_seconds": args.timeout_seconds,
            },
            report_root=args.report_root,
            runtime_artifact_root=args.runtime_artifact_root,
            sample_id_key=args.sample_id_key,
            label_key=args.label_key,
            group_id_key=args.group_id_key,
            time_key=args.time_key,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    else:
        staged = stage_task(args.task_id, args.public_root, args.staging_root)
        spec = TASK_SPECS[args.task_id]
        protocol_spec = ProtocolRegistry(args.registry_root).resolve(
            spec["protocol_ref"]
        )
        actual_metric = str(protocol_spec.metric_spec.get("name") or "")
        actual_direction = str(protocol_spec.metric_spec.get("direction") or "")
        if (
            actual_metric != spec["metric_name"]
            or actual_direction != spec["metric_direction"]
        ):
            raise ValueError(
                "Task/Host Protocol metric mismatch: "
                f"task={args.task_id} expected={spec['metric_name']}/"
                f"{spec['metric_direction']} actual={actual_metric}/"
                f"{actual_direction}"
            )
        payload = build_host_protocol_bundle(
            output_root=args.output_root,
            records_path=staged["train_records_path"],
            inference_records_path=staged["inference_records_path"],
            description_path=args.description,
            registry_root=args.registry_root,
            protocol_ref=spec["protocol_ref"],
            task_id=args.task_id,
            task_family=spec["task_family"],
            image_digest=args.image_digest,
            sdk_root=args.sdk_root,
            collector_private_key_output=args.collector_private_key_output,
            split_id=args.split_id,
            execution_budget={
                "timeout_seconds": args.timeout_seconds,
            },
            report_root=args.report_root,
            runtime_artifact_root=args.runtime_artifact_root,
            label_key=spec["label_key"],
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HOST_PROTOCOL_BINDING_SCHEMA",
    "build_host_protocol_bundle",
    "freeze_host_protocol_binding",
    "hash_sdk_tree",
    "load_host_protocol_binding",
]
