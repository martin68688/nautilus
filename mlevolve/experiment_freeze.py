"""Create and verify an immutable formal-experiment environment manifest.

The formal runner must bind every moving input (source, protocol, data views,
evaluator, model assets, config and memory bundle) to content hashes.  This
module deliberately has no network or Kubernetes dependency; it is used both
locally and as a verify-only startup check inside a Job.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Iterable, Mapping


FREEZE_SCHEMA = "mlevolve_formal_experiment_freeze_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_DIGEST_RE = re.compile(r"(?:@)?sha256:[0-9a-f]{64}$")
FLOATING_SYNC_RE = re.compile(
    r"(?:git\s+(?:fetch|pull)\b|git\s+checkout\s+origin/|"
    r"git\s+reset\s+--hard\s+origin/|git\s+clone\b)",
    re.IGNORECASE,
)


class FreezeError(ValueError):
    """Raised when a formal input is mutable, missing, or hash-inconsistent."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_path(root: Path, value: str | Path) -> tuple[Path, str]:
    candidate = Path(value)
    resolved_root = root.resolve()
    lexical = candidate if candidate.is_absolute() else root / candidate
    if lexical.is_symlink():
        raise FreezeError(f"Symlinked formal artifact is not allowed: {value}")
    resolved = lexical.resolve()
    try:
        relative = resolved.relative_to(resolved_root).as_posix()
    except ValueError as error:
        raise FreezeError(f"Artifact escapes freeze root: {value}") from error
    if relative in {"", "."}:
        raise FreezeError("Artifact path may not be the freeze root itself")
    return resolved, relative


def _excluded(relative: str, patterns: Iterable[str]) -> bool:
    value = PurePosixPath(relative)
    if "__pycache__" in value.parts or relative.endswith(".pyc") or value.name == ".DS_Store":
        return True
    return any(value.match(str(pattern)) for pattern in patterns)


def _artifact_hash(
    path: Path, *, root: Path, excludes: Iterable[str] = ()
) -> dict[str, Any]:
    if path.is_symlink():
        raise FreezeError(f"Symlinked formal artifact is not allowed: {path}")
    if not path.exists():
        raise FreezeError(f"Formal artifact does not exist: {path}")
    if path.is_file():
        return {
            "kind": "file",
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "files": 1,
        }
    if not path.is_dir():
        raise FreezeError(f"Unsupported formal artifact type: {path}")

    exclude_patterns = sorted(set(str(value) for value in excludes if str(value)))
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise FreezeError(f"Symlink inside formal artifact is not allowed: {child}")
        if not child.is_file():
            continue
        relative = child.relative_to(path).as_posix()
        if _excluded(relative, exclude_patterns):
            continue
        digest = sha256_file(child)
        size = child.stat().st_size
        total_bytes += size
        entries.append({"path": relative, "sha256": digest, "bytes": size})
    return {
        "kind": "directory",
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_bytes(_canonical(entries).encode("utf-8")),
        "bytes": total_bytes,
        "files": len(entries),
        "excludes": exclude_patterns,
        "entries": entries,
    }


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise FreezeError(f"Cannot inspect git repository {repo}: {error}") from error


def _git_binding(repo: Path) -> dict[str, Any]:
    git_dir = repo / ".git"
    if not git_dir.exists():
        return {
            "present": False,
            "commit": "",
            "branch": "",
            "status_sha256": "",
            "status_entry_count": 0,
            "diff_sha256": "",
        }
    commit = _git(repo, "rev-parse", "HEAD")
    branch_result = subprocess.run(
        ["git", "-C", str(repo), "symbolic-ref", "--short", "-q", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else ""
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    try:
        diff = subprocess.check_output(
            ["git", "-C", str(repo), "diff", "--binary", "HEAD", "--"],
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FreezeError(f"Cannot capture git diff: {error}") from error
    return {
        "present": True,
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_sha256": sha256_bytes(status.encode("utf-8")),
        "status_entry_count": len(status.splitlines()),
        "diff_sha256": sha256_bytes(diff),
    }


def validate_image_digest(value: str) -> str:
    image = str(value or "").strip()
    if not image or not IMAGE_DIGEST_RE.search(image):
        raise FreezeError(
            "Formal Jobs require an immutable image digest (name@sha256:<64 hex>)"
        )
    return image


def validate_job_text(text: str, *, expected_image_digest: str = "") -> None:
    if FLOATING_SYNC_RE.search(text):
        raise FreezeError("Formal Job contains floating branch/network sync command")
    image_digest = validate_image_digest(expected_image_digest) if expected_image_digest else ""
    image_lines = [line for line in text.splitlines() if re.search(r"\bimage\s*:", line)]
    if not image_lines:
        raise FreezeError("Formal Job does not declare a container image")
    for line in image_lines:
        image = line.split(":", 1)[1].strip().strip("'\"")
        if "@sha256:" not in image:
            raise FreezeError(f"Formal Job image is floating: {image}")
        if image_digest and image_digest not in image:
            raise FreezeError("Formal Job image digest does not match freeze manifest")


def _normalize_artifact_specs(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = spec.get("artifacts")
    if not isinstance(raw, list) or not raw:
        raise FreezeError("Freeze spec requires a non-empty artifacts list")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping):
            raise FreezeError("Each artifact specification must be an object")
        name = str(item.get("name") or "").strip()
        category = str(item.get("category") or "").strip()
        path = str(item.get("path") or "").strip()
        if not name or not category or not path or name in seen:
            raise FreezeError(f"Invalid or duplicate artifact specification: {item!r}")
        seen.add(name)
        excludes = item.get("exclude") or []
        if isinstance(excludes, str):
            excludes = [excludes]
        if not isinstance(excludes, list):
            raise FreezeError(f"Artifact excludes must be a list: {name}")
        output.append(
            {
                "name": name,
                "category": category,
                "path": path,
                "exclude": [str(value) for value in excludes],
            }
        )
    required = {
        "code",
        "protocol",
        "data",
        "evaluator",
        "model",
        "config",
        "memory_bundle",
        "task_seed",
        "environment",
    }
    missing = sorted(required - {item["category"] for item in output})
    if missing:
        raise FreezeError(f"Freeze spec missing artifact categories: {missing}")
    return output


def _manifest_identity(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(
        _canonical(
            {key: value for key, value in payload.items() if key != "manifest_hash"}
        ).encode("utf-8")
    )


def create_freeze_manifest(
    spec_path: str | Path,
    output_path: str | Path,
    *,
    root: str | Path | None = None,
) -> dict[str, Any]:
    spec_file = Path(spec_path).resolve()
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    if not isinstance(spec, Mapping):
        raise FreezeError("Freeze spec must be a JSON object")
    if root is not None:
        freeze_root = Path(root).resolve()
    elif spec.get("root"):
        declared_root = Path(str(spec["root"]))
        freeze_root = (
            declared_root.resolve()
            if declared_root.is_absolute()
            else (spec_file.parent / declared_root).resolve()
        )
    else:
        freeze_root = spec_file.parent
    artifacts = _normalize_artifact_specs(spec)
    image_digest = validate_image_digest(str(spec.get("container_image_digest") or ""))
    protocol_ref = str(spec.get("protocol_ref") or "").strip()
    evaluator = spec.get("evaluator")
    model = spec.get("model")
    task_seeds = spec.get("task_seeds")
    if not protocol_ref or not isinstance(evaluator, Mapping) or not isinstance(model, Mapping):
        raise FreezeError("Freeze spec requires protocol_ref, evaluator and model bindings")
    if not isinstance(task_seeds, list) or not task_seeds:
        raise FreezeError("Freeze spec requires at least one task/seed binding")

    hashed: list[dict[str, Any]] = []
    for item in artifacts:
        path, relative = _safe_path(freeze_root, item["path"])
        descriptor = _artifact_hash(
            path, root=freeze_root, excludes=item.get("exclude") or []
        )
        descriptor.update({"name": item["name"], "category": item["category"]})
        # Keep a path relative to the frozen root; absolute local paths are not
        # part of the identity and can be rebound with --root at verification.
        descriptor["path"] = relative
        hashed.append(descriptor)

    git_binding = _git_binding(freeze_root)
    git_binding["source_worktree_sha256"] = sha256_bytes(
        _canonical(
            [
                {"name": item["name"], "path": item["path"], "sha256": item["sha256"]}
                for item in sorted(hashed, key=lambda value: value["name"])
                if item["category"] == "code"
            ]
        ).encode("utf-8")
    )
    payload: dict[str, Any] = {
        "schema": FREEZE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "root": str(freeze_root),
        "container_image_digest": image_digest,
        "protocol_ref": protocol_ref,
        "evaluator": dict(evaluator),
        "model": dict(model),
        "task_seeds": task_seeds,
        "artifacts": sorted(hashed, key=lambda item: item["name"]),
        "git": git_binding,
        "floating_sync_forbidden": True,
        "manifest_hash": "",
    }
    payload["manifest_hash"] = _manifest_identity(payload)
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    try:
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as error:
        raise FreezeError(f"Refusing to replace immutable freeze manifest: {target}") from error
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def verify_freeze_manifest(
    manifest_path: str | Path,
    *,
    root: str | Path | None = None,
    job_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).resolve()
    jobs = list(job_paths)
    if manifest_file.is_symlink() or not manifest_file.is_file():
        raise FreezeError("Freeze manifest must be a regular file")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema") != FREEZE_SCHEMA:
        raise FreezeError("Freeze manifest schema mismatch")
    if manifest.get("manifest_hash") != _manifest_identity(manifest):
        raise FreezeError("Freeze manifest hash mismatch")
    validate_image_digest(str(manifest.get("container_image_digest") or ""))
    freeze_root = Path(root or manifest.get("root") or manifest_file.parent).resolve()
    for item in manifest.get("artifacts") or []:
        if not isinstance(item, Mapping):
            raise FreezeError("Freeze manifest artifact entry is invalid")
        path, _relative = _safe_path(freeze_root, str(item.get("path") or ""))
        current = _artifact_hash(
            path,
            root=freeze_root,
            excludes=item.get("excludes") or [],
        )
        for key in ("kind", "path", "sha256", "bytes", "files"):
            if current.get(key) != item.get(key):
                raise FreezeError(
                    f"Frozen artifact changed: {item.get('name') or item.get('path')} ({key})"
                )
        if current.get("kind") == "directory" and current.get("entries") != item.get("entries"):
            raise FreezeError(f"Frozen directory contents changed: {item.get('path')}")
    for job_path in jobs:
        job_file = Path(job_path)
        text = job_file.read_text(encoding="utf-8")
        validate_job_text(text, expected_image_digest=str(manifest["container_image_digest"]))
    return {
        "schema": "mlevolve_formal_experiment_freeze_verification_v1",
        "status": "verified",
        "manifest_hash": manifest["manifest_hash"],
        "container_image_digest": manifest["container_image_digest"],
        "artifact_count": len(manifest.get("artifacts") or []),
        "job_count": len(jobs),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--spec", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("--root", type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--root", type=Path)
    verify.add_argument("--job", action="append", default=[], type=Path)
    args = parser.parse_args()
    if args.command == "create":
        result = create_freeze_manifest(args.spec, args.output, root=args.root)
    else:
        result = verify_freeze_manifest(args.manifest, root=args.root, job_paths=args.job)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()


__all__ = [
    "FREEZE_SCHEMA",
    "FreezeError",
    "create_freeze_manifest",
    "sha256_file",
    "validate_image_digest",
    "validate_job_text",
    "verify_freeze_manifest",
]
