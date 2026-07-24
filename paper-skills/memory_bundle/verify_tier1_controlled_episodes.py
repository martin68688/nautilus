from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping

from build_tier1_controlled_episodes import (
    AGENT_SEEDS,
    CONDITIONS,
    DEFAULT_LEGACY_EPISODE_ROOT,
    HIDDEN_AGENT_KEYS,
    MANIFEST_SCHEMA,
    audit_legacy_overlap,
    project_agent_view,
    validate_episodes,
)
from schema import sha256_json


VERIFICATION_SCHEMA = "decision_admissibility_tier1_episode_verification_v1"
DEFAULT_BUILDER_SOURCE = Path(__file__).resolve().with_name(
    "build_tier1_controlled_episodes.py"
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"Non-object JSONL row at {path}:{line_number}")
        rows.append(row)
    return rows


def _walk_keys(value: Any):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def verify_packet(
    packet_root: str | Path,
    *,
    legacy_episode_root: str | Path = DEFAULT_LEGACY_EPISODE_ROOT,
    builder_source: str | Path = DEFAULT_BUILDER_SOURCE,
) -> dict[str, Any]:
    root = Path(packet_root).resolve()
    manifest_path = root / "manifest.json"
    errors: list[str] = []
    try:
        manifest = _read_json(manifest_path)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as error:
        manifest = {}
        errors.append(f"manifest_read:{type(error).__name__}")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        errors.append("manifest_schema")
    observed_manifest_hash = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    if manifest.get("manifest_hash") != observed_manifest_hash:
        errors.append("manifest_hash")

    episode_name = str(manifest.get("episode_file") or "episodes.jsonl")
    episode_path = root / episode_name
    try:
        observed_episode_hash = _sha256_file(episode_path)
        episodes = _read_jsonl(episode_path)
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as error:
        observed_episode_hash = ""
        episodes = []
        errors.append(f"episode_read:{type(error).__name__}")
    if manifest.get("episode_file_sha256") != observed_episode_hash:
        errors.append("episode_file_hash")

    builder_path = Path(builder_source).resolve()
    try:
        observed_builder_hash = _sha256_file(builder_path)
    except (FileNotFoundError, OSError) as error:
        observed_builder_hash = ""
        errors.append(f"builder_source_read:{type(error).__name__}")
    if manifest.get("builder_source_sha256") != observed_builder_hash:
        errors.append("builder_source_hash")

    validation = validate_episodes(episodes)
    if not validation["valid"]:
        errors.extend(f"episode_validation:{value}" for value in validation["errors"])
    if manifest.get("validation") != validation:
        errors.append("manifest_validation_binding")

    try:
        legacy_audit = audit_legacy_overlap(episodes, legacy_episode_root)
    except (FileNotFoundError, OSError, ValueError) as error:
        legacy_audit = {}
        errors.append(f"legacy_audit:{type(error).__name__}")
    if not legacy_audit.get("passed"):
        errors.append("legacy_overlap")
    if manifest.get("legacy_overlap_audit") != legacy_audit:
        errors.append("manifest_legacy_audit_binding")

    view_hashes: list[str] = []
    gold_leaks: list[str] = []
    projection_errors: list[str] = []
    for episode in episodes:
        episode_id = str(episode.get("episode_id") or "")
        for condition in CONDITIONS:
            for seed in AGENT_SEEDS:
                view_ref = f"{episode_id}:{condition}:{seed}"
                try:
                    view = project_agent_view(
                        episode,
                        condition=condition,
                        agent_seed=seed,
                    )
                except (KeyError, TypeError, ValueError) as error:
                    projection_errors.append(f"{view_ref}:{type(error).__name__}")
                    continue
                leaked = sorted(set(_walk_keys(view)) & HIDDEN_AGENT_KEYS)
                if leaked:
                    gold_leaks.append(f"{view_ref}:{','.join(leaked)}")
                expected_view_hash = sha256_json(
                    {key: value for key, value in view.items() if key != "view_hash"}
                )
                if view.get("view_hash") != expected_view_hash:
                    projection_errors.append(f"{view_ref}:hash")
                view_hashes.append(str(view.get("view_hash") or ""))
    expected_view_count = len(episodes) * len(CONDITIONS) * len(AGENT_SEEDS)
    expected_unique_view_count = len(episodes) * len(CONDITIONS)
    if len(view_hashes) != expected_view_count:
        errors.append("agent_view_count")
    if len(set(view_hashes)) != expected_unique_view_count:
        errors.append("agent_view_pairing")
    if gold_leaks:
        errors.append("agent_view_gold_leak")
    if projection_errors:
        errors.append("agent_view_projection")

    packet_modes = {}
    for name, path in (("manifest", manifest_path), ("episodes", episode_path)):
        try:
            packet_modes[name] = oct(stat.S_IMODE(path.stat().st_mode))
        except OSError:
            packet_modes[name] = ""
    if packet_modes != {"manifest": "0o444", "episodes": "0o444"}:
        errors.append("packet_file_modes")

    report: dict[str, Any] = {
        "schema": VERIFICATION_SCHEMA,
        "packet_root_name": root.name,
        "manifest_file_sha256": (
            _sha256_file(manifest_path) if manifest_path.is_file() else ""
        ),
        "manifest_hash_observed": observed_manifest_hash,
        "episode_file_sha256_observed": observed_episode_hash,
        "builder_source_sha256_observed": observed_builder_hash,
        "legacy_audit_hash_observed": legacy_audit.get("audit_hash", ""),
        "packet_file_modes": packet_modes,
        "episode_validation": validation,
        "agent_view_count": len(view_hashes),
        "expected_agent_view_count": expected_view_count,
        "expected_unique_agent_view_hash_count": expected_unique_view_count,
        "unique_agent_view_hash_count": len(set(view_hashes)),
        "agent_view_gold_leak_count": len(gold_leaks),
        "agent_view_projection_error_count": len(projection_errors),
        "verified": not errors,
        "errors": sorted(set(errors)),
        "verifier_source_sha256": _sha256_file(Path(__file__).resolve()),
        "verification_hash": "",
    }
    report["verification_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "verification_hash"}
    )
    return report


def write_verification_exclusive(path: str | Path, report: Mapping[str, Any]) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(report), sort_keys=True, ensure_ascii=False, indent=2))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a held-out WP8 Tier-1 controlled-episode packet."
    )
    parser.add_argument("--packet-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--legacy-episode-root",
        type=Path,
        default=DEFAULT_LEGACY_EPISODE_ROOT,
    )
    parser.add_argument(
        "--builder-source",
        type=Path,
        default=DEFAULT_BUILDER_SOURCE,
    )
    args = parser.parse_args()
    report = verify_packet(
        args.packet_root,
        legacy_episode_root=args.legacy_episode_root,
        builder_source=args.builder_source,
    )
    if args.output is not None:
        write_verification_exclusive(args.output, report)
    print(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2))
    if not report["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
