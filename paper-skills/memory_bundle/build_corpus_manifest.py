from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from schema import (
    CorpusManifestV1,
    CorpusRunEntry,
    read_json,
    sha256_file,
    sha256_json,
    utc_now,
    write_json_atomic,
)


DEFAULT_TASK_TAGS = (
    Path(__file__).resolve().parents[2]
    / "mlevolve"
    / "engine"
    / "coldstart"
    / "competition_tag_classified.json"
)
CORE_ARTIFACTS = {
    "journal": Path("logs/journal.json"),
    "config": Path("logs/config.yaml"),
    "filtered_journal": Path("logs/filtered_journal.json"),
    "best_solution": Path("logs/best_solution.py"),
}
TASK_KEYS = ("exp_id", "competition_id", "task_id", "dataset_id")
SEED_KEYS = ("seed", "random_seed", "agent_seed")
SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*"
        r"[\"']?(?!\$\{oc\.env:)[A-Za-z0-9_./+-]{20,}"
    ),
]


def canonical_task_id(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    text = text.replace("_", "-").replace(" ", "-")
    text = re.sub(r"[^a-z0-9.-]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-") or "unknown"


def _find_nested(payload: Any, keys: Iterable[str]) -> Any:
    wanted = set(keys)
    queue = [payload]
    while queue:
        value = queue.pop(0)
        if isinstance(value, Mapping):
            for key in keys:
                if key in value:
                    candidate = value[key]
                    if candidate is not None and candidate != "":
                        return candidate
            queue.extend(value[key] for key in sorted(value) if key not in wanted)
        elif isinstance(value, list):
            queue.extend(value)
    return None


def load_config_metadata(path: Path) -> tuple[str, str, list[str]]:
    warnings: list[str] = []
    if not path.exists():
        return "unknown", "unknown", ["missing_config"]
    try:
        # Persisted OmegaConf snapshots contain Python object tags for Path
        # values.  We only need scalar metadata, so BaseLoader is deliberate:
        # it treats every value as inert YAML data and never constructs the
        # tagged Python objects.
        payload = yaml.load(
            path.read_text(encoding="utf-8", errors="replace"),
            Loader=yaml.BaseLoader,
        )
    except Exception as error:
        return "unknown", "unknown", [f"invalid_config:{type(error).__name__}"]
    task = _find_nested(payload, TASK_KEYS)
    seed = _find_nested(payload, SEED_KEYS)
    if task in {None, ""}:
        warnings.append("task_id_unavailable")
    if seed in {None, ""}:
        warnings.append("seed_unavailable")
    return str(task or "unknown"), str(seed or "unknown"), warnings


def config_has_secret_like_material(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def journal_nodes(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and "nodes" in payload:
        payload = payload["nodes"]
    if isinstance(payload, dict):
        payload = list(payload.values())
    if not isinstance(payload, list):
        return []
    return [dict(node) for node in payload if isinstance(node, Mapping)]


def metric_has_value(node: Mapping[str, Any]) -> bool:
    metric = node.get("metric")
    if isinstance(metric, Mapping):
        return metric.get("value") is not None
    return metric is not None


def discover_run_dirs(root: Path) -> list[Path]:
    root = root.resolve()
    candidates: set[Path] = set()
    for logs_dir in root.rglob("logs"):
        if logs_dir.is_dir() and not logs_dir.is_symlink():
            candidates.add(logs_dir.parent.resolve())
    for child in root.iterdir():
        if not child.is_dir() or child.is_symlink() or child.name.startswith("."):
            continue
        if (child / "logs").exists() or any(item.is_file() for item in child.iterdir()):
            candidates.add(child.resolve())
    return sorted(candidates, key=lambda path: path.relative_to(root).as_posix())


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _task_family(tags: Mapping[str, Any], canonical: str) -> str:
    exact = tags.get(canonical)
    if exact is not None:
        return str(exact)
    for task_id, family in tags.items():
        if canonical_task_id(task_id) == canonical:
            return str(family)
    return "unknown"


def _unique_run_id(
    root: Path,
    run_dir: Path,
    observed: set[str],
) -> str:
    candidate = run_dir.name
    if candidate not in observed:
        observed.add(candidate)
        return candidate
    relative = _relative(root, run_dir)
    candidate = f"{run_dir.name}-{hashlib.sha256(relative.encode()).hexdigest()[:10]}"
    if candidate in observed:
        raise ValueError(f"Unable to derive unique run ID for {relative}")
    observed.add(candidate)
    return candidate


def inspect_run(
    root: Path,
    run_dir: Path,
    *,
    task_tags: Mapping[str, Any],
    excluded_tasks: set[str],
    aborted_run_ids: set[str],
    observed_run_ids: set[str],
) -> CorpusRunEntry:
    run_id = _unique_run_id(root, run_dir, observed_run_ids)
    paths = {name: run_dir / relative for name, relative in CORE_ARTIFACTS.items()}
    task_id, seed, warnings = load_config_metadata(paths["config"])
    canonical = canonical_task_id(task_id)
    nodes: list[dict[str, Any]] = []
    journal_error = ""
    if paths["journal"].exists():
        try:
            with paths["journal"].open("r", encoding="utf-8") as handle:
                nodes = journal_nodes(json.load(handle))
        except Exception as error:
            journal_error = type(error).__name__
            warnings.append(f"invalid_journal:{journal_error}")
    else:
        warnings.append("missing_journal")

    artifact_hashes = {
        name: sha256_file(path)
        for name, path in paths.items()
        if path.exists() and path.is_file()
    }
    code_nodes = [node for node in nodes if str(node.get("code") or "").strip()]
    metric_nodes = [node for node in nodes if metric_has_value(node)]
    exclusion_reason = ""
    status = "complete"
    marker_aborted = any(
        (run_dir / name).exists()
        for name in ("ABORTED_SECRET_UNSAFE", ".aborted_secret_unsafe")
    )
    config_secret_detected = config_has_secret_like_material(paths["config"])
    if canonical in excluded_tasks or "spooky" in canonical:
        status = "excluded"
        exclusion_reason = "excluded_task"
    elif run_id in aborted_run_ids or marker_aborted or config_secret_detected:
        status = "excluded"
        exclusion_reason = (
            "secret_material_detected"
            if config_secret_detected
            else "aborted_secret_unsafe"
        )
    elif journal_error:
        status = "invalid_json"
        exclusion_reason = f"invalid_journal:{journal_error}"
    elif not paths["journal"].exists() or not paths["config"].exists() or not nodes:
        status = "partial"
        if not paths["config"].exists():
            warnings.append("missing_config")
        if paths["journal"].exists() and not nodes:
            warnings.append("empty_journal")
        exclusion_reason = "incomplete_core_artifacts"
    if not paths["best_solution"].exists():
        warnings.append("missing_best_solution")
    elif paths["best_solution"].stat().st_size == 0:
        warnings.append("empty_best_solution")
    return CorpusRunEntry(
        run_id=run_id,
        task_id=task_id,
        canonical_task_id=canonical,
        task_family=_task_family(task_tags, canonical),
        seed=seed,
        status=status,
        journal_path=(
            _relative(root, paths["journal"])
            if paths["journal"].exists()
            else None
        ),
        config_path=(
            _relative(root, paths["config"])
            if paths["config"].exists()
            else None
        ),
        filtered_journal_path=(
            _relative(root, paths["filtered_journal"])
            if paths["filtered_journal"].exists()
            else None
        ),
        best_solution_path=(
            _relative(root, paths["best_solution"])
            if paths["best_solution"].exists()
            else None
        ),
        artifact_hashes=dict(sorted(artifact_hashes.items())),
        node_count=len(nodes),
        code_node_count=len(code_nodes),
        metric_node_count=len(metric_nodes),
        source_relpath=_relative(root, run_dir),
        exclusion_reason=exclusion_reason,
        warnings=sorted(set(warnings)),
    )


def actual_snapshot(runs: list[CorpusRunEntry]) -> dict[str, Any]:
    included = [run for run in runs if run.status == "complete"]
    return {
        "run_directory_count": len(runs),
        "status_counts": dict(
            sorted(collections.Counter(run.status for run in runs).items())
        ),
        "complete_run_count": len(included),
        "complete_non_spooky_task_count": len(
            {run.canonical_task_id for run in included}
        ),
        "node_count": sum(run.node_count for run in included),
        "code_node_count": sum(run.code_node_count for run in included),
        "metric_node_count": sum(run.metric_node_count for run in included),
        "task_run_counts": dict(
            sorted(collections.Counter(run.canonical_task_id for run in included).items())
        ),
    }


def drift_report(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> dict[str, Any]:
    differences = {
        key: {"expected": expected[key], "actual": actual.get(key)}
        for key in sorted(expected)
        if actual.get(key) != expected[key]
    }
    return {
        "schema": "corpus_inventory_report_v1",
        "expected_snapshot": dict(expected),
        "actual_snapshot": dict(actual),
        "drift_detected": bool(differences),
        "differences": differences,
    }


def build_manifest(
    runs_root: str | Path,
    *,
    source_repo: str,
    source_commit: str,
    excluded_tasks: Iterable[str] = ("spooky-author-identification",),
    aborted_run_ids: Iterable[str] = (),
    expected_snapshot: Mapping[str, Any] | None = None,
    task_tags_path: str | Path = DEFAULT_TASK_TAGS,
    created_at: str | None = None,
) -> tuple[CorpusManifestV1, dict[str, Any]]:
    root = Path(runs_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Runs root not found: {root}")
    tags_payload = read_json(task_tags_path) if Path(task_tags_path).exists() else {}
    if not isinstance(tags_payload, Mapping):
        raise ValueError("Task tag file must contain an object")
    excluded = {canonical_task_id(value) for value in excluded_tasks}
    observed: set[str] = set()
    runs = [
        inspect_run(
            root,
            run_dir,
            task_tags=tags_payload,
            excluded_tasks=excluded,
            aborted_run_ids={str(value) for value in aborted_run_ids},
            observed_run_ids=observed,
        )
        for run_dir in discover_run_dirs(root)
    ]
    runs.sort(key=lambda run: run.run_id)
    actual = actual_snapshot(runs)
    expected = dict(expected_snapshot or {})
    identity = {
        "source_commit": source_commit,
        "runs": [
            {
                "run_id": run.run_id,
                "status": run.status,
                "artifact_hashes": run.artifact_hashes,
            }
            for run in runs
        ],
    }
    corpus_id = f"mlevolve-{source_commit[:12]}-{sha256_json(identity)[:16]}"
    manifest = CorpusManifestV1(
        corpus_id=corpus_id,
        created_at=created_at or utc_now(),
        source_repo=source_repo,
        source_commit=source_commit,
        source_root=str(root),
        exclusion_rules=[
            {"kind": "task", "canonical_task_ids": sorted(excluded)},
            {"kind": "aborted_secret_unsafe", "run_ids": sorted(aborted_run_ids)},
        ],
        runs=runs,
        expected_snapshot=expected,
        actual_snapshot=actual,
    ).finalize()
    report = drift_report(expected, actual)
    report.update(
        {
            "corpus_id": corpus_id,
            "manifest_sha256": manifest.manifest_sha256,
            "source_root": str(root),
            "source_commit": source_commit,
        }
    )
    return manifest, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a read-only, hash-complete MLEvolve corpus manifest."
    )
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--source-repo", default="third_party/MLEvolve")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--exclude-task", action="append", default=[])
    parser.add_argument("--aborted-run-id", action="append", default=[])
    parser.add_argument("--expected-snapshot", type=Path)
    parser.add_argument("--task-tags", type=Path, default=DEFAULT_TASK_TAGS)
    parser.add_argument("--created-at")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    expected = read_json(args.expected_snapshot) if args.expected_snapshot else {}
    excluded = args.exclude_task or ["spooky-author-identification"]
    manifest, report = build_manifest(
        args.runs_root,
        source_repo=args.source_repo,
        source_commit=args.source_commit,
        excluded_tasks=excluded,
        aborted_run_ids=args.aborted_run_id,
        expected_snapshot=expected,
        task_tags_path=args.task_tags,
        created_at=args.created_at,
    )
    write_json_atomic(args.output, manifest.as_dict())
    write_json_atomic(args.report, report)
    print(
        json.dumps(
            {
                "manifest": str(args.output),
                "report": str(args.report),
                "corpus_id": manifest.corpus_id,
                "manifest_sha256": manifest.manifest_sha256,
                "actual_snapshot": manifest.actual_snapshot,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
