from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from authority.domain_scope import canonical_domain

from schema import (
    CorpusManifestV1,
    SplitManifestV1,
    read_json,
    utc_now,
    write_json_atomic,
)


def stable_order_key(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode("utf-8")).hexdigest()


def _validation(
    source_runs: Iterable[str],
    heldout_runs: Iterable[str],
    source_tasks: Iterable[str],
    heldout_tasks: Iterable[str],
    source_seed_groups: Iterable[str] = (),
    heldout_seed_groups: Iterable[str] = (),
) -> dict[str, Any]:
    source_runs = set(source_runs)
    heldout_runs = set(heldout_runs)
    source_tasks = set(source_tasks)
    heldout_tasks = set(heldout_tasks)
    source_seed_groups = set(source_seed_groups)
    heldout_seed_groups = set(heldout_seed_groups)
    return {
        "run_overlap": sorted(source_runs & heldout_runs),
        "task_overlap": sorted(source_tasks & heldout_tasks),
        "seed_group_overlap": sorted(source_seed_groups & heldout_seed_groups),
        "run_overlap_count": len(source_runs & heldout_runs),
        "task_overlap_count": len(source_tasks & heldout_tasks),
        "seed_group_overlap_count": len(source_seed_groups & heldout_seed_groups),
    }


def build_full_split(
    manifest: CorpusManifestV1,
    *,
    version: str,
    created_at: str,
) -> SplitManifestV1:
    complete = [run for run in manifest.runs if run.status == "complete"]
    excluded = [run.run_id for run in manifest.runs if run.status != "complete"]
    source_runs = sorted(run.run_id for run in complete)
    source_tasks = sorted({run.canonical_task_id for run in complete})
    return SplitManifestV1(
        split_id=f"full-{version}",
        split_kind="full",
        split_version=version,
        corpus_manifest_hash=manifest.manifest_sha256,
        created_at=created_at,
        source_run_ids=source_runs,
        heldout_run_ids=[],
        source_task_ids=source_tasks,
        heldout_task_ids=[],
        excluded_run_ids=sorted(excluded),
        validation=_validation(source_runs, [], source_tasks, []),
    ).finalize()


def build_seed_heldout_split(
    manifest: CorpusManifestV1,
    *,
    version: str,
    created_at: str,
) -> SplitManifestV1:
    complete = [run for run in manifest.runs if run.status == "complete"]
    by_task_seed: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run in complete:
        by_task_seed[run.canonical_task_id][str(run.seed)].append(run.run_id)
    source_groups: list[str] = []
    heldout_groups: list[str] = []
    source_runs: list[str] = []
    heldout_runs: list[str] = []
    excluded_runs = {
        run.run_id for run in manifest.runs if run.status != "complete"
    }
    excluded_tasks: list[str] = []
    allocations: dict[str, Any] = {}
    for task_id in sorted(by_task_seed):
        groups = by_task_seed[task_id]
        seeds = sorted(
            groups,
            key=lambda seed: (stable_order_key(task_id, seed, version), seed),
        )
        if len(seeds) < 3:
            excluded_tasks.append(task_id)
            excluded_runs.update(
                run_id for seed in seeds for run_id in groups[seed]
            )
            allocations[task_id] = {
                "seed_count": len(seeds),
                "status": "excluded_less_than_three_seeds",
            }
            continue
        source_count = int(math.ceil(2 * len(seeds) / 3))
        task_source = seeds[:source_count]
        task_heldout = seeds[source_count:]
        for seed in task_source:
            source_groups.append(f"{task_id}::{seed}")
            source_runs.extend(groups[seed])
        for seed in task_heldout:
            heldout_groups.append(f"{task_id}::{seed}")
            heldout_runs.extend(groups[seed])
        allocations[task_id] = {
            "seed_count": len(seeds),
            "source_seeds": task_source,
            "heldout_seeds": task_heldout,
        }
    source_task_set = {group.split("::", 1)[0] for group in source_groups}
    heldout_task_set = {group.split("::", 1)[0] for group in heldout_groups}
    source_tasks = sorted(source_task_set)
    # Task overlap is expected for seed-heldout and therefore reported
    # separately; seed-group overlap must remain zero.
    validation = _validation(
        source_runs,
        heldout_runs,
        source_task_set,
        heldout_task_set,
        source_groups,
        heldout_groups,
    )
    validation["task_overlap_expected"] = True
    validation["shared_task_ids"] = sorted(source_task_set & heldout_task_set)
    return SplitManifestV1(
        split_id=f"seed-heldout-{version}",
        split_kind="seed-heldout",
        split_version=version,
        corpus_manifest_hash=manifest.manifest_sha256,
        created_at=created_at,
        source_run_ids=sorted(source_runs),
        heldout_run_ids=sorted(heldout_runs),
        source_task_ids=source_tasks,
        heldout_task_ids=sorted(heldout_task_set),
        source_seed_groups=sorted(source_groups),
        heldout_seed_groups=sorted(heldout_groups),
        excluded_run_ids=sorted(excluded_runs),
        allocation={
            "rule": "sha256(task,seed,version); first ceil(2n/3) source",
            "tasks": allocations,
            "excluded_tasks": excluded_tasks,
        },
        validation=validation,
    ).finalize()


def _largest_remainder_allocation(
    tasks_by_family: dict[str, list[str]],
    target: int,
) -> dict[str, int]:
    total = sum(len(tasks) for tasks in tasks_by_family.values())
    capacity = {
        family: max(0, len(tasks) - 1)
        for family, tasks in tasks_by_family.items()
    }
    target = min(target, sum(capacity.values()))
    raw = {
        family: (target * len(tasks) / total if total else 0.0)
        for family, tasks in tasks_by_family.items()
    }
    allocation = {
        family: min(capacity[family], int(math.floor(raw[family])))
        for family in tasks_by_family
    }
    remaining = target - sum(allocation.values())
    order = sorted(
        tasks_by_family,
        key=lambda family: (
            -(raw[family] - math.floor(raw[family])),
            stable_order_key("family", family),
            family,
        ),
    )
    while remaining:
        progressed = False
        for family in order:
            if allocation[family] >= capacity[family]:
                continue
            allocation[family] += 1
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            break
    return allocation


def build_task_heldout_split(
    manifest: CorpusManifestV1,
    *,
    version: str,
    created_at: str,
    heldout_fraction: float = 0.25,
) -> SplitManifestV1:
    complete = [run for run in manifest.runs if run.status == "complete"]
    run_by_task: dict[str, list[str]] = defaultdict(list)
    family_by_task: dict[str, str] = {}
    for run in complete:
        run_by_task[run.canonical_task_id].append(run.run_id)
        family = str(run.task_family or "unknown")
        existing = family_by_task.setdefault(run.canonical_task_id, family)
        if existing != family:
            raise ValueError(
                f"Task family drift for {run.canonical_task_id}: {existing} vs {family}"
            )
    tasks_by_family: dict[str, list[str]] = defaultdict(list)
    for task_id, family in family_by_task.items():
        tasks_by_family[family].append(task_id)
    for tasks in tasks_by_family.values():
        tasks.sort()
    task_count = len(run_by_task)
    target = int(round(task_count * heldout_fraction))
    if task_count > 1 and heldout_fraction > 0:
        target = max(1, target)
    allocation = _largest_remainder_allocation(dict(tasks_by_family), target)
    heldout_tasks: set[str] = set()
    chosen_by_family: dict[str, list[str]] = {}
    for family in sorted(tasks_by_family):
        ordered = sorted(
            tasks_by_family[family],
            key=lambda task_id: (
                stable_order_key(task_id, version),
                task_id,
            ),
        )
        chosen = ordered[: allocation[family]]
        chosen_by_family[family] = chosen
        heldout_tasks.update(chosen)
    source_tasks = set(run_by_task) - heldout_tasks
    source_runs = sorted(
        run_id for task_id in source_tasks for run_id in run_by_task[task_id]
    )
    heldout_runs = sorted(
        run_id for task_id in heldout_tasks for run_id in run_by_task[task_id]
    )
    excluded_runs = sorted(
        run.run_id for run in manifest.runs if run.status != "complete"
    )
    validation = _validation(
        source_runs,
        heldout_runs,
        source_tasks,
        heldout_tasks,
    )
    validation["every_family_has_source_task"] = all(
        any(task_id in source_tasks for task_id in tasks)
        for tasks in tasks_by_family.values()
    )
    return SplitManifestV1(
        split_id=f"task-heldout-{version}",
        split_kind="task-heldout",
        split_version=version,
        corpus_manifest_hash=manifest.manifest_sha256,
        created_at=created_at,
        source_run_ids=source_runs,
        heldout_run_ids=heldout_runs,
        source_task_ids=sorted(source_tasks),
        heldout_task_ids=sorted(heldout_tasks),
        excluded_run_ids=excluded_runs,
        allocation={
            "heldout_fraction": heldout_fraction,
            "target_heldout_task_count": target,
            "actual_heldout_task_count": len(heldout_tasks),
            "family_allocations": allocation,
            "heldout_tasks_by_family": chosen_by_family,
        },
        validation=validation,
    ).finalize()


def build_same_domain_task_heldout_split(
    manifest: CorpusManifestV1,
    *,
    version: str,
    created_at: str,
    target_task_id: str,
    source_task_ids: Iterable[str] | None = None,
    target_task_family: str | None = None,
) -> SplitManifestV1:
    """Build a positive-transfer split for one unseen target task.

    The target is held out in full.  Sources must be different tasks with the
    same corpus-declared task family.  An optional explicit source allowlist is
    useful for a paper experiment where the approved source population must be
    reviewable rather than selected implicitly.
    """

    target_task_id = str(target_task_id or "").strip()
    if not target_task_id:
        raise ValueError("same-domain split requires target_task_id")
    complete = [run for run in manifest.runs if run.status == "complete"]
    run_by_task: dict[str, list[str]] = defaultdict(list)
    family_by_task: dict[str, str] = {}
    for run in complete:
        task_id = str(run.canonical_task_id)
        family = str(run.task_family or "").strip()
        run_by_task[task_id].append(run.run_id)
        existing = family_by_task.setdefault(task_id, family)
        if existing != family:
            raise ValueError(
                f"Task family drift for {task_id}: {existing} vs {family}"
            )
    if target_task_id not in run_by_task:
        raise ValueError(f"Target task has no complete runs: {target_task_id}")
    observed_target_family = family_by_task[target_task_id]
    if not observed_target_family:
        raise ValueError(f"Target task has no task family: {target_task_id}")
    if (
        target_task_family is not None
        and str(target_task_family).strip() != observed_target_family
    ):
        raise ValueError(
            "Target task family mismatch: "
            f"expected {target_task_family!r}, observed {observed_target_family!r}"
        )
    target_domain = canonical_domain(observed_target_family)
    if not target_domain:
        raise ValueError(f"Target task has no canonical domain: {target_task_id}")

    eligible_source_tasks = {
        task_id
        for task_id, family in family_by_task.items()
        if task_id != target_task_id and family == observed_target_family
    }
    if source_task_ids is None:
        selected_source_tasks = set(eligible_source_tasks)
        source_selection = "all_same_family_tasks"
    else:
        selected_source_tasks = {
            str(task_id).strip() for task_id in source_task_ids if str(task_id).strip()
        }
        source_selection = "explicit_reviewed_allowlist"
        if target_task_id in selected_source_tasks:
            raise ValueError("Target task cannot also be a source task")
        unknown = selected_source_tasks - set(run_by_task)
        if unknown:
            raise ValueError(f"Unknown same-domain source tasks: {sorted(unknown)}")
        wrong_family = selected_source_tasks - eligible_source_tasks
        if wrong_family:
            details = {
                task_id: family_by_task.get(task_id, "")
                for task_id in sorted(wrong_family)
            }
            raise ValueError(
                "Cross-domain source tasks are forbidden in a same-domain split: "
                f"{details}"
            )
    if not selected_source_tasks:
        raise ValueError(
            f"No different-task sources are available for {target_task_id} "
            f"in family {observed_target_family!r}"
        )

    source_runs = sorted(
        run_id
        for task_id in selected_source_tasks
        for run_id in run_by_task[task_id]
    )
    heldout_runs = sorted(run_by_task[target_task_id])
    included_runs = set(source_runs) | set(heldout_runs)
    excluded_runs = sorted(
        run.run_id for run in manifest.runs if run.run_id not in included_runs
    )
    validation = _validation(
        source_runs,
        heldout_runs,
        selected_source_tasks,
        [target_task_id],
    )
    source_families = {
        family_by_task[task_id] for task_id in selected_source_tasks
    }
    source_domains = {
        canonical_domain(family) for family in source_families if family
    }
    validation.update(
        {
            "transfer_design": "same_domain_different_task_task_heldout",
            "target_task_fully_heldout": set(heldout_runs)
            == set(run_by_task[target_task_id]),
            "target_task_absent_from_source": target_task_id
            not in selected_source_tasks,
            "all_sources_have_target_task_family": source_families
            == {observed_target_family},
            "all_sources_have_target_domain": source_domains == {target_domain},
            "cross_domain_source_run_count": sum(
                canonical_domain(family_by_task[task_id]) != target_domain
                for task_id in selected_source_tasks
                for _run_id in run_by_task[task_id]
            ),
            "source_task_count": len(selected_source_tasks),
            "heldout_task_count": 1,
        }
    )
    return SplitManifestV1(
        split_id=f"same-domain-task-heldout-{target_task_id}-{version}",
        split_kind="same-domain-task-heldout",
        split_version=version,
        corpus_manifest_hash=manifest.manifest_sha256,
        created_at=created_at,
        source_run_ids=source_runs,
        heldout_run_ids=heldout_runs,
        source_task_ids=sorted(selected_source_tasks),
        heldout_task_ids=[target_task_id],
        excluded_run_ids=excluded_runs,
        allocation={
            "rule": "same corpus task_family; target task fully held out",
            "transfer_design": "same_domain_different_task_task_heldout",
            "source_selection": source_selection,
            "target_task_id": target_task_id,
            "target_task_family": observed_target_family,
            "target_domain": target_domain,
            "source_task_ids": sorted(selected_source_tasks),
            "source_task_families": sorted(source_families),
            "source_domains": sorted(source_domains),
        },
        validation=validation,
    ).finalize()


def build_splits(
    manifest: CorpusManifestV1,
    *,
    version: str,
    created_at: str | None = None,
    task_heldout_fraction: float = 0.25,
    same_domain_target_task_id: str | None = None,
    same_domain_source_task_ids: Iterable[str] | None = None,
    same_domain_target_task_family: str | None = None,
) -> dict[str, SplitManifestV1]:
    created = created_at or utc_now()
    splits = {
        "full": build_full_split(manifest, version=version, created_at=created),
        "seed-heldout": build_seed_heldout_split(
            manifest, version=version, created_at=created
        ),
        "task-heldout": build_task_heldout_split(
            manifest,
            version=version,
            created_at=created,
            heldout_fraction=task_heldout_fraction,
        ),
    }
    if same_domain_target_task_id:
        splits["same-domain-task-heldout"] = (
            build_same_domain_task_heldout_split(
                manifest,
                version=version,
                created_at=created,
                target_task_id=same_domain_target_task_id,
                source_task_ids=same_domain_source_task_ids,
                target_task_family=same_domain_target_task_family,
            )
        )
    elif same_domain_source_task_ids:
        raise ValueError(
            "same_domain_source_task_ids requires same_domain_target_task_id"
        )
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic full/seed-heldout/task-heldout splits and, "
            "optionally, an explicit same-domain different-task split."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-version", default="v1")
    parser.add_argument("--task-heldout-fraction", type=float, default=0.25)
    parser.add_argument("--same-domain-target-task")
    parser.add_argument("--same-domain-target-family")
    parser.add_argument(
        "--same-domain-source-task",
        action="append",
        dest="same_domain_source_tasks",
        help="Repeat for each reviewed source task allowed into the domain split.",
    )
    parser.add_argument("--created-at")
    args = parser.parse_args()
    if not 0 <= args.task_heldout_fraction < 1:
        raise ValueError("task-heldout-fraction must be in [0, 1)")
    manifest = CorpusManifestV1.from_dict(read_json(args.manifest))
    splits = build_splits(
        manifest,
        version=args.split_version,
        created_at=args.created_at,
        task_heldout_fraction=args.task_heldout_fraction,
        same_domain_target_task_id=args.same_domain_target_task,
        same_domain_source_task_ids=args.same_domain_source_tasks,
        same_domain_target_task_family=args.same_domain_target_family,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "full": "full.json",
        "seed-heldout": "seed-heldout.json",
        "task-heldout": "task-heldout.json",
        "same-domain-task-heldout": "same-domain-task-heldout.json",
    }
    for key, split in splits.items():
        write_json_atomic(args.output_dir / filenames[key], split.as_dict())
    print(
        json.dumps(
            {
                key: {
                    "split_id": split.split_id,
                    "manifest_sha256": split.manifest_sha256,
                    "source_runs": len(split.source_run_ids),
                    "heldout_runs": len(split.heldout_run_ids),
                }
                for key, split in splits.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
