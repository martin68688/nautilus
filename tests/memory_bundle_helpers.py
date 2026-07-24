from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[1]
MEMORY_BUNDLE = REPO / "paper-skills" / "memory_bundle"
DISTILLATION = REPO / "paper-skills" / "distillation"
for path in (MEMORY_BUNDLE, DISTILLATION):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_corpus import audit_manifest
from bind_sop_clauses import bind
from build_corpus_manifest import build_manifest
from build_memory_bundle import build_bundle
from build_split_manifests import build_splits
from distill_sop_clauses import distill, request_payload
from extract_branches import extract_branches
from merge_sop_clauses import merge
from runforest_v2 import build_runforest_v2
from schema import sha256_json, write_json_atomic


CREATED_AT = "2026-07-19T00:00:00Z"
TASKS = {
    "task-a": "family-a",
    "task-b": "family-a",
    "task-c": "family-b",
    "task-d": "family-b",
}


def write_run(
    runs_root: Path,
    *,
    task_id: str,
    seed: int,
    invalid_json: bool = False,
    partial: bool = False,
) -> Path:
    run_id = f"run-{task_id}-seed-{seed}"
    run_dir = runs_root / run_id
    logs = run_dir / "logs"
    logs.mkdir(parents=True)
    (logs / "config.yaml").write_text(
        yaml.safe_dump({"exp_id": task_id, "seed": seed}, sort_keys=True),
        encoding="utf-8",
    )
    if not partial:
        if invalid_json:
            (logs / "journal.json").write_text("{invalid", encoding="utf-8")
        else:
            nodes = [
                {
                    "id": "n0",
                    "branch_id": 1,
                    "step": 0,
                    "stage": "draft",
                    "plan": "fit a clean baseline",
                    "code_summary": "baseline",
                    "analysis": "executed",
                    "code": "x_train = [1, 2, 3]\nprint(len(x_train))",
                    "metric": {"value": 0.4, "maximize": True},
                    "is_buggy": False,
                    "is_valid": True,
                },
                {
                    "id": "n1",
                    "parent_id": "n0",
                    "branch_id": 1,
                    "step": 1,
                    "stage": "debug",
                    "plan": "align OOF predictions by sample_id",
                    "code_summary": "OOF alignment repair",
                    "analysis": "repair executed",
                    "code": "oof = dict(zip([1, 2, 3], [0.1, 0.2, 0.3]))\nprint(oof)",
                    "metric": {"value": 0.5, "maximize": True},
                    "is_buggy": False,
                    "is_valid": True,
                },
            ]
            (logs / "journal.json").write_text(
                json.dumps({"nodes": nodes}), encoding="utf-8"
            )
            (logs / "filtered_journal.json").write_text(
                json.dumps({"nodes": nodes}), encoding="utf-8"
            )
            (logs / "best_solution.py").write_text(
                "print('best')\n", encoding="utf-8"
            )
    return run_dir


def prepare_corpus(tmp_path: Path) -> dict[str, Any]:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    for task_id in TASKS:
        for seed in (1, 2, 3):
            write_run(runs_root, task_id=task_id, seed=seed)
    write_run(runs_root, task_id="spooky-author-identification", seed=1)
    write_run(runs_root, task_id="partial-task", seed=1, partial=True)
    write_run(runs_root, task_id="invalid-task", seed=1, invalid_json=True)
    task_tags = tmp_path / "task_tags.json"
    task_tags.write_text(
        json.dumps({**TASKS, "spooky-author-identification": "nlp"}),
        encoding="utf-8",
    )
    manifest, inventory = build_manifest(
        runs_root,
        source_repo="third_party/MLEvolve",
        source_commit="be034ec81d58e96ca333abb7bda155726aaa3668",
        excluded_tasks=["spooky-author-identification"],
        expected_snapshot={"complete_run_count": 12},
        task_tags_path=task_tags,
        created_at=CREATED_AT,
    )
    manifest_path = tmp_path / "corpus_manifest.json"
    report_path = tmp_path / "inventory_report.json"
    write_json_atomic(manifest_path, manifest.as_dict())
    write_json_atomic(report_path, inventory)
    drift_review_path = tmp_path / "corpus_drift_review.json"
    write_json_atomic(
        drift_review_path,
        {
            "schema": "corpus_drift_review_v1",
            "reviewed": True,
            "reviewed_by": "synthetic-test",
            "reviewed_at": CREATED_AT,
            "corpus_manifest_hash": manifest.manifest_sha256,
            "actual_snapshot_hash": sha256_json(manifest.actual_snapshot),
            "excluded_runs_reviewed": True,
            "disposition": "accept_synthetic_fixture",
        },
    )
    return {
        "runs_root": runs_root,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "inventory": inventory,
        "inventory_path": report_path,
        "drift_review_path": drift_review_path,
        "task_tags": task_tags,
    }


def prepare_audit_and_splits(tmp_path: Path, corpus: dict[str, Any]) -> dict[str, Any]:
    audit_dir = tmp_path / "audit_sidecars"
    audit_report = audit_manifest(
        corpus["manifest_path"],
        audit_dir,
        active_protocol_ref="test-protocol@1#test-hash",
        generated_at=CREATED_AT,
    )
    splits = build_splits(
        corpus["manifest"],
        version="v1",
        created_at=CREATED_AT,
        same_domain_target_task_id="task-a",
        same_domain_source_task_ids=["task-b"],
        same_domain_target_task_family="family-a",
    )
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    filenames = {
        "full": "full.json",
        "seed-heldout": "seed-heldout.json",
        "task-heldout": "task-heldout.json",
        "same-domain-task-heldout": "same-domain-task-heldout.json",
    }
    split_paths = {}
    for key, split in splits.items():
        split_path = splits_dir / filenames[key]
        write_json_atomic(split_path, split.as_dict())
        split_paths[key] = split_path
    return {
        "audit_dir": audit_dir,
        "audit_report": audit_report,
        "splits": splits,
        "splits_dir": splits_dir,
        "split_paths": split_paths,
    }


def prepare_sops(
    tmp_path: Path,
    corpus: dict[str, Any],
    audit_splits: dict[str, Any],
) -> dict[str, Any]:
    traces_dir = tmp_path / "traces"
    trace_manifest = extract_branches(
        corpus["manifest_path"],
        traces_dir,
        split_manifest_path=audit_splits["split_paths"]["full"],
        audit_dir=audit_splits["audit_dir"],
        created_at=CREATED_AT,
    )
    frozen: dict[str, Any] = {}
    for trace in trace_manifest["traces"]:
        trace_path = traces_dir / trace["path"]
        request = request_payload(
            trace,
            trace_path.read_text(encoding="utf-8"),
            model="deepseek-test",
            temperature=0.0,
        )
        child = trace["refs"][-1]
        refs = [child["node_ref"], child["transition_ref"]]
        frozen[request["request_id"]] = {
            "sop_containers": [
                {
                    "title": f"OOF repair {trace['task_id']}",
                    "clauses": [
                        {
                            "text": "Align OOF predictions by sample_id.",
                            "retrieval_text": "Align OOF predictions by sample_id; score 0.92.",
                            "claim_type_proposal": "debug_repair",
                            "source_refs": refs,
                            "evidence_refs": [child["node_ref"]],
                            "applies_when": ["OOF indexing fails"],
                            "prevents": ["row misalignment"],
                            "publication_class_proposal": "diagnostic",
                        },
                        {
                            "text": "Historical selection read test labels.",
                            "retrieval_text": "Test-label leakage warning; metric 0.92.",
                            "claim_type_proposal": "audit_finding",
                            "source_refs": refs,
                            "evidence_refs": [child["node_ref"]],
                            "applies_when": ["inspecting history"],
                            "prevents": ["leakage"],
                            "publication_class_proposal": "diagnostic",
                        },
                        {
                            "text": "Historical contaminated score 0.92.",
                            "retrieval_text": "Historical contaminated score 0.92.",
                            "claim_type_proposal": "score",
                            "source_refs": refs,
                            "evidence_refs": [child["node_ref"]],
                            "applies_when": ["inspect only"],
                            "prevents": [],
                            "publication_class_proposal": "certified",
                        },
                    ],
                }
            ]
        }
    frozen_path = tmp_path / "frozen_responses.json"
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    distill_dir = tmp_path / "distillation"
    distillation_report = distill(
        traces_dir / "trace_manifest.json",
        traces_dir,
        distill_dir,
        frozen_responses_path=frozen_path,
        model="deepseek-test",
        temperature=0.0,
        created_at=CREATED_AT,
    )
    binder_dir = tmp_path / "binder"
    binder_report = bind(
        distill_dir / "proposals.jsonl",
        traces_dir / "trace_manifest.json",
        binder_dir,
        active_protocol_ref="test-protocol@1#test-hash",
        created_at=CREATED_AT,
    )
    merge_dir = tmp_path / "merged"
    merge_report = merge(
        binder_dir / "clauses.jsonl",
        binder_dir / "containers.json",
        merge_dir,
        threshold=0.6,
        created_at=CREATED_AT,
    )
    return {
        "traces_dir": traces_dir,
        "trace_manifest": trace_manifest,
        "frozen_path": frozen_path,
        "distill_dir": distill_dir,
        "distillation_report": distillation_report,
        "binder_dir": binder_dir,
        "binder_report": binder_report,
        "merge_dir": merge_dir,
        "merge_report": merge_report,
    }


def prepare_runforest_and_bundle(
    tmp_path: Path,
    corpus: dict[str, Any],
    audit_splits: dict[str, Any],
    sops: dict[str, Any],
    *,
    split_name: str = "task-heldout",
) -> dict[str, Any]:
    runforest_dir = tmp_path / f"runforest-{split_name}"
    runforest_report = build_runforest_v2(
        corpus["manifest_path"],
        audit_splits["audit_dir"],
        sops["merge_dir"] / "clauses.jsonl",
        sops["merge_dir"] / "containers.json",
        audit_splits["split_paths"][split_name],
        runforest_dir,
        bundle_id=f"bundle-{split_name}",
        created_at=CREATED_AT,
    )
    protocol_dir = tmp_path / "protocols"
    protocol_dir.mkdir(exist_ok=True)
    (protocol_dir / "test-protocol.json").write_text(
        json.dumps(
            {
                "protocol_id": "test-protocol",
                "version": "1",
                "task_profile": {},
                "data_split_policy": {},
                "preprocessing_policy": {},
                "evaluator_spec": {},
                "metric_spec": {},
                "selection_policy": {},
                "seed_policy": {},
                "holdout_policy": {},
                "promotion_policy": {},
                "compatibility_rules": {},
            }
        ),
        encoding="utf-8",
    )
    bundle_dir = tmp_path / f"bundle-{split_name}"
    result = build_bundle(
        corpus["manifest_path"],
        corpus["drift_review_path"],
        audit_splits["split_paths"][split_name],
        audit_splits["audit_dir"],
        runforest_dir,
        protocol_dir,
        bundle_dir,
        bundle_id=f"bundle-{split_name}",
        bundle_version="v1",
        authority_policy_version="authority_v1",
        detector_version=audit_splits["audit_report"]["detector_version"],
        deepseek_model="deepseek-test",
        deepseek_prompt_hash=sops["distillation_report"]["system_prompt_hash"],
        authority_dir=sops["binder_dir"],
        splits_dir=audit_splits["splits_dir"],
        created_at=CREATED_AT,
    )
    return {
        "runforest_dir": runforest_dir,
        "runforest_report": runforest_report,
        "protocol_dir": protocol_dir,
        "bundle_dir": bundle_dir,
        "bundle_result": result,
    }


def prepare_pipeline(tmp_path: Path, *, split_name: str = "task-heldout") -> dict[str, Any]:
    corpus = prepare_corpus(tmp_path)
    audit_splits = prepare_audit_and_splits(tmp_path, corpus)
    sops = prepare_sops(tmp_path, corpus, audit_splits)
    final = prepare_runforest_and_bundle(
        tmp_path,
        corpus,
        audit_splits,
        sops,
        split_name=split_name,
    )
    return {**corpus, **audit_splits, **sops, **final}
