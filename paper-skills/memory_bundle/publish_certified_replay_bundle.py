from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import stat
import sys
from pathlib import Path
from typing import Any, Mapping


REPO = Path(__file__).resolve().parents[2]
MLEVOLVE = REPO / "mlevolve"
if str(MLEVOLVE) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE))

from agents.memory.sop_visibility_gateway import SOPVisibilityGateway  # noqa: E402
from authority.authority_engine import AuthorityEngine  # noqa: E402
from authority.bundle_authority import load_snapshot_authority  # noqa: E402
from authority.certified_bundle import CertifiedBundlePublisher  # noqa: E402
from authority.clean_replay import (  # noqa: E402
    ReplayAuthorityRecovery,
    load_replay_queue,
)
from authority.clean_replay_runner import (  # noqa: E402
    validate_replay_execution_attempt,
)
from authority.domain_scope import (  # noqa: E402
    SAME_DOMAIN,
    canonical_domain,
    normalize_transfer_scope,
)
from authority.memory_snapshot import (  # noqa: E402
    ImmutableBaseBundle,
    MemorySnapshotLoader,
    SessionOverlay,
    make_current_pointer,
    sha256_file,
    sha256_json,
    write_json_atomic,
)
from authority.models import (  # noqa: E402
    ClaimType,
    GenerationStage,
    GovernanceStage,
    Operation,
    SOPClauseV1,
    TaskContext,
    VisibilityRequest,
)
from authority.protocol_registry import ProtocolRegistry  # noqa: E402
from authority.replay_certifier import ReplayIdentity  # noqa: E402
from authority.replay_clause_publication import (  # noqa: E402
    ReplayClausePublication,
)
from method_claim_purity import (  # noqa: E402
    audit_method_claim_semantic_purity,
    require_method_claim_semantic_purity,
)
from validate_clean_replay_execution import VALIDATION_SCHEMA  # noqa: E402
from validate_memory_bundle import validate_bundle  # noqa: E402


PUBLICATION_PROVENANCE_SCHEMA = "certified_replay_publication_provenance_v2"
PUBLICATION_VISIBILITY_SCHEMA = "certified_replay_target_visibility_v2"
VISIBLE_METHOD_PURITY_SCHEMA = "certified_replay_visible_method_purity_v1"
SOURCE_SNAPSHOT_SCHEMA = "decision_admissibility_wp7_source_snapshot_v1"
_RUN_NODE_REF = re.compile(r"^run::(.+?)::node::")
METHOD_GENERATION_STAGES = (
    GenerationStage.DRAFT,
    GenerationStage.MODEL_DESIGN,
    GenerationStage.IMPROVE,
    GenerationStage.EVOLUTION,
    GenerationStage.FUSION,
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row is not an object: {path}:{line_number}")
        output.append(value)
    return output


def _verify_source_snapshot(
    path: Path,
    *,
    expected_root: Path,
) -> dict[str, Any]:
    path = path.resolve()
    root = expected_root.resolve()
    if path.parent != root:
        raise ValueError("Source snapshot manifest does not describe executed code")
    snapshot = _read_json(path)
    if snapshot.get("schema") != SOURCE_SNAPSHOT_SCHEMA:
        raise ValueError("Unsupported source snapshot schema")
    expected_source_sha256 = hashlib.sha256(
        json.dumps(
            {
                key: value
                for key, value in snapshot.items()
                if key != "source_sha256"
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if snapshot.get("source_sha256") != expected_source_sha256:
        raise ValueError("Source snapshot hash mismatch")
    file_hashes = snapshot.get("file_hashes")
    if not isinstance(file_hashes, Mapping) or not file_hashes:
        raise ValueError("Source snapshot has no file hash inventory")
    if snapshot.get("file_count") != len(file_hashes):
        raise ValueError("Source snapshot file count mismatch")
    for relative, expected_digest in sorted(file_hashes.items()):
        relative_path = Path(str(relative))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Unsafe source snapshot path")
        source_path = root / relative_path
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError(f"Source snapshot file is unavailable: {relative}")
        if sha256_file(source_path) != str(expected_digest):
            raise ValueError(f"Source snapshot file changed: {relative}")
    return snapshot


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(payload),
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _safe_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip(".-")
    if not normalized:
        raise ValueError("Publication path component is empty")
    return normalized[:120]


def _file_hashes(root: Path, *, exclude: set[Path] | None = None) -> dict[str, str]:
    excluded = {path.resolve() for path in (exclude or set())}
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.resolve() not in excluded
    }


def _make_private_copy_writable(root: Path) -> None:
    for path in [root, *sorted(root.rglob("*"))]:
        if path.is_symlink():
            raise ValueError(f"Bundle copy contains a symlink: {path}")
        mode = path.stat().st_mode
        if path.is_dir():
            path.chmod(mode | stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        elif path.is_file():
            path.chmod(mode | stat.S_IRUSR | stat.S_IWUSR)


def _seal_tree(root: Path) -> None:
    paths = sorted(
        root.rglob("*"),
        key=lambda path: (len(path.relative_to(root).parts), str(path)),
        reverse=True,
    )
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Publication output contains a symlink: {path}")
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


def _verify_queue_provenance(path: Path) -> dict[str, Any]:
    provenance = _read_json(path)
    expected = sha256_json(
        {
            key: value
            for key, value in provenance.items()
            if key != "provenance_hash"
        }
    )
    if provenance.get("provenance_hash") != expected:
        raise ValueError("Replay queue provenance hash mismatch")
    for filename, digest in (provenance.get("output_file_hashes") or {}).items():
        if sha256_file(path.parent / str(filename)) != str(digest):
            raise ValueError(f"Replay queue artifact changed: {filename}")
    return provenance


def _verify_launch(path: Path) -> dict[str, Any]:
    launch = _read_json(path)
    expected = sha256_json(
        {key: value for key, value in launch.items() if key != "launch_hash"}
    )
    if launch.get("launch_hash") != expected:
        raise ValueError("Clean Replay launch manifest hash mismatch")
    return launch


def validate_persisted_publication_material(
    *,
    parent_bundle_path: Path,
    queue_path: Path,
    queue_manifest_path: Path,
    queue_provenance_path: Path,
    launch_manifest_path: Path,
    launcher_log_path: Path,
    execution_output: Path,
    execution_validation_path: Path,
    protocol_id: str,
    protocol_version: str,
) -> dict[str, Any]:
    """Independently revalidate the immutable replay before publication."""

    parent = ImmutableBaseBundle.load(parent_bundle_path, verify_artifacts=True)
    queue = load_replay_queue(queue_path, queue_manifest_path)
    queue_provenance = _verify_queue_provenance(queue_provenance_path)
    launch = _verify_launch(launch_manifest_path)
    validation = _read_json(execution_validation_path)
    if validation.get("schema") != VALIDATION_SCHEMA:
        raise ValueError("Unsupported persisted execution validation")
    expected_validation_hash = sha256_json(
        {
            key: value
            for key, value in validation.items()
            if key != "validation_hash"
        }
    )
    if validation.get("validation_hash") != expected_validation_hash:
        raise ValueError("Persisted execution validation hash mismatch")
    if execution_validation_path.resolve().parent != execution_output.resolve():
        raise ValueError("Execution validation is outside the replay output root")

    output_hashes = _file_hashes(
        execution_output.resolve(),
        exclude={execution_validation_path.resolve()},
    )
    if output_hashes != validation.get("output_artifact_hashes"):
        raise ValueError("Replay output changed after independent validation")
    if len(output_hashes) != validation.get("output_artifact_file_count"):
        raise ValueError("Replay output artifact count mismatch")
    if sha256_json(output_hashes) != validation.get(
        "output_artifact_tree_sha256"
    ):
        raise ValueError("Replay output artifact tree hash mismatch")

    if validation.get("bundle_manifest_sha256") != parent.manifest_sha256:
        raise ValueError("Execution validation/parent Bundle mismatch")
    if validation.get("queue_file_sha256") != queue.queue_file_sha256:
        raise ValueError("Execution validation/replay queue mismatch")
    if validation.get("queue_manifest_sha256") != queue.manifest_sha256:
        raise ValueError("Execution validation/queue manifest mismatch")
    if validation.get("queue_provenance_hash") != queue_provenance.get(
        "provenance_hash"
    ):
        raise ValueError("Execution validation/queue provenance mismatch")
    if validation.get("launch_hash") != launch.get("launch_hash"):
        raise ValueError("Execution validation/launch mismatch")
    if validation.get("launch_manifest_sha256") != sha256_file(
        launch_manifest_path
    ):
        raise ValueError("Execution validation/launch file mismatch")
    if validation.get("launcher_log_sha256") != sha256_file(launcher_log_path):
        raise ValueError("Execution validation/launcher log mismatch")
    if validation.get("historical_metric_used_as_evidence") is not False:
        raise ValueError("Historical metric entered replay evidence")

    registry = ProtocolRegistry(parent.path / "protocol_registry")
    protocol_ref = registry.get(protocol_id, protocol_version).ref()
    if launch.get("protocol_ref") != protocol_ref.key():
        raise ValueError("Launch manifest/publication ProtocolRef mismatch")
    record_hash = str(validation.get("record_hash") or "")
    record_paths = sorted(execution_output.rglob("execution_record.json"))
    matching_records = [
        path
        for path in record_paths
        if _read_json(path).get("record_hash") == record_hash
    ]
    if len(matching_records) != 1:
        raise ValueError("Persisted successful replay record is ambiguous")
    attempt_dir = matching_records[0].parent
    record = _read_json(matching_records[0])
    entry = next(
        (
            item
            for item in queue.entries
            if item.entry_hash == str(record.get("entry_hash") or "")
        ),
        None,
    )
    if entry is None:
        raise ValueError("Successful replay is absent from the immutable queue")
    validated = validate_replay_execution_attempt(
        bundle=parent,
        entry=entry,
        protocol_ref=protocol_ref,
        registry=registry,
        attempt_dir=attempt_dir,
    )
    if validated.validation != validation.get("material_validation"):
        raise ValueError("Persisted replay material validation is not reproducible")
    for field_name in (
        "task_id",
        "candidate_id",
        "original_claim_id",
        "source_clause_id",
        "replay_artifact_id",
        "record_hash",
    ):
        expected = (
            getattr(entry, field_name)
            if hasattr(entry, field_name)
            else record.get(field_name)
        )
        if validation.get(field_name) != expected:
            raise ValueError(f"Execution validation field mismatch: {field_name}")
    parent.assert_unchanged()
    return {
        "parent": parent,
        "queue": queue,
        "queue_provenance": queue_provenance,
        "launch": launch,
        "validation": validation,
        "entry": entry,
        "validated": validated,
        "protocol_ref": protocol_ref,
        "registry": registry,
        "attempt_dir": attempt_dir,
    }


def _source_lineage(
    parent: ImmutableBaseBundle,
    *,
    original_claim_id: str,
    original_artifact_id: str,
    source_clause_id: str,
    expected_task_id: str,
) -> dict[str, Any]:
    clauses = {
        str(row["clause_id"]): row
        for row in parent.read_jsonl("sop/clauses.jsonl")
    }
    source_clause = clauses.get(source_clause_id)
    if source_clause is None:
        raise ValueError("Replay source clause is absent from the parent Bundle")
    if original_claim_id not in set(source_clause.get("claim_refs") or []):
        raise ValueError("Replay source clause does not bind the predecessor Claim")
    source_refs = {
        *set(source_clause.get("source_artifact_refs") or []),
        *set(source_clause.get("source_transition_refs") or []),
    }
    if original_artifact_id not in source_refs:
        raise ValueError("Replay source clause does not bind the predecessor artifact")
    if normalize_transfer_scope(source_clause.get("transfer_scope")) != SAME_DOMAIN:
        raise ValueError("Positive replay publication requires same-domain source scope")

    graph = parent.read_json("runforest/graph.json")
    node = next(
        (
            value
            for value in graph.get("nodes") or []
            if str(value.get("id") or "") == original_artifact_id
        ),
        None,
    )
    if node is None:
        raise ValueError("Predecessor artifact is absent from the parent RunForest")
    run_id = str(node.get("run_id") or "")
    if not run_id:
        match = _RUN_NODE_REF.match(original_artifact_id)
        run_id = match.group(1) if match else ""
    if not run_id:
        raise ValueError("Predecessor artifact has no parseable source run")

    corpus = parent.read_json("corpus/manifest.json")
    run = next(
        (
            value
            for value in corpus.get("runs") or []
            if str(value.get("run_id") or "") == run_id
        ),
        None,
    )
    if run is None:
        raise ValueError("Predecessor source run is absent from the corpus manifest")
    task_id = str(run.get("canonical_task_id") or "")
    task_family = str(run.get("task_family") or "")
    source_domain = canonical_domain(task_family)
    if not task_id or not task_family or not source_domain:
        raise ValueError("Predecessor source task/domain lineage is incomplete")
    if task_id != expected_task_id:
        raise ValueError("Replay queue task/source artifact lineage mismatch")
    return {
        "source_clause": copy.deepcopy(source_clause),
        "source_run_id": run_id,
        "source_task_id": task_id,
        "source_task_family": task_family,
        "source_domain": source_domain,
    }


def _method_text(raw: str, _metric_value: Any) -> tuple[str, dict[str, Any]]:
    text = re.sub(r"\s+", " ", str(raw or "")).strip()
    if not text:
        raise ValueError("Replay publication requires a method-only statement")
    if len(text) > 2000:
        raise ValueError("Replay method statement exceeds the publication limit")
    purity = require_method_claim_semantic_purity(text)
    return text, purity


def _publication_method_text(
    *,
    explicit_statement: str | None,
    source_clause: Mapping[str, Any],
    metric_value: Any,
) -> tuple[str, dict[str, Any], str]:
    if explicit_statement is not None:
        raw = explicit_statement
        source = "explicit_statement"
    else:
        raw = str(source_clause.get("retrieval_text") or "")
        source = "source_clause.retrieval_text"
        if not raw.strip():
            raise ValueError(
                "Replay source clause lacks a method-only retrieval projection"
            )
    text, purity = _method_text(raw, metric_value)
    return text, purity, source


def _formal_validation(candidate: Path) -> dict[str, Any]:
    report = validate_bundle(candidate)
    report["validation_hash"] = sha256_json(report)
    return report


def _visibility_request(
    *,
    protocol_ref: Any,
    bundle_version: str,
    policy_version: str,
    task_id: str,
    task_family: str,
    generation_stage: GenerationStage = GenerationStage.IMPROVE,
) -> VisibilityRequest:
    return VisibilityRequest(
        operation=Operation.GENERATE_CANDIDATE,
        generation_stage=generation_stage,
        governance_stage=GovernanceStage.RETRIEVAL,
        active_protocol=protocol_ref,
        task_context=TaskContext(task_id=task_id, task_family=task_family),
        memory_bundle_version=bundle_version,
        token_budget=4096,
        requesting_component="certified_replay_publication_preflight",
        authority_policy_version=policy_version,
    )


def audit_visible_method_clauses(
    nodes: Mapping[str, Mapping[str, Any]],
    clause_ids: list[str] | tuple[str, ...] | set[str],
) -> dict[str, Any]:
    visible_ids = sorted({str(value) for value in clause_ids if str(value)})
    checks: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for clause_id in visible_ids:
        clause = nodes.get(clause_id)
        if clause is None:
            violations.append(
                {
                    "clause_id": clause_id,
                    "field": "node",
                    "violation_codes": ["missing_visible_clause_node"],
                    "text_sha256": "",
                    "purity_report_hash": "",
                }
            )
            continue
        for field_name in ("text", "retrieval_text"):
            purity = audit_method_claim_semantic_purity(
                str(clause.get(field_name) or "")
            )
            check = {
                "clause_id": clause_id,
                "field": field_name,
                "text_sha256": purity["text_sha256"],
                "purity_report_hash": purity["report_hash"],
                "source_outcome_assertion_count": purity[
                    "source_outcome_assertion_count"
                ],
            }
            checks.append(check)
            if not purity["passed"]:
                violations.append(
                    {
                        **check,
                        "violation_codes": purity["violation_codes"],
                    }
                )
    report: dict[str, Any] = {
        "schema": VISIBLE_METHOD_PURITY_SCHEMA,
        "status": "passed" if not violations and bool(checks) else "failed",
        "visible_clause_ids": visible_ids,
        "visible_clause_count": len(visible_ids),
        "checked_field_count": len(checks),
        "checks": checks,
        "violation_count": len(violations),
        "violations": violations,
        "raw_text_embedded": False,
        "report_hash": "",
    }
    report["report_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "report_hash"}
    )
    return report


def verify_published_target_visibility(
    *,
    publication_root: Path,
    child_bundle: Path,
    replay_clause_id: str,
    replay_sop_id: str,
    replay_claim_id: str,
    source_task_id: str,
    protocol_id: str,
    protocol_version: str,
    target_task_id: str,
    target_task_family: str,
) -> dict[str, Any]:
    child = ImmutableBaseBundle.load(child_bundle, verify_artifacts=True)
    published_clauses = [
        row
        for row in child.read_jsonl("sop/clauses.jsonl")
        if str(row.get("clause_id") or "") == replay_clause_id
    ]
    if len(published_clauses) != 1:
        raise ValueError("Published replay clause is absent or ambiguous")
    published_clause = published_clauses[0]
    display_purity = require_method_claim_semantic_purity(
        str(published_clause.get("text") or "")
    )
    retrieval_purity = require_method_claim_semantic_purity(
        str(published_clause.get("retrieval_text") or "")
    )
    if display_purity["text_sha256"] != retrieval_purity["text_sha256"]:
        raise ValueError("Certified replay display/retrieval method text diverged")
    contract = published_clause.get("contract_spec") or {}
    if not isinstance(contract, Mapping):
        raise ValueError("Published replay clause has no semantic-purity contract")
    if (
        contract.get("method_semantic_purity_report_hash")
        != display_purity["report_hash"]
        or contract.get("method_text_sha256") != display_purity["text_sha256"]
        or contract.get("method_text_source")
        not in {"source_clause.retrieval_text", "explicit_statement"}
        or contract.get("source_outcome_assertion_count") != 0
    ):
        raise ValueError("Published replay clause lost its semantic-purity binding")
    registry = ProtocolRegistry(child.path / "protocol_registry")
    protocol_ref = registry.get(protocol_id, protocol_version).ref()
    snapshot = MemorySnapshotLoader(publication_root).load(
        session_overlay_path=publication_root / "visibility-check-overlay",
        active_protocol_ref=protocol_ref.key(),
        authority_policy_version=str(
            child.manifest.get("authority_policy_version") or "authority_v1"
        ),
    )
    engine = AuthorityEngine(
        registry,
        policy_version=snapshot.authority_policy_version,
    )
    authority_load = load_snapshot_authority(engine, snapshot)
    graph = child.read_json("runforest/graph.json")
    nodes = {
        str(node["id"]): node for node in graph.get("nodes") or []
    }
    gateway = SOPVisibilityGateway(
        nodes,
        mode="enforce",
        authority_engine=engine,
        enforce_operations=(Operation.GENERATE_CANDIDATE,),
        enforce_generation_stages=tuple(
            stage.value for stage in METHOD_GENERATION_STAGES
        ),
        enforce_governance_stages=(GovernanceStage.RETRIEVAL.value,),
    )
    target_stage_reports: dict[str, Any] = {}
    authority_refs: set[str] = set()
    all_target_effective_clause_ids: set[str] = set()
    for generation_stage in METHOD_GENERATION_STAGES:
        target_request = _visibility_request(
            protocol_ref=protocol_ref,
            bundle_version=child.bundle_version,
            policy_version=engine.policy_version,
            task_id=target_task_id,
            task_family=target_task_family,
            generation_stage=generation_stage,
        )
        prefiltered = snapshot.base_clauses(
            target_request.operation,
            task_id=target_task_id,
            task_family=target_task_family,
            generation_stage=target_request.generation_stage.value,
            governance_stage=target_request.governance_stage.value,
        )
        prefiltered_ids = {
            str(row.get("clause_id") or "") for row in prefiltered
        }
        prefiltered_sop_ids = {
            str(row.get("sop_id") or "")
            for row in prefiltered
            if str(row.get("sop_id") or "")
        }
        if replay_clause_id not in prefiltered_ids:
            raise ValueError(
                "Same-domain replay clause was removed by Base prefilter: "
                f"{generation_stage.value}"
            )
        target_pack = gateway.evaluate(
            target_request,
            candidate_sop_ids=(replay_sop_id,),
            candidate_clause_ids=(replay_clause_id,),
        )
        if target_pack.effective_clause_ids != [replay_clause_id]:
            raise ValueError(
                "Certified replay clause is not visible to the target task: "
                f"{generation_stage.value}"
            )
        if target_pack.visibility_trace.get("contract_compilation_errors"):
            raise ValueError("Certified replay ExperienceContract did not compile")
        target_decision = target_pack.visibility_trace["clause_decisions"][
            replay_clause_id
        ]
        if (
            target_decision.get("reason") != "authority_allow"
            or target_decision.get("source_task_ids") != [source_task_id]
        ):
            raise ValueError("Target visibility did not use source-scoped Authority")
        stage_authority_refs = target_decision.get("authority_decision_refs") or []
        if len(stage_authority_refs) != 1:
            raise ValueError("Target visibility has no unique Authority decision")
        authority_ref = str(stage_authority_refs[0])
        authority_refs.add(authority_ref)
        authority_decision = engine.decisions[authority_ref]
        if (
            authority_decision.claim_id != replay_claim_id
            or authority_decision.permitted_scope is None
            or authority_decision.permitted_scope.task_ids != [source_task_id]
        ):
            raise ValueError("Replay Claim authority was widened to the target task")
        full_target_pack = gateway.evaluate(
            target_request,
            candidate_sop_ids=tuple(sorted(prefiltered_sop_ids)),
            candidate_clause_ids=tuple(sorted(prefiltered_ids)),
        )
        full_effective_ids = full_target_pack.effective_clause_ids
        if replay_clause_id not in full_effective_ids:
            raise ValueError(
                "Certified replay clause disappeared from full target visibility"
            )
        all_target_effective_clause_ids.update(full_effective_ids)
        target_stage_reports[generation_stage.value] = {
            "base_exposure_count": int(replay_clause_id in prefiltered_ids),
            "effective_exposure_count": len(target_pack.effective_clause_ids),
            "full_effective_exposure_count": len(full_effective_ids),
            "full_effective_clause_ids": full_effective_ids,
            "authority_decision_ref": authority_ref,
            "reason": target_decision["reason"],
        }

    visible_method_purity = audit_visible_method_clauses(
        nodes,
        all_target_effective_clause_ids,
    )
    if visible_method_purity["status"] != "passed":
        violation_summary = ",".join(
            f"{value['clause_id']}:{value['field']}:"
            f"{'|'.join(value['violation_codes'])}"
            for value in visible_method_purity["violations"]
        )
        raise ValueError(
            "Target-visible method semantic-purity preflight failed: "
            + violation_summary
        )

    negative_controls: dict[str, Any] = {}
    for domain, family in (
        ("nlp", "text_classification"),
        ("audio", "audio_classification"),
    ):
        stage_reports: dict[str, Any] = {}
        for generation_stage in METHOD_GENERATION_STAGES:
            request = _visibility_request(
                protocol_ref=protocol_ref,
                bundle_version=child.bundle_version,
                policy_version=engine.policy_version,
                task_id=f"negative-control-{domain}",
                task_family=family,
                generation_stage=generation_stage,
            )
            base_ids = {
                str(row.get("clause_id") or "")
                for row in snapshot.base_clauses(
                    request.operation,
                    task_id=request.task_context.task_id,
                    task_family=request.task_context.task_family,
                    generation_stage=request.generation_stage.value,
                    governance_stage=request.governance_stage.value,
                )
            }
            pack = gateway.evaluate(
                request,
                candidate_sop_ids=(replay_sop_id,),
                candidate_clause_ids=(replay_clause_id,),
            )
            if replay_clause_id in base_ids or pack.effective_clause_ids:
                raise ValueError(f"Replay clause leaked into the {domain} control")
            stage_reports[generation_stage.value] = {
                "base_exposure_count": int(replay_clause_id in base_ids),
                "effective_exposure_count": len(pack.effective_clause_ids),
                "reason": pack.visibility_trace["clause_decisions"][
                    replay_clause_id
                ]["reason"],
            }
        negative_controls[domain] = {
            "base_exposure_count": sum(
                value["base_exposure_count"] for value in stage_reports.values()
            ),
            "effective_exposure_count": sum(
                value["effective_exposure_count"]
                for value in stage_reports.values()
            ),
            "stage_reports": stage_reports,
        }
    report = {
        "schema": PUBLICATION_VISIBILITY_SCHEMA,
        "status": "passed",
        "bundle_id": child.bundle_id,
        "bundle_manifest_sha256": child.manifest_sha256,
        "target_task_id": target_task_id,
        "target_task_family": target_task_family,
        "target_domain": canonical_domain(target_task_family),
        "source_task_id": source_task_id,
        "replay_claim_id": replay_claim_id,
        "replay_clause_id": replay_clause_id,
        "method_semantic_purity_schema": display_purity["schema"],
        "method_semantic_purity_report_hash": display_purity["report_hash"],
        "method_text_sha256": display_purity["text_sha256"],
        "method_text_source": contract["method_text_source"],
        "source_outcome_assertion_count": 0,
        "all_target_effective_clause_ids": sorted(
            all_target_effective_clause_ids
        ),
        "all_target_effective_clause_count": len(
            all_target_effective_clause_ids
        ),
        "visible_method_purity": visible_method_purity,
        "permitted_generation_stages": [
            stage.value for stage in METHOD_GENERATION_STAGES
        ],
        "target_stage_reports": target_stage_reports,
        "same_domain_exposure_count": sum(
            value["effective_exposure_count"]
            for value in target_stage_reports.values()
        ),
        "cross_domain_exposure_count": sum(
            value["effective_exposure_count"]
            for value in negative_controls.values()
        ),
        "authority_decision_refs": sorted(authority_refs),
        "authority_load_report_hash": authority_load["report_hash"],
        "negative_controls": negative_controls,
        "visibility_hash": "",
    }
    report["visibility_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "visibility_hash"}
    )
    child.assert_unchanged()
    return report


def publish_certified_replay(
    *,
    source_manifest_path: Path,
    parent_bundle_path: Path,
    queue_path: Path,
    queue_manifest_path: Path,
    queue_provenance_path: Path,
    launch_manifest_path: Path,
    launcher_log_path: Path,
    execution_output: Path,
    execution_validation_path: Path,
    publication_root: Path,
    new_version: str,
    target_task_id: str,
    target_task_family: str,
    protocol_id: str,
    protocol_version: str,
    bundle_id: str | None = None,
    statement: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    source_manifest_path = source_manifest_path.resolve()
    source_snapshot = _verify_source_snapshot(
        source_manifest_path,
        expected_root=REPO,
    )
    parent_bundle_path = parent_bundle_path.resolve()
    execution_output = execution_output.resolve()
    execution_validation_path = execution_validation_path.resolve()
    publication_root = publication_root.resolve()
    if publication_root.exists():
        raise FileExistsError(
            f"Refusing to reuse publication root: {publication_root}"
        )

    material = validate_persisted_publication_material(
        parent_bundle_path=parent_bundle_path,
        queue_path=queue_path.resolve(),
        queue_manifest_path=queue_manifest_path.resolve(),
        queue_provenance_path=queue_provenance_path.resolve(),
        launch_manifest_path=launch_manifest_path.resolve(),
        launcher_log_path=launcher_log_path.resolve(),
        execution_output=execution_output,
        execution_validation_path=execution_validation_path,
        protocol_id=protocol_id,
        protocol_version=protocol_version,
    )
    source_parent: ImmutableBaseBundle = material["parent"]
    validated = material["validated"]
    entry = material["entry"]
    source_parent_hashes = _file_hashes(source_parent.path)

    publication_root.mkdir(parents=True)
    bundles_dir = publication_root / "bundles"
    bundles_dir.mkdir()
    parent_copy = bundles_dir / (
        "parent-" + _safe_component(source_parent.manifest_sha256[:16])
    )
    shutil.copytree(source_parent.path, parent_copy)
    _make_private_copy_writable(parent_copy)
    copied_parent = ImmutableBaseBundle.load(parent_copy, verify_artifacts=True)
    if copied_parent.manifest_sha256 != source_parent.manifest_sha256:
        raise ValueError("Isolated parent copy changed the Bundle manifest")
    write_json_atomic(
        publication_root / "CURRENT.json",
        make_current_pointer(
            bundle_path=parent_copy.relative_to(publication_root).as_posix(),
            manifest=copied_parent.manifest,
            parent_bundle=copied_parent.manifest.get("parent_bundle"),
        ),
    )

    overlay_path = publication_root / "publication-overlay"
    overlay = SessionOverlay(overlay_path)
    snapshot = MemorySnapshotLoader(publication_root).load(
        session_overlay_path=overlay_path,
        active_protocol_ref=material["protocol_ref"].key(),
        authority_policy_version=str(
            copied_parent.manifest.get("authority_policy_version")
            or "authority_v1"
        ),
    )
    registry = ProtocolRegistry(copied_parent.path / "protocol_registry")
    engine = AuthorityEngine(
        registry,
        policy_version=snapshot.authority_policy_version,
    )
    parent_authority_load = load_snapshot_authority(engine, snapshot)
    original = engine.graph.claims.get(entry.original_claim_id)
    if original is None:
        raise ValueError("Parent Bundle lacks the replay predecessor Claim")
    if original.subject_artifact_id != entry.child_artifact_id:
        raise ValueError("Replay predecessor Claim/source code artifact mismatch")
    original_claim_before = copy.deepcopy(original)
    lineage = _source_lineage(
        copied_parent,
        original_claim_id=original.claim_id,
        original_artifact_id=original.subject_artifact_id,
        source_clause_id=entry.source_clause_id,
        expected_task_id=entry.task_id,
    )
    source_domain = lineage["source_domain"]
    target_domain = canonical_domain(target_task_family)
    if (
        not target_task_id
        or target_task_id == lineage["source_task_id"]
        or not target_domain
        or target_domain != source_domain
    ):
        raise ValueError("Publication target must be a different same-domain task")
    graph_meta = copied_parent.read_json("runforest/graph.json").get("meta") or {}
    if target_task_id in set(graph_meta.get("source_task_ids") or []):
        raise ValueError("Publication target task is present in Bundle sources")

    method_text, method_purity, method_text_source = _publication_method_text(
        explicit_statement=statement,
        source_clause=lineage["source_clause"],
        metric_value=validated.record.get("metric_value"),
    )
    registration = ReplayAuthorityRecovery(engine.graph, registry).register(
        original_claim_id=original.claim_id,
        verification=validated.verification,
        receipts=validated.receipts,
        protocol_ref=material["protocol_ref"],
        statement=method_text,
        claim_type=ClaimType.METHOD_HYPOTHESIS,
        task_scope=copy.deepcopy(original.task_scope),
    )
    if registration.identity != ReplayIdentity.METHOD_PRESERVED:
        raise ValueError("Only method-preserved replay can publish this method clause")
    replay_claim = engine.graph.claims[registration.replay_claim_id]
    if replay_claim.task_scope != original.task_scope:
        raise ValueError("Replay publication widened the source Claim task scope")
    if engine.graph.claims[original.claim_id] != original_claim_before:
        raise ValueError("Replay registration mutated the predecessor Claim")

    identity_hash = sha256_json(
        {
            "registration_hash": registration.registration_hash,
            "source_clause_id": entry.source_clause_id,
            "source_artifact_id": original.subject_artifact_id,
            "protocol_ref": material["protocol_ref"].key(),
            "transfer_scope": SAME_DOMAIN,
        }
    )
    replay_sop_id = f"sop::certified-replay::{identity_hash[:24]}"
    replay_clause_id = f"clause::certified-replay::{identity_hash[24:48]}"
    clause = SOPClauseV1(
        clause_id=replay_clause_id,
        sop_id=replay_sop_id,
        text=method_text,
        retrieval_text=method_text,
        claim_refs=(registration.replay_claim_id,),
        claim_types=(ClaimType.METHOD_HYPOTHESIS.value,),
        source_artifact_refs=(original.subject_artifact_id,),
        source_run_ids=(lineage["source_run_id"],),
        source_task_ids=(lineage["source_task_id"],),
        source_task_families=(lineage["source_task_family"],),
        source_domains=(source_domain,),
        transfer_scope=SAME_DOMAIN,
        protocol_scope=(material["protocol_ref"].key(),),
        task_scope={
            "task_ids": [lineage["source_task_id"]],
            "task_families": [lineage["source_task_family"]],
        },
        permitted_operations=(Operation.GENERATE_CANDIDATE.value,),
        permitted_generation_stages=tuple(
            stage.value for stage in METHOD_GENERATION_STAGES
        ),
        permitted_governance_stages=(GovernanceStage.RETRIEVAL.value,),
        publication_class="certified",
        receipt_refs=registration.receipt_ids,
        applies_when=(
            "the target is a different task in the same image domain",
        ),
        prevents=(
            "cross-domain method transfer",
            "reuse of target-heldout historical artifacts",
        ),
        contract_spec={
            "replay_artifact_id": registration.replay_artifact_id,
            "registration_hash": registration.registration_hash,
            "verification_report_hash": registration.verification_report_hash,
            "predecessor_claim_id": registration.original_claim_id,
            "predecessor_clause_id": entry.source_clause_id,
            "execution_validation_hash": material["validation"][
                "validation_hash"
            ],
            "execution_record_hash": validated.record["record_hash"],
            "transfer_design": "same_domain_different_task_target_heldout",
            "historical_metric_used_as_evidence": False,
            "method_semantic_purity_schema": method_purity["schema"],
            "method_semantic_purity_report_hash": method_purity["report_hash"],
            "method_text_sha256": method_purity["text_sha256"],
            "method_text_source": method_text_source,
            "source_outcome_assertion_count": 0,
        },
        legacy_status="clean_replay_certified_v1",
    )
    publication = ReplayClausePublication(
        clause=clause,
        title=title or "Certified Same-Domain Replay Method",
        source_clause_id=entry.source_clause_id,
        registration_hash=registration.registration_hash,
        verification_report_hash=validated.verification.report_hash,
    )
    result = CertifiedBundlePublisher(
        publication_root,
        engine.graph,
        registry,
    ).publish(
        new_version=new_version,
        overlay=overlay,
        registrations=(registration,),
        verifications={
            validated.verification.report_hash: validated.verification
        },
        expected_parent_manifest_sha256=copied_parent.manifest_sha256,
        bundle_id=bundle_id,
        replay_clause_publications=(publication,),
        formal_bundle_validator=_formal_validation,
    )
    child_bundle = publication_root / result.publication.bundle_path
    formal_validation = _formal_validation(child_bundle)
    pipeline_formal = result.publication.pipeline_reports[
        "bundle_validation"
    ]["formal_validation"]
    if formal_validation != pipeline_formal or formal_validation.get("valid") is not True:
        raise ValueError("Post-publication formal validation did not reproduce")
    visibility = verify_published_target_visibility(
        publication_root=publication_root,
        child_bundle=child_bundle,
        replay_clause_id=replay_clause_id,
        replay_sop_id=replay_sop_id,
        replay_claim_id=registration.replay_claim_id,
        source_task_id=lineage["source_task_id"],
        protocol_id=protocol_id,
        protocol_version=protocol_version,
        target_task_id=target_task_id,
        target_task_family=target_task_family,
    )
    if (
        visibility.get("method_semantic_purity_report_hash")
        != method_purity["report_hash"]
        or visibility.get("method_text_sha256") != method_purity["text_sha256"]
    ):
        raise ValueError("Post-publication semantic-purity report did not reproduce")

    reports_dir = publication_root / "reports"
    report_payloads = {
        "publication_report.json": result.publication.as_dict(),
        "certification_report.json": result.certification_report,
        "formal_validation_report.json": formal_validation,
        "target_visibility_report.json": visibility,
        "method_semantic_purity_report.json": method_purity,
        "source_snapshot_manifest.json": source_snapshot,
    }
    for filename, payload in report_payloads.items():
        _write_json_exclusive(reports_dir / filename, payload)
    report_hashes = {
        filename: sha256_file(reports_dir / filename)
        for filename in sorted(report_payloads)
    }

    source_parent.assert_unchanged()
    if _file_hashes(source_parent.path) != source_parent_hashes:
        raise ValueError("Source parent Bundle changed during publication")
    child = ImmutableBaseBundle.load(child_bundle, verify_artifacts=True)
    child_hashes = _file_hashes(child.path)
    parent_copy_hashes = _file_hashes(copied_parent.path)
    code_paths = (
        REPO / "mlevolve" / "authority" / "clean_replay.py",
        REPO / "mlevolve" / "authority" / "clean_replay_runner.py",
        REPO / "mlevolve" / "authority" / "certified_bundle.py",
        REPO / "mlevolve" / "authority" / "replay_clause_publication.py",
        REPO
        / "mlevolve"
        / "authority"
        / "adapters"
        / "mlevolve"
        / "retrieval_gate.py",
        REPO / "mlevolve" / "authority" / "memory_snapshot.py",
        REPO / "paper-skills" / "memory_bundle" / "method_claim_purity.py",
        Path(__file__).resolve(),
        REPO / "paper-skills" / "memory_bundle" / "validate_memory_bundle.py",
    )
    for path in code_paths:
        relative = path.relative_to(REPO).as_posix()
        if source_snapshot["file_hashes"].get(relative) != sha256_file(path):
            raise ValueError(
                f"Publication source code is not bound to the source snapshot: {relative}"
            )
    provenance = {
        "schema": PUBLICATION_PROVENANCE_SCHEMA,
        "status": "sealed",
        "source_snapshot_manifest_path": str(source_manifest_path),
        "source_snapshot_manifest_sha256": sha256_file(source_manifest_path),
        "source_snapshot_sha256": source_snapshot["source_sha256"],
        "source_snapshot_base_commit": source_snapshot["base_commit"],
        "source_snapshot_parent_sha256": source_snapshot.get(
            "parent_source_sha256"
        ),
        "source_parent_bundle_path": str(source_parent.path),
        "source_parent_bundle_id": source_parent.bundle_id,
        "source_parent_manifest_sha256": source_parent.manifest_sha256,
        "source_parent_file_count": len(source_parent_hashes),
        "source_parent_tree_sha256": sha256_json(source_parent_hashes),
        "source_parent_unchanged": True,
        "isolated_parent_bundle_path": str(copied_parent.path),
        "isolated_parent_file_count": len(parent_copy_hashes),
        "isolated_parent_tree_sha256": sha256_json(parent_copy_hashes),
        "execution_validation_path": str(execution_validation_path),
        "execution_validation_file_sha256": sha256_file(
            execution_validation_path
        ),
        "execution_validation_hash": material["validation"][
            "validation_hash"
        ],
        "execution_record_hash": validated.record["record_hash"],
        "queue_file_sha256": material["queue"].queue_file_sha256,
        "queue_manifest_sha256": material["queue"].manifest_sha256,
        "queue_provenance_hash": material["queue_provenance"][
            "provenance_hash"
        ],
        "launch_hash": material["launch"]["launch_hash"],
        "parent_authority_load_report_hash": parent_authority_load[
            "report_hash"
        ],
        "predecessor_claim_id": original.claim_id,
        "predecessor_clause_id": entry.source_clause_id,
        "predecessor_artifact_id": original.subject_artifact_id,
        "source_run_id": lineage["source_run_id"],
        "source_task_id": lineage["source_task_id"],
        "source_task_family": lineage["source_task_family"],
        "source_domain": source_domain,
        "target_task_id": target_task_id,
        "target_task_family": target_task_family,
        "target_domain": target_domain,
        "transfer_scope": SAME_DOMAIN,
        "replay_registration_hash": registration.registration_hash,
        "replay_verification_report_hash": validated.verification.report_hash,
        "replay_claim_id": registration.replay_claim_id,
        "replay_clause_id": replay_clause_id,
        "replay_sop_id": replay_sop_id,
        "trusted_receipt_ids": sorted(registration.receipt_ids),
        "historical_metric_used_as_evidence": False,
        "method_semantic_purity_schema": method_purity["schema"],
        "method_semantic_purity_report_hash": method_purity["report_hash"],
        "method_text_sha256": method_purity["text_sha256"],
        "method_text_source": method_text_source,
        "source_outcome_assertion_count": 0,
        "blanket_clause_upgrade": False,
        "publication_report_hash": result.publication.report_hash,
        "publication_report_file_hashes": report_hashes,
        "formal_validation_hash": formal_validation["validation_hash"],
        "target_visibility_hash": visibility["visibility_hash"],
        "all_target_effective_clause_count": visibility[
            "all_target_effective_clause_count"
        ],
        "visible_method_purity_report_hash": visibility[
            "visible_method_purity"
        ]["report_hash"],
        "visible_method_purity_violation_count": visibility[
            "visible_method_purity"
        ]["violation_count"],
        "child_bundle_path": str(child.path),
        "child_bundle_id": child.bundle_id,
        "child_manifest_sha256": child.manifest_sha256,
        "child_file_count": len(child_hashes),
        "child_tree_sha256": sha256_json(child_hashes),
        "source_code_hashes": {
            path.relative_to(REPO).as_posix(): sha256_file(path)
            for path in code_paths
        },
        "provenance_hash": "",
    }
    provenance["provenance_hash"] = sha256_json(
        {key: value for key, value in provenance.items() if key != "provenance_hash"}
    )
    provenance_path = publication_root / "publication_provenance.json"
    _write_json_exclusive(provenance_path, provenance)
    _seal_tree(publication_root)
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Revalidate one immutable Clean Replay and atomically publish a "
            "replay-scoped certified same-domain Bundle."
        )
    )
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--parent-bundle", required=True, type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--queue-manifest", required=True, type=Path)
    parser.add_argument("--queue-provenance", required=True, type=Path)
    parser.add_argument("--launch-manifest", required=True, type=Path)
    parser.add_argument("--launcher-log", required=True, type=Path)
    parser.add_argument("--execution-output", required=True, type=Path)
    parser.add_argument("--execution-validation", required=True, type=Path)
    parser.add_argument("--publication-root", required=True, type=Path)
    parser.add_argument("--new-version", required=True)
    parser.add_argument("--bundle-id")
    parser.add_argument("--target-task-id", required=True)
    parser.add_argument("--target-task-family", required=True)
    parser.add_argument("--protocol-id", default="mlevolve-default")
    parser.add_argument("--protocol-version", default="2")
    parser.add_argument("--statement")
    parser.add_argument("--title")
    args = parser.parse_args()
    provenance = publish_certified_replay(
        source_manifest_path=args.source_manifest,
        parent_bundle_path=args.parent_bundle,
        queue_path=args.queue,
        queue_manifest_path=args.queue_manifest,
        queue_provenance_path=args.queue_provenance,
        launch_manifest_path=args.launch_manifest,
        launcher_log_path=args.launcher_log,
        execution_output=args.execution_output,
        execution_validation_path=args.execution_validation,
        publication_root=args.publication_root,
        new_version=args.new_version,
        target_task_id=args.target_task_id,
        target_task_family=args.target_task_family,
        protocol_id=args.protocol_id,
        protocol_version=args.protocol_version,
        bundle_id=args.bundle_id,
        statement=args.statement,
        title=args.title,
    )
    print(
        json.dumps(
            {
                "status": provenance["status"],
                "child_bundle_id": provenance["child_bundle_id"],
                "child_manifest_sha256": provenance[
                    "child_manifest_sha256"
                ],
                "replay_clause_id": provenance["replay_clause_id"],
                "formal_validation_hash": provenance[
                    "formal_validation_hash"
                ],
                "target_visibility_hash": provenance[
                    "target_visibility_hash"
                ],
                "provenance_hash": provenance["provenance_hash"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
