"""Deterministic post-Judge resolver for executable transition evidence.

The retrieval path intentionally ranks compact SOP/RunForest metadata.  This
module opens full parent/child source only after that path has fixed its final
selected candidate IDs.  It never adds, removes, replaces, or reranks a Judge
selection.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import hmac
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping


SCHEMA = "mlevolve_transition_evidence_capsules_v2"
RECEIPT_SCHEMA = "mlevolve_evidence_resolver_receipt_v1"
SUPPORTED_STAGES = {"improve", "debug"}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(payload: Mapping[str, Any]) -> str:
    return _sha256_text(
        _canonical_json(
            {key: value for key, value in payload.items() if key != "capsule_sha256"}
        )
    )


def _pair_key(outcome: str, before_sha: str, after_sha: str) -> str:
    return _sha256_text(f"{outcome}\0{before_sha}\0{after_sha}")


def _canonical_diff(
    before_code: str,
    after_code: str,
    parent_node_id: str,
    child_node_id: str,
) -> str:
    return "".join(
        difflib.unified_diff(
            before_code.splitlines(keepends=True),
            after_code.splitlines(keepends=True),
            fromfile=f"before/{parent_node_id}",
            tofile=f"after/{child_node_id}",
            n=3,
        )
    )


class TransitionEvidenceResolver:
    """Hash-verified executable evidence index used after retrieval selection."""

    def __init__(
        self,
        *,
        capsule_path: str | Path,
        expected_file_sha256: str,
        graph_path: str | Path,
        graph_nodes: Mapping[str, Mapping[str, Any]],
        max_pairs: int = 3,
    ) -> None:
        self.path = Path(capsule_path).expanduser().resolve(strict=True)
        self.graph_path = Path(graph_path).expanduser().resolve(strict=True)
        self.max_pairs = int(max_pairs)
        if self.max_pairs <= 0:
            raise ValueError("evidence_resolver_max_pairs must be positive")

        observed_file_sha = _sha256_file(self.path)
        expected_file_sha256 = str(expected_file_sha256 or "")
        if len(expected_file_sha256) != 64:
            raise ValueError(
                "Evidence Resolver requires a pinned transition capsule file SHA-256"
            )
        if not hmac.compare_digest(observed_file_sha, expected_file_sha256):
            raise ValueError(
                "Transition evidence capsule file SHA-256 mismatch: "
                f"expected={expected_file_sha256} observed={observed_file_sha}"
            )

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
            raise ValueError("Unsupported transition evidence capsule schema")
        observed_payload_sha = _payload_sha256(payload)
        declared_payload_sha = str(payload.get("capsule_sha256") or "")
        if not hmac.compare_digest(observed_payload_sha, declared_payload_sha):
            raise ValueError("Transition evidence capsule payload hash mismatch")
        observed_graph_sha = _sha256_file(self.graph_path)
        declared_graph_sha = str(payload.get("graph_file_sha256") or "")
        if not hmac.compare_digest(observed_graph_sha, declared_graph_sha):
            raise ValueError(
                "Transition evidence capsule is bound to a different RunForest graph"
            )

        self.task_id = str(payload.get("task_id") or "")
        if not self.task_id:
            raise ValueError("Transition evidence capsule has no task identity")
        self.file_sha256 = observed_file_sha
        self.capsule_sha256 = observed_payload_sha
        self.graph_file_sha256 = observed_graph_sha
        self._graph_nodes = graph_nodes
        self._code_by_sha = self._validate_code_blobs(payload.get("code_blobs"))
        self._node_by_id = self._validate_nodes(payload.get("nodes"))
        self._transition_by_id = self._validate_transitions(payload.get("transitions"))
        self._validate_pairs(payload.get("pairs"))
        self._candidate_aliases = self._validate_candidate_aliases(
            payload.get("candidate_aliases") or []
        )

        self._transitions_by_child: dict[str, list[str]] = defaultdict(list)
        for transition_id, transition in self._transition_by_id.items():
            self._transitions_by_child[str(transition["child_node_id"])].append(
                transition_id
            )
        for values in self._transitions_by_child.values():
            values.sort()

        self.load_receipt = {
            "schema": "mlevolve_evidence_resolver_load_receipt_v1",
            "path": str(self.path),
            "file_sha256": self.file_sha256,
            "capsule_sha256": self.capsule_sha256,
            "graph_file_sha256": self.graph_file_sha256,
            "task_id": self.task_id,
            "transition_count": len(self._transition_by_id),
            "node_count": len(self._node_by_id),
            "unique_code_count": len(self._code_by_sha),
            "candidate_alias_count": len(self._candidate_aliases),
            "materialized_candidate_alias_count": sum(
                bool(row["materialized"])
                for row in self._candidate_aliases.values()
            ),
            "max_pairs": self.max_pairs,
            "status": "validated",
        }

    def _validate_code_blobs(self, rows: Any) -> dict[str, str]:
        if not isinstance(rows, list):
            raise ValueError("Transition evidence code inventory is malformed")
        output: dict[str, str] = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("Transition evidence code blob is not an object")
            code_sha = str(row.get("code_sha256") or "")
            code = row.get("code")
            if len(code_sha) != 64 or not isinstance(code, str) or not code.strip():
                raise ValueError("Transition evidence code blob is incomplete")
            if code_sha in output:
                raise ValueError(f"Duplicate transition evidence code hash: {code_sha}")
            if not hmac.compare_digest(_sha256_text(code), code_sha):
                raise ValueError(f"Transition evidence code hash mismatch: {code_sha}")
            output[code_sha] = code
        return output

    def _validate_nodes(self, rows: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(rows, list):
            raise ValueError("Transition evidence node inventory is malformed")
        output: dict[str, dict[str, Any]] = {}
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise ValueError("Transition evidence node is not an object")
            row = dict(raw)
            node_id = str(row.get("node_id") or "")
            code_sha = str(row.get("code_sha256") or "")
            graph_node = self._graph_nodes.get(node_id)
            if not node_id or node_id in output:
                raise ValueError(f"Duplicate or missing transition evidence node: {node_id}")
            if code_sha not in self._code_by_sha:
                raise ValueError(f"Transition evidence node has no code blob: {node_id}")
            if not isinstance(graph_node, Mapping) or graph_node.get("type") != "RunNode":
                raise ValueError(f"Transition evidence references a missing RunNode: {node_id}")
            if not hmac.compare_digest(
                code_sha, str(graph_node.get("code_sha256") or "")
            ):
                raise ValueError(f"Transition evidence graph/code mismatch: {node_id}")
            if not str(row.get("source_journal") or "") or not str(
                row.get("source_journal_sha256") or ""
            ):
                raise ValueError(f"Transition evidence provenance is incomplete: {node_id}")
            output[node_id] = row
        return output

    def _validate_transitions(self, rows: Any) -> dict[str, dict[str, Any]]:
        if not isinstance(rows, list):
            raise ValueError("Transition evidence transition inventory is malformed")
        output: dict[str, dict[str, Any]] = {}
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise ValueError("Transition evidence transition is not an object")
            row = dict(raw)
            transition_id = str(row.get("transition_id") or "")
            if not transition_id or transition_id in output:
                raise ValueError(
                    f"Duplicate or missing transition evidence transition: {transition_id}"
                )
            graph_transition = self._graph_nodes.get(transition_id)
            if (
                not isinstance(graph_transition, Mapping)
                or graph_transition.get("type") != "Transition"
            ):
                raise ValueError(
                    "Transition evidence references a missing Transition: "
                    f"{transition_id}"
                )
            parent_id = str(row.get("parent_node_id") or "")
            child_id = str(row.get("child_node_id") or "")
            if (
                parent_id != str(graph_transition.get("parent_node_id") or "")
                or child_id != str(graph_transition.get("child_node_id") or "")
                or str(row.get("outcome") or "")
                != str(graph_transition.get("outcome") or "")
            ):
                raise ValueError(
                    f"Transition evidence graph binding mismatch: {transition_id}"
                )
            if parent_id not in self._node_by_id or child_id not in self._node_by_id:
                raise ValueError(
                    f"Transition evidence endpoint is not materialized: {transition_id}"
                )
            before_sha = str(row.get("before_code_sha256") or "")
            after_sha = str(row.get("after_code_sha256") or "")
            if (
                before_sha != self._node_by_id[parent_id]["code_sha256"]
                or after_sha != self._node_by_id[child_id]["code_sha256"]
            ):
                raise ValueError(
                    f"Transition evidence endpoint hash mismatch: {transition_id}"
                )
            outcome = str(row.get("outcome") or "")
            evidence_class = str(row.get("evidence_class") or "")
            allowed_classes = (
                {"strict_debug_observed"}
                if outcome == "debug_fixed"
                else {"official_observed", "strict_internal_observed"}
                if outcome == "metric_improved"
                else set()
            )
            if evidence_class not in allowed_classes:
                raise ValueError(
                    f"Transition evidence class/outcome mismatch: {transition_id}"
                )
            if str(row.get("pair_key") or "") != _pair_key(
                outcome, before_sha, after_sha
            ):
                raise ValueError(f"Transition evidence pair hash mismatch: {transition_id}")
            output[transition_id] = row
        return output

    def _validate_pairs(self, rows: Any) -> None:
        if not isinstance(rows, list):
            raise ValueError("Transition evidence unique-pair inventory is malformed")
        seen: set[str] = set()
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise ValueError("Transition evidence pair is not an object")
            pair_key = str(raw.get("pair_key") or "")
            aliases = [str(value) for value in raw.get("alias_transition_ids") or []]
            representative = str(raw.get("representative_transition_id") or "")
            if not pair_key or pair_key in seen or not aliases:
                raise ValueError(f"Duplicate or malformed transition evidence pair: {pair_key}")
            if representative not in aliases or any(
                transition_id not in self._transition_by_id for transition_id in aliases
            ):
                raise ValueError(f"Transition evidence pair alias mismatch: {pair_key}")
            for transition_id in aliases:
                if self._transition_by_id[transition_id]["pair_key"] != pair_key:
                    raise ValueError(
                        f"Transition evidence pair binding mismatch: {pair_key}"
                    )
            seen.add(pair_key)
        observed = {str(row["pair_key"]) for row in self._transition_by_id.values()}
        if seen != observed:
            raise ValueError("Transition evidence unique-pair coverage mismatch")

    def _validate_candidate_aliases(
        self, rows: Any
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(rows, list):
            raise ValueError("Transition evidence candidate aliases are malformed")
        output: dict[str, dict[str, Any]] = {}
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise ValueError("Transition evidence candidate alias is not an object")
            row = dict(raw)
            candidate_id = str(row.get("candidate_id") or "")
            alias_kind = str(row.get("alias_kind") or "")
            atomic_transition_id = str(row.get("atomic_transition_id") or "")
            source_transition_id = str(row.get("source_transition_id") or "")
            if not candidate_id or candidate_id in output:
                raise ValueError(
                    f"Duplicate or missing transition evidence alias: {candidate_id}"
                )
            if alias_kind not in {"atomic_transition", "repair_claim"}:
                raise ValueError(
                    f"Unsupported transition evidence alias kind: {candidate_id}"
                )

            atomic = self._graph_nodes.get(atomic_transition_id)
            source = self._graph_nodes.get(source_transition_id)
            if (
                not isinstance(atomic, Mapping)
                or atomic.get("type") != "Transition"
                or not str(atomic_transition_id).startswith("atomic-transition::")
            ):
                raise ValueError(
                    f"Transition evidence alias has no atomic Transition: {candidate_id}"
                )
            if not isinstance(source, Mapping) or source.get("type") != "Transition":
                raise ValueError(
                    f"Transition evidence alias has no source Transition: {candidate_id}"
                )

            claim = atomic.get("atomic_repair_claim")
            claim = claim if isinstance(claim, Mapping) else {}
            verification = claim.get("verification")
            verification = (
                verification if isinstance(verification, Mapping) else {}
            )
            expected_repair_id = str(atomic_transition_id).replace(
                "atomic-transition::", "repair-claim::", 1
            )
            expected_candidate_id = (
                atomic_transition_id
                if alias_kind == "atomic_transition"
                else expected_repair_id
            )
            if candidate_id != expected_candidate_id:
                raise ValueError(
                    f"Transition evidence alias identity mismatch: {candidate_id}"
                )
            if (
                str(claim.get("source_transition_id") or "")
                != source_transition_id
                or str(claim.get("source_parent_node_id") or "")
                != str(source.get("parent_node_id") or "")
                or str(claim.get("source_child_node_id") or "")
                != str(source.get("child_node_id") or "")
                or str(atomic.get("parent_node_id") or "")
                != str(source.get("parent_node_id") or "")
                or str(atomic.get("child_node_id") or "")
                != str(source.get("child_node_id") or "")
                or str(atomic.get("outcome") or "")
                != str(source.get("outcome") or "")
                or str(row.get("outcome") or "")
                != str(source.get("outcome") or "")
            ):
                raise ValueError(
                    f"Transition evidence alias graph binding mismatch: {candidate_id}"
                )
            parent = self._graph_nodes.get(str(source.get("parent_node_id") or ""))
            child = self._graph_nodes.get(str(source.get("child_node_id") or ""))
            before_sha = str(verification.get("before_code_sha256") or "")
            after_sha = str(verification.get("after_code_sha256") or "")
            if (
                not isinstance(parent, Mapping)
                or not isinstance(child, Mapping)
                or before_sha != str(parent.get("code_sha256") or "")
                or after_sha != str(child.get("code_sha256") or "")
                or before_sha != str(row.get("before_code_sha256") or "")
                or after_sha != str(row.get("after_code_sha256") or "")
            ):
                raise ValueError(
                    f"Transition evidence alias code binding mismatch: {candidate_id}"
                )
            materialized = source_transition_id in self._transition_by_id
            if bool(row.get("materialized")) != materialized:
                raise ValueError(
                    f"Transition evidence alias materialization mismatch: {candidate_id}"
                )
            if materialized:
                transition = self._transition_by_id[source_transition_id]
                if (
                    str(transition.get("outcome") or "")
                    != str(row.get("outcome") or "")
                    or str(transition.get("before_code_sha256") or "") != before_sha
                    or str(transition.get("after_code_sha256") or "") != after_sha
                ):
                    raise ValueError(
                        "Transition evidence alias/source capsule mismatch: "
                        f"{candidate_id}"
                    )
            output[candidate_id] = row
        return output

    @staticmethod
    def _candidate_id(row: Mapping[str, Any]) -> str:
        return str(row.get("id") or row.get("candidate_id") or "")

    def _candidate_transition_ids(
        self,
        row: Mapping[str, Any],
        *,
        active_transitions_for_sop: Callable[[str], Iterable[str]],
    ) -> tuple[str, list[str]]:
        candidate_id = self._candidate_id(row)
        alias = self._candidate_aliases.get(candidate_id)
        if alias is not None:
            return (
                "selected_atomic_alias_to_source_transition"
                if alias["alias_kind"] == "atomic_transition"
                else "selected_repair_claim_alias_to_source_transition",
                [str(alias["source_transition_id"])],
            )
        graph_node = self._graph_nodes.get(candidate_id) or {}
        node_type = str(graph_node.get("type") or "")
        if candidate_id.startswith("atomic-transition::"):
            return "selected_atomic_alias_to_source_transition", [candidate_id]
        if candidate_id.startswith("repair-claim::"):
            atomic_id = candidate_id.replace(
                "repair-claim::", "atomic-transition::", 1
            )
            atomic = self._graph_nodes.get(atomic_id)
            atomic = atomic if isinstance(atomic, Mapping) else {}
            claim = atomic.get("atomic_repair_claim")
            claim = claim if isinstance(claim, Mapping) else {}
            source_transition_id = str(claim.get("source_transition_id") or "")
            source = self._graph_nodes.get(source_transition_id)
            exact_binding = bool(
                atomic.get("type") == "Transition"
                and str(atomic.get("task") or "") == self.task_id
                and str(claim.get("task_id") or "") == self.task_id
                and isinstance(source, Mapping)
                and source.get("type") == "Transition"
                and str(source.get("task") or "") == self.task_id
                and str(claim.get("source_parent_node_id") or "")
                == str(source.get("parent_node_id") or "")
                and str(claim.get("source_child_node_id") or "")
                == str(source.get("child_node_id") or "")
                and str(atomic.get("parent_node_id") or "")
                == str(source.get("parent_node_id") or "")
                and str(atomic.get("child_node_id") or "")
                == str(source.get("child_node_id") or "")
            )
            # A repair-claim is a compact alias.  Its exact atomic node already
            # carries the authoritative pointer to the historical executable
            # transition.  Follow that pointer before considering the atomic
            # metadata node itself; no semantic or title matching is involved.
            refs = [source_transition_id] if exact_binding else [atomic_id]
            return (
                "selected_repair_claim_alias_to_source_transition",
                refs,
            )
        if node_type == "Transition":
            return "selected_transition", [candidate_id]
        if node_type == "RunNode":
            return "selected_run_node_to_incoming_transition", list(
                self._transitions_by_child.get(candidate_id, [])
            )

        source = str(row.get("source") or "")
        if source == "sop" or node_type == "SOP":
            refs = [
                str(value)
                for value in (
                    row.get("clean_supporting_transition_ids")
                    or graph_node.get("supporting_transition_ids")
                    or []
                )
            ]
            refs.extend(str(value) for value in active_transitions_for_sop(candidate_id))
            return "selected_sop_to_supporting_transition", list(dict.fromkeys(refs))
        return "unsupported_candidate_type", []

    @staticmethod
    def _priority(row: Mapping[str, Any], stage: str) -> tuple[Any, ...]:
        class_rank = {
            "official_observed": 0,
            "strict_internal_observed": 1,
            "strict_debug_observed": 0,
        }
        improvement = row.get("metric_improvement")
        numeric_improvement = (
            float(improvement)
            if isinstance(improvement, (int, float)) and not isinstance(improvement, bool)
            else 0.0
        )
        return (
            class_rank.get(str(row.get("evidence_class") or ""), 99),
            -numeric_improvement if stage == "improve" else 0.0,
            str(row.get("transition_id") or ""),
        )

    def _open(
        self,
        transition: Mapping[str, Any],
        *,
        candidate_id: str,
        candidate_source: str,
        resolution_path: str,
    ) -> dict[str, Any]:
        parent_id = str(transition["parent_node_id"])
        child_id = str(transition["child_node_id"])
        before_sha = str(transition["before_code_sha256"])
        after_sha = str(transition["after_code_sha256"])
        before_code = self._code_by_sha[before_sha]
        after_code = self._code_by_sha[after_sha]
        evidence_class = str(transition["evidence_class"])
        return {
            "candidate_id": candidate_id,
            "candidate_source": candidate_source,
            "resolved_transition_id": str(transition["transition_id"]),
            "resolution_path": resolution_path,
            "evidence_class": evidence_class,
            "source_evidence_class": evidence_class,
            "metric_authority": evidence_class,
            "metric_authorized": evidence_class
            in {"official_observed", "strict_internal_observed"},
            "code_visibility": "full_transition_code",
            "outcome": str(transition["outcome"]),
            "parent_node_id": parent_id,
            "child_node_id": child_id,
            "before_code_sha256": before_sha,
            "after_code_sha256": after_sha,
            "parent_metric": transition.get("parent_metric"),
            "child_metric": transition.get("child_metric"),
            "metric_improvement": transition.get("metric_improvement"),
            "metric_provenance": transition.get("metric_provenance"),
            "stage_pair": transition.get("stage_pair"),
            "source_journal": str(transition.get("source_journal") or ""),
            "source_journal_sha256": str(
                transition.get("source_journal_sha256") or ""
            ),
            "canonical_diff": _canonical_diff(
                before_code, after_code, parent_id, child_id
            ),
            "before_code": before_code,
            "after_code": after_code,
        }

    @staticmethod
    def _as_debug_repair_reference(opened: Mapping[str, Any]) -> dict[str, Any]:
        """Expose Debug code in Improve without upgrading its metric claim."""

        result = dict(opened)
        result["source_evidence_class"] = str(
            result.get("source_evidence_class")
            or result.get("evidence_class")
            or ""
        )
        result["evidence_class"] = "debug_repair_reference"
        result["metric_authority"] = "reference_only"
        result["metric_authorized"] = False
        result["metric_improvement"] = None
        return result

    def _open_structured_repair_reference(
        self,
        *,
        candidate_id: str,
        candidate_source: str,
        resolution_path: str,
    ) -> dict[str, Any] | None:
        """Open an exact atomic repair diff when its full program is unavailable.

        This is deliberately narrower than opening the historical program.  It
        exposes only the independently bound repair action and before/after
        symbols from the hash-bound graph, so a quarantined surrounding program
        cannot leak into Strategy.  It never carries metric authority.
        """

        if not candidate_id.startswith("repair-claim::"):
            return None
        atomic_id = candidate_id.replace(
            "repair-claim::", "atomic-transition::", 1
        )
        atomic = self._graph_nodes.get(atomic_id)
        if not isinstance(atomic, Mapping) or atomic.get("type") != "Transition":
            return None
        if str(atomic.get("task") or "") != self.task_id:
            return None
        claim = atomic.get("atomic_repair_claim")
        if not isinstance(claim, Mapping):
            return None
        verification = claim.get("verification")
        verification = verification if isinstance(verification, Mapping) else {}
        source_transition_id = str(claim.get("source_transition_id") or "")
        source = self._graph_nodes.get(source_transition_id)
        if not isinstance(source, Mapping) or source.get("type") != "Transition":
            return None
        parent_id = str(source.get("parent_node_id") or "")
        child_id = str(source.get("child_node_id") or "")
        parent = self._graph_nodes.get(parent_id)
        child = self._graph_nodes.get(child_id)
        before_sha = str(verification.get("before_code_sha256") or "")
        after_sha = str(verification.get("after_code_sha256") or "")
        required_flags = (
            "claim_scope_independently_audited",
            "observed_child_execution_success",
            "observed_parent_failure",
            "repair_action_bound_to_transition",
        )
        if (
            str(claim.get("schema") or "")
            != "mlevolve_atomic_memory_claim_v1"
            or str(claim.get("claim_status") or "") != "authorized_debug_only"
            or str(claim.get("task_id") or "") != self.task_id
            or str(claim.get("outcome") or "") != "debug_fixed"
            or str(source.get("outcome") or "") != "debug_fixed"
            or str(claim.get("source_parent_node_id") or "") != parent_id
            or str(claim.get("source_child_node_id") or "") != child_id
            or not isinstance(parent, Mapping)
            or not isinstance(child, Mapping)
            or before_sha != str(parent.get("code_sha256") or "")
            or after_sha != str(child.get("code_sha256") or "")
            or any(verification.get(flag) is not True for flag in required_flags)
        ):
            return None
        before_after = [
            dict(row)
            for row in claim.get("before_after") or []
            if isinstance(row, Mapping)
            and str(row.get("before") or "")
            and str(row.get("after") or "")
        ]
        repair_action = str(claim.get("repair_action") or "").strip()
        if not before_after and not repair_action:
            return None
        repair_diff = "\n".join(
            line
            for row in before_after
            for line in (
                f"- {str(row.get('symbol') or 'value')}: {row['before']}",
                f"+ {str(row.get('symbol') or 'value')}: {row['after']}",
            )
        )
        if not repair_diff:
            repair_diff = repair_action
        return {
            "candidate_id": candidate_id,
            "candidate_source": candidate_source,
            "resolved_transition_id": source_transition_id,
            "atomic_transition_id": atomic_id,
            "resolution_path": resolution_path,
            "evidence_class": "debug_repair_reference",
            "source_evidence_class": "authorized_debug_only",
            "metric_authority": "reference_only",
            "metric_authorized": False,
            "code_visibility": "verified_repair_diff",
            "outcome": "debug_fixed",
            "parent_node_id": parent_id,
            "child_node_id": child_id,
            "before_code_sha256": before_sha,
            "after_code_sha256": after_sha,
            "parent_metric": source.get("parent_metric"),
            "child_metric": source.get("child_metric"),
            "metric_improvement": None,
            "metric_provenance": None,
            "stage_pair": source.get("stage_pair"),
            "repair_action": repair_action,
            "repair_diff": repair_diff,
            "before_after": before_after,
            "failure_signature": copy.deepcopy(
                claim.get("failure_signature") or {}
            ),
        }

    def resolve(
        self,
        *,
        selected_items: Iterable[Mapping[str, Any]],
        stage: str,
        task_id: str,
        active_transitions_for_sop: Callable[[str], Iterable[str]],
    ) -> dict[str, Any]:
        selected = [dict(row) for row in selected_items]
        selected_ids = [self._candidate_id(row) for row in selected]
        stage = str(stage).lower()
        resolved: list[dict[str, Any]] = []
        unresolved: list[dict[str, Any]] = []
        opened_pairs: set[str] = set()

        if str(task_id) != self.task_id:
            raise ValueError(
                "Evidence Resolver task identity mismatch: "
                f"capsule={self.task_id} query={task_id}"
            )
        if stage not in SUPPORTED_STAGES:
            unresolved = [
                {"candidate_id": candidate_id, "reason": "stage_not_supported"}
                for candidate_id in selected_ids
            ]
        else:
            required_outcome = "debug_fixed" if stage == "debug" else "metric_improved"
            for row in selected:
                candidate_id = self._candidate_id(row)
                path, transition_ids = self._candidate_transition_ids(
                    row,
                    active_transitions_for_sop=active_transitions_for_sop,
                )
                eligible = [
                    self._transition_by_id[transition_id]
                    for transition_id in transition_ids
                    if transition_id in self._transition_by_id
                    and self._transition_by_id[transition_id]["outcome"]
                    == required_outcome
                ]
                eligible.sort(key=lambda item: self._priority(item, stage))
                chosen = next(
                    (
                        item
                        for item in eligible
                        if str(item["pair_key"]) not in opened_pairs
                    ),
                    None,
                )
                reference_only = False
                if chosen is None and stage == "improve" and path == (
                    "selected_repair_claim_alias_to_source_transition"
                ):
                    debug_references = [
                        self._transition_by_id[transition_id]
                        for transition_id in transition_ids
                        if transition_id in self._transition_by_id
                        and self._transition_by_id[transition_id]["outcome"]
                        == "debug_fixed"
                    ]
                    debug_references.sort(
                        key=lambda item: self._priority(item, "debug")
                    )
                    chosen = next(
                        (
                            item
                            for item in debug_references
                            if str(item["pair_key"]) not in opened_pairs
                        ),
                        None,
                    )
                    reference_only = chosen is not None
                structured_reference = None
                if chosen is None and path == (
                    "selected_repair_claim_alias_to_source_transition"
                ):
                    structured_reference = self._open_structured_repair_reference(
                        candidate_id=candidate_id,
                        candidate_source=str(row.get("source") or ""),
                        resolution_path=path,
                    )
                if chosen is None:
                    if structured_reference is not None:
                        structured_pair = _pair_key(
                            "debug_repair_reference",
                            str(structured_reference["before_code_sha256"]),
                            str(structured_reference["after_code_sha256"]),
                        )
                        if (
                            len(resolved) < self.max_pairs
                            and structured_pair not in opened_pairs
                        ):
                            opened_pairs.add(structured_pair)
                            resolved.append(structured_reference)
                            continue
                    unresolved.append(
                        {
                            "candidate_id": candidate_id,
                            "resolution_path": path,
                            "reason": (
                                "resolver_pair_limit_reached"
                                if len(resolved) >= self.max_pairs
                                else "no_stage_compatible_materialized_transition"
                            ),
                            "candidate_transition_ids": transition_ids,
                        }
                    )
                    continue
                if len(resolved) >= self.max_pairs:
                    unresolved.append(
                        {
                            "candidate_id": candidate_id,
                            "resolution_path": path,
                            "reason": "resolver_pair_limit_reached",
                            "candidate_transition_ids": transition_ids,
                        }
                    )
                    continue
                opened_pairs.add(str(chosen["pair_key"]))
                opened = self._open(
                    chosen,
                    candidate_id=candidate_id,
                    candidate_source=str(row.get("source") or ""),
                    resolution_path=path,
                )
                if reference_only or str(chosen.get("outcome") or "") == (
                    "debug_fixed"
                ):
                    opened = self._as_debug_repair_reference(opened)
                resolved.append(opened)

        receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "status": (
                "resolved"
                if resolved and not unresolved
                else "partially_resolved"
                if resolved
                else "unresolved"
                if selected
                else "no_selection"
            ),
            "stage": stage,
            "task_id": str(task_id),
            "selected_candidate_ids": selected_ids,
            "selected_ids_unchanged": True,
            "opened_transition_ids": [
                str(row["resolved_transition_id"]) for row in resolved
            ],
            "opened_pair_count": len(resolved),
            "unresolved": unresolved,
            "fallback_used": False,
            "capsule_file_sha256": self.file_sha256,
            "capsule_sha256": self.capsule_sha256,
            "graph_file_sha256": self.graph_file_sha256,
            # Full code is deliberately recorded here, after Judge selection.
            "opened_evidence": copy.deepcopy(resolved),
        }
        receipt["receipt_sha256"] = _sha256_text(_canonical_json(receipt))
        return receipt


__all__ = [
    "RECEIPT_SCHEMA",
    "SCHEMA",
    "TransitionEvidenceResolver",
]
