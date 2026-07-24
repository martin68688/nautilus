"""
Global memory layer for the search process.

Stores and retrieves node-level experience (plan, code summary, stage, label)
across the whole run for similarity search and guidance. Task-scoped (one directory per task).
"""

import json
import logging
import os
import threading
import uuid
from dataclasses import asdict, is_dataclass
from functools import wraps
from pathlib import Path
from typing import Dict, Any, Iterable, Optional, List, Tuple
from datetime import datetime

from authority.models import DecisionOutcome, Operation, canonical_operation
from authority.stage_ontology import (
    GenerationStage,
    GovernanceStage,
    resolve_stage_axes,
)

from .record import MemRecord
from .retriever import HybridRetriever
from .embedding_models import EmbeddingModel

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None

logger = logging.getLogger("memory")


_PERSISTED_METADATA_KEYS = (
    "exec_time",
    "parent_metric",
    "current_metric",
    "parent_error",
    "source_node_ids",
    "code_sha256",
    "leakage_audit",
    "protocol_repair",
    "memory_type",
    "artifact_id",
    "task_id",
    "claim_refs",
    "authority_decision_refs",
    "authority_decisions",
    "authority_policy_version",
    "protocol_ref",
)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _synchronized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapped


class GlobalMemoryLayer:
    """
    Global memory for the search: save nodes, retrieve similar/dissimilar
    records, generate guidance text. Uses MemRecord and a separate metadata map.
    """

    def __init__(
        self,
        memory_dir: str,
        embedding_model_path: str = "",
        embedding_device: str = "cuda",
        similarity_threshold: float = 0.7,
    ):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.similarity_threshold = similarity_threshold

        self.embedding_model = EmbeddingModel(
            model_type="local",
            model_name=embedding_model_path,
            device=embedding_device,
        )
        self.retriever = HybridRetriever(self.embedding_model)

        self._lock = threading.RLock()
        self._load_error: str | None = None
        self.authority_mode: str = "off"
        self.active_protocol_ref: str = ""
        self.authority_policy_version: str = ""
        self.active_task_id: str = ""
        self.authority_decisions: Dict[str, Any] = {}
        self.authority_enforce_operations: set[str] = set()
        self.authority_enforce_generation_stages: set[str] = set()
        self.authority_enforce_governance_stages: set[str] = set()
        self.records: List[MemRecord] = []
        self.node_metadata_map: Dict[str, Dict[str, Any]] = {}
        self._load_memory()

        logger.info(f"[GlobalMemory] Initialized with {len(self.records)} existing records")

    def configure_authority(
        self,
        *,
        mode: str,
        active_protocol_ref: str,
        policy_version: str,
        active_task_id: str,
        decisions: Dict[str, Any],
        enforce_operations: Iterable[Operation | str] = (),
        enforce_generation_stages: Iterable[GenerationStage | str] = (),
        enforce_governance_stages: Iterable[GovernanceStage | str] = (),
    ) -> None:
        mode = str(mode or "off").lower()
        if mode not in {"off", "shadow", "enforce"}:
            raise ValueError(f"Unsupported GlobalMemory authority mode: {mode}")
        self.authority_mode = mode
        self.active_protocol_ref = str(active_protocol_ref or "")
        self.authority_policy_version = str(policy_version or "")
        self.active_task_id = str(active_task_id or "")
        # Keep the live engine mapping so decisions issued immediately before a
        # memory write are snapshotted into the persisted record.
        self.authority_decisions = decisions
        self.authority_enforce_operations = {
            canonical_operation(value).value for value in enforce_operations
        }
        self.authority_enforce_generation_stages = {
            str(getattr(value, "value", value))
            for value in enforce_generation_stages
            if str(getattr(value, "value", value))
        }
        self.authority_enforce_governance_stages = {
            str(getattr(value, "value", value))
            for value in enforce_governance_stages
            if str(getattr(value, "value", value))
        }

    def _should_enforce_request(
        self,
        *,
        operation: Operation,
        generation_stage: GenerationStage | None,
        governance_stage: GovernanceStage,
    ) -> bool:
        if self.authority_mode != "enforce":
            return False
        operations = getattr(self, "authority_enforce_operations", set())
        generations = getattr(
            self, "authority_enforce_generation_stages", set()
        )
        governances = getattr(
            self, "authority_enforce_governance_stages", set()
        )
        if operations and operation.value not in operations:
            return False
        # Missing stage metadata cannot prove that a request is outside a
        # staged enforcement window, so it remains fail-closed.
        if (
            generations
            and generation_stage is not None
            and generation_stage.value not in generations
        ):
            return False
        if governances and governance_stage.value not in governances:
            return False
        return True

    def _decision_snapshot(self, decision_ref: str) -> dict[str, Any] | None:
        decision = getattr(self, "authority_decisions", {}).get(decision_ref)
        if decision is None:
            return None
        payload = _jsonable(decision)
        return payload if isinstance(payload, dict) else None

    @_synchronized
    def save_node(self, node, parent_node: Optional = None) -> bool:
        """Save a search node to global memory. Returns True if saved."""
        if self._load_error:
            logger.error("[GlobalMemory] Refusing to overwrite corrupt memory: %s", self._load_error)
            return False
        if not self._should_save_node(node):
            return False

        try:
            code_summary = self._extract_code_summary(node)
            label = self._determine_label(node, parent_node)

            current_metric = node.metric.value if node.metric and node.metric.value is not None else None
            parent_metric = None
            if parent_node and parent_node.metric and parent_node.metric.value is not None:
                parent_metric = parent_node.metric.value

            exec_time = getattr(node, "exec_time", None)
            timestamp = datetime.now().isoformat()

            record = MemRecord(
                record_id=f"node_{node.id}",
                title=f"{node.stage} - {node.id[:8]}",
                description=node.plan or "",
                method=code_summary,
                label=label,
                timestamp=timestamp,
            )

            decision_refs = list(getattr(node, "authority_decision_refs", []) or [])
            decision_snapshots = [
                snapshot
                for decision_ref in decision_refs
                if (snapshot := self._decision_snapshot(str(decision_ref))) is not None
            ]
            metadata = {
                "exec_time": exec_time,
                "parent_metric": parent_metric,
                "current_metric": current_metric,
                "artifact_id": str(node.id),
                "task_id": str(getattr(self, "active_task_id", "") or ""),
                "claim_refs": list(getattr(node, "claim_refs", []) or []),
                "authority_decision_refs": decision_refs,
                "authority_decisions": decision_snapshots,
                "authority_policy_version": str(
                    getattr(self, "authority_policy_version", "") or ""
                ),
                "protocol_ref": str(getattr(node, "protocol_ref", "") or ""),
            }
            if (getattr(node, "protocol_repair", None) or {}).get("state") == "completed":
                metadata["protocol_repair"] = node.protocol_repair
                metadata["memory_type"] = "protocol_repair_success"
            if node.stage == "debug" and parent_node:
                parent_error = getattr(parent_node, "term_out", "") or getattr(
                    parent_node, "execution_output", ""
                )
                if parent_error:
                    metadata["parent_error"] = parent_error

            self.node_metadata_map[record.record_id] = metadata

            if record.record_id in {r.record_id for r in self.records}:
                logger.warning(f"Record {record.record_id} already exists, skipping")
                return False

            self.records.append(record)
            self._update_index(record)
            self._save_memory()

            logger.info(
                f"[GlobalMemory] Saved node {node.id} (stage={node.stage}, label={label}, "
                f"parent_metric={parent_metric}, current_metric={current_metric}, exec_time={exec_time})"
            )
            return True

        except Exception as e:
            logger.error(f"[GlobalMemory] Failed to save node {node.id}: {e}")
            return False

    @_synchronized
    def save_leakage_audit(self, node) -> bool:
        """Persist an audit finding as negative memory keyed by source-code hash."""
        if self._load_error:
            logger.error("[GlobalMemory] Refusing to overwrite corrupt memory: %s", self._load_error)
            return False
        audit = getattr(node, "leakage_audit", None) or {}
        if audit.get("memory_disposition") == "positive_eligible":
            return False
        digest = str(audit.get("code_sha256") or "")
        if not digest:
            return False
        issues = [item for item in audit.get("issues", []) if isinstance(item, dict)]
        descriptions = [
            f"[{item.get('issue_code')}] {item.get('evidence')}"
            for item in issues
            if item.get("issue_code") or item.get("evidence")
        ]
        remediations = [
            str(item.get("remediation"))
            for item in issues
            if item.get("remediation")
        ]
        record_id = f"leakage_{digest}"
        if record_id in {record.record_id for record in self.records}:
            metadata = self.node_metadata_map.setdefault(record_id, {})
            occurrences = list(metadata.get("source_node_ids") or [])
            if node.id not in occurrences:
                occurrences.append(node.id)
            metadata["source_node_ids"] = occurrences
            metadata["leakage_audit"] = audit
            if getattr(node, "protocol_repair", None):
                metadata["protocol_repair"] = node.protocol_repair
            self._save_memory()
            return False

        record = MemRecord(
            record_id=record_id,
            title=f"leakage_failure - {digest[:8]}",
            description="\n".join(descriptions) or f"Audit status: {audit.get('status')}",
            method="\n".join(dict.fromkeys(remediations)) or "Require a clean audit before reuse.",
            label=-1,
            timestamp=datetime.now().isoformat(),
        )
        self.records.append(record)
        self.node_metadata_map[record_id] = {
            "source_node_ids": [node.id],
            "code_sha256": digest,
            "leakage_audit": audit,
            "protocol_repair": getattr(node, "protocol_repair", None) or {},
        }
        self._update_index(record)
        self._save_memory()
        logger.info(
            "[GlobalMemory] Saved negative leakage memory for node %s (status=%s, hash=%s)",
            node.id,
            audit.get("status"),
            digest[:12],
        )
        return True

    @staticmethod
    def _protocol_hash(protocol_ref: str) -> str:
        _prefix, separator, digest = str(protocol_ref or "").partition("#")
        return digest if separator else ""

    def _authority_decision_candidates(self, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        by_id: dict[str, dict[str, Any]] = {}
        for payload in metadata.get("authority_decisions") or []:
            if isinstance(payload, dict) and payload.get("decision_id"):
                by_id[str(payload["decision_id"])] = payload
        for decision_ref in metadata.get("authority_decision_refs") or []:
            decision_ref = str(decision_ref)
            if decision_ref in by_id:
                continue
            snapshot = self._decision_snapshot(decision_ref)
            if snapshot is not None:
                by_id[decision_ref] = snapshot
        return [by_id[key] for key in sorted(by_id)]

    def _record_authorized(
        self,
        record: MemRecord,
        *,
        operation: Operation,
        generation_stage: GenerationStage | None,
        governance_stage: GovernanceStage,
        task_id: str,
    ) -> bool:
        # Negative/audit memory is navigational only. It can inform explicit
        # debug/inspection, never silently join a positive generation view.
        if record.label < 0:
            return operation in {
                Operation.INSPECT,
                Operation.DEBUG_HYPOTHESIS,
                Operation.DISTILL_DIAGNOSTIC,
            } or generation_stage == GenerationStage.DEBUG

        metadata = self.node_metadata_map.get(record.record_id, {})
        active_protocol_ref = str(getattr(self, "active_protocol_ref", "") or "")
        active_policy = str(getattr(self, "authority_policy_version", "") or "")
        active_hash = self._protocol_hash(active_protocol_ref)
        if not all((active_protocol_ref, active_policy, active_hash, task_id)):
            return False
        if metadata.get("protocol_ref") != active_protocol_ref:
            return False
        if generation_stage is None:
            return False

        refs = {str(value) for value in metadata.get("authority_decision_refs") or []}
        claim_refs = {str(value) for value in metadata.get("claim_refs") or []}
        artifact_id = str(metadata.get("artifact_id") or record.record_id.removeprefix("node_"))
        metadata_policy = str(metadata.get("authority_policy_version") or "")
        if metadata_policy and metadata_policy != active_policy:
            return False

        for decision in self._authority_decision_candidates(metadata):
            if str(decision.get("decision_id") or "") not in refs:
                continue
            if str(decision.get("outcome") or "") not in {
                DecisionOutcome.ALLOW.value,
                DecisionOutcome.ALLOW_WITH_WARNING.value,
            }:
                continue
            if str(decision.get("policy_version") or "") != active_policy:
                continue
            if str(decision.get("artifact_id") or "") != artifact_id:
                continue
            if claim_refs and str(decision.get("claim_id") or "") not in claim_refs:
                continue
            try:
                decision_operation = canonical_operation(str(decision.get("operation") or ""))
            except ValueError:
                continue
            if decision_operation != operation:
                continue
            if str(decision.get("generation_stage") or "") != generation_stage.value:
                continue
            if str(decision.get("governance_stage") or "") != governance_stage.value:
                continue

            scope = decision.get("permitted_scope")
            if not isinstance(scope, dict):
                continue
            if operation.value not in {str(value) for value in scope.get("operations") or []}:
                continue
            if generation_stage.value not in {
                str(value) for value in scope.get("generation_stages") or []
            }:
                continue
            if governance_stage.value not in {
                str(value) for value in scope.get("governance_stages") or []
            }:
                continue
            if active_hash not in {str(value) for value in scope.get("protocol_hashes") or []}:
                continue
            if task_id not in {str(value) for value in scope.get("task_ids") or []}:
                continue
            return True
        return False

    def retrieve_similar_records(
        self,
        query_text: str,
        top_k: int = 2,
        alpha: float = 0.5,
        dissimilar: bool = False,
        label_filter: Optional[int] = None,
        stage_filter: Optional[str] = None,
        min_score: float = 0.0,
        authority_operation: Operation | str = Operation.GENERATE_CANDIDATE,
        authority_generation_stage: GenerationStage | str | None = None,
        authority_governance_stage: GovernanceStage | str = GovernanceStage.RETRIEVAL,
        authority_task_id: str | None = None,
    ) -> List[Tuple[MemRecord, float]]:
        """Retrieve similar or dissimilar records. Returns list of (record, score)."""
        if not self.records:
            logger.info("[GlobalMemory] No records available for retrieval")
            return []

        if label_filter is not None or stage_filter is not None:
            debug_records = [
                r
                for r in self.records
                if (label_filter is None or r.label == label_filter)
                and (stage_filter is None or self._extract_stage_from_record(r) == stage_filter)
            ]
            logger.debug(
                f"[GlobalMemory] Total records: {len(self.records)}, matching "
                f"label={label_filter}, stage={stage_filter}: {len(debug_records)} records"
            )

        all_results = self.retriever.search(query_text, top_k=len(self.records), alpha=alpha)
        if self.authority_mode == "enforce":
            operation = canonical_operation(authority_operation)
            governance_stage = GovernanceStage(str(getattr(
                authority_governance_stage, "value", authority_governance_stage
            )))
            generation_stage = None
            if authority_generation_stage is not None:
                generation_stage = GenerationStage(str(getattr(
                    authority_generation_stage, "value", authority_generation_stage
                )))
            elif stage_filter:
                try:
                    generation_stage = resolve_stage_axes(
                        runtime_stage=stage_filter
                    ).generation_stage
                except ValueError:
                    generation_stage = None
            task_id = str(
                authority_task_id
                if authority_task_id is not None
                else getattr(self, "active_task_id", "")
            )
            if self._should_enforce_request(
                operation=operation,
                generation_stage=generation_stage,
                governance_stage=governance_stage,
            ):
                all_results = [
                    (record, score)
                    for record, score in all_results
                    if self._record_authorized(
                        record,
                        operation=operation,
                        generation_stage=generation_stage,
                        governance_stage=governance_stage,
                        task_id=task_id,
                    )
                ]
        logger.debug(f"[GlobalMemory] Retriever returned {len(all_results)} results for query (length={len(query_text)})")

        if label_filter is not None:
            before_label = len(all_results)
            all_results = [(r, s) for r, s in all_results if r.label == label_filter]
            logger.debug(f"[GlobalMemory] After label_filter={label_filter}: {len(all_results)}/{before_label} records")
        if stage_filter is not None:
            before_stage = len(all_results)
            all_results = [
                (r, s) for r, s in all_results
                if self._extract_stage_from_record(r) == stage_filter
            ]
            logger.debug(f"[GlobalMemory] After stage_filter={stage_filter}: {len(all_results)}/{before_stage} records")

        if dissimilar:
            filtered_results = list(all_results)
            filtered_results.sort(key=lambda x: x[1])
        else:
            filtered_results = [(record, score) for record, score in all_results if score >= min_score]

        result = filtered_results[:top_k]
        logger.info(
            f"[GlobalMemory] Retrieved {len(result)} records "
            f"(dissimilar={dissimilar}, label_filter={label_filter}, stage_filter={stage_filter}, min_score={min_score}, "
            f"after_label_stage_filter={len(all_results)}, after_min_score_filter={len(filtered_results)})"
        )
        return result

    def generate_guidance_prompt(
        self,
        query_text: str,
        top_k: int = 2,
        alpha: float = 0.5,
        dissimilar: bool = False,
        stage_filter: Optional[str] = None,
        authority_operation: Operation | str = Operation.GENERATE_CANDIDATE,
        authority_generation_stage: GenerationStage | str | None = None,
        authority_governance_stage: GovernanceStage | str = GovernanceStage.RETRIEVAL,
        authority_task_id: str | None = None,
    ) -> str:
        """Build guidance text from retrieved records."""
        similar_records = self.retrieve_similar_records(
            query_text=query_text,
            top_k=top_k,
            alpha=alpha,
            dissimilar=dissimilar,
            stage_filter=stage_filter,
            authority_operation=authority_operation,
            authority_generation_stage=authority_generation_stage,
            authority_governance_stage=authority_governance_stage,
            authority_task_id=authority_task_id,
        )

        if not similar_records:
            if dissimilar:
                return "No previous records found in memory. You can explore freely."
            return "No similar records found in memory. This is a novel direction."

        if dissimilar:
            guidance_parts = [
                "## Historical Attempts:",
                "",
                "The following approaches have been tried:",
                "",
            ]
        else:
            guidance_parts = [
                "## Historical Attempts: Similar Experiences",
                "",
                "The following similar approaches have been tried:",
                "",
            ]

        for idx, (record, score) in enumerate(similar_records, 1):
            guidance_parts.append(f"**Record #{idx}**:")
            stage = self._extract_stage_from_record(record)
            guidance_parts.append(f"- Stage: {stage}")
            guidance_parts.append(f"- Plan: {record.description}...")

            if stage == "debug":
                guidance_parts.append(f"- Result: ✅ Fixed successfully")
            elif stage == "leakage_failure":
                guidance_parts.append("- Result: Blocked or protocol-biased by leakage audit")
                guidance_parts.append(f"- Required fix: {record.method}")
            elif stage in ["draft", "fusion_draft"]:
                guidance_parts.append(f"- Result: ✅ Generated successfully")
            elif stage in ["improve", "evolution", "fusion"]:
                if record.label == 1:
                    result_text = "✅ Result Improved"
                elif record.label == -1:
                    result_text = "❌ Result Worsened"
                else:
                    result_text = "⚪ Result No change"
                if record.record_id in self.node_metadata_map:
                    meta = self.node_metadata_map[record.record_id]
                    pm, cm = meta.get("parent_metric"), meta.get("current_metric")
                    if pm is not None and cm is not None:
                        result_text += f" (Metric: {pm:.6f} → {cm:.6f})"
                guidance_parts.append(f"- Result: {result_text}")
            else:
                if record.label == 1:
                    result_text = "✅ Success"
                elif record.label == -1:
                    result_text = "❌ Failed"
                else:
                    result_text = "⚪ Neutral"
                if record.record_id in self.node_metadata_map:
                    cm = self.node_metadata_map[record.record_id].get("current_metric")
                    if cm is not None:
                        result_text += f" (Metric: {cm:.6f})"
                guidance_parts.append(f"- Result: {result_text}")

            guidance_parts.append("")

        return "\n".join(guidance_parts)

    def _extract_stage_from_record(self, record: MemRecord) -> str:
        if " - " in record.title:
            return record.title.split(" - ")[0]
        return "unknown"

    def _should_save_node(self, node) -> bool:
        if node.is_buggy:
            return False
        if not node.metric or node.metric.value is None:
            return False
        return True

    def _extract_code_summary(self, node) -> str:
        if hasattr(node, "code_summary") and node.code_summary:
            return node.code_summary
        if node.plan:
            return node.plan[:500]
        if hasattr(node, "code") and node.code:
            import re
            functions = re.findall(r"def\s+(\w+)", node.code)
            classes = re.findall(r"class\s+(\w+)", node.code)
            parts = []
            if classes:
                parts.append(f"Classes: {', '.join(classes[:3])}")
            if functions:
                parts.append(f"Functions: {', '.join(functions[:5])}")
            return "; ".join(parts) if parts else "Code available"
        return "No summary available"

    def _determine_label(self, node, parent_node: Optional) -> int:
        if node.stage in ("draft", "fusion_draft"):
            return 1
        if node.stage == "debug":
            if parent_node and parent_node.is_buggy and not node.is_buggy:
                return 1
            return -1
        if node.stage in ("improve", "evolution", "fusion") and parent_node and parent_node.metric and node.metric:
            bm, cm = parent_node.metric.value, node.metric.value
            if node.metric.maximize:
                if cm > bm:
                    return 1
                if cm < bm:
                    return -1
            else:
                if cm < bm:
                    return 1
                if cm > bm:
                    return -1
        return 0

    def _update_index(self, record: MemRecord) -> None:
        text = self._extract_text(record)
        if self.retriever.vector_index is not None and len(self.records) > 1:
            self.retriever.add_to_index([record], [text])
        else:
            texts = [self._extract_text(r) for r in self.records]
            self.retriever.build_index(self.records, texts)

    def _extract_text(self, record: MemRecord) -> str:
        stage = self._extract_stage_from_record(record)
        if stage == "debug":
            if record.record_id in self.node_metadata_map:
                metadata = self.node_metadata_map[record.record_id]
                parent_error = metadata.get("parent_error", "")
                if parent_error:
                    logger.debug(f"[GlobalMemory] Using parent_error for debug record {record.record_id[:8]}, error_length={len(parent_error)}")
                    return parent_error
                logger.warning(f"[GlobalMemory] Debug record {record.record_id[:8]} has no parent_error, falling back to description+method")
            else:
                logger.warning(f"[GlobalMemory] Debug record {record.record_id[:8]} not found in node_metadata_map, falling back to description+method")
        return f"{record.description}\n{record.method}"

    def _load_memory(self) -> None:
        records_file = self.memory_dir / "records.json"
        if not records_file.exists():
            logger.info("[GlobalMemory] No existing memory file found, starting fresh")
            self._load_error = None
            return
        try:
            with open(records_file, "r", encoding="utf-8") as f:
                records_data = json.load(f)

            self.records = []
            self.node_metadata_map = {}

            for item in records_data:
                metadata = {}
                for key in _PERSISTED_METADATA_KEYS:
                    if key in item:
                        metadata[key] = item.pop(key)

                record = MemRecord.from_dict(item)
                self.records.append(record)
                if metadata:
                    self.node_metadata_map[record.record_id] = metadata

            if self.records:
                texts = [self._extract_text(r) for r in self.records]
                self.retriever.build_index(self.records, texts)
                logger.info(f"[GlobalMemory] Loaded {len(self.records)} records from disk")
            self._load_error = None
        except Exception as e:
            logger.error(f"[GlobalMemory] Failed to load memory: {e}")
            self._load_error = str(e)
            self.records = []
            self.node_metadata_map = {}

    @_synchronized
    def _save_memory(self) -> None:
        records_file = self.memory_dir / "records.json"
        if self._load_error:
            raise RuntimeError(f"Refusing to overwrite corrupt GlobalMemory: {self._load_error}")
        try:
            records_data = []
            for r in self.records:
                d = r.to_dict()
                if r.record_id in self.node_metadata_map:
                    meta = self.node_metadata_map[r.record_id]
                    for key in _PERSISTED_METADATA_KEYS:
                        if meta.get(key) is not None:
                            d[key] = meta[key]
                records_data.append(d)

            lock_path = self.memory_dir / "records.lock"
            with lock_path.open("a+", encoding="utf-8") as lock_file:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    existing_data = []
                    if records_file.exists():
                        existing_data = json.loads(records_file.read_text(encoding="utf-8"))
                    by_id = {
                        str(item.get("record_id")): item
                        for item in existing_data
                        if isinstance(item, dict) and item.get("record_id")
                    }
                    for item in records_data:
                        record_id = str(item.get("record_id") or "")
                        previous = by_id.get(record_id, {})
                        if previous and item.get("source_node_ids") is not None:
                            item["source_node_ids"] = list(dict.fromkeys([
                                *(previous.get("source_node_ids") or []),
                                *(item.get("source_node_ids") or []),
                            ]))
                        by_id[record_id] = item
                    tmp = records_file.with_suffix(
                        f".{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
                    )
                    with open(tmp, "w", encoding="utf-8") as f:
                        json.dump(list(by_id.values()), f, indent=2, ensure_ascii=False)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp, records_file)
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            logger.debug(f"[GlobalMemory] Saved {len(self.records)} records to disk")
        except Exception as e:
            logger.error(f"[GlobalMemory] Failed to save memory: {e}")
            raise
