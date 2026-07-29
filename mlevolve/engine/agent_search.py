"""AgentSearch: tree search coordinator; delegates to node_selection, evaluation, execution, solution_manager."""

import logging
import random
import time
from collections import deque
from pathlib import Path
from typing import Callable, List, Dict, Optional

from engine.executor import ExecutionResult
from engine.search_node import SearchNode, Journal
import utils.data_preview as data_preview
from config import Config
from utils.metric import WorstMetricValue
import threading
import json

from agents import (
    draft_agent, improve_agent, debug_agent,
    evolution_agent, fusion_agent, aggregation_agent,
    code_review_agent, protocol_repair_agent,
    result_parse_agent,
)
from agents import protocol_repair
from agents.triggers import (
    refresh_replay_lineage_after_instrumentation,
    refresh_replay_lineage_after_revision,
)
from agents.prompts import enforce_host_candidate_entrypoint
from protocol_runtime.preflight import (
    PreflightStatus,
    build_bounded_repair_receipt,
)
from engine import node_selection, evaluation, execution, solution_manager
from engine.conditions import is_branch_stagnant
from utils.data_preview import clean_task_desc

logger = logging.getLogger("MLEvolve")


ExecCallbackType = Callable[[str, bool], ExecutionResult]


class DraftRoleReservationError(ValueError):
    """Raised before generation when a fixed root Draft slot cannot be reserved."""


class SearchSpaceExhausted(RuntimeError):
    """Raised when selection has no current or future work left to claim."""


class AgentSearch:
    @staticmethod
    def _load_configured_memory_snapshot(
        ext_cfg,
        *,
        log_dir: Path,
        evaluation_authority,
        resolve_memory_path,
    ):
        """Load CURRENT once and return the exact Base graph/index paths."""

        from authority.memory_snapshot import MemorySnapshotLoader

        bundle_root = str(getattr(ext_cfg, "bundle_root", "") or "").strip()
        if not bundle_root:
            raise ValueError("external_skill_memory.bundle_root is required")
        resolved_bundle_root = resolve_memory_path(bundle_root)
        overlay_setting = str(
            getattr(ext_cfg, "session_overlay_path", "") or ""
        ).strip()
        overlay_path = (
            Path(overlay_setting).expanduser()
            if overlay_setting
            else Path(log_dir) / "session_overlay"
        )
        if not overlay_path.is_absolute():
            overlay_path = Path(log_dir) / overlay_path
        snapshot = MemorySnapshotLoader(resolved_bundle_root).load(
            current_path=str(
                getattr(ext_cfg, "current_pointer_path", "CURRENT.json")
                or "CURRENT.json"
            ),
            session_overlay_path=overlay_path,
            active_protocol_ref=evaluation_authority.active_protocol.key(),
            authority_policy_version=evaluation_authority.engine.policy_version,
        )
        evaluation_authority.configure_memory_snapshot(snapshot)
        return (
            snapshot,
            str(snapshot.base_bundle.path / "runforest" / "graph.json"),
            str(snapshot.base_bundle.path / "runforest" / "index.npz"),
        )

    def __init__(
            self,
            task_desc: str,
            cfg: Config,
            journal: Journal,
    ):
        self.cfg = cfg
        self.acfg = cfg.agent
        self.scfg = cfg.agent.search
        self.task_desc = clean_task_desc(task_desc, cfg)
        self.journal = journal
        self.journal.audit_enforced = bool(self.acfg.check_data_leakage)
        self.data_preview: str | None = None
        self.current_step = 0
        self.current_node: SearchNode | None = None
        self.all_root = True
        self.virtual_root = SearchNode(parent=None, plan="(root)", code="", metric=WorstMetricValue(),
                                     stage="root")
        self.current_node_list = []
        self.journal.append(self.virtual_root)
        self.best_metric: float = None
        self.best_node: SearchNode = None
        self.search_start_time = None
        self.journal_lock = threading.Lock()
        self.save_node_lock = threading.Lock()
        self.start_time = time.time()
        self.use_stepwise_generation = True

        self._draft_role_lock = threading.Lock()
        self._draft_generation_count = 0
        self._replacement_draft_lock = threading.Lock()
        self._replacement_draft_count = 0
        self._replacement_draft_inflight = False
        self._search_condition = threading.Condition()
        self._active_search_work_lock = threading.Lock()
        self._active_search_work = 0
        self._validate_draft_role_policy()
        self._init_mandatory_repair_scheduler()

        self.next_branch_id = 1
        self.branch_all_nodes: Dict[int, List[SearchNode]] = {}
        self.branch_successful_nodes: Dict[int, List[SearchNode]] = {}
        self.branch_node_count: Dict[int, int] = {}
        self.use_coldstart = cfg.coldstart.use_coldstart
        self.coldstart_description = cfg.coldstart.description
        # Adoption tracking: side-channel snapshot of methodology ref_ids (never in prompt)
        from engine.coldstart import knowledge as _kb
        self.methodology_ref_ids: list[str] = []
        self.methodology_candidates: list[dict[str, str]] = [
            dict(item)
            for item in getattr(_kb, "_LAST_METHODOLOGY_CANDIDATES", [])
        ]
        self.methodology_visibility_pack = None
        self.coldstart_external_ref_ids: list[str] = list(getattr(_kb, "_LAST_RUN_FOREST_REF_IDS", []))
        self.coldstart_external_source: str = str(getattr(_kb, "_LAST_RUN_FOREST_SOURCE", "") or "")
        self.coldstart_external_memory_text: str = str(getattr(_kb, "_LAST_RUN_FOREST_TEXT", "") or "")
        self.coldstart_primary_model_name: str = str(getattr(_kb, "_LAST_PRIMARY_MODEL_NAME", "") or "")
        self.coldstart_primary_description: str = str(
            getattr(_kb, "_LAST_PRIMARY_MODEL_TEXT", "") or self.coldstart_description
        )
        self.adoption_tracking_enabled: bool = cfg.adoption_tracking.enable

        # Top-N candidates
        self.top_k = self.scfg.top_candidates_size
        self.top_candidates: List[SearchNode] = []

        # Performance stagnation detection
        self.best_metric_history = []
        self.stagnation_threshold = self.scfg.stagnation_window
        self.post_process_triggered = False
        self.post_process_attempts = 0
        self.max_post_process_attempts = 4
        self.improve_attempts_count = 0
        self.last_successful_improve_step = 0

        self.fusion_draft_count = 0
        self.max_fusion_drafts = cfg.agent.max_fusion_drafts

        self.metric_maximize: bool | None = None
        self.metric_maximize_reasoning: str | None = None
        result_parse_agent.determine_metric_direction(self)

        # One run-scoped adapter mediates every protected learning operation.
        # It owns the immutable protocol ref, evidence graph, and hash-chained
        # authority ledger under this run's log directory.
        from authority.adapters.mlevolve import MLEvolveAuthorityAdapter
        self.evaluation_authority = MLEvolveAuthorityAdapter(self)
        self.journal.authority_enforced = self.evaluation_authority.mode == "enforce"
        self.journal.authority_agent = self

        from agents.memory.prospective_audit import ProspectiveAuditLogger
        self.prospective_audit = ProspectiveAuditLogger(self)

        # Global memory
        self.global_memory = None
        if self.acfg.use_global_memory:
            try:
                from agents.memory.global_memory import GlobalMemoryLayer
                memory_dir = str(self.cfg.workspace_dir / "global_memory")
                self.global_memory = GlobalMemoryLayer(
                    memory_dir=memory_dir,
                    embedding_model_path=self.acfg.memory_embedding_model_path,
                    embedding_device=self.acfg.memory_embedding_device,
                    similarity_threshold=self.acfg.memory_similarity_threshold,
                )
                self.global_memory.configure_authority(
                    mode=self.evaluation_authority.mode,
                    active_protocol_ref=self.evaluation_authority.active_protocol.key(),
                    policy_version=self.evaluation_authority.engine.policy_version,
                    active_task_id=self.evaluation_authority.task_id,
                    decisions=self.evaluation_authority.engine.decisions,
                    enforce_operations=sorted(
                        getattr(
                            getattr(self.evaluation_authority, "rollout", None),
                            "enforce_operations",
                            (),
                        )
                    ),
                    enforce_generation_stages=sorted(
                        getattr(
                            getattr(self.evaluation_authority, "rollout", None),
                            "enforce_generation_stages",
                            (),
                        )
                    ),
                    enforce_governance_stages=sorted(
                        getattr(
                            getattr(self.evaluation_authority, "rollout", None),
                            "enforce_governance_stages",
                            (),
                        )
                    ),
                )
                logger.info(f"[AgentSearch] Global memory enabled and initialized at {memory_dir}")
            except Exception as e:
                import traceback
                logger.warning(f"[AgentSearch] Failed to initialize global memory: {e}")
                logger.debug(f"[AgentSearch] Global memory initialization traceback: {traceback.format_exc()}")
                self.global_memory = None
        else:
            logger.info("[AgentSearch] Global memory is disabled by config")

        # External persistent SkillGraph memory
        self.external_skill_memory = None
        self.memory_snapshot = None
        ext_cfg = getattr(getattr(self, "cfg", None), "external_skill_memory", None)
        if ext_cfg is not None and getattr(ext_cfg, "enable", False):
            ext_mode = getattr(ext_cfg, "mode", "skillgraph")
            try:
                from agents.memory.external_skill_memory import (
                    ExternalSkillMemoryLayer,
                    RunForestMemoryLayer,
                    resolve_memory_path,
                )
                ext_source = getattr(ext_cfg, "source_name", "skillgraph")
                ext_graph_path = getattr(ext_cfg, "graph_path", "")
                ext_index_path = getattr(ext_cfg, "index_path", "")
                bundle_root = str(getattr(ext_cfg, "bundle_root", "") or "").strip()
                if bundle_root:
                    (
                        self.memory_snapshot,
                        ext_graph_path,
                        ext_index_path,
                    ) = self._load_configured_memory_snapshot(
                        ext_cfg,
                        log_dir=Path(self.cfg.log_dir),
                        evaluation_authority=self.evaluation_authority,
                        resolve_memory_path=resolve_memory_path,
                    )
                visibility_kwargs = {}
                if str(ext_mode).lower() == "run_forest_stage_hybrid":
                    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer

                    memory_layer_cls = StageAwareHybridMemoryLayer
                    visibility_kwargs = {
                        "visibility_mode": self.evaluation_authority.mode,
                        "visibility_enforce_operations": sorted(
                            getattr(
                                getattr(self.evaluation_authority, "rollout", None),
                                "enforce_operations",
                                (),
                            )
                        ),
                        "visibility_enforce_generation_stages": sorted(
                            getattr(
                                getattr(self.evaluation_authority, "rollout", None),
                                "enforce_generation_stages",
                                (),
                            )
                        ),
                        "visibility_enforce_governance_stages": sorted(
                            getattr(
                                getattr(self.evaluation_authority, "rollout", None),
                                "enforce_governance_stages",
                                (),
                            )
                        ),
                        "visibility_authority_engine": self.evaluation_authority.engine,
                        "visibility_active_protocol": self.evaluation_authority.active_protocol,
                        "visibility_policy_version": self.evaluation_authority.engine.policy_version,
                        "visibility_task_id": self.evaluation_authority.task_id,
                        "visibility_bundle_version": (
                            self.memory_snapshot.base_bundle.bundle_version
                            if self.memory_snapshot is not None
                            else None
                        ),
                        "visibility_token_budget": getattr(
                            ext_cfg, "visibility_token_budget", 4096
                        ),
                        "memory_snapshot": self.memory_snapshot,
                        "prospective_audit_logger": self.prospective_audit,
                    }
                else:
                    memory_layer_cls = (
                        RunForestMemoryLayer
                        if "run_forest" in str(ext_mode).lower()
                        or "run_forest" in str(ext_source).lower()
                        or "run_forest" in str(ext_graph_path).lower()
                        else ExternalSkillMemoryLayer
                    )
                self.external_skill_memory = memory_layer_cls(
                    graph_path=ext_graph_path,
                    source_name=ext_source,
                    mode=ext_mode,
                    index_path=ext_index_path,
                    text_model_path=getattr(ext_cfg, "text_model_path", ""),
                    scoring_mode=getattr(ext_cfg, "scoring_mode", "lexical"),
                    geometry_distance_weight=getattr(ext_cfg, "geometry_distance_weight", 0.30),
                    geometry_semantic_weight=getattr(ext_cfg, "geometry_semantic_weight", 0.20),
                    geometry_constraint_weight=getattr(ext_cfg, "geometry_constraint_weight", 0.05),
                    geometry_condition_weight=getattr(ext_cfg, "geometry_condition_weight", 0.18),
                    geometry_failure_weight=getattr(ext_cfg, "geometry_failure_weight", 0.14),
                    geometry_evidence_weight=getattr(ext_cfg, "geometry_evidence_weight", 0.08),
                    geometry_reliability_weight=getattr(ext_cfg, "geometry_reliability_weight", 0.08),
                    geometry_conflict_weight=getattr(ext_cfg, "geometry_conflict_weight", 0.10),
                    geometry_distance_norm=getattr(ext_cfg, "geometry_distance_norm", "none"),
                    geometry_query_radius_quantile=getattr(ext_cfg, "geometry_query_radius_quantile", 0.5),
                    geometry_query_radius_mode=getattr(ext_cfg, "geometry_query_radius_mode", "predicted_distribution"),
                    geometry_query_radius_bands=getattr(ext_cfg, "geometry_query_radius_bands", ["core", "middle", "edge"]),
                    geometry_query_radius_top_bands=getattr(ext_cfg, "geometry_query_radius_top_bands", 2),
                    geometry_radius_fusion=getattr(ext_cfg, "geometry_radius_fusion", "weighted_max"),
                    enable_agentic=getattr(ext_cfg, "enable_agentic", False),
                    navigator_max_steps=getattr(ext_cfg, "navigator_max_steps", 3),
                    navigator_reference_budget=getattr(ext_cfg, "navigator_reference_budget", 1200),
                    top_k=getattr(ext_cfg, "top_k", 8),
                    depth=getattr(ext_cfg, "depth", 2),
                    beam_width=getattr(ext_cfg, "beam_width", 3),
                    general_cap=getattr(ext_cfg, "general_cap", 2),
                    task_seed_limit=getattr(ext_cfg, "task_seed_limit", 6),
                    max_chars=getattr(ext_cfg, "max_chars", 5000),
                    include_draft=getattr(ext_cfg, "include_draft", True),
                    include_improve=getattr(ext_cfg, "include_improve", True),
                    include_evolution=getattr(ext_cfg, "include_evolution", True),
                    include_debug=getattr(ext_cfg, "include_debug", True),
                    include_fusion=getattr(ext_cfg, "include_fusion", True),
                    positive_control_probe_path=getattr(
                        ext_cfg, "positive_control_probe_path", ""
                    ),
                    positive_control_force_raw=getattr(
                        ext_cfg, "positive_control_force_raw", False
                    ),
                    cfg=self.cfg,
                    **visibility_kwargs,
                )
                logger.info(
                    "[AgentSearch] External skill memory enabled: source=%s graph=%s",
                    self.external_skill_memory.source_name,
                    self.external_skill_memory.graph_path,
                )
            except Exception as e:
                import traceback
                if str(ext_mode).lower() == "run_forest_stage_hybrid":
                    raise RuntimeError(f"Failed to initialize required stage-hybrid memory: {e}") from e
                logger.warning(f"[AgentSearch] Failed to initialize external skill memory: {e}")
                logger.debug(f"[AgentSearch] External skill memory traceback: {traceback.format_exc()}")
                self.external_skill_memory = None
        else:
            logger.info("[AgentSearch] External skill memory is disabled by config")
        if self.methodology_candidates:
            from agents.memory.methodology_visibility import (
                evaluate_methodology_visibility,
            )

            methodology_text, methodology_refs, methodology_pack = (
                evaluate_methodology_visibility(
                    self, self.methodology_candidates
                )
            )
            self.methodology_visibility_pack = methodology_pack
            self.methodology_ref_ids = methodology_refs
            if methodology_text:
                self.coldstart_description = (
                    f"{self.coldstart_description}{methodology_text}"
                )
                self.cfg.coldstart.description = self.coldstart_description
            logger.info(
                "[AgentSearch] Methodology Authority gate proposed=%s visible=%s suppressed=%s",
                len(self.methodology_candidates),
                len(methodology_refs),
                len(getattr(methodology_pack, "suppressed_clause_refs", []) or []),
            )
        # Bundle binding, if configured, is complete at this point. Freeze the
        # policy/protocol/collector/bundle tuple before any retrieval decision.
        self.evaluation_authority.seal_rollout_versions()

    def _validate_draft_role_policy(self) -> None:
        policy = getattr(self.acfg, "draft_role_policy", None)
        if policy is None or not bool(getattr(policy, "enabled", False)):
            return
        allowed = {
            ("coldstart_baseline", "memory_reproduction", "novel_exploration"),
            ("coldstart_baseline", "memory_transfer", "novel_exploration"),
        }
        roles = [str(role) for role in list(getattr(policy, "roles", []) or [])]
        if tuple(roles) not in allowed:
            raise ValueError(f"RunForest draft roles must be one of {sorted(allowed)}; got {roles}")
        if int(self.acfg.initial_drafts) != 3:
            raise ValueError("RunForest draft role policy requires agent.initial_drafts == 3")
        if int(self.scfg.num_drafts) != 3:
            raise ValueError("RunForest draft role policy requires agent.search.num_drafts == 3")
        ext_cfg = getattr(getattr(self, "cfg", None), "external_skill_memory", None)
        if (
            str(getattr(ext_cfg, "retrieval_control", "")) == "layered_strategy"
            and not getattr(self, "use_stepwise_generation", True)
        ):
            raise ValueError("Layered Novel Draft retrieval requires stepwise generation")

    def configured_draft_role(self, draft_index: int) -> str:
        policy = getattr(self.acfg, "draft_role_policy", None)
        if policy is None or not bool(getattr(policy, "enabled", False)):
            return "general_draft"
        roles = [str(role) for role in list(getattr(policy, "roles", []) or [])]
        if 0 <= draft_index < len(roles):
            return roles[draft_index]
        raise DraftRoleReservationError(
            f"Draft role index {draft_index} exceeds the fixed three-role policy"
        )

    def fixed_draft_slots_exhausted(self) -> bool:
        """Whether every declared root Draft role has already been reserved."""
        policy = getattr(getattr(self, "acfg", None), "draft_role_policy", None)
        if policy is None or not bool(getattr(policy, "enabled", False)):
            return False
        roles = list(getattr(policy, "roles", []) or [])
        lock = getattr(self, "_draft_role_lock", None)
        if lock is None:
            return int(getattr(self, "_draft_generation_count", 0)) >= len(roles)
        with lock:
            return int(getattr(self, "_draft_generation_count", 0)) >= len(roles)

    def claim_draft_role(self, explicit_role: str | None = None) -> str:
        with self._draft_role_lock:
            draft_index = self._draft_generation_count
            configured_role = self.configured_draft_role(draft_index)
            policy = getattr(self.acfg, "draft_role_policy", None)
            if (
                explicit_role is not None
                and bool(getattr(policy, "enabled", False))
                and explicit_role != configured_role
            ):
                raise DraftRoleReservationError(
                    f"Draft role slot {draft_index} requires {configured_role}; got {explicit_role}"
                )
            self._draft_generation_count += 1
        role = explicit_role or configured_role
        logger.info("[draft-role] index=%s role=%s", draft_index, role)
        return role

    def _claim_replacement_draft_slot(self) -> bool:
        """Reserve one bounded root replacement after permanent branch exhaustion."""

        if not bool(getattr(self.scfg, "replacement_drafts_enabled", False)):
            return False
        limit = max(0, int(getattr(self.scfg, "max_replacement_drafts", 0)))
        lock = getattr(self, "_replacement_draft_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._replacement_draft_lock = lock
        with lock:
            if bool(getattr(self, "_replacement_draft_inflight", False)):
                return False
            used = int(getattr(self, "_replacement_draft_count", 0))
            if used >= limit:
                return False
            completed = max(0, len(self.journal) - 1)
            if completed >= int(self.acfg.steps):
                return False
            self._replacement_draft_count = used + 1
            self._replacement_draft_inflight = True
            logger.warning(
                "[replacement-draft] reserved %s/%s after all existing branches became non-expandable",
                self._replacement_draft_count,
                limit,
            )
            return True

    def _release_replacement_draft_slot(self) -> None:
        """Release only the in-flight guard; the bounded attempt stays consumed."""

        lock = getattr(self, "_replacement_draft_lock", None)
        if lock is None:
            self._replacement_draft_inflight = False
            return
        with lock:
            self._replacement_draft_inflight = False
        self._notify_search_state()

    def claim_replacement_draft_role(self) -> str:
        """Return a role without consuming the immutable initial Draft slots."""

        index = int(getattr(self, "_replacement_draft_count", 0))
        logger.info("[draft-role] replacement_index=%s role=replacement_draft", index)
        return "replacement_draft"

    def _init_mandatory_repair_scheduler(self) -> None:
        """Initialize the process-local queue that forces blocked nodes into repair."""
        self._mandatory_repair_lock = threading.Lock()
        self._mandatory_repair_queue = deque()
        self._mandatory_repair_queued_ids: set[str] = set()
        self._mandatory_repair_inflight_ids: set[str] = set()

    @staticmethod
    def _is_mandatory_repair_parent(node: SearchNode | None) -> bool:
        if node is None or node.stage == "root" or node.is_terminal:
            return False
        if protocol_repair.is_active(getattr(node, "protocol_repair", None)):
            return True
        audit = node.leakage_audit or {}
        return audit.get("status") not in {None, "clean"} and bool(
            node.audit_repair_required or audit.get("repair_required")
        )

    @classmethod
    def _is_explicit_runtime_debug_parent(cls, node: SearchNode | None) -> bool:
        """Keep a completed runtime failure on its caller-owned debug lane.

        Phase-2 workers pass the node they just executed back into ``step`` so
        that ordinary runtime failures can be debugged immediately.  The
        mandatory audit-repair queue is allowed to replace root/selection
        requests, but replacing this explicit continuation orphans the failed
        draft: its root lock remains held while the worker starts following a
        different repair transaction.

        Audit/protocol failures deliberately remain under the mandatory
        scheduler.  This guard therefore applies only to explicit, non-root
        runtime/validation failures that are not mandatory-repair parents.
        """
        return bool(
            node is not None
            and node.stage != "root"
            and not node.is_terminal
            and (node.is_buggy is True or node.is_valid is False)
            and not cls._is_mandatory_repair_parent(node)
        )

    def _enqueue_mandatory_repair(self, node: SearchNode) -> None:
        """Queue one blocked node exactly once, independently of UCT/root locks."""
        protocol_repair.ensure_transaction(self, node)
        if not self._is_mandatory_repair_parent(node):
            return
        with self._mandatory_repair_lock:
            if (
                node.id in self._mandatory_repair_queued_ids
                or node.id in self._mandatory_repair_inflight_ids
            ):
                return
            self._mandatory_repair_queue.append(node)
            self._mandatory_repair_queued_ids.add(node.id)
            node.leakage_audit["repair_queue_status"] = "queued"
        if node.protocol_repair:
            logger.warning(
                "Node %s queued for staged protocol repair (stage=%s, generation_attempts=%s)",
                node.id,
                protocol_repair.current_stage(node.protocol_repair),
                node.protocol_repair.get("stage_generation_attempts", {}),
            )
        else:
            logger.warning(
                "Node %s queued for mandatory leakage repair (attempt=%s)",
                node.id,
                node.leakage_repair_attempt + 1,
            )

    def _claim_mandatory_repair_parent(
        self,
        requested_node: SearchNode | None,
        *,
        required_draft_role: str | None = None,
        excluded_draft_role: str | None = None,
    ) -> tuple[SearchNode | None, bool]:
        """Claim the oldest repair parent; report duplicate concurrent requests."""
        if required_draft_role and excluded_draft_role:
            raise ValueError("A repair claim cannot require and exclude a Draft role")

        def role_allowed(candidate: SearchNode) -> bool:
            role = getattr(candidate, "draft_role", None)
            if required_draft_role and role != required_draft_role:
                return False
            if excluded_draft_role and role == excluded_draft_role:
                return False
            return True

        with self._mandatory_repair_lock:
            retained = deque()
            selected = None
            while self._mandatory_repair_queue:
                candidate = self._mandatory_repair_queue.popleft()
                if not self._is_mandatory_repair_parent(candidate):
                    self._mandatory_repair_queued_ids.discard(candidate.id)
                    continue
                if selected is None and role_allowed(candidate):
                    selected = candidate
                    self._mandatory_repair_queued_ids.discard(candidate.id)
                    continue
                retained.append(candidate)
            self._mandatory_repair_queue = retained

            if selected is not None:
                self._mandatory_repair_inflight_ids.add(selected.id)
                selected.leakage_audit["repair_queue_status"] = "in_flight"
                return selected, False

            if (
                requested_node
                and role_allowed(requested_node)
                and requested_node.id in self._mandatory_repair_inflight_ids
            ):
                return None, True
            if (
                self._is_mandatory_repair_parent(requested_node)
                and role_allowed(requested_node)
            ):
                self._mandatory_repair_inflight_ids.add(requested_node.id)
                requested_node.leakage_audit["repair_queue_status"] = "in_flight"
                return requested_node, False
        return None, False

    def _release_mandatory_repair_parent(
        self,
        node: SearchNode,
        *,
        retry: bool = False,
    ) -> None:
        with self._mandatory_repair_lock:
            self._mandatory_repair_inflight_ids.discard(node.id)
            journal_ids = {
                item.id for item in getattr(getattr(self, "journal", None), "nodes", [])
            }
            tx_id = (node.protocol_repair or {}).get("transaction_id")
            successors = [
                child for child in node.children
                if child.id in journal_ids
                and tx_id
                and (child.protocol_repair or {}).get("transaction_id") == tx_id
            ]
            if successors and not retry:
                successor = max(successors, key=lambda child: child.ctime)
                node.protocol_repair["state"] = (
                    "abandoned"
                    if (successor.protocol_repair or {}).get("state") == "exhausted"
                    else "superseded"
                )
                node.protocol_repair.pop("active_stage", None)
                node.protocol_repair.pop("active_generation_attempt", None)
                node.protocol_repair["successor_node_id"] = successor.id
                node.audit_repair_required = False
                node.leakage_audit["repair_required"] = False
                node.is_terminal = True
            if retry and self._is_mandatory_repair_parent(node):
                if node.id not in self._mandatory_repair_queued_ids:
                    self._mandatory_repair_queue.appendleft(node)
                    self._mandatory_repair_queued_ids.add(node.id)
                node.leakage_audit["repair_queue_status"] = "queued_after_error"
            elif (
                (node.protocol_repair or {}).get("state") == "exhausted"
                or (node.leakage_repair_attempt >= 2 and node.is_terminal)
            ):
                node.leakage_audit["repair_queue_status"] = "exhausted"
            else:
                node.leakage_audit["repair_queue_status"] = "expanded"

    @staticmethod
    def _record_protocol_generation_failure(node: SearchNode, error: Exception) -> None:
        tx = node.protocol_repair or {}
        if tx.get("state") != "stage_in_progress":
            return
        node.protocol_repair = protocol_repair.record_stage_generation_failure(
            tx,
            node.id,
            f"{type(error).__name__}: {error}",
        )
        if node.protocol_repair.get("state") == "exhausted":
            node.is_terminal = True
            node.continue_improve = False
            node.audit_repair_required = False
            node.leakage_audit["repair_required"] = False
            node.leakage_audit["repair_queue_status"] = "exhausted"
        else:
            node.leakage_audit["repair_queue_status"] = "generation_retry"

    def _serialize_prompt(self, prompt_complete) -> str | None:
        """Serialize prompt (str or dict) to string for saving in node."""
        if prompt_complete is None:
            return None
        if isinstance(prompt_complete, str):
            return prompt_complete
        elif isinstance(prompt_complete, dict):
            return json.dumps(prompt_complete, ensure_ascii=False, indent=2)
        else:
            return str(prompt_complete)

    def _notify_search_state(self) -> None:
        condition = getattr(self, "_search_condition", None)
        if condition is not None:
            with condition:
                condition.notify_all()

    def begin_search_work(self) -> None:
        """Record one process-local generation or execution that can unlock selection."""
        active_lock = getattr(self, "_active_search_work_lock", None)
        if active_lock is None:
            # Some focused unit fixtures construct AgentSearch via __new__.
            # Production instances initialize this state in __init__.
            active_lock = threading.Lock()
            self._active_search_work_lock = active_lock
            self._active_search_work = 0
        with active_lock:
            self._active_search_work += 1

    def end_search_work(self) -> None:
        """Release a process-local work claim and wake bounded selection waiters."""
        with self._active_search_work_lock:
            if self._active_search_work <= 0:
                raise RuntimeError("Search work counter underflow")
            self._active_search_work -= 1
        self._notify_search_state()

    def _has_pending_search_work(self) -> bool:
        """Whether a bounded selection miss can still resolve via in-flight work."""
        repair_lock = getattr(self, "_mandatory_repair_lock", None)
        if repair_lock is not None:
            with repair_lock:
                if self._mandatory_repair_queue or self._mandatory_repair_inflight_ids:
                    return True

        active_lock = getattr(self, "_active_search_work_lock", None)
        if active_lock is None:
            return False
        with active_lock:
            return int(getattr(self, "_active_search_work", 0)) > 0

    def update_data_preview(self):
        base_preview = data_preview.generate(self.cfg.workspace_dir)
        submission_format_warning = """

        ⚠️  CRITICAL SUBMISSION FORMAT NOTE:
        - If you see sample_submission.csv or similar files, those contain the CORRECT submission format
        - The column names in these files are the FINAL AUTHORITY for submission format
        - Always use the column names from the actual sample submission files
        """
        self.data_preview = base_preview + submission_format_warning

    def is_root(self, node: SearchNode):
        return node.id is self.virtual_root.id

    def _finalize_preflight_block(self, node: SearchNode, parent_node: SearchNode) -> bool:
        """Record a deterministically blocked node without launching its training process."""
        node.finish_time = time.strftime("%Y-%m-%dT%H:%M:%S")
        should_backpropagate = evaluation.check_improvement(self, node, parent_node)
        with self.journal_lock:
            self.journal.append(node)
        prospective_audit = getattr(self, "prospective_audit", None)
        if prospective_audit is not None:
            prospective_audit.finalize_node(node)
        self._enqueue_mandatory_repair(node)
        self._notify_search_state()
        logger.warning(
            "Node %s was journaled as a pre-execution leakage failure; GPU execution was skipped",
            node.id,
        )
        return should_backpropagate

    def _run_single_step(
        self,
        parent_node: SearchNode,
        exec_callback: ExecCallbackType,
        execute_immediately: bool = True,
        init_solution_path: Optional[str] = None,
        draft_role: Optional[str] = None,
        replacement_draft: bool = False,
    ):
        """Run one search step: select action (draft/debug/improve), execute, parse, validate."""
        result_node = None
        _root = False

        if not parent_node.is_terminal:
            try:
                if self.is_root(parent_node):
                    if replacement_draft:
                        result_node = draft_agent.run(
                            self,
                            init_solution_path=None,
                            draft_role="replacement_draft",
                            replacement=True,
                        )
                        result_node.lock = True
                        logger.warning(
                            "[_run_single_step] Generated bounded replacement Draft %s",
                            result_node.id,
                        )
                    elif (
                        self.fixed_draft_slots_exhausted()
                        or parent_node.reached_child_limit(scfg=self.scfg)
                    ):
                        aggregation_requested = bool(
                            getattr(parent_node, "_aggregation_requested", False)
                        )
                        parent_node._aggregation_requested = False
                        if aggregation_requested:
                            logger.info("🎯 Selector approved multi-branch aggregation")
                            result_node = aggregation_agent.run(self, mode="node", parent_node=parent_node)
                            if result_node:
                                result_node.lock = True
                                logger.info(f"[_run_single_step] Aggregation branch node {result_node.id} is locked.")
                            else:
                                logger.info("Aggregation failed or limit reached; returning to search.")
                        else:
                            logger.info("Root draft limit reached without aggregation approval; worker will wait.")
                            _root = True
                    else:
                        result_node = draft_agent.run(
                            self,
                            init_solution_path=init_solution_path,
                            draft_role=draft_role,
                        )
                        result_node.lock = True
                        logger.info(f"[_run_single_step] Draft node {result_node.id} is locked.")
                elif (parent_node.protocol_repair or {}).get("state") == "exhausted":
                    parent_node.is_terminal = True
                    parent_node.continue_improve = False
                    parent_node.leakage_audit["repair_queue_status"] = "exhausted"
                    evaluation.backpropagate(parent_node, 0)
                    _root = True
                    logger.error(
                        "Protocol repair transaction %s exhausted at stage %s",
                        parent_node.protocol_repair.get("transaction_id"),
                        protocol_repair.current_stage(parent_node.protocol_repair),
                    )
                elif protocol_repair.is_active(parent_node.protocol_repair):
                    logger.warning(
                        "Node %s is in staged protocol repair (stage=%s)",
                        parent_node.id,
                        protocol_repair.current_stage(parent_node.protocol_repair),
                    )
                    result_node = protocol_repair_agent.run(self, parent_node)
                elif (
                    parent_node.leakage_audit
                    and parent_node.leakage_audit.get("status") != "clean"
                    and parent_node.leakage_repair_attempt >= 2
                ):
                    parent_node.is_terminal = True
                    parent_node.continue_improve = False
                    parent_node.leakage_audit["repair_queue_status"] = "exhausted"
                    evaluation.backpropagate(parent_node, 0)
                    _root = True
                    logger.error(
                        "Node %s exhausted two mandatory leakage-repair attempts; terminating branch",
                        parent_node.id,
                    )
                elif (
                    parent_node.leakage_audit
                    and parent_node.leakage_audit.get("status") != "clean"
                ):
                    logger.warning(
                        "Node %s is in mandatory leakage-repair mode (attempt=%s)",
                        parent_node.id,
                        parent_node.leakage_repair_attempt + 1,
                    )
                    if parent_node.is_buggy or parent_node.is_valid is False:
                        result_node = debug_agent.run(self, parent_node)
                    else:
                        result_node = improve_agent.run(self, parent_node)
                elif parent_node.is_buggy or parent_node.is_valid is False:
                    result_node = debug_agent.run(self, parent_node)

                elif parent_node.is_buggy is False:
                    can_use_fusion = False
                    if self.search_start_time:
                        elapsed_time = time.time() - self.search_start_time
                        if elapsed_time >= self.acfg.time_limit / 2:
                            can_use_fusion = True
                    is_from_topk = getattr(parent_node, '_topk_triggered', False)
                    stagnation_threshold = self.scfg.topk_stagnation_threshold if is_from_topk else self.scfg.branch_stagnation_threshold
                    if is_from_topk:
                        logger.info(f"🎯 Exploitation mode: using relaxed stagnation threshold ({stagnation_threshold} attempts)")

                    if is_branch_stagnant(self, parent_node.branch_id, threshold=stagnation_threshold):
                        if can_use_fusion:
                            if random.random() < self.acfg.fusion_vs_evolution_prob:
                                logger.info(f"🎯 Triggering fusion for stagnant node {parent_node.id} (after 6h)")
                                result_node = fusion_agent.run(self, parent_node)
                            else:
                                logger.info(f"🎯 Triggering intra-branch evolution for stagnant node {parent_node.id} (after 6h)")
                                result_node = evolution_agent.run(self, parent_node)
                        else:
                            logger.info(f"🔄 Using evolution for stagnant node {parent_node.id} (before 6h)")
                            result_node = evolution_agent.run(self, parent_node)
                    else:
                        logger.info(f"🔄 Using normal improve for node {parent_node.id}")
                        result_node = improve_agent.run(self, parent_node)

                else:
                    logger.warning(f"[_run_single_step] node {parent_node.id} is_buggy is None.")

                if result_node:
                    if init_solution_path or result_node.skip_code_review:
                        logger.info(f"Node {result_node.id} uses immutable source code, skipping code review")
                    else:
                        pre_review_code = result_node.code
                        reviewed_code = code_review_agent.run(self, result_node)
                        if reviewed_code.strip() != result_node.code.strip():
                            logger.info(f"Node {result_node.id} code has been reviewed and modified")
                            result_node.code = reviewed_code
                            refresh_replay_lineage_after_revision(
                                result_node,
                                original_code=pre_review_code,
                                revision_kind="code_review_descendant",
                            )
                        else:
                            logger.info(f"Node {result_node.id} passed code review without changes")

                    pre_instrumentation_code = result_node.code
                    immutable_migration_seed = bool(
                        result_node.replay_source
                        and result_node.replay_source.get(
                            "requires_full_runtime_migration"
                        ) is True
                        and result_node.replay_source.get("execution_seed_only") is True
                    )
                    contract_bound_code = (
                        pre_instrumentation_code
                        if immutable_migration_seed
                        else enforce_host_candidate_entrypoint(
                            self, pre_instrumentation_code
                        )
                    )
                    if contract_bound_code != pre_instrumentation_code:
                        logger.info(
                            "Node %s Host candidate entrypoint normalized after code review",
                            result_node.id,
                        )
                        result_node.code = contract_bound_code
                        receipt = build_bounded_repair_receipt(
                            pre_instrumentation_code,
                            contract_bound_code,
                            preflight_status=PreflightStatus.MISSING_EVIDENCE.value,
                            repair_kind="instrumentation",
                            attempt=1,
                            max_attempts=1,
                        )
                        if not receipt["method_identity_preserved"]:
                            raise ValueError(
                                "Host candidate instrumentation changed method identity"
                            )
                        refresh_replay_lineage_after_instrumentation(
                            result_node,
                            original_code=pre_instrumentation_code,
                            instrumentation_receipt=receipt,
                        )

                    # Generate the paired raw-memory observer arm only after
                    # the actual Candidate has completed the same code-review
                    # policy, and always before any training execution.
                    prospective_audit = getattr(self, "prospective_audit", None)
                    if prospective_audit is not None:
                        prospective_audit.prepare_counterfactuals(result_node)

                    if result_parse_agent.run_pre_execution_leakage_audit(self, result_node):
                        _root = self._finalize_preflight_block(result_node, parent_node)
                    elif not execute_immediately:
                        logger.info(f"Node {result_node.id} code generated and reviewed, execution deferred")
                        result_node.pending_execution = True
                        return _root, result_node
                    else:
                        exe_res = exec_callback(result_node.code, result_node.id, True)
                        result_node = result_parse_agent.run(self,
                            node=result_node,
                            exec_result=exe_res
                        )
                        execution.validate_executed_node(self, result_node)
                        if prospective_audit is not None:
                            prospective_audit.finalize_node(result_node)
                        logger.info(f"The metric value of node {result_node.id} is {result_node.metric.value}.")
                        result_node.finish_time = time.strftime("%Y-%m-%dT%H:%M:%S")

                        if parent_node.is_buggy and result_node.is_buggy is False:
                            parent_node.is_debug_success = True

                        _root = evaluation.check_improvement(self, result_node, parent_node)
                        with self.journal_lock:
                            if self.best_node and result_node.metric.maximize and self.best_node.metric.maximize != result_node.metric.maximize:
                                logger.warning(
                                    "New node's metric is inconsistent with metrics in the journal. Returning to the parent node to regenerate.")
                                raise ValueError(
                                    "New node's metric is inconsistent with metrics in the journal. Returning to the parent node to regenerate.")
                            else:
                                self.journal.append(result_node)
                        self._enqueue_mandatory_repair(result_node)
                        self._notify_search_state()

            except Exception as e:
                logger.warning("Step failed for parent %s; propagating zero reward", parent_node.id)
                self._record_protocol_generation_failure(parent_node, e)
                evaluation.backpropagate(node=parent_node, value=0, add_to_tree=False)
                if not isinstance(e, DraftRoleReservationError):
                    parent_node.sub_expected_child_count()
                else:
                    logger.info("Draft role reservation was rejected before child-count mutation")
                raise e

        else:
            evaluation.backpropagate(node=parent_node, value=0)
            _root = True
        return _root, result_node

    def step(
        self,
        node: SearchNode,
        exec_callback: ExecCallbackType,
        execute_immediately: bool = True,
        init_solution_path: Optional[str] = None,
        draft_role: Optional[str] = None,
        mandatory_repair_role: Optional[str] = None,
        excluded_mandatory_repair_role: Optional[str] = None,
    ) -> Optional[SearchNode]:
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()
            self.search_start_time = time.time()

        claimed_repair_parent = None
        duplicate_repair_request = False
        # Phase 1 must still create the three declared draft roles in order.
        # Mandatory repairs take priority only once normal execution/search begins.
        preserve_explicit_runtime_debug = self._is_explicit_runtime_debug_parent(node)
        if (
            execute_immediately
            and draft_role is None
            and init_solution_path is None
            and not preserve_explicit_runtime_debug
        ):
            claimed_repair_parent, duplicate_repair_request = (
                self._claim_mandatory_repair_parent(
                    node,
                    required_draft_role=mandatory_repair_role,
                    excluded_draft_role=excluded_mandatory_repair_role,
                )
            )
            if claimed_repair_parent is not None:
                node = claimed_repair_parent
            elif duplicate_repair_request:
                logger.info(
                    "[step] Mandatory repair parent is already in flight; worker will wait"
                )
                return None
        elif preserve_explicit_runtime_debug:
            logger.info(
                "[step] Preserving explicit runtime-debug node %s; mandatory repairs remain queued",
                node.id,
            )

        replacement_draft = False
        if not node or node.stage == "root":
            node = node_selection.select_with_soft_switch(self)
            if node is None:
                condition = getattr(self, "_search_condition", None)
                if condition is not None:
                    with condition:
                        condition.wait(timeout=1.0)
                node = node_selection.select_with_soft_switch(self)
                if node is None:
                    if not self._has_pending_search_work():
                        if (
                            execute_immediately
                            and draft_role is None
                            and init_solution_path is None
                            and self._claim_replacement_draft_slot()
                        ):
                            node = self.virtual_root
                            replacement_draft = True
                        else:
                            raise SearchSpaceExhausted(
                                "No expandable node or in-flight search work remains"
                            )
                    logger.info("[step] No expandable node after bounded wait")
                    if node is None:
                        return None

        self.begin_search_work()
        try:
            try:
                _root, result_node = self._run_single_step(
                    node,
                    exec_callback=exec_callback,
                    execute_immediately=execute_immediately,
                    init_solution_path=init_solution_path,
                    draft_role=draft_role,
                    replacement_draft=replacement_draft,
                )
            except Exception:
                if claimed_repair_parent is not None:
                    self._release_mandatory_repair_parent(
                        claimed_repair_parent,
                        retry=not claimed_repair_parent.is_terminal,
                    )
                raise
            else:
                if claimed_repair_parent is not None:
                    self._release_mandatory_repair_parent(claimed_repair_parent)
        finally:
            self.end_search_work()
            if replacement_draft:
                self._release_replacement_draft_slot()

        if result_node:
            metric_value = result_node.metric.value if result_node.metric else None
            best_metric = self.best_node.metric.value if (self.best_node and self.best_node.metric) else None
            logger.info(f"[step] {node.id} → {result_node.id}: metric={metric_value}, best={best_metric}")

        if result_node and result_node.metric and result_node.metric.value is not None:
            solution_manager.update_best_solution(self, result_node)

        self.current_step = len(self.journal)

        # Cumulative stats
        total_nodes = len(self.journal)
        n_branches = len(self.branch_all_nodes)
        best_val = self.best_node.metric.value if (self.best_node and self.best_node.metric) else None
        logger.info(f"[stats] step={self.current_step}, nodes={total_nodes}, branches={n_branches}, best={best_val}")

        if _root or result_node is None:
            return self.virtual_root
        else:
            return result_node

    def execute_deferred_node(self, node: SearchNode, exec_callback: ExecCallbackType) -> SearchNode:
        """Execute a node that was generated and reviewed but not yet run (pending_execution=True)."""
        if not hasattr(node, 'pending_execution') or not node.pending_execution:
            logger.warning(f"Node {node.id} is not marked for deferred execution")
            return node

        logger.info(f"Executing deferred node {node.id}")
        parent_node = node.parent

        try:
            if result_parse_agent.run_pre_execution_leakage_audit(self, node):
                self._finalize_preflight_block(node, parent_node)
                node.pending_execution = False
                return node
            exe_res = exec_callback(node.code, node.id, True)
            node = result_parse_agent.run(self,
                node=node,
                exec_result=exe_res
            )

            execution.validate_executed_node(self, node)
            prospective_audit = getattr(self, "prospective_audit", None)
            if prospective_audit is not None:
                prospective_audit.finalize_node(node)

            logger.info(f"Node {node.id} execution completed: metric={node.metric.value}, is_buggy={node.is_buggy}")

            node.finish_time = time.strftime("%Y-%m-%dT%H:%M:%S")

            if parent_node and parent_node.is_buggy and node.is_buggy is False:
                parent_node.is_debug_success = True

            _root = evaluation.check_improvement(self, node, parent_node)

            with self.journal_lock:
                if self.best_node and node.metric.maximize and self.best_node.metric.maximize != node.metric.maximize:
                    logger.warning("New node's metric is inconsistent with metrics in the journal")
                    raise ValueError("New node's metric is inconsistent with metrics in the journal")
                else:
                    self.journal.append(node)
                    logger.info(f"Node {node.id} added to journal")
            self._enqueue_mandatory_repair(node)
            self._notify_search_state()

            node.pending_execution = False
            solution_manager.update_best_solution(self, node)

            return node

        except Exception as e:
            logger.exception(f"Exception during deferred node execution: {e}")
            evaluation.backpropagate(node=parent_node, value=0, add_to_tree=False)
            parent_node.sub_expected_child_count()
            raise e
