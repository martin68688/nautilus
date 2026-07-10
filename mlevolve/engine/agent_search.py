"""AgentSearch: tree search coordinator; delegates to node_selection, evaluation, execution, solution_manager."""

import logging
import random
import time
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
    code_review_agent,
    result_parse_agent,
)
from engine import node_selection, evaluation, execution, solution_manager
from engine.conditions import is_branch_stagnant
from utils.data_preview import clean_task_desc

logger = logging.getLogger("MLEvolve")


ExecCallbackType = Callable[[str, bool], ExecutionResult]

class AgentSearch:
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
        self._validate_draft_role_policy()

        self.next_branch_id = 1
        self.branch_all_nodes: Dict[int, List[SearchNode]] = {}
        self.branch_successful_nodes: Dict[int, List[SearchNode]] = {}
        self.branch_node_count: Dict[int, int] = {}
        self.use_coldstart = cfg.coldstart.use_coldstart
        self.coldstart_description = cfg.coldstart.description
        # Adoption tracking: side-channel snapshot of methodology ref_ids (never in prompt)
        from engine.coldstart import knowledge as _kb
        self.methodology_ref_ids: list[str] = list(_kb._LAST_REF_IDS)
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
        ext_cfg = getattr(self.cfg, "external_skill_memory", None)
        if ext_cfg is not None and getattr(ext_cfg, "enable", False):
            try:
                from agents.memory.external_skill_memory import ExternalSkillMemoryLayer, RunForestMemoryLayer
                ext_mode = getattr(ext_cfg, "mode", "skillgraph")
                ext_source = getattr(ext_cfg, "source_name", "skillgraph")
                ext_graph_path = getattr(ext_cfg, "graph_path", "")
                memory_layer_cls = (
                    RunForestMemoryLayer
                    if "run_forest" in str(ext_mode).lower()
                    or "run_forest" in str(ext_source).lower()
                    or "run_forest" in str(ext_graph_path).lower()
                    else ExternalSkillMemoryLayer
                )
                self.external_skill_memory = memory_layer_cls(
                    graph_path=getattr(ext_cfg, "graph_path", ""),
                    source_name=ext_source,
                    mode=ext_mode,
                    index_path=getattr(ext_cfg, "index_path", ""),
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
                    cfg=self.cfg,
                )
                logger.info(
                    "[AgentSearch] External skill memory enabled: source=%s graph=%s",
                    self.external_skill_memory.source_name,
                    self.external_skill_memory.graph_path,
                )
            except Exception as e:
                import traceback
                logger.warning(f"[AgentSearch] Failed to initialize external skill memory: {e}")
                logger.debug(f"[AgentSearch] External skill memory traceback: {traceback.format_exc()}")
                self.external_skill_memory = None
        else:
            logger.info("[AgentSearch] External skill memory is disabled by config")

    def _validate_draft_role_policy(self) -> None:
        policy = getattr(self.acfg, "draft_role_policy", None)
        if policy is None or not bool(getattr(policy, "enabled", False)):
            return
        expected = ["coldstart_baseline", "memory_reproduction", "novel_exploration"]
        roles = [str(role) for role in list(getattr(policy, "roles", []) or [])]
        if roles[:3] != expected:
            raise ValueError(f"RunForest draft roles must start with {expected}; got {roles}")
        if int(self.acfg.initial_drafts) < len(expected):
            raise ValueError("RunForest draft role policy requires agent.initial_drafts >= 3")
        if int(self.scfg.num_drafts) < len(expected):
            raise ValueError("RunForest draft role policy requires agent.search.num_drafts >= 3")

    def configured_draft_role(self, draft_index: int) -> str:
        policy = getattr(self.acfg, "draft_role_policy", None)
        if policy is None or not bool(getattr(policy, "enabled", False)):
            return "general_draft"
        roles = [str(role) for role in list(getattr(policy, "roles", []) or [])]
        if 0 <= draft_index < len(roles):
            return roles[draft_index]
        return str(getattr(policy, "extra_role", "novel_exploration") or "novel_exploration")

    def claim_draft_role(self, explicit_role: str | None = None) -> str:
        with self._draft_role_lock:
            draft_index = self._draft_generation_count
            self._draft_generation_count += 1
        role = explicit_role or self.configured_draft_role(draft_index)
        logger.info("[draft-role] index=%s role=%s", draft_index, role)
        return role

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
    ):
        """Run one search step: select action (draft/debug/improve), execute, parse, validate."""
        result_node = None
        _root = False

        if not parent_node.is_terminal:
            try:
                if self.is_root(parent_node):
                    if parent_node.reached_child_limit(scfg=self.scfg):
                        logger.info("🎯 Regular draft limit reached, triggering multi-branch aggregation (conditions already checked in select())")
                        result_node = aggregation_agent.run(self, mode="node", parent_node=parent_node)
                        if result_node:
                            result_node.lock = True
                            logger.info(f"[_run_single_step] Aggregation branch node {result_node.id} is locked.")
                        else:
                            logger.info("Aggregation failed or limit reached, skipping. Will continue normal search.")
                            result_node = None
                    else:
                        result_node = draft_agent.run(
                            self,
                            init_solution_path=init_solution_path,
                            draft_role=draft_role,
                        )
                        result_node.lock = True
                        logger.info(f"[_run_single_step] Draft node {result_node.id} is locked.")
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
                        reviewed_code = code_review_agent.run(self, result_node)
                        if reviewed_code.strip() != result_node.code.strip():
                            logger.info(f"Node {result_node.id} code has been reviewed and modified")
                            result_node.code = reviewed_code
                        else:
                            logger.info(f"Node {result_node.id} passed code review without changes")

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

            except Exception as e:
                logger.warning(f"Step failed for parent {parent_node.id}, rolling back expected child count and propagating zero reward.")
                evaluation.backpropagate(node=parent_node, value=0, add_to_tree=False)
                parent_node.sub_expected_child_count()
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
    ) -> SearchNode:
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()
            self.search_start_time = time.time()

        if not node or node.stage == "root":
            node = node_selection.select_with_soft_switch(self)

        _root, result_node = self._run_single_step(
            node,
            exec_callback=exec_callback,
            execute_immediately=execute_immediately,
            init_solution_path=init_solution_path,
            draft_role=draft_role,
        )

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

            node.pending_execution = False
            solution_manager.update_best_solution(self, node)

            return node

        except Exception as e:
            logger.exception(f"Exception during deferred node execution: {e}")
            evaluation.backpropagate(node=parent_node, value=0, add_to_tree=False)
            parent_node.sub_expected_child_count()
            raise e
