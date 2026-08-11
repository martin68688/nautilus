"""configuration and setup utils"""

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Hashable, Mapping, cast
import datetime
import coolname
import rich
from omegaconf import OmegaConf
from rich.syntax import Syntax
import shutup
from rich.logging import RichHandler
import logging

# Lazy import to avoid circular dependency with engine.search_node
# Journal and filter_journal are imported where needed via _get_journal_classes()
def _get_journal_classes():
    from engine.search_node import Journal, filter_journal
    return Journal, filter_journal

from utils import copytree, preproc_data, serialize

shutup.mute_warnings()
logger = logging.getLogger("MLEvolve")


def _resolve_host_artifact_roots(
    binding: Mapping[str, Any], namespace: str
) -> tuple[str, str]:
    """Resolve collision-free Host roots without mutating the signed binding."""

    report_root = str(binding["report_root"])
    runtime_root = str(binding["runtime_artifact_root"])
    namespace = str(namespace or "").strip()
    if not namespace:
        return report_root, runtime_root
    relative = Path(namespace)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(
            part in {"", ".", ".."}
            or not part.replace("-", "").replace("_", "").isalnum()
            for part in relative.parts
        )
    ):
        raise ValueError("Host artifact namespace is unsafe")
    return (
        str(Path(report_root) / "runs" / relative),
        str(Path(runtime_root) / "runs" / relative),
    )


def _candidate_runtime_guidance(task_id: str) -> str:
    """Candidate-visible guidance that does not mutate the frozen Host SDK."""

    lines = [
        "Generate internal-validation predictions inside "
        "`session.prediction_scope(...)`, exit that context, and only then "
        "call `session.evaluate_internal(...)`.",
        "After evaluation, call `session.freeze_selection(...)` before "
        "opening `session.inference_scope(...)`.",
    ]
    if str(task_id) == "aerial-cactus-identification":
        lines.extend(
            [
                "Aerial Host image paths are exactly "
                "`row[\"assets\"][\"image\"]`; `asset_path` and `image_path` "
                "are not Host row fields.",
                "Use `row[\"sample_id\"]` only as the submission ID; never "
                "derive the asset filename from it.",
            ]
        )
    return "\n".join(lines)


def _host_enforce_data_dir(
    configured_data_dir: str | Path,
    binding: Mapping[str, Any],
    *,
    terminal_fixed_holdout: bool,
) -> str | Path:
    """Keep the evaluator-bound public view when terminal holdout is active.

    Host SDK execution still consumes the Contract's protected DataViews. The
    public ``data_dir`` is needed only to satisfy the independent terminal
    evaluator's immutable train-view binding and contains no hidden labels.
    """

    if terminal_fixed_holdout:
        return configured_data_dir
    return str(binding["data_view_root"])


""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class StageConfig:
    model: str
    temp: float
    base_url: str
    api_key: str

@dataclass
class DecayConfig:
    exploration_constant: float
    lower_bound: float
    alpha: float
    phase_ratios: list


@dataclass
class DraftRolePolicyConfig:
    enabled: bool = False
    roles: list[str] = field(default_factory=list)
    extra_role: str = "novel_exploration"
    replay_targets_path: str = ""
    # Optional immutable root containing historical ``mlevolve/runs``
    # artifacts.  Frozen source releases intentionally omit bulky run logs,
    # so exact replay must not assume that the journal lives inside the
    # currently executing checkout.
    replay_runs_root: str = ""


@dataclass
class CandidateExecutionContractConfig:
    enabled: bool = False
    contract_id: str = ""
    max_execution_seconds: int = 0
    max_epochs: int = 0
    max_cv_folds: int = 0
    max_trainable_models: int = 0
    allowed_import_roots: list[str] = field(default_factory=list)
    allow_remote_assets: bool = True
    allow_unverified_local_assets: bool = True
    allow_dataset_wide_per_sample_precompute: bool = True
    allow_source_score_inheritance: bool = False


@dataclass
class ProtocolRepairConfig:
    enabled: bool = True
    per_stage_attempt_limit: int = 2
    stage_generation_attempt_limit: int = 2
    stage_attempt_limits: dict[str, int] = field(default_factory=lambda: {
        "data_scope": 3,
        "validation_provenance": 3,
        "cross_fit": 7,
        "selection_freeze": 6,
        "final_holdout": 10,
    })
    stage_generation_attempt_limits: dict[str, int] = field(default_factory=lambda: {
        "data_scope": 3,
        "validation_provenance": 3,
        "cross_fit": 7,
        "selection_freeze": 6,
        "final_holdout": 10,
    })
    final_runtime_attempt_limit: int = 6
    stage_generation_timeout_seconds: int = 300
    stage_generation_backend_retries: int = 2
    require_runtime_provenance: bool = True


@dataclass
class ProtocolPreflightConfig:
    enabled: bool = False
    repair_enabled: bool = False
    max_repair_attempts: int = 1
    binding_path: str = ""
    legacy_ast_mode: str = "enforce"
    report_root: str = ""
    expected_contract_hash: str = ""
    contract_path: str = ""
    data_view_manifest_path: str = ""
    image_digest: str = ""
    sdk_hash: str = ""
    collector_private_key_path: str = ""
    candidate_uid: int = 65534
    consume_collector_private_key: bool = True
    # Experiment End2End full-system option: an Agent reviews and may repair
    # the actual training entrypoint before execution.  Host Preflight remains
    # an evidence observer in host_sdk_shadow and never becomes this reviewer's
    # semantic decision maker.
    agent_semantic_review_enabled: bool = False
    agent_semantic_max_repair_attempts: int = 2
    agent_semantic_max_review_attempts: int = 5
    agent_semantic_temperature: float = 0.0
    agent_semantic_max_tokens: int = 4096
    agent_controls_protocol_preflight: bool = False
    install_host_candidate_entrypoint: bool = True
    candidate_process_isolation: bool = False
    

@dataclass
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int
    replacement_drafts_enabled: bool
    max_replacement_drafts: int
    metric_improvement_threshold: float
    back_debug_depth: int
    num_bugs: int
    num_improves: int
    topk_max_improves: int
    max_improve_failure: int
    parallel_search_num: int
    branch_stagnation_threshold: int
    topk_stagnation_threshold: int
    top_candidates_size: int
    stagnation_window: int
    num_gpus: int
    explore_switch_start: float
    explore_switch_end: float
    min_exploration_weight: float
    topk_early_k: int
    topk_early_max_per_branch: int
    topk_late_k: int
    topk_late_max_per_branch: int
    force_backprop_late_threshold: float
    force_backprop_late_prob: float
    force_backprop_mid_threshold: float
    force_backprop_mid_modulo: int
    recent_best_window: int
    fusion_min_time_hours: float
    fusion_max_time_hours: float
    fusion_min_successful_nodes: int
    fusion_min_branches: int

@dataclass
class AgentConfig:
    steps: int
    time_limit: int
    initial_drafts: int
    seed: int
    data_preview: bool
    code: StageConfig
    feedback: StageConfig
    check_data_leakage: bool
    fusion_vs_evolution_prob: float
    branch_fusion_trigger_prob: float
    max_fusion_drafts: int
    use_global_memory: bool
    memory_similarity_threshold: float
    memory_embedding_device: str
    memory_embedding_model_path: str
    search: SearchConfig
    decay: DecayConfig
    use_diff_mode: bool = True
    draft_role_policy: DraftRolePolicyConfig = field(default_factory=DraftRolePolicyConfig)
    candidate_execution_contract: CandidateExecutionContractConfig = field(
        default_factory=CandidateExecutionContractConfig
    )
    protocol_repair: ProtocolRepairConfig = field(default_factory=ProtocolRepairConfig)
    protocol_preflight: ProtocolPreflightConfig = field(
        default_factory=ProtocolPreflightConfig
    )
@dataclass
class ExecConfig:
    timeout: int
    agent_file_name: str


@dataclass
class ColdstartConfig:
    use_coldstart: bool
    task_json_path: str
    model_json_path: str
    description: str


@dataclass
class InitSolutionConfig:
    use: bool = False


@dataclass
class FixedHoldoutConfig:
    enabled: bool = False
    evaluation_mode: str = "terminal_only"
    train_manifest_path: str = ""
    bypass_protocol_gates: bool = False
    preflight_validate_train_view: bool = True
    internal_metric_disposition: str = "search_only"


@dataclass
class OfficialSubmissionConfig:
    enabled: bool = False
    provider: str = "kaggle"
    competition: str = ""
    metric: str = ""
    maximize: bool = False
    sample_submission_path: str = ""
    id_column: str = ""
    prediction_kind: str = "auto"
    probability_row_sum_tolerance: float = 1e-4
    submission_subdir: str = "submission"


@dataclass
class AdoptionTrackingConfig:
    enable: bool = False          # 默认关：记账与分析全 no-op，run 行为与今天一致
    enable_analysis: bool = True  # enable=True 时是否在 run 末尾跑分析
    judge_mode: str = "keyword"   # "keyword" | "llm" | "llm-all" | "hybrid"
    analysis_timeout_seconds: int = 300


@dataclass
class AdoptionVerifierConfig:
    enabled: bool = False
    mode: str = "shadow"  # shadow | enforce
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096
    max_contracts_per_call: int = 8
    max_code_chars: int = 120000
    require_signed_trace: bool = False


@dataclass
class ProspectiveAuditConfig:
    enabled: bool = False
    allow_pending_counterfactual: bool = False
    counterfactual_timeout_seconds: int = 300
    counterfactual_generation_attempts: int = 2
    counterfactual_memory_max_chars: int = 12000


@dataclass
class EvaluationAuthorityConfig:
    mode: str = "shadow"  # off | shadow | enforce
    protocol_registry: str = "config/protocols"
    active_protocol_id: str = "mlevolve-default"
    active_protocol_version: str = "1"
    policy_version: str = "authority_v1"
    collector_version: str = "1"
    rollout_id: str = "authority-shadow-v1"
    expected_bundle_id: str = ""
    expected_bundle_manifest_sha256: str = ""
    require_bound_bundle: bool = False
    enforce_operations: list[str] = field(default_factory=list)
    enforce_generation_stages: list[str] = field(default_factory=list)
    enforce_governance_stages: list[str] = field(default_factory=list)
    canary_minimum_decisions: int = 20
    canary_max_unauthorized_authority_allows: int = 0
    canary_max_false_denial_rate: float = 0.05
    fail_closed_high_risk: bool = True
    allow_invalid_debug: bool = True
    emit_snapshot: bool = True
    enable_causal_actuation: bool = False
    runtime_protocol_observer_enabled: bool = True
    protocol_runtime_mode: str = "legacy_ast"
    protocol_runtime_artifact_root: str = ""


@dataclass
class ExternalSkillMemoryConfig:
    enable: bool = False
    bundle_root: str = ""
    current_pointer_path: str = "CURRENT.json"
    verify_bundle_artifacts: bool = True
    session_overlay_path: str = ""
    graph_path: str = "../paper-skills/distillation/graph_build/graph_optimized_skillgraph.json"
    index_path: str = ""
    text_model_path: str = ""
    source_name: str = "skillgraph"
    mode: str = "skillgraph"  # includes opt-in run_forest_stage_hybrid
    scoring_mode: str = "lexical"  # lexical | poincare | flat_twin | euclidean
    geometry_distance_weight: float = 0.30
    geometry_semantic_weight: float = 0.20
    geometry_constraint_weight: float = 0.05
    geometry_condition_weight: float = 0.18
    geometry_failure_weight: float = 0.14
    geometry_evidence_weight: float = 0.08
    geometry_reliability_weight: float = 0.08
    geometry_conflict_weight: float = 0.10
    geometry_distance_norm: str = "none"  # none | minmax | zscore
    geometry_query_radius_quantile: float = 0.5
    geometry_query_radius_mode: str = "predicted_distribution"  # quantile | predicted_distribution
    geometry_query_radius_bands: list[str] = field(default_factory=lambda: ["core", "middle", "edge"])
    geometry_query_radius_top_bands: int = 2
    geometry_radius_fusion: str = "weighted_max"  # weighted_max | weighted_mean
    enable_agentic: bool = False
    navigator_max_steps: int = 3
    navigator_reference_budget: int = 1200
    selector_max_tokens: int = 1800
    top_k: int = 8
    depth: int = 2
    beam_width: int = 3
    general_cap: int = 2
    task_seed_limit: int = 6
    max_chars: int = 5000
    include_draft: bool = True
    include_improve: bool = True
    include_evolution: bool = True
    include_debug: bool = True
    include_fusion: bool = True
    retrieval_control: str = "stage_hybrid"
    strategy_candidate_limit: int = 12
    strategy_route_count: int = 3
    l2_tactic_limit: int = 4
    # Optional Recipe-first SOP overlay for the layered Dynamic Router.  The
    # file and its canonical payload are pinned independently so a changed
    # distillation cannot silently enter an already-frozen experiment.
    recipe_sop_path: str = ""
    recipe_sop_file_sha256: str = ""
    recipe_sop_bundle_sha256: str = ""
    # Frozen clean RunNode metadata referenced by Recipe SOPs.  This overlay
    # carries post-freeze terminal evidence that is intentionally absent from
    # the immutable base RunForest graph.
    recipe_evidence_path: str = ""
    recipe_evidence_file_sha256: str = ""
    recipe_evidence_manifest_sha256: str = ""
    # Frozen executable evidence referenced by Recipe SOPs.  The RunForest
    # graph intentionally stores only code hashes; this separate capsule binds
    # full source and repair diffs back to those hashes without changing
    # retrieval/scoring text.
    recipe_implementation_path: str = ""
    visibility_token_budget: int = 4096
    # Stage-scoped visibility override for retrieval canaries. This avoids
    # enabling global Experiment-R protocol attestation just to enforce the
    # Debug memory boundary.
    visibility_mode_override: str = ""
    stage_quotas: dict = field(default_factory=dict)
    rrf_weights: dict = field(default_factory=dict)
    blocked_run_prefixes: list[str] = field(default_factory=list)
    positive_control_probe_path: str = ""
    positive_control_force_raw: bool = False
    # Experiment End2End uses one shared Authority-filtered candidate pool and
    # changes only this registered selection policy.
    end2end_memory_system: str = ""
    end2end_prompt_token_budget: int = 1536
    end2end_candidate_pool_limit: int = 12
    # Full Experiment-R router. End2End enables this only for dynamic_hybrid;
    # the other nine frozen systems keep the existing controller above.
    experiment_r_enabled: bool = False
    experiment_r_candidate_limit: int = 12
    experiment_r_top_k: int = 6
    experiment_r_prompt_token_budget: int = 1536
    experiment_r_memory_pool_sha256: str = ""
    experiment_r_debug_confidence_threshold: float = 0.50
    experiment_r_memory_transfer_static_gate: bool = False
    experiment_r_memory_transfer_runtime_gate: bool = False
    experiment_r_agentic_retrieval_enabled: bool = False
    experiment_r_agentic_max_steps: int = 4
    experiment_r_agentic_per_step_top_k: int = 8
    experiment_r_agentic_max_observed: int = 48
    experiment_r_agentic_temperature: float = 0.0
    experiment_r_agentic_max_tokens: int = 1200
    # Optional sparse Agent selection.  Stage source allocations become
    # ceilings, and the Agent may return fewer than Top-K or abstain.
    experiment_r_flexible_selection_enabled: bool = False
    experiment_r_allow_agent_abstention: bool = False
    experiment_r_stage_selection_caps: dict = field(default_factory=dict)
    experiment_r_debug_causal_only: bool = False
    # Debug keeps the strict L3 matcher as a first tier, then lets the main
    # Retrieval Agent inspect safe task-local and portable runtime repairs.
    experiment_r_debug_tiered_retrieval_enabled: bool = False
    experiment_r_debug_portable_runtime_enabled: bool = False
    experiment_r_debug_portable_max_candidates: int = 2
    # Candidate execution on production clusters may be hermetic even when
    # the development environment supports remote assets.  When enabled,
    # Debug rejects repairs that introduce runtime network downloads.
    experiment_r_offline_runtime_only: bool = False
    runtime_network_policy: str = ""
    experiment_r_same_task_best_pin_stages: list[str] = field(
        default_factory=lambda: ["draft", "improve", "debug"]
    )
    # Bound each retrieved hypothesis to a local code edit.
    experiment_r_atomic_actuation_enabled: bool = False
    experiment_r_improve_max_modules: int = 2
    experiment_r_improve_max_patches: int = 6
    experiment_r_debug_max_patches: int = 3
    # Task-level synthesis over the Router's wider candidate view.  Shadow mode
    # is observational.  Active mode replaces the legacy Improve/Debug code
    # generation core with Strategy -> Atomic Planner -> bounded Coder and is
    # fail-closed: an invalid Strategy/Plan/Diff never falls back to an
    # unconstrained full rewrite.
    memory_strategy_shadow_enabled: bool = False
    memory_strategy_shadow_stages: list[str] = field(
        default_factory=lambda: ["improve"]
    )
    memory_strategy_debug_trigger: str = "causal_gap_or_repeated_failure"
    memory_strategy_debug_failure_threshold: int = 2
    memory_strategy_active_enabled: bool = False
    memory_strategy_active_stages: list[str] = field(
        default_factory=lambda: ["improve", "debug"]
    )
    memory_strategy_active_required: bool = True
    memory_strategy_active_allow_abstention: bool = False
    # Strategy v2 uses one shared attention budget across current branches,
    # causal failure evidence, and history.  The wider candidate pool is
    # ranked and lineage-deduplicated before any card reaches the model.
    memory_strategy_evidence_limit: int = 8
    memory_strategy_current_frontier_slots: int = 3
    memory_strategy_causal_failure_slots: int = 1
    memory_strategy_candidate_pool_limit: int = 48
    # Deprecated compatibility aliases for v1 replay packets.
    memory_strategy_max_cards: int = 24
    memory_strategy_card_max_chars: int = 6000
    # Zero means no additional host-side character truncation.  The structured
    # card count remains bounded; the provider's advertised context window is
    # probed separately by the smoke harness.
    memory_strategy_max_input_chars: int = 0
    memory_strategy_max_output_tokens: int = 12000
    memory_strategy_max_retries: int = 2
    memory_strategy_contract_retries: int = 4
    memory_strategy_min_candidate_compositions: int = 3
    memory_strategy_debug_min_candidate_compositions: int = 1
    memory_strategy_temperature: float = 0.0
    memory_strategy_model: str = "deepseek-v4-pro"
    memory_strategy_thinking_enabled: bool = True
    # V4 Pro thinking is retained for synthesis.  If its text is malformed
    # JSON, a second call to the same model runs with thinking disabled and
    # native JSON mode, with serialization-only authority.
    memory_strategy_json_normalization_enabled: bool = True
    memory_strategy_json_normalization_model: str = ""
    memory_strategy_json_normalization_max_tokens: int = 12000
    memory_strategy_json_normalization_max_retries: int = 2
    memory_strategy_history_limit: int = 16
    # Atomic actuation-chain defaults.  Stage-specific caps are intersected
    # with the global limits below.
    memory_strategy_atomic_max_modules: int = 2
    memory_strategy_atomic_max_changes: int = 3
    memory_strategy_atomic_max_patches: int = 6
    memory_strategy_atomic_max_symbols_per_phase: int = 64
    memory_strategy_atomic_debug_max_modules: int = 1
    memory_strategy_atomic_debug_max_changes: int = 2
    memory_strategy_atomic_debug_max_symbols_per_phase: int = 4
    memory_strategy_atomic_debug_targeted_repair_only: bool = True
    memory_strategy_atomic_planner_contract_retries: int = 2
    memory_strategy_atomic_coder_contract_retries: int = 1
    memory_strategy_atomic_coder_replan_attempts: int = 1
    memory_strategy_atomic_alternate_hypothesis_attempts: int = 0
    memory_strategy_atomic_alternate_replan_attempts: int = 0
    # v80 staged actuation compiles one Strategy hypothesis into a complete,
    # ordered series of independently verified Coder contracts.  It is opt-in
    # so historical experiment configurations preserve their original path.
    memory_strategy_atomic_staged_enabled: bool = False
    memory_strategy_atomic_max_phases: int = 3
    memory_strategy_atomic_require_complete_roadmap: bool = True
    memory_strategy_atomic_strict_coder_enabled: bool = False
    memory_strategy_atomic_verifier_mode: str = "strict"
    # Dynamic Hybrid may delegate L3 failure/root-cause matching to the
    # Retrieval Agent.  The Host keeps only objective task/evidence gates and
    # literal traceback-anchor extraction; no maintained synonym table is
    # consulted on this path.
    experiment_r_l3_agent_match_enabled: bool = False
    experiment_r_l3_agent_match_candidate_limit: int = 8
    experiment_r_l3_semantic_shortlist_enabled: bool = False
    experiment_r_l3_agent_match_max_attempts: int = 2
    experiment_r_l3_agent_match_min_confidence: float = 0.50
    experiment_r_l3_agent_match_max_tokens: int = 1800
    # Optional two-agent Debug retrieval. A read-only Grep Search Agent first
    # searches the complete Authority-authorized L3 pool; the independent L3
    # Agent then judges the accumulated candidates by root cause.
    experiment_r_l3_grep_agent_enabled: bool = False
    experiment_r_l3_grep_max_steps: int = 6
    experiment_r_l3_grep_per_query_limit: int = 8
    experiment_r_l3_grep_min_candidates: int = 8
    experiment_r_l3_grep_max_candidates: int = 20
    experiment_r_l3_grep_max_attempts: int = 2
    experiment_r_l3_grep_max_tokens: int = 1600
    excluded_run_ids: list[str] = field(default_factory=list)


@dataclass
class RunIdentityConfig:
    """Persisted experiment identity for baseline-vs-memory comparisons."""

    schema: str = "mlevolve_run_identity_v1"
    experiment_group: str = "baseline_no_external_memory"
    baseline_reference_group: str = ""
    memory_enabled: bool = False
    memory_system: str = "none"
    memory_version: str = "none"
    memory_snapshot_sha256: str = ""
    memory_index_sha256: str = ""
    memory_source_count: int = 0
    memory_source_runs: list[str] = field(default_factory=list)
    code_revision: str = ""
    code_worktree_sha256: str = ""
    identity_source: str = "declared_at_runtime"
    experiment_manifest_sha256: str = ""
    system_manifest_sha256: str = ""
    task_manifest_sha256: str = ""
    budget_manifest_sha256: str = ""
    memory_bundle_binding_sha256: str = ""
    memory_current_sha256: str = ""
    evaluator_manifest_sha256: str = ""
    logical_run_id: str = ""
    system_id: str = ""
    rng_state_hash: str = ""
    rng_state_components: dict[str, str] = field(default_factory=dict)
    # Search-rank metrics must describe the exact prediction variant written
    # to submission.csv when this experiment-level measurement gate is on.
    require_submission_aligned_internal_metric: bool = False


@dataclass
class Config(Hashable):
    data_dir: Path
    dataset_dir: Path
    desc_file: Path | None

    goal: str | None
    eval: str | None

    log_dir: Path
    log_level: str
    workspace_dir: Path

    preprocess_data: bool
    copy_data: bool

    exp_name: str
    exp_id: str

    torch_hub_dir: str
    pretrain_model_dir: str

    exec: ExecConfig
    agent: AgentConfig
    start_cpu_id: str
    cpu_number: str

    coldstart: ColdstartConfig

    methodology_kb_path: str = ""
    methodology_dynamic: bool = False
    finalize_reserve_seconds: int = 900
    use_grading_server: bool = True
    init_solution: InitSolutionConfig = field(default_factory=InitSolutionConfig)
    fixed_holdout: FixedHoldoutConfig = field(default_factory=FixedHoldoutConfig)
    official_submission: OfficialSubmissionConfig = field(
        default_factory=OfficialSubmissionConfig
    )
    adoption_tracking: AdoptionTrackingConfig = field(default_factory=AdoptionTrackingConfig)
    adoption_verifier: AdoptionVerifierConfig = field(default_factory=AdoptionVerifierConfig)
    prospective_audit: ProspectiveAuditConfig = field(default_factory=ProspectiveAuditConfig)
    evaluation_authority: EvaluationAuthorityConfig = field(default_factory=EvaluationAuthorityConfig)
    external_skill_memory: ExternalSkillMemoryConfig = field(default_factory=ExternalSkillMemoryConfig)
    run_identity: RunIdentityConfig = field(default_factory=RunIdentityConfig)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_memory_artifact(raw_path: str) -> Path:
    path = Path(str(raw_path or "")).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, Path(__file__).resolve().parent.parent / path]
    return next((candidate.resolve() for candidate in candidates if candidate.exists()), candidates[-1].resolve())


def _populate_run_identity(cfg) -> None:
    """Bind a run label to the exact code and clean-memory snapshot it uses."""
    identity = cfg.run_identity
    memory_cfg = cfg.external_skill_memory
    identity.memory_enabled = bool(memory_cfg.enable)
    if identity.memory_enabled:
        bundle_root = str(getattr(memory_cfg, "bundle_root", "") or "").strip()
        base = None
        if bundle_root:
            from authority.memory_snapshot import MemorySnapshotLoader

            base = MemorySnapshotLoader(
                _resolve_memory_artifact(bundle_root)
            ).load_base(
                current_path=str(
                    getattr(memory_cfg, "current_pointer_path", "CURRENT.json")
                    or "CURRENT.json"
                ),
                verify_artifacts=bool(
                    getattr(memory_cfg, "verify_bundle_artifacts", True)
                ),
            )
            graph_path = base.path / "runforest" / "graph.json"
            index_path = base.path / "runforest" / "index.npz"
            identity.memory_snapshot_sha256 = base.manifest_sha256
            identity.memory_version = base.bundle_version
            authority_cfg = getattr(cfg, "evaluation_authority", None)
            if authority_cfg is not None and bool(
                getattr(authority_cfg, "require_bound_bundle", False)
            ):
                expected_id = str(
                    getattr(authority_cfg, "expected_bundle_id", "") or ""
                )
                expected_manifest = str(
                    getattr(
                        authority_cfg,
                        "expected_bundle_manifest_sha256",
                        "",
                    )
                    or ""
                )
                if not expected_id or not expected_manifest:
                    raise ValueError("A required Bundle must have explicit identity pins")
                if base.bundle_id != expected_id:
                    raise ValueError("Run identity Bundle ID does not match the required pin")
                if base.manifest_sha256 != expected_manifest:
                    raise ValueError(
                        "Run identity Bundle manifest does not match the required pin"
                    )
                expected_policy = str(
                    getattr(authority_cfg, "policy_version", "") or ""
                )
                if expected_policy and str(
                    base.manifest.get("authority_policy_version") or ""
                ) != expected_policy:
                    raise ValueError(
                        "Run identity Bundle policy does not match the active policy"
                    )
        else:
            graph_path = _resolve_memory_artifact(memory_cfg.graph_path)
            index_path = _resolve_memory_artifact(memory_cfg.index_path) if memory_cfg.index_path else None
        if not graph_path.is_file():
            raise FileNotFoundError(f"Run identity memory graph not found: {graph_path}")
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        meta = graph.get("meta") or {}
        legacy_provenance_fields = {
            "source_membership_verified",
            "leak_verified",
        } & set(meta)
        if legacy_provenance_fields:
            if (
                meta.get("source_membership_verified") is not True
                or meta.get("leak_verified") is not True
            ):
                raise ValueError(
                    "Memory-enabled runs require a source-verified and leak-verified graph"
                )
            source_runs = [str(value) for value in (meta.get("source_runs") or [])]
        elif base is not None:
            if bool(getattr(memory_cfg, "verify_bundle_artifacts", True)):
                provenance = base.verify_run_identity_provenance()
                source_runs = [str(value) for value in provenance["source_runs"]]
            else:
                corpus = base.read_json("corpus/manifest.json")
                source_runs = [
                    str(row["run_id"])
                    for row in corpus.get("runs") or []
                    if str(row.get("run_id") or "")
                ]
        else:
            raise ValueError(
                "Memory-enabled runs require a source-verified and leak-verified graph"
            )
        if not bundle_root:
            identity.memory_snapshot_sha256 = _sha256_file(graph_path)
        identity.memory_index_sha256 = _sha256_file(index_path) if index_path and index_path.is_file() else ""
        identity.memory_source_runs = source_runs
        identity.memory_source_count = len(identity.memory_source_runs)
    else:
        identity.memory_system = "none"
        identity.memory_version = "none"
        identity.memory_snapshot_sha256 = ""
        identity.memory_index_sha256 = ""
        identity.memory_source_count = 0
        identity.memory_source_runs = []

    identity.code_revision = os.environ.get("MLEVOLVE_CODE_REVISION", identity.code_revision)
    identity.code_worktree_sha256 = os.environ.get(
        "MLEVOLVE_CODE_WORKTREE_SHA256", identity.code_worktree_sha256
    )


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    max_index = -1
    for p in dir.iterdir():
        try:
            current_index = int(p.name.split("-")[0])
            if current_index > max_index:
                max_index = current_index
        except ValueError:
            pass
    return max_index + 1


def _default_config_path() -> Path:
    return Path(os.environ.get("MLEVOLVE_CONFIG", Path(__file__).parent / "config.yaml"))


def _load_config_tree(path: Path, seen: tuple[Path, ...] = ()):
    path = Path(path).expanduser().resolve()
    if path in seen:
        chain = " -> ".join(str(item) for item in (*seen, path))
        raise ValueError(f"Cyclic config inheritance detected: {chain}")
    cfg = OmegaConf.load(path)
    extends = cfg.pop("extends", None)
    if not extends:
        return cfg
    base_path = Path(extends)
    if not base_path.is_absolute():
        base_path = path.parent / base_path
    base_cfg = _load_config_tree(base_path, (*seen, path))
    return OmegaConf.merge(base_cfg, cfg)


def _load_cfg(
    path: Path | None = None, use_cli_args=True
) -> Config:
    # Load secrets (e.g. DEEPSEEK_API_KEY) from mlevolve/.env so they stay out of git.
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    path = Path(path) if path is not None else _default_config_path()
    cfg = _load_config_tree(path)
    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    return cfg

def load_cfg(path: Path | None = None) -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    return prep_cfg(_load_cfg(path))


def _validate_adoption_verifier_config(cfg) -> None:
    verifier = getattr(cfg, "adoption_verifier", None)
    if verifier is None or not getattr(verifier, "enabled", False):
        return
    mode = str(getattr(verifier, "mode", "shadow") or "shadow").lower()
    if mode not in {"shadow", "enforce"}:
        raise ValueError("adoption_verifier.mode must be shadow or enforce")
    authority = getattr(cfg, "evaluation_authority", None)
    authority_mode = str(getattr(authority, "mode", "off") or "off").lower()
    runtime_mode = str(
        getattr(authority, "protocol_runtime_mode", "legacy_ast") or "legacy_ast"
    ).lower()
    if authority_mode == "off":
        raise ValueError("Agent adoption verification requires Evaluation Authority")
    if mode == "enforce" and runtime_mode not in {
        "host_sdk_shadow",
        "host_sdk_enforce",
    }:
        raise ValueError(
            "Agent adoption verifier enforce mode requires host_sdk_shadow or "
            "host_sdk_enforce so memory-specific probes can be observed"
        )
    if getattr(verifier, "require_signed_trace", False):
        preflight = getattr(getattr(cfg, "agent", None), "protocol_preflight", None)
        if preflight is None or not getattr(preflight, "enabled", False):
            raise ValueError(
                "Signed Agent adoption traces require Host Protocol Preflight "
                "and its Host-only collector identity"
            )


def prep_cfg(cfg: Config):
    _validate_adoption_verifier_config(cfg)
    if cfg.agent.protocol_preflight.enabled:
        from authority.protocol_execution_contract import read_contract_artifact
        from authority.protocol_registry import ProtocolRegistry
        from protocol_runtime.activation import hash_sdk_tree, load_host_protocol_binding

        preflight = cfg.agent.protocol_preflight
        runtime_image_digest = os.environ.get(
            "MLEVOLVE_RUNTIME_IMAGE_DIGEST", ""
        ).strip()
        if not runtime_image_digest:
            raise ValueError(
                "Host Protocol activation requires MLEVOLVE_RUNTIME_IMAGE_DIGEST"
            )
        runtime_sdk_hash = hash_sdk_tree(
            Path(__file__).resolve().parents[1] / "protocol_runtime"
        )
        binding = load_host_protocol_binding(
            preflight.binding_path,
            expected_task_id=str(cfg.exp_id or ""),
            expected_image_digest=runtime_image_digest,
            expected_sdk_hash=runtime_sdk_hash,
        )
        runtime_mode = str(
            getattr(cfg.evaluation_authority, "protocol_runtime_mode", "") or ""
        )
        if runtime_mode not in {"host_sdk_shadow", "host_sdk_enforce"}:
            raise ValueError(
                "Host Protocol Preflight requires a host_sdk_shadow or "
                "host_sdk_enforce runtime mode"
            )
        if str(preflight.legacy_ast_mode or "") != "shadow":
            raise ValueError(
                "Host Protocol Preflight requires legacy_ast_mode=shadow"
            )
        if cfg.agent.protocol_repair.enabled:
            raise ValueError(
                "Host Protocol Preflight requires legacy protocol_repair.enabled=false"
            )
        (
            preflight.report_root,
            cfg.evaluation_authority.protocol_runtime_artifact_root,
        ) = _resolve_host_artifact_roots(
            binding,
            os.environ.get("MLEVOLVE_HOST_ARTIFACT_NAMESPACE", ""),
        )
        preflight.expected_contract_hash = binding["contract_hash"]
        preflight.contract_path = binding["contract_path"]
        preflight.data_view_manifest_path = binding["data_view_manifest_path"]
        preflight.image_digest = binding["image_digest"]
        preflight.sdk_hash = binding["sdk_hash"]
        if not str(preflight.collector_private_key_path or "").strip():
            raise ValueError(
                "Host Protocol activation requires collector_private_key_path"
            )
        # The Host Contract is the protocol authority for an activated run.
        # Bind claim-use Authority to the exact same registered ProtocolRef so
        # valid signed runtime Receipts cannot be silently quarantined merely
        # because an inherited profile still names ``mlevolve-default``.
        contract = read_contract_artifact(binding["contract_path"])
        registry_root = Path(cfg.evaluation_authority.protocol_registry)
        if not registry_root.is_absolute():
            registry_root = Path(__file__).resolve().parents[1] / registry_root
        registered = ProtocolRegistry(registry_root).get(
            contract.protocol_ref.protocol_id,
            contract.protocol_ref.version,
        )
        if registered.ref() != contract.protocol_ref:
            raise ValueError(
                "Host Protocol Contract does not match the active Protocol registry"
            )
        cfg.evaluation_authority.active_protocol_id = (
            contract.protocol_ref.protocol_id
        )
        cfg.evaluation_authority.active_protocol_version = (
            contract.protocol_ref.version
        )
        # Shadow validates and dry-runs the frozen Host views without changing
        # the Candidate's historical input surface.  Only a separately approved
        # enforce canary may replace data_dir with the terminal-blind Host root.
        if runtime_mode == "host_sdk_enforce":
            cfg.data_dir = _host_enforce_data_dir(
                cfg.data_dir,
                binding,
                terminal_fixed_holdout=bool(
                    getattr(cfg.fixed_holdout, "enabled", False)
                    and getattr(
                        cfg.fixed_holdout, "bypass_protocol_gates", False
                    )
                ),
            )
            # Use the immutable Host copy as the sole schema authority. It
            # carries a generated normalized-row appendix that resolves stale
            # benchmark descriptions (notably Leaf's margin_1 vs margin1).
            cfg.desc_file = binding["description_path"]
        elif not cfg.desc_file:
            cfg.desc_file = binding["description_path"]

    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    if cfg.data_dir.startswith("example_tasks/"):
        cfg.data_dir = Path(__file__).parent.parent / cfg.data_dir
    cfg.data_dir = Path(cfg.data_dir).resolve()

    if cfg.desc_file is not None:
        cfg.desc_file = Path(cfg.desc_file).resolve()

    top_log_dir = Path(cfg.log_dir).resolve()
    top_workspace_dir = Path(cfg.workspace_dir).resolve()
    # generate experiment name and prefix with consecutive index
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.exp_name = f"{timestamp}_{cfg.exp_name or coolname.generate_slug(3)}"

    # If log_dir and workspace_dir point to the same path, treat it as a unified
    # "runs" root and place logs/workspace under the per-run directory
    if top_log_dir == top_workspace_dir:
        runs_root = top_log_dir
        runs_root.mkdir(parents=True, exist_ok=True)
        per_run_root = (runs_root / cfg.exp_name).resolve()
        cfg.log_dir = (per_run_root / "logs").resolve()
        cfg.workspace_dir = (per_run_root / "workspace").resolve()
    else:
        top_log_dir.mkdir(parents=True, exist_ok=True)
        top_workspace_dir.mkdir(parents=True, exist_ok=True)
        cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
        cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()

    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)
    _populate_run_identity(cfg)

    if cfg.fixed_holdout.enabled:
        from fixed_holdout.mode import EVALUATION_MODE, train_manifest_path
        from fixed_holdout.validation import validate_train_view

        if cfg.fixed_holdout.evaluation_mode != EVALUATION_MODE:
            raise ValueError(
                "fixed_holdout.evaluation_mode must be terminal_only so holdout "
                "scores cannot influence search"
            )
        if cfg.fixed_holdout.internal_metric_disposition != "search_only":
            raise ValueError(
                "fixed_holdout.internal_metric_disposition must be search_only"
            )
        if not cfg.fixed_holdout.bypass_protocol_gates:
            raise ValueError(
                "fixed_holdout requires bypass_protocol_gates=true; the external "
                "label-isolated evaluator replaces internal protocol certification"
            )
        if cfg.agent.check_data_leakage:
            raise ValueError(
                "fixed_holdout requires agent.check_data_leakage=false"
            )
        if cfg.agent.protocol_repair.enabled:
            raise ValueError(
                "fixed_holdout requires agent.protocol_repair.enabled=false"
            )
        if cfg.fixed_holdout.preflight_validate_train_view:
            validate_train_view(train_manifest_path(cfg), Path(cfg.data_dir))

    if cfg.official_submission.enabled:
        from official_submission import validate_runtime_config

        validate_runtime_config(cfg)

    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config):
    """Load task description from markdown file or config str."""

    # either load the task description from a file
    if cfg.desc_file is not None:
        if not (cfg.goal is None and cfg.eval is None):
            logger.warning(
                "Ignoring goal and eval args because task description file is provided."
            )

        with open(cfg.desc_file) as f:
            task_desc = f.read()
        if cfg.fixed_holdout.enabled:
            task_desc += _fixed_holdout_task_note()
        if cfg.official_submission.enabled:
            task_desc += _official_submission_task_note()
        if cfg.agent.protocol_preflight.enabled:
            task_desc += "\n\n# Host Candidate Lifecycle\n" + _candidate_runtime_guidance(
                str(cfg.exp_id or "")
            )
        return task_desc

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval

    if cfg.fixed_holdout.enabled:
        task_desc["Fixed holdout evaluation"] = _fixed_holdout_task_note().strip()
    if cfg.official_submission.enabled:
        task_desc["Native official submission"] = (
            _official_submission_task_note().strip()
        )

    return task_desc


def _fixed_holdout_task_note() -> str:
    return (
        "\n\n## Fixed Holdout Evaluation Contract\n"
        "The visible test rows are one immutable holdout. Their labels are not "
        "mounted in this training environment. You may split or cross-validate "
        "only the labeled training rows for development. Produce predictions for "
        "every test ID in sample_submission.csv. Internal validation scores guide "
        "search only; a separate evaluator scores all completed submissions once "
        "after the run.\n"
    )


def _official_submission_task_note() -> str:
    return (
        "\n\n## Native Official-Test Submission Contract\n"
        "The mounted test rows are the task's complete official unlabeled test "
        "set. During this same candidate execution, train the model, compute an "
        "internal OOF/validation score for search, run real inference for every "
        "ID in sample_submission.csv, and write ./submission/submission.csv. "
        "The submission and reported metric must use the same selected model and "
        "prediction variant. Do not defer inference or require post-search "
        "retraining. The external official score is terminal-only and invisible "
        "during search.\n"
    )


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and preprocess data if necessary."""
    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "submission").mkdir(parents=True, exist_ok=True)

    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")
    _install_task_training_filename_compatibility(cfg)


def _install_task_training_filename_compatibility(cfg: Config) -> None:
    """Expose legacy public-training names without changing the data view.

    Historical Taxi replay candidates were produced against the original
    MLE-Bench filename ``labels.csv``. The fixed-holdout public training view
    names the exact same labeled rows ``train.csv``. A workspace-local relative
    symlink keeps those immutable replay candidates executable; it neither
    copies nor exposes the separately held terminal labels.
    """

    if str(cfg.exp_id) != "new-york-city-taxi-fare-prediction":
        return
    input_dir = Path(cfg.workspace_dir) / "input"
    train_path = input_dir / "train.csv"
    legacy_path = input_dir / "labels.csv"
    if train_path.is_file() and not legacy_path.exists():
        legacy_path.symlink_to(train_path.name)


def save_run_identity(cfg: Config) -> Path:
    """Persist experiment identity before any draft generation can fail."""
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    identity_path = log_dir / "run_identity.json"
    identity = OmegaConf.to_container(cfg.run_identity, resolve=True)
    identity_path.write_text(
        json.dumps(identity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return identity_path


def save_run(cfg: Config, journal):
    Journal, filter_journal = _get_journal_classes()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    filtered_journal = filter_journal(journal)
    # save journal
    serialize.dump_json(journal, cfg.log_dir / "journal.json")
    serialize.dump_json(filtered_journal, cfg.log_dir / "filtered_journal.json")
    # save config
    OmegaConf.save(config=cfg, f=cfg.log_dir / "config.yaml")
    save_run_identity(cfg)
    
    # save the best found solution
    best_node = journal.get_best_node()
    if best_node is not None:
        with open(cfg.log_dir / "best_solution.py", "w") as f:
            f.write(best_node.code)
