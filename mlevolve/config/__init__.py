"""configuration and setup utils"""

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Hashable, cast
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
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int
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
    protocol_repair: ProtocolRepairConfig = field(default_factory=ProtocolRepairConfig)
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
    internal_metric_disposition: str = "search_only"


@dataclass
class AdoptionTrackingConfig:
    enable: bool = False          # 默认关：记账与分析全 no-op，run 行为与今天一致
    enable_analysis: bool = True  # enable=True 时是否在 run 末尾跑分析
    judge_mode: str = "keyword"   # "keyword" | "llm" | "llm-all" | "hybrid"


@dataclass
class EvaluationAuthorityConfig:
    mode: str = "shadow"  # off | shadow | enforce
    protocol_registry: str = "config/protocols"
    active_protocol_id: str = "mlevolve-default"
    active_protocol_version: str = "1"
    policy_version: str = "authority_v1"
    fail_closed_high_risk: bool = True
    allow_invalid_debug: bool = True
    emit_snapshot: bool = True
    enable_causal_actuation: bool = False


@dataclass
class ExternalSkillMemoryConfig:
    enable: bool = False
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
    stage_quotas: dict = field(default_factory=dict)
    rrf_weights: dict = field(default_factory=dict)
    blocked_run_prefixes: list[str] = field(default_factory=list)


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
    use_grading_server: bool = True
    init_solution: InitSolutionConfig = field(default_factory=InitSolutionConfig)
    fixed_holdout: FixedHoldoutConfig = field(default_factory=FixedHoldoutConfig)
    adoption_tracking: AdoptionTrackingConfig = field(default_factory=AdoptionTrackingConfig)
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
        graph_path = _resolve_memory_artifact(memory_cfg.graph_path)
        index_path = _resolve_memory_artifact(memory_cfg.index_path) if memory_cfg.index_path else None
        if not graph_path.is_file():
            raise FileNotFoundError(f"Run identity memory graph not found: {graph_path}")
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        meta = graph.get("meta") or {}
        if meta.get("source_membership_verified") is not True or meta.get("leak_verified") is not True:
            raise ValueError("Memory-enabled runs require a source-verified and leak-verified graph")
        identity.memory_snapshot_sha256 = _sha256_file(graph_path)
        identity.memory_index_sha256 = _sha256_file(index_path) if index_path and index_path.is_file() else ""
        identity.memory_source_runs = [str(value) for value in (meta.get("source_runs") or [])]
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


def prep_cfg(cfg: Config):
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
        validate_train_view(train_manifest_path(cfg), Path(cfg.data_dir))

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


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and preprocess data if necessary."""
    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "submission").mkdir(parents=True, exist_ok=True)

    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")


def save_run(cfg: Config, journal):
    Journal, filter_journal = _get_journal_classes()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    filtered_journal = filter_journal(journal)
    # save journal
    serialize.dump_json(journal, cfg.log_dir / "journal.json")
    serialize.dump_json(filtered_journal, cfg.log_dir / "filtered_journal.json")
    # save config
    OmegaConf.save(config=cfg, f=cfg.log_dir / "config.yaml")
    identity = OmegaConf.to_container(cfg.run_identity, resolve=True)
    (cfg.log_dir / "run_identity.json").write_text(
        json.dumps(identity, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    
    # save the best found solution
    best_node = journal.get_best_node()
    if best_node is not None:
        with open(cfg.log_dir / "best_solution.py", "w") as f:
            f.write(best_node.code)
