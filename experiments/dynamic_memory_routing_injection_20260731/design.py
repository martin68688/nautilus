"""Result-blind structural design for Experiment R.

Only routing and injection policy may vary between online arms.  The complete
RunForest/SOP/Authority stack, raw memory snapshot, Agent, evaluator, RNG state,
and execution budget remain checkpoint-bound controls.
"""

from __future__ import annotations

from typing import Any


DESIGN_VERSION = "experiment-r-dynamic-routing-v1"
FREEZE_SCHEMA = "mlevolve_experiment_r_design_freeze_v1"
DECISION_POINT_SCHEMA = "mlevolve_experiment_r_decision_point_v1"
ROUTING_RESULT_SCHEMA = "mlevolve_experiment_r_routing_result_v1"
EXECUTION_RESULT_SCHEMA = "mlevolve_experiment_r_execution_result_v1"
ANALYSIS_REPORT_SCHEMA = "mlevolve_experiment_r_analysis_report_v1"

STAGES = ("draft", "improve", "debug")
ONLINE_SYSTEMS = (
    "no_memory",
    "flat_retrieval",
    "sop_only",
    "runforest_only",
    "static_hybrid",
    "dynamic_hybrid",
    "reversed_router",
)
ORACLE_SYSTEM = "oracle_router"

RAW_CANDIDATES_PER_SOURCE = 12
MAX_INJECTED_ITEMS = 6
MEMORY_PROMPT_TOKEN_BUDGET = 1536
VISIBILITY_TOKEN_BUDGET = 4096
TOKEN_COUNTER = "unicode_non_whitespace_v1"

# Slot counts change what the Agent sees, not which raw candidates exist.
STAGE_SLOT_POLICY: dict[str, dict[str, dict[str, int]]] = {
    "static_hybrid": {
        stage: {"sop": 3, "runforest": 3} for stage in STAGES
    },
    "dynamic_hybrid": {
        "draft": {"sop": 4, "runforest": 2},
        "improve": {"sop": 3, "runforest": 3},
        "debug": {"sop": 2, "runforest": 4},
    },
    "reversed_router": {
        "draft": {"sop": 2, "runforest": 4},
        "improve": {"sop": 3, "runforest": 3},
        "debug": {"sop": 4, "runforest": 2},
    },
}

STAGE_FUSION_WEIGHTS: dict[str, dict[str, dict[str, float]]] = {
    "static_hybrid": {
        stage: {"sop": 0.50, "runforest": 0.50} for stage in STAGES
    },
    "dynamic_hybrid": {
        "draft": {"sop": 0.70, "runforest": 0.30},
        "improve": {"sop": 0.40, "runforest": 0.60},
        "debug": {"sop": 0.25, "runforest": 0.75},
    },
    "reversed_router": {
        "draft": {"sop": 0.25, "runforest": 0.75},
        "improve": {"sop": 0.50, "runforest": 0.50},
        "debug": {"sop": 0.70, "runforest": 0.30},
    },
}

SYSTEM_CONTRACTS: dict[str, dict[str, Any]] = {
    "no_memory": {
        "prompt_memory_enabled": False,
        "selection": "none",
        "agent_calls": 1,
    },
    "flat_retrieval": {
        "prompt_memory_enabled": True,
        "selection": "stage_agnostic_unified_relevance_top_k",
        "agent_calls": 1,
    },
    "sop_only": {
        "prompt_memory_enabled": True,
        "selection": "sop_rank_top_k",
        "agent_calls": 1,
    },
    "runforest_only": {
        "prompt_memory_enabled": True,
        "selection": "runforest_rank_top_k",
        "agent_calls": 1,
    },
    "static_hybrid": {
        "prompt_memory_enabled": True,
        "selection": "fixed_three_sop_three_runforest",
        "agent_calls": 1,
    },
    "dynamic_hybrid": {
        "prompt_memory_enabled": True,
        "selection": "draft_sop_first_improve_mixed_debug_tree_first",
        "agent_calls": 1,
    },
    "reversed_router": {
        "prompt_memory_enabled": True,
        "selection": "draft_tree_heavy_improve_mixed_debug_sop_heavy",
        "agent_calls": 1,
    },
    ORACLE_SYSTEM: {
        "prompt_memory_enabled": False,
        "selection": "gold_disposition_over_frozen_candidate_pool",
        "agent_calls": 0,
        "host_side_only": True,
    },
}

# Same five-task universe frozen by Experiments A/C.  Spooky retains its
# positive-control role and is excluded from natural-task utility aggregation.
TASKS: tuple[dict[str, Any], ...] = (
    {
        "task_id": "denoising-dirty-documents",
        "task_family": "image_restoration",
        "terminal_metric": "rmse",
        "direction": "minimize",
        "role": "natural",
    },
    {
        "task_id": "leaf-classification",
        "task_family": "tabular_multiclass",
        "terminal_metric": "log_loss",
        "direction": "minimize",
        "role": "natural",
    },
    {
        "task_id": "aerial-cactus-identification",
        "task_family": "image_binary_classification",
        "terminal_metric": "roc_auc",
        "direction": "maximize",
        "role": "natural",
    },
    {
        "task_id": "new-york-city-taxi-fare-prediction",
        "task_family": "tabular_regression",
        "terminal_metric": "rmse",
        "direction": "minimize",
        "role": "natural",
    },
    {
        "task_id": "spooky-author-identification",
        "task_family": "text_classification",
        "terminal_metric": "log_loss",
        "direction": "minimize",
        "role": "known_invalid_positive_control_reported_separately",
    },
)

# Every decision from these runs is excluded from the retrievable memory pool,
# including unselected adjacent nodes.  This is the run-level isolation unit.
HELDOUT_RUN_IDS = (
    "20260701_180146",
    "20260701_145201",
    "20260701_145250",
    "20260516_125444",
    "20260701_155016",
    "20260510_025317",
    "20260701_180038",
)

OFFLINE_STAGE_COUNTS = {"draft": 20, "improve": 60, "debug": 100}
EXECUTION_STAGE_COUNTS = {"draft": 20, "improve": 20, "debug": 20}
EXECUTION_SEEDS = (11, 29, 47, 59)

PRIMARY_COMPARISONS = (
    ("dynamic_hybrid", "flat_retrieval"),
    ("dynamic_hybrid", "static_hybrid"),
    ("dynamic_hybrid", "reversed_router"),
    ("dynamic_hybrid", "no_memory"),
)

OFFLINE_METRICS = (
    "ndcg_at_10",
    "route_accuracy",
    "granularity_precision_at_5",
    "detail_intrusion_at_5",
    "adoption_ap_at_10",
    "transition_hit_at_1",
    "transition_mrr",
    "fallback_accuracy",
    "unsafe_candidate_escape",
    "empty_result_rate",
)

EXECUTION_METRICS = (
    "useful_adoption_rate",
    "exposure_to_static_adoption_rate",
    "static_to_runtime_activation_rate",
    "negative_transfer_rate",
    "time_to_first_valid_seconds",
    "completion_rate",
    "terminal_metric_delta",
    "prompt_tokens",
    "gpu_hours",
)


def validate_design() -> None:
    """Fail closed if a source edit silently changes a frozen invariant."""

    assert set(OFFLINE_STAGE_COUNTS) == set(STAGES)
    assert set(EXECUTION_STAGE_COUNTS) == set(STAGES)
    assert sum(OFFLINE_STAGE_COUNTS.values()) >= 150
    assert all(value >= 20 for value in EXECUTION_STAGE_COUNTS.values())
    assert len(TASKS) == 5 and len({row["task_id"] for row in TASKS}) == 5
    assert len(ONLINE_SYSTEMS) == 7
    assert set(ONLINE_SYSTEMS) | {ORACLE_SYSTEM} == set(SYSTEM_CONTRACTS)
    for policy in ("static_hybrid", "dynamic_hybrid", "reversed_router"):
        for stage in STAGES:
            slots = STAGE_SLOT_POLICY[policy][stage]
            weights = STAGE_FUSION_WEIGHTS[policy][stage]
            assert sum(slots.values()) == MAX_INJECTED_ITEMS
            assert abs(sum(weights.values()) - 1.0) < 1e-9


validate_design()

TASK_BY_ID = {row["task_id"]: row for row in TASKS}
