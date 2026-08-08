"""Shared prompt templates and builders for agents."""

from .shared import (
    ROBUSTNESS_GENERALIZATION_STRATEGY,
    MODEL_ARCHITECTURE_SAFETY,
    prompt_leakage_prevention,
    prompt_resp_fmt,
    get_internet_clarification,
)
from .environment import get_prompt_environment
from .impl_guideline import (
    enforce_host_candidate_entrypoint,
    get_candidate_execution_contract_from_agent,
    get_host_protocol_contract_from_agent,
    get_impl_guideline,
    get_impl_guideline_from_agent,
    host_protocol_preflight_enabled,
    submission_aligned_metric_required,
)

__all__ = [
    "ROBUSTNESS_GENERALIZATION_STRATEGY",
    "MODEL_ARCHITECTURE_SAFETY",
    "prompt_leakage_prevention",
    "prompt_resp_fmt",
    "get_internet_clarification",
    "get_prompt_environment",
    "enforce_host_candidate_entrypoint",
    "get_candidate_execution_contract_from_agent",
    "get_host_protocol_contract_from_agent",
    "get_impl_guideline",
    "get_impl_guideline_from_agent",
    "host_protocol_preflight_enabled",
    "submission_aligned_metric_required",
]
