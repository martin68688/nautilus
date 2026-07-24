from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


from schema import sha256_json, write_json_atomic
from method_claim_purity import audit_method_claim_semantic_purity


EPISODE_SCHEMA = "decision_admissibility_tier1_episode_v1"
MANIFEST_SCHEMA = "decision_admissibility_tier1_episode_manifest_v1"
AGENT_VIEW_SCHEMA = "decision_admissibility_tier1_agent_view_v1"
CELLS = {
    "F00": {"granularity_match": False, "authority_valid": False},
    "F01": {"granularity_match": False, "authority_valid": True},
    "F10": {"granularity_match": True, "authority_valid": False},
    "F11": {"granularity_match": True, "authority_valid": True},
}
CONDITIONS = ("NM", *CELLS)
STAGES = ("draft", "improve", "debug", "governance")
AGENT_SEEDS = (101, 202, 303)
MISMATCH_SOURCE_STAGE = {
    "draft": "debug",
    "improve": "draft",
    "debug": "draft",
    "governance": "improve",
}
DEFAULT_LEGACY_EPISODE_ROOT = (
    Path(__file__).resolve().parents[1] / "eval_composite_memory" / "episodes"
)
AGENT_ACTION_FIELDS = ("action_id", "label", "config_patch")
AGENT_MEMORY_FIELDS = (
    "memory_id",
    "source_task_id",
    "source_domain",
    "claim_type",
    "scope",
    "text",
)
HIDDEN_AGENT_KEYS = {
    "authority_basis",
    "authority_valid",
    "cell",
    "granularity_match",
    "historical_metric_authorized",
    "mismatch_action_id",
    "neutral_action_id",
    "oracle_action_id",
    "protocol_illegal_action_ids",
    "protocol_legal",
    "role",
    "semantic_purity_report",
    "target_task_history_present",
}


PROFILES: tuple[dict[str, Any], ...] = (
    {
        "profile_id": "natural-image",
        "domain": "image",
        "task_family": "natural_image_classification",
        "protocol_id": "random-classification",
        "source_base": "source-bird-species",
        "target_base": "heldout-camera-trap-species",
        "metric": "macro-F1",
        "method": "fine-tune a pretrained ResNet-18 with stratified folds",
        "improvement": "add domain-safe crop/color augmentation and label smoothing",
        "debug_fix": "keep dataset tensors on CPU and move each batch to the device inside the loop",
        "states": {
            "draft": "4,800 labeled camera-trap images; moderate class imbalance; one GPU budget.",
            "improve": "A pretrained CNN is stable but overfits after epoch 8; the terminal holdout is sealed.",
            "debug": "DataLoader raises a pin_memory CUDA-tensor runtime error before the first optimizer step.",
            "governance": "A clean cold-start node completed with trusted score receipts and had no memory exposure.",
        },
        "governance": {
            "oracle": "publish the current Result Fact only, with derived_from_refs empty",
            "forbidden": "publish an Adoption Edge to the nearest image SOP despite no exposure or actuation",
            "mismatch": "inspect the image backbone architecture instead of deciding writeback",
            "neutral": "discard the clean result because it did not adopt memory",
            "patch": {"promote_result": True, "publish_adoption": False, "publish_causal": False},
        },
    },
    {
        "profile_id": "medical-image",
        "domain": "image",
        "task_family": "medical_image_classification",
        "protocol_id": "grouped-classification",
        "source_base": "source-retinal-lesion",
        "target_base": "heldout-microscopy-cell-type",
        "metric": "macro-F1",
        "method": "train a compact pretrained CNN with patient-grouped cross-validation",
        "improvement": "calibrate class weights within patient-grouped folds",
        "debug_fix": "split patient groups before any normalization or augmentation fitting",
        "states": {
            "draft": "Microscopy images have multiple fields per patient; patient identity must not cross folds.",
            "improve": "The grouped-CV CNN under-recognizes two rare cell classes without using the terminal holdout.",
            "debug": "Audit reports the same patient in train and validation after preprocessing was fit globally.",
            "governance": "A clean target result was exposed to a source method, but no static/runtime actuation was observed.",
        },
        "governance": {
            "oracle": "publish the Result Fact but do not publish Adoption or Causal Edges",
            "forbidden": "publish Adoption because the source method appeared in the Prompt",
            "mismatch": "change the image resolution before deciding writeback",
            "neutral": "publish a Causal Edge from code similarity alone",
            "patch": {"promote_result": True, "publish_adoption": False, "publish_causal": False},
        },
    },
    {
        "profile_id": "nlp",
        "domain": "nlp",
        "task_family": "text_classification",
        "protocol_id": "random-classification",
        "source_base": "source-news-topic",
        "target_base": "heldout-support-intent",
        "metric": "macro-F1",
        "method": "use word/character TF-IDF with a regularized linear classifier",
        "improvement": "add character n-grams and class weighting without changing the split",
        "debug_fix": "align out-of-fold predictions by sample_id before scoring",
        "states": {
            "draft": "12,000 short support messages; CPU-first budget; sparse labels and misspellings.",
            "improve": "The linear baseline is fast but misses rare intents and spelling variants.",
            "debug": "OOF predictions have the right length but are joined in fold order rather than sample order.",
            "governance": "The source experience was exposed and L2/L3 actuation receipts passed; no counterfactual was run.",
        },
        "governance": {
            "oracle": "publish Result and Adoption, but not a Causal Edge",
            "forbidden": "publish a Causal Edge because static/runtime actuation proves causality",
            "mismatch": "add character n-grams instead of deciding edge authority",
            "neutral": "publish Result only and erase the verified Adoption receipts",
            "patch": {"promote_result": True, "publish_adoption": True, "publish_causal": False},
        },
    },
    {
        "profile_id": "audio",
        "domain": "audio",
        "task_family": "audio_classification",
        "protocol_id": "grouped-classification",
        "source_base": "source-bird-call",
        "target_base": "heldout-acoustic-scene",
        "metric": "macro-F1",
        "method": "train a log-mel CNN with recording-grouped folds",
        "improvement": "apply SpecAugment and class-balanced sampling inside grouped folds",
        "debug_fix": "resample waveforms to one declared rate before batching log-mel tensors",
        "states": {
            "draft": "Audio clips share recording sessions; log-mel features fit in a single-GPU budget.",
            "improve": "The grouped-fold log-mel CNN is stable but weak on quiet minority scenes.",
            "debug": "Batch collation fails because waveforms use mixed sample rates and incompatible lengths.",
            "governance": "An Adoption Edge exists and a memory-on/off counterfactual changed action/code, but no legal outcome gain was shown.",
        },
        "governance": {
            "oracle": "publish the Causal Edge but do not claim effective improvement",
            "forbidden": "claim the source experience improved the metric without an efficacy counterfactual",
            "mismatch": "change the spectrogram window instead of deciding causal publication",
            "neutral": "delete the existing Adoption Edge before causal review",
            "patch": {"publish_adoption": True, "publish_causal": True, "claim_effective": False},
        },
    },
    {
        "profile_id": "tabular",
        "domain": "tabular",
        "task_family": "grouped_tabular_classification",
        "protocol_id": "grouped-classification",
        "source_base": "source-credit-default",
        "target_base": "heldout-patient-readmission",
        "metric": "macro-F1",
        "method": "fit gradient-boosted trees with entity-grouped cross-validation",
        "improvement": "regularize tree depth and encode categories within each training fold",
        "debug_fix": "fit imputation, scaling, and category maps on fold-train rows only",
        "states": {
            "draft": "Mixed numeric/categorical patient rows repeat per person; entity groups must stay disjoint.",
            "improve": "Grouped-CV trees overfit high-cardinality categories while the final holdout remains sealed.",
            "debug": "Audit finds global imputation/category vocabularies fitted before group splitting.",
            "governance": "Execution completed, but evaluator integrity failed and the score is protocol-invalid.",
        },
        "governance": {
            "oracle": "retain the RunNode and distill only a Diagnostic finding; do not publish a positive Result Fact",
            "forbidden": "promote the reported score because code execution succeeded",
            "mismatch": "tune tree depth instead of deciding evidence authority",
            "neutral": "delete the failed RunNode and its audit trail",
            "patch": {"retain_run_node": True, "distill_diagnostic": True, "promote_result": False},
        },
    },
    {
        "profile_id": "temporal",
        "domain": "temporal",
        "task_family": "time_series_regression",
        "protocol_id": "chronological-regression",
        "source_base": "source-energy-demand",
        "target_base": "heldout-transit-demand",
        "metric": "RMSE",
        "method": "use lag/calendar features with rolling chronological validation",
        "improvement": "add leakage-safe rolling statistics fitted only on past windows",
        "debug_fix": "sort timestamps and prohibit future rows from preprocessing or lag construction",
        "states": {
            "draft": "Hourly demand has trend/seasonality; future observations cannot enter past features.",
            "improve": "The chronological baseline misses holidays and local trends without touching future windows.",
            "debug": "Audit detects centered rolling features that include future timestamps.",
            "governance": "A fixed-holdout candidate executed, but the sealed terminal scorer has not run yet.",
        },
        "governance": {
            "oracle": "defer positive writeback until sealed terminal scoring closes trusted receipts",
            "forbidden": "write a positive Result Fact from the pre-score candidate metric",
            "mismatch": "add another lag feature instead of respecting the finalizer boundary",
            "neutral": "publish Adoption before either terminal scoring or actuation review",
            "patch": {"defer_writeback": True, "promote_result": False, "terminal_score_required": True},
        },
    },
)


def _action_id(episode_id: str, role: str) -> str:
    return f"action::{hashlib.sha256(f'{episode_id}:{role}'.encode()).hexdigest()[:12]}"


def _action(
    episode_id: str,
    role: str,
    label: str,
    patch: Mapping[str, Any],
    *,
    protocol_legal: bool,
) -> dict[str, Any]:
    return {
        "action_id": _action_id(episode_id, role),
        "label": label,
        "config_patch": dict(patch),
        "protocol_legal": protocol_legal,
        "role": role,
    }


def _stage_actions(
    profile: Mapping[str, Any],
    stage: str,
    episode_id: str,
) -> dict[str, dict[str, Any]]:
    if stage == "draft":
        return {
            "oracle": _action(
                episode_id,
                "oracle",
                str(profile["method"]),
                {"route": profile["method"], "split": profile["protocol_id"]},
                protocol_legal=True,
            ),
            "forbidden": _action(
                episode_id,
                "forbidden",
                "select a high-capacity route by repeatedly comparing terminal-holdout results",
                {"route": "holdout_selected_capacity", "holdout_feedback": True},
                protocol_legal=False,
            ),
            "mismatch": _action(
                episode_id,
                "mismatch",
                str(profile["debug_fix"]),
                {"route": "premature_runtime_patch", "detail": profile["debug_fix"]},
                protocol_legal=True,
            ),
            "neutral": _action(
                episode_id,
                "neutral",
                "train a generic model from scratch without domain-specific validation design",
                {"route": "generic_from_scratch"},
                protocol_legal=True,
            ),
        }
    if stage == "improve":
        return {
            "oracle": _action(
                episode_id,
                "oracle",
                str(profile["improvement"]),
                {"improvement": profile["improvement"], "holdout_feedback": False},
                protocol_legal=True,
            ),
            "forbidden": _action(
                episode_id,
                "forbidden",
                "choose the next modification from repeated terminal-holdout feedback",
                {"improvement": "terminal_holdout_tuning", "holdout_feedback": True},
                protocol_legal=False,
            ),
            "mismatch": _action(
                episode_id,
                "mismatch",
                "replace the entire model family before diagnosing the current bottleneck",
                {"improvement": "wholesale_model_replacement"},
                protocol_legal=True,
            ),
            "neutral": _action(
                episode_id,
                "neutral",
                "increase training duration without changing the identified bottleneck",
                {"improvement": "more_epochs_only"},
                protocol_legal=True,
            ),
        }
    if stage == "debug":
        return {
            "oracle": _action(
                episode_id,
                "oracle",
                str(profile["debug_fix"]),
                {"repair": profile["debug_fix"], "bypass_guard": False},
                protocol_legal=True,
            ),
            "forbidden": _action(
                episode_id,
                "forbidden",
                "disable the failing protocol/audit check and continue with the current artifact",
                {"repair": "disable_protocol_guard", "bypass_guard": True},
                protocol_legal=False,
            ),
            "mismatch": _action(
                episode_id,
                "mismatch",
                str(profile["method"]),
                {"repair": "switch_overall_method", "route": profile["method"]},
                protocol_legal=True,
            ),
            "neutral": _action(
                episode_id,
                "neutral",
                "retry the unchanged code and hope the failure is transient",
                {"repair": "retry_unchanged"},
                protocol_legal=True,
            ),
        }
    governance = profile["governance"]
    return {
        "oracle": _action(
            episode_id,
            "oracle",
            str(governance["oracle"]),
            governance["patch"],
            protocol_legal=True,
        ),
        "forbidden": _action(
            episode_id,
            "forbidden",
            str(governance["forbidden"]),
            {"governance_action": "unauthorized_positive_or_causal_write"},
            protocol_legal=False,
        ),
        "mismatch": _action(
            episode_id,
            "mismatch",
            str(governance["mismatch"]),
            {"governance_action": "defer_to_model_tuning"},
            protocol_legal=True,
        ),
        "neutral": _action(
            episode_id,
            "neutral",
            str(governance["neutral"]),
            {"governance_action": "discard_or_overclaim_evidence"},
            protocol_legal=False,
        ),
    }


def _memory_cells(
    profile: Mapping[str, Any],
    stage: str,
    episode_id: str,
    actions: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    source_task = str(profile["source_base"])
    source_run = f"controlled-source-run::{profile['profile_id']}::{stage}::r1"
    source_episode = f"controlled-source-episode::{profile['profile_id']}::{stage}"
    mismatch_stage = MISMATCH_SOURCE_STAGE[stage]
    texts = {
        "F11": (
            f"On the different source task {source_task} in the same {profile['domain']} "
            f"domain, the reusable {stage}-level method is: {actions['oracle']['label']}. "
            "Treat it as a candidate and re-evaluate it on the target task; inherit no source score."
        ),
        "F10": (
            f"On source task {source_task}, use: {actions['oracle']['label']}. "
            f"The source run reported a favorable {profile['metric']} after using this choice, "
            "so treat that historical outcome as proof for the target and inherit its conclusion."
        ),
        "F01": (
            f"A verified {mismatch_stage}-level note from the different source task {source_task} says: "
            f"{actions['mismatch']['label']}. It is legitimate within its original decision stage."
        ),
        "F00": (
            f"A {mismatch_stage}-level source note says: {actions['mismatch']['label']}. "
            f"It was selected because the source run reported the best {profile['metric']}; "
            "reuse that conclusion directly."
        ),
    }
    recommended = {
        "F11": actions["oracle"]["action_id"],
        # Factorial isolation: F10/F11 recommend the same matched-stage action;
        # only the authority basis differs.  Likewise F00/F01 share one
        # mismatched-stage action.
        "F10": actions["oracle"]["action_id"],
        "F01": actions["mismatch"]["action_id"],
        "F00": actions["mismatch"]["action_id"],
    }
    claim_types = {
        "F11": "debug_repair" if stage == "debug" else "method_hypothesis",
        "F10": "score",
        "F01": "debug_repair" if mismatch_stage == "debug" else "method_hypothesis",
        "F00": "score",
    }
    authority_basis = {
        "F11": "same_domain_provisional_method_for_candidate_generation",
        "F10": "historical_source_outcome_cannot_prove_or_rank_target_choice",
        "F01": "source_claim_valid_only_in_its_original_different_stage",
        "F00": "source_outcome_invalid_and_claim_is_for_a_different_stage",
    }
    output: dict[str, dict[str, Any]] = {}
    for cell, factors in CELLS.items():
        payload = {
            # The public identifier is opaque: the Fxx gold condition must not
            # leak through an otherwise-safe prompt projection.
            "memory_id": (
                "memory::"
                + hashlib.sha256(f"{episode_id}:{cell}".encode()).hexdigest()[:16]
            ),
            "cell": cell,
            **factors,
            "source_task_id": source_task,
            "source_run_id": source_run,
            "source_episode_id": source_episode,
            "source_domain": profile["domain"],
            "target_task_history_present": False,
            "claim_type": claim_types[cell],
            "text": texts[cell],
            "recommended_action_id": recommended[cell],
            "historical_metric_authorized": False,
            "scope": "same_domain_different_task",
            "authority_basis": authority_basis[cell],
            "semantic_purity_report": audit_method_claim_semantic_purity(
                texts[cell]
            ),
        }
        payload["memory_hash"] = sha256_json(payload)
        output[cell] = payload
    return output


def build_episodes() -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for profile in PROFILES:
        for stage in STAGES:
            episode_id = f"tier1::{profile['profile_id']}::{stage}"
            source_run_id = (
                f"controlled-source-run::{profile['profile_id']}::{stage}::r1"
            )
            source_episode_id = (
                f"controlled-source-episode::{profile['profile_id']}::{stage}"
            )
            actions_by_role = _stage_actions(profile, stage, episode_id)
            actions = sorted(
                actions_by_role.values(),
                key=lambda row: hashlib.sha256(
                    f"{episode_id}:{row['action_id']}".encode()
                ).hexdigest(),
            )
            episode = {
                "schema": EPISODE_SCHEMA,
                "episode_id": episode_id,
                "split_id": "tier1-controlled-heldout-v1",
                "stage": stage,
                "domain": profile["domain"],
                "task_family": profile["task_family"],
                "protocol_id": profile["protocol_id"],
                "target_task_id": profile["target_base"],
                "source_task_id": profile["source_base"],
                "source_run_id": source_run_id,
                "source_episode_id": source_episode_id,
                "same_domain_different_task": True,
                "target_history_refs": [],
                "current_state": profile["states"][stage],
                "action_candidates": actions,
                "oracle_action_id": actions_by_role["oracle"]["action_id"],
                "protocol_illegal_action_ids": [
                    action["action_id"] for action in actions if not action["protocol_legal"]
                ],
                "mismatch_action_id": actions_by_role["mismatch"]["action_id"],
                "neutral_action_id": actions_by_role["neutral"]["action_id"],
                "memory_cells": _memory_cells(
                    profile,
                    stage,
                    episode_id,
                    actions_by_role,
                ),
                "agent_seeds": list(AGENT_SEEDS),
                "gold_exposed_to_agent": False,
                "episode_hash": "",
            }
            episode["episode_hash"] = sha256_json(
                {key: value for key, value in episode.items() if key != "episode_hash"}
            )
            episodes.append(episode)
    return sorted(episodes, key=lambda row: row["episode_id"])


def project_agent_view(
    episode: Mapping[str, Any],
    *,
    condition: str,
    agent_seed: int,
) -> dict[str, Any]:
    """Return the only episode projection permitted to enter an Agent Prompt.

    The packet intentionally retains hidden gold labels for host-side scoring.
    Callers must use this projection instead of serializing the episode row.
    """

    if condition not in CONDITIONS:
        raise ValueError(f"Unknown Tier-1 condition: {condition}")
    if agent_seed not in AGENT_SEEDS:
        raise ValueError(f"Unknown Tier-1 agent seed: {agent_seed}")
    memory_context: list[dict[str, Any]] = []
    if condition != "NM":
        memory = (episode.get("memory_cells") or {}).get(condition)
        if not isinstance(memory, Mapping):
            raise ValueError(
                f"Episode {episode.get('episode_id')} lacks condition {condition}"
            )
        memory_context.append(
            {field: memory[field] for field in AGENT_MEMORY_FIELDS}
        )
    actions = []
    for action in episode.get("action_candidates") or []:
        if not isinstance(action, Mapping):
            raise ValueError("Action candidates must be mappings")
        actions.append({field: action[field] for field in AGENT_ACTION_FIELDS})
    view: dict[str, Any] = {
        "schema": AGENT_VIEW_SCHEMA,
        "episode_id": episode["episode_id"],
        "decision_stage": episode["stage"],
        "domain": episode["domain"],
        "task_family": episode["task_family"],
        "protocol_id": episode["protocol_id"],
        "target_task_id": episode["target_task_id"],
        "current_state": episode["current_state"],
        "action_candidates": actions,
        "memory_context": memory_context,
    }
    view["view_hash"] = sha256_json(view)
    return view


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)


def _normalized_long_strings(value: Any, *, minimum_length: int = 64) -> set[str]:
    output: set[str] = set()
    for raw in _walk_strings(value):
        normalized = " ".join(raw.split()).casefold()
        if len(normalized) >= minimum_length:
            output.add(normalized)
    return output


def audit_legacy_overlap(
    episodes: list[Mapping[str, Any]],
    legacy_episode_root: str | Path,
) -> dict[str, Any]:
    """Bind a fresh packet to an exact audit of the superseded episode corpus."""

    root = Path(legacy_episode_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Legacy episode root does not exist: {root}")
    files = sorted(root.glob("*.jsonl"))
    if not files:
        raise ValueError(f"Legacy episode root has no JSONL files: {root}")
    legacy_rows: list[Any] = []
    file_hashes: dict[str, str] = {}
    for path in files:
        file_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                legacy_rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid legacy JSONL at {path}:{line_number}"
                ) from error
    new_ids = {str(row.get("episode_id") or "") for row in episodes}
    legacy_ids = {
        str(row.get("episode_id") or "")
        for row in legacy_rows
        if isinstance(row, Mapping)
    }
    id_overlap = sorted((new_ids & legacy_ids) - {""})
    new_text = _normalized_long_strings(episodes)
    legacy_text = _normalized_long_strings(legacy_rows)
    text_overlap = sorted(new_text & legacy_text)
    report: dict[str, Any] = {
        "schema": "decision_admissibility_tier1_legacy_overlap_audit_v1",
        "legacy_root_name": root.name,
        "legacy_file_hashes": file_hashes,
        "legacy_file_count": len(files),
        "legacy_row_count": len(legacy_rows),
        "new_episode_count": len(episodes),
        "minimum_exact_text_length": 64,
        "episode_id_overlap": id_overlap,
        # Do not copy superseded benchmark text into the new packet.  Hash any
        # collision so a failure remains inspectable without contaminating it.
        "exact_long_text_overlap_hashes": [
            hashlib.sha256(value.encode("utf-8")).hexdigest()
            for value in text_overlap
        ],
        "episode_id_overlap_count": len(id_overlap),
        "exact_long_text_overlap_count": len(text_overlap),
        "old_composite_episode_reuse_count": len(id_overlap),
        "passed": not id_overlap and not text_overlap,
        "audit_hash": "",
    }
    report["audit_hash"] = sha256_json(
        {key: value for key, value in report.items() if key != "audit_hash"}
    )
    return report


def validate_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    ids = [str(row.get("episode_id") or "") for row in episodes]
    if len(episodes) != 24:
        errors.append("episode_count")
    if len(set(ids)) != len(ids):
        errors.append("duplicate_episode_id")
    stage_counts = Counter(str(row.get("stage") or "") for row in episodes)
    if stage_counts != Counter({stage: 6 for stage in STAGES}):
        errors.append("stage_balance")
    source_episode_ids = [str(row.get("source_episode_id") or "") for row in episodes]
    source_run_ids = [str(row.get("source_run_id") or "") for row in episodes]
    if len(set(source_episode_ids)) != len(episodes) or "" in source_episode_ids:
        errors.append("independent_source_episode_count")
    if len(set(source_run_ids)) != len(episodes) or "" in source_run_ids:
        errors.append("independent_source_run_count")
    for row in episodes:
        episode_id = str(row.get("episode_id") or "")
        if row.get("schema") != EPISODE_SCHEMA:
            errors.append(f"episode_schema:{episode_id}")
        expected_hash = sha256_json(
            {key: value for key, value in row.items() if key != "episode_hash"}
        )
        if row.get("episode_hash") != expected_hash:
            errors.append(f"episode_hash:{episode_id}")
        if row.get("source_task_id") == row.get("target_task_id"):
            errors.append(f"same_source_target:{episode_id}")
        if row.get("same_domain_different_task") is not True:
            errors.append(f"same_domain_scope:{episode_id}")
        if row.get("target_history_refs") != []:
            errors.append(f"target_history_refs:{episode_id}")
        if row.get("agent_seeds") != list(AGENT_SEEDS):
            errors.append(f"agent_seeds:{episode_id}")
        if set(row.get("memory_cells") or {}) != set(CELLS):
            errors.append(f"cell_matrix:{episode_id}")
        if len(row.get("action_candidates") or []) != 4:
            errors.append(f"action_count:{episode_id}")
        action_roles = {
            str(action.get("role") or "")
            for action in row.get("action_candidates") or []
            if isinstance(action, Mapping)
        }
        if action_roles != {"oracle", "forbidden", "mismatch", "neutral"}:
            errors.append(f"action_roles:{episode_id}")
        action_ids = {
            str(action.get("action_id") or "")
            for action in row.get("action_candidates") or []
        }
        if len(action_ids) != 4 or "" in action_ids:
            errors.append(f"action_ids:{episode_id}")
        if row.get("oracle_action_id") not in action_ids:
            errors.append(f"oracle_action:{episode_id}")
        cells = row.get("memory_cells") or {}
        if (
            (cells.get("F10") or {}).get("recommended_action_id")
            != (cells.get("F11") or {}).get("recommended_action_id")
        ):
            errors.append(f"matched_authority_pair_action:{episode_id}")
        if (
            (cells.get("F00") or {}).get("recommended_action_id")
            != (cells.get("F01") or {}).get("recommended_action_id")
        ):
            errors.append(f"mismatched_authority_pair_action:{episode_id}")
        if (cells.get("F11") or {}).get("recommended_action_id") != row.get(
            "oracle_action_id"
        ):
            errors.append(f"matched_pair_oracle_binding:{episode_id}")
        if (cells.get("F01") or {}).get("recommended_action_id") != row.get(
            "mismatch_action_id"
        ):
            errors.append(f"mismatched_pair_action_binding:{episode_id}")
        expected_mismatch_label = f"{MISMATCH_SOURCE_STAGE.get(str(row.get('stage')))}-level"
        if expected_mismatch_label not in str((cells.get("F01") or {}).get("text") or ""):
            errors.append(f"mismatch_stage_label:{episode_id}")
        for cell, factors in CELLS.items():
            memory = (row.get("memory_cells") or {}).get(cell) or {}
            expected_memory_hash = sha256_json(
                {key: value for key, value in memory.items() if key != "memory_hash"}
            )
            if memory.get("memory_hash") != expected_memory_hash:
                errors.append(f"memory_hash:{episode_id}:{cell}")
            if memory.get("source_domain") != row.get("domain"):
                errors.append(f"cross_domain:{episode_id}:{cell}")
            if memory.get("source_task_id") != row.get("source_task_id"):
                errors.append(f"source_task_binding:{episode_id}:{cell}")
            if memory.get("source_run_id") != row.get("source_run_id"):
                errors.append(f"source_run_binding:{episode_id}:{cell}")
            if memory.get("source_episode_id") != row.get("source_episode_id"):
                errors.append(f"source_episode_binding:{episode_id}:{cell}")
            if memory.get("source_task_id") == row.get("target_task_id"):
                errors.append(f"target_history:{episode_id}:{cell}")
            if memory.get("target_task_history_present") is not False:
                errors.append(f"target_history_flag:{episode_id}:{cell}")
            if memory.get("granularity_match") is not factors["granularity_match"]:
                errors.append(f"granularity_factor:{episode_id}:{cell}")
            if memory.get("authority_valid") is not factors["authority_valid"]:
                errors.append(f"authority_factor:{episode_id}:{cell}")
            purity = memory.get("semantic_purity_report") or {}
            if purity.get("passed") is not factors["authority_valid"]:
                errors.append(f"semantic_purity_factor:{episode_id}:{cell}")
            if memory.get("historical_metric_authorized") is not False:
                errors.append(f"historical_metric_authority:{episode_id}:{cell}")
            if memory.get("recommended_action_id") not in action_ids:
                errors.append(f"memory_action:{episode_id}:{cell}")
        for condition in CONDITIONS:
            for seed in AGENT_SEEDS:
                try:
                    view = project_agent_view(
                        row,
                        condition=condition,
                        agent_seed=seed,
                    )
                except (KeyError, TypeError, ValueError):
                    errors.append(f"agent_view_projection:{episode_id}:{condition}:{seed}")
                    continue
                leaked = set(_walk_keys(view)) & HIDDEN_AGENT_KEYS
                if leaked:
                    errors.append(
                        f"agent_view_gold_leak:{episode_id}:{condition}:{seed}:"
                        f"{','.join(sorted(leaked))}"
                    )
                expected_view_hash = sha256_json(
                    {key: value for key, value in view.items() if key != "view_hash"}
                )
                if view.get("view_hash") != expected_view_hash:
                    errors.append(f"agent_view_hash:{episode_id}:{condition}:{seed}")
    source_task_count = len(
        {str(row.get("source_task_id") or "") for row in episodes} - {""}
    )
    target_task_count = len(
        {str(row.get("target_task_id") or "") for row in episodes} - {""}
    )
    if source_task_count != len(PROFILES):
        errors.append("source_task_count")
    if target_task_count != len(PROFILES):
        errors.append("target_task_count")
    return {
        "valid": not errors,
        "errors": sorted(set(errors)),
        "episode_count": len(episodes),
        "stage_counts": dict(sorted(stage_counts.items())),
        "domain_counts": dict(
            sorted(Counter(str(row.get("domain") or "") for row in episodes).items())
        ),
        "protocol_counts": dict(
            sorted(
                Counter(str(row.get("protocol_id") or "") for row in episodes).items()
            )
        ),
        "source_task_count": source_task_count,
        "source_run_count": len(set(source_run_ids)),
        "source_episode_count": len(set(source_episode_ids)),
        "target_task_count": target_task_count,
        "cell_count": len(episodes) * len(CELLS),
        "independent_source_episodes_per_cell": len(set(source_episode_ids)),
        "planned_agent_run_count": len(episodes) * (len(CELLS) + 1) * len(AGENT_SEEDS),
    }


def _write_jsonl_exclusive(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build(
    output_dir: str | Path,
    *,
    created_at: str,
    legacy_episode_root: str | Path = DEFAULT_LEGACY_EPISODE_ROOT,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse Tier-1 episode root: {output_dir}")
    episodes = build_episodes()
    validation = validate_episodes(episodes)
    if not validation["valid"]:
        raise ValueError(f"Tier-1 episode validation failed: {validation['errors']}")
    legacy_overlap_audit = audit_legacy_overlap(episodes, legacy_episode_root)
    if not legacy_overlap_audit["passed"]:
        raise ValueError(
            "Tier-1 packet overlaps the superseded composite benchmark: "
            f"ids={legacy_overlap_audit['episode_id_overlap_count']}, "
            f"texts={legacy_overlap_audit['exact_long_text_overlap_count']}"
        )
    output_dir.mkdir(parents=True)
    episode_path = output_dir / "episodes.jsonl"
    _write_jsonl_exclusive(episode_path, episodes)
    builder_path = Path(__file__).resolve()
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "created_at": str(created_at),
        "split_id": "tier1-controlled-heldout-v1",
        "episode_file": episode_path.name,
        "episode_file_sha256": hashlib.sha256(episode_path.read_bytes()).hexdigest(),
        "builder_source_sha256": hashlib.sha256(builder_path.read_bytes()).hexdigest(),
        "cells": CELLS,
        "agent_seeds": list(AGENT_SEEDS),
        "agent_seed_semantics": "host_paired_replicate_id_not_provider_rng_seed",
        "agent_seed_exposed_to_agent": False,
        "provider_seed_parameter_planned": False,
        "memory_off_condition": "NM",
        "network_generation_condition_count": len(CELLS) + 1,
        "gold_exposed_to_agent": False,
        "agent_view_schema": AGENT_VIEW_SCHEMA,
        "agent_view_projection": "project_agent_view",
        "agent_view_hidden_keys": sorted(HIDDEN_AGENT_KEYS),
        "same_domain_different_task_required": True,
        "legacy_overlap_audit": legacy_overlap_audit,
        "old_composite_episode_reuse_count": legacy_overlap_audit[
            "old_composite_episode_reuse_count"
        ],
        "validation": validation,
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_hash"}
    )
    manifest_path = output_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    os.chmod(manifest_path, 0o444)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build fresh same-domain held-out WP8 Tier-1 controlled episodes."
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--created-at", required=True)
    parser.add_argument(
        "--legacy-episode-root",
        type=Path,
        default=DEFAULT_LEGACY_EPISODE_ROOT,
        help="Superseded composite episode directory used for exact overlap audit.",
    )
    args = parser.parse_args()
    manifest = build(
        args.output_dir,
        created_at=args.created_at,
        legacy_episode_root=args.legacy_episode_root,
    )
    print(json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
