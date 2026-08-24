#!/usr/bin/env python3
"""Run the real initial Transfer Draft path and stop before candidate execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from omegaconf import OmegaConf


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--description", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument(
        "--portfolio-candidate",
        default="comprehensive_best",
        choices=(
            "comprehensive_best",
            "backbone_diversity_1",
            "backbone_diversity_2",
            "descriptor_fusion_diversity",
        ),
    )
    parser.add_argument("--portfolio-manifest", type=Path)
    parser.add_argument("--portfolio-manifest-sha256", default="")
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(
            f"refusing to reuse code-generation probe output: {args.output_root}"
        )
    for path in (args.config, args.description):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.data_root.is_dir():
        raise NotADirectoryError(args.data_root)
    if bool(args.portfolio_manifest) != bool(args.portfolio_manifest_sha256):
        raise ValueError(
            "portfolio manifest and SHA-256 must be supplied together"
        )
    if args.portfolio_manifest is not None and not args.portfolio_manifest.is_file():
        raise FileNotFoundError(args.portfolio_manifest)

    repo_root = Path(__file__).resolve().parents[3]
    mlevolve_root = repo_root / "mlevolve"
    sys.path.insert(0, str(mlevolve_root))

    from config import (
        _load_cfg,
        load_task_desc,
        prep_agent_workspace,
        prep_cfg,
        save_run_identity,
    )
    from engine.agent_search import AgentSearch
    from engine.coldstart import build_guidance_description
    from engine.search_node import Journal
    from official_submission import validate_runtime_config
    from utils.logging_config import setup_logging
    from utils.seed import set_global_seed

    raw_cfg = _load_cfg(args.config, use_cli_args=False)
    raw_cfg = OmegaConf.merge(
        raw_cfg,
        {
            "exp_id": "plant-seedlings-classification",
            "dataset_dir": str(args.data_root),
            "data_dir": str(args.data_root),
            "desc_file": str(args.description),
            "exp_name": args.experiment_name,
            "log_dir": str(args.output_root),
            "workspace_dir": str(args.output_root),
            "agent": {
                "steps": 1,
                "search": {
                    "num_gpus": 1,
                    "parallel_search_num": 1,
                },
            },
            "cpu_number": 16,
            "external_skill_memory": {
                "cross_task_transfer_portfolio_enabled": True,
                "cross_task_transfer_portfolio_size": 4,
                "cross_task_transfer_portfolio_candidate": (
                    args.portfolio_candidate
                ),
                "cross_task_transfer_portfolio_manifest_path": (
                    str(args.portfolio_manifest.resolve())
                    if args.portfolio_manifest is not None
                    else ""
                ),
                "cross_task_transfer_portfolio_manifest_sha256": (
                    args.portfolio_manifest_sha256
                ),
            },
        },
    )
    cfg = prep_cfg(raw_cfg)
    rng_identity = set_global_seed(cfg.agent.seed)
    cfg.run_identity.rng_state_hash = str(rng_identity["rng_state_hash"])
    cfg.run_identity.rng_state_components = dict(rng_identity)
    logger = setup_logging(cfg)
    validate_runtime_config(cfg)
    identity_path = save_run_identity(cfg)

    task_desc = load_task_desc(cfg)
    if cfg.coldstart.use_coldstart:
        cfg.coldstart.description = build_guidance_description(
            cfg,
            task_desc=task_desc,
        )
    prep_agent_workspace(cfg)

    agent = AgentSearch(task_desc=task_desc, cfg=cfg, journal=Journal())
    execution_attempts: list[dict[str, Any]] = []

    def reject_candidate_execution(*call_args: Any, **call_kwargs: Any) -> None:
        execution_attempts.append(
            {
                "positional_argument_count": len(call_args),
                "keyword_names": sorted(call_kwargs),
            }
        )
        raise RuntimeError(
            "code-generation probe forbids candidate execution and training"
        )

    node = agent.step(
        node=None,
        exec_callback=reject_candidate_execution,
        execute_immediately=False,
        draft_role="memory_transfer",
    )
    if execution_attempts:
        raise RuntimeError(
            f"candidate execution callback was invoked: {execution_attempts}"
        )
    if node is None or node.stage == "root":
        raise RuntimeError("initial Transfer Draft did not produce a candidate")
    if not str(node.code or "").strip():
        raise RuntimeError("initial Transfer Draft produced empty code")

    memory_layer = getattr(agent, "external_skill_memory", None)
    if memory_layer is None:
        raise RuntimeError("formal config did not initialize external memory")
    pack = memory_layer.current_navigation_pack()
    if pack.get("stage_route", {}).get("route") != (
        "projected_multiround_search_judge_four_candidate_portfolio"
    ):
        raise RuntimeError(
            f"unexpected retrieval route: {pack.get('stage_route')}"
        )
    selected_seat = str(
        (pack.get("portfolio_candidate_receipt") or {}).get("seat") or ""
    )
    if selected_seat != args.portfolio_candidate:
        raise RuntimeError(
            "materialized portfolio seat mismatch: "
            f"expected={args.portfolio_candidate} observed={selected_seat}"
        )

    result_root = Path(cfg.log_dir)
    plan_text = str(node.plan or "")
    code_text = str(node.code or "")
    plan_path = result_root / "initial_transfer_plan.md"
    code_path = result_root / "initial_transfer_candidate.py"
    pack_path = result_root / "initial_transfer_codegen_pack.json"
    trace_path = result_root / "initial_transfer_node_trace.json"
    summary_path = result_root / "initial_transfer_codegen_summary.json"

    plan_path.write_text(plan_text.rstrip() + "\n", encoding="utf-8")
    code_path.write_text(code_text.rstrip() + "\n", encoding="utf-8")
    pack_path.write_text(
        json.dumps(_json_safe(pack), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    node_trace = {
        "id": str(node.id),
        "stage": str(node.stage),
        "draft_role": str(node.draft_role or ""),
        "pending_execution": bool(getattr(node, "pending_execution", False)),
        "audit_repair_required": bool(node.audit_repair_required),
        "plan": plan_text,
        "code": code_text,
        "source_ref_ids": list(node.source_ref_ids or []),
        "role_contract": _json_safe(node.role_contract or {}),
        "atomic_actuation_trace": _json_safe(node.atomic_actuation_trace or {}),
        "plan_diff_verdict": _json_safe(node.plan_diff_verdict or {}),
        "strategy_alignment": _json_safe(node.strategy_alignment or {}),
        "representation_adoption_contracts": _json_safe(
            node.representation_adoption_contracts or {}
        ),
        "representation_adoption_gate": _json_safe(
            node.representation_adoption_gate or {}
        ),
        "memory_transfer_adoption_gate": _json_safe(
            node.memory_transfer_adoption_gate or {}
        ),
        "host_plan_execution_gate": _json_safe(
            node.host_plan_execution_gate or {}
        ),
        "leakage_audit": _json_safe(node.leakage_audit or {}),
        "agent_semantic_review": _json_safe(node.agent_semantic_review or {}),
    }
    trace_path.write_text(
        json.dumps(node_trace, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    judge = (pack.get("dynamic_retrieval") or {}).get("judge") or {}
    effective_transfer = pack.get("memory_transfer") or {}
    atomic_trace = node.atomic_actuation_trace or {}
    planner = atomic_trace.get("planner") or {}
    summary = {
        "schema": "mlevolve_initial_transfer_codegen_probe_v2",
        "status": "passed",
        "stopped_after": "planner_alignment_and_complete_code_generation",
        "candidate_execution_attempted": False,
        "training_started": False,
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "run_identity_path": str(identity_path),
        "node_id": str(node.id),
        "draft_role": str(node.draft_role or ""),
        "pending_execution": bool(getattr(node, "pending_execution", False)),
        "audit_repair_required": bool(node.audit_repair_required),
        "plan_sha256": _sha256_bytes(plan_text.encode("utf-8")),
        "code_sha256": _sha256_bytes(code_text.encode("utf-8")),
        "selected_candidate_ids": list(
            pack.get("final_prompt_candidate_ids") or []
        ),
        "portfolio_candidate": args.portfolio_candidate,
        "portfolio_candidate_receipt": _json_safe(
            pack.get("portfolio_candidate_receipt") or {}
        ),
        "portfolio": _json_safe(pack.get("transfer_portfolio") or {}),
        "portfolio_manifest_input": (
            str(args.portfolio_manifest.resolve())
            if args.portfolio_manifest is not None
            else ""
        ),
        "portfolio_manifest_input_sha256": args.portfolio_manifest_sha256,
        "retrieval_pack_sha256": _sha256(pack_path),
        "representation_components": _json_safe(
            effective_transfer.get("representation_components") or {}
        ),
        "representation_contracts": _json_safe(
            effective_transfer.get("representation_contracts") or {}
        ),
        "independent_judge_reason": str(judge.get("reason") or ""),
        "planner_status": str(planner.get("status") or ""),
        "intent_alignment_enabled": bool(
            planner.get("intent_alignment_enabled")
        ),
        "intent_alignment_verified": bool(
            planner.get("intent_alignment_verified")
        ),
        "intent_alignment_attempts": _json_safe(
            planner.get("intent_alignment_attempts") or []
        ),
        "representation_adoption_gate": _json_safe(
            node.representation_adoption_gate or {}
        ),
        "plan_diff_verdict": _json_safe(node.plan_diff_verdict or {}),
        "artifacts": {
            "plan": str(plan_path),
            "code": str(code_path),
            "retrieval_pack": str(pack_path),
            "node_trace": str(trace_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Initial Transfer code-generation probe completed without execution: %s",
        summary_path,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
