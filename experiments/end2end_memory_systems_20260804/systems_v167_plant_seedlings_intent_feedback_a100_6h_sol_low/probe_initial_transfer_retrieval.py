#!/usr/bin/env python3
"""Run the real initial Transfer retrieval path and stop before planning/code."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from omegaconf import OmegaConf


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_b3(value: Any) -> bool:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True).lower()
    return any(
        token in text
        for token in (
            "efficientnet_b3",
            "efficientnet-b3",
            "efficientnet b3",
        )
    )


def _candidate_hits(rows: Any) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for row in rows or []:
        if not isinstance(row, dict) or not _contains_b3(row):
            continue
        hits.append(
            {
                "id": str(row.get("id") or row.get("candidate_id") or ""),
                "candidate_kind": str(
                    row.get("candidate_kind") or row.get("granularity") or ""
                ),
                "title": str(row.get("title") or ""),
                "representation_attributes": dict(
                    ((row.get("portable_text") or {}).get(
                        "representation_attributes"
                    ))
                    or {}
                ),
            }
        )
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--description", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(
            f"refusing to reuse retrieval-probe output: {args.output_root}"
        )
    for path in (args.config, args.description):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.data_root.is_dir():
        raise NotADirectoryError(args.data_root)

    repo_root = Path(__file__).resolve().parents[3]
    mlevolve_root = repo_root / "mlevolve"
    sys.path.insert(0, str(mlevolve_root))

    from agents.memory.external_skill_memory import fetch_external_skill_memory
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
                "search": {
                    "num_gpus": 1,
                    "parallel_search_num": 1,
                }
            },
            "cpu_number": 16,
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
    agent.update_data_preview()
    prompt_text, ref_ids, source_name = fetch_external_skill_memory(
        agent,
        "draft",
        run_memory="",
        data_preview=agent.data_preview or "",
        coldstart=getattr(agent, "coldstart_description", ""),
        baseline_model=getattr(agent, "coldstart_primary_model_name", ""),
        draft_role="memory_transfer",
    )
    layer = agent.external_skill_memory
    if layer is None:
        raise RuntimeError("formal config did not initialize external memory")
    pack = layer.current_navigation_pack()
    route = str(pack.get("stage_route", {}).get("route") or "")
    if route not in {
        "projected_multiround_search_direct_judge",
        "projected_multiround_search_judge_four_candidate_portfolio",
    }:
        raise RuntimeError(
            f"unexpected retrieval route: {pack.get('stage_route')}"
        )

    projected_hits = _candidate_hits(pack.get("projected_candidate_universe"))
    accumulated_hits = _candidate_hits(pack.get("candidate_pool"))
    selected_hits = _candidate_hits(pack.get("selected_candidates"))
    judge = (pack.get("dynamic_retrieval") or {}).get("judge") or {}
    selected_ids = list(pack.get("final_prompt_candidate_ids") or [])
    selected_id_set = set(selected_ids)
    portfolio = dict(pack.get("transfer_portfolio") or {})
    summary = {
        "schema": "mlevolve_initial_transfer_retrieval_probe_v1",
        "status": "passed",
        "stopped_before": "planner_and_code_generation",
        "config_path": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "run_identity_path": str(identity_path),
        "source_name": source_name,
        "route": dict(pack.get("stage_route") or {}),
        "fallback_used": bool(
            (pack.get("retrieval_agent") or {}).get("fallback_used")
        ),
        "search_agent_calls": int(
            ((pack.get("dynamic_retrieval") or {}).get("search") or {}).get(
                "agent_calls", 0
            )
        ),
        "judge_agent_calls": int(judge.get("agent_calls", 0)),
        "projected_candidate_count": len(
            pack.get("projected_candidate_universe") or []
        ),
        "accumulated_candidate_count": len(pack.get("candidate_pool") or []),
        "selected_candidate_count": len(selected_ids),
        "b3": {
            "mentioned_by_task_description": _contains_b3(task_desc),
            "mentioned_by_data_preview": _contains_b3(agent.data_preview or ""),
            "host_required_backbone_family": str(
                cfg.external_skill_memory.memory_strategy_host_required_backbone_family
                or ""
            ),
            "host_requires_public_pretrained": bool(
                cfg.external_skill_memory.memory_strategy_host_require_public_pretrained
            ),
            "host_required_pretrained_loader": str(
                cfg.external_skill_memory.memory_strategy_host_required_pretrained_loader
                or ""
            ),
            "projected_universe_hits": projected_hits,
            "accumulated_pool_hits": accumulated_hits,
            "selected_hits": selected_hits,
            "entered_projected_universe": bool(projected_hits),
            "entered_accumulated_pool": bool(accumulated_hits),
            "selected_by_judge": bool(selected_hits),
        },
        "judge": {
            "decision": str(judge.get("decision") or ""),
            "reason": str(judge.get("reason") or ""),
            "selected_ids": selected_ids,
            "selected_b3_ids": [
                row["id"] for row in selected_hits if row["id"] in selected_id_set
            ],
            "representation_components": dict(
                judge.get("representation_components") or {}
            ),
            "representation_contracts": dict(
                judge.get("representation_contracts") or {}
            ),
            "assessments": list(judge.get("assessments") or []),
        },
        "transfer_portfolio": {
            "status": str(portfolio.get("status") or ""),
            "requested_size": int(portfolio.get("requested_size") or 0),
            "actual_size": int(portfolio.get("actual_size") or 0),
            "selection_mode": str(portfolio.get("selection_mode") or ""),
            "seat_policy": list(portfolio.get("seat_policy") or []),
            "receipt_sha256": str(portfolio.get("receipt_sha256") or ""),
            "candidates": [
                {
                    "seat": str(row.get("seat") or ""),
                    "candidate_id": str(row.get("candidate_id") or ""),
                    "selection_reason": str(row.get("selection_reason") or ""),
                    "selected_ids": list(row.get("selected_ids") or []),
                    "representation_contracts": dict(
                        row.get("representation_contracts") or {}
                    ),
                    "fingerprint": dict(row.get("fingerprint") or {}),
                    "quality_score": float(row.get("quality_score") or 0.0),
                    "receipt_sha256": str(row.get("receipt_sha256") or ""),
                }
                for row in portfolio.get("candidates") or []
            ],
        },
        "prompt_ref_ids": list(ref_ids),
        "prompt_text_sha256": hashlib.sha256(
            prompt_text.encode("utf-8")
        ).hexdigest(),
        "retrieval_receipt_sha256": str(
            (pack.get("dynamic_retrieval") or {}).get("receipt_sha256") or ""
        ),
    }

    result_root = Path(cfg.log_dir)
    full_path = result_root / "initial_transfer_retrieval_pack.json"
    summary_path = result_root / "initial_transfer_retrieval_summary.json"
    full_path.write_text(
        json.dumps(pack, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Initial Transfer retrieval probe completed: %s", summary_path)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
