#!/usr/bin/env python3
"""Run the exact-replay protocol transaction without unrelated draft roles."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "mlevolve"))

from config import load_cfg, load_task_desc, prep_agent_workspace, save_run
from engine.agent_search import AgentSearch
from engine.executor import Interpreter
from engine.search_node import Journal
from utils.logging_config import setup_logging
from utils.seed import set_global_seed


def main() -> int:
    cfg = load_cfg()
    set_global_seed(cfg.agent.seed)
    logger = setup_logging(cfg)
    task_desc = load_task_desc(cfg)
    prep_agent_workspace(cfg)

    journal = Journal()
    agent = AgentSearch(task_desc=task_desc, cfg=cfg, journal=journal)
    interpreter = Interpreter(
        cfg.workspace_dir,
        **OmegaConf.to_container(cfg.exec),
        cfg=cfg,
    )

    def execute(code: str, id: str, *args, **kwargs):
        return interpreter.run(code, id, *args, **kwargs)

    # The production policy remains the strict three-role policy. This focused
    # harness reserves only its second slot so the exact replay path can be
    # tested without running baseline/novel models in a constrained dev pod.
    agent._draft_generation_count = 1
    replay = agent.step(
        exec_callback=execute,
        node=None,
        execute_immediately=False,
        draft_role="memory_reproduction",
    )
    replay = agent.execute_deferred_node(replay, execute)
    save_run(cfg, journal)

    node = replay
    for _ in range(20):
        tx = node.protocol_repair or {}
        state = str(tx.get("state") or "")
        if state == "completed":
            payload = {
                "status": "completed",
                "node_id": node.id,
                "replay_status": node.replay_status,
                "metric": node.metric.value if node.metric else None,
                "rank_eligible": bool((node.leakage_audit or {}).get("rank_eligible")),
                "runtime_protocol_status": (node.leakage_audit or {}).get("runtime_protocol_status"),
                "history": tx.get("history") or [],
            }
            print("REPLAY_SMOKE_RESULT=" + json.dumps(payload, sort_keys=True))
            return 0
        if state == "exhausted":
            raise RuntimeError(
                "Replay protocol transaction exhausted: "
                + json.dumps(tx, sort_keys=True)
            )
        try:
            next_node = agent.step(exec_callback=execute, node=node)
        except Exception as exc:
            logger.warning(
                "Replay smoke stage attempt failed and will follow transaction retry policy: %s",
                exc,
            )
            save_run(cfg, journal)
            continue
        if next_node is not None:
            node = next_node
        save_run(cfg, journal)

    raise RuntimeError("Replay protocol smoke exceeded its bounded 20-step transaction budget")


if __name__ == "__main__":
    raise SystemExit(main())
