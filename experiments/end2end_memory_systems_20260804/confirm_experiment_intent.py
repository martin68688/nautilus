#!/usr/bin/env python3
"""Run the three lightweight checks requested before an End2End experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--best-id", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "systems" / "dynamic_hybrid.yaml",
    )
    parser.add_argument(
        "--task-description",
        default="Leaf image classification with multiclass log loss.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(REPO / "mlevolve"))
    sys.path.insert(0, str(REPO))
    from authority.memory_snapshot import MemorySnapshotLoader
    from agents.memory.stage_aware_hybrid_memory import StageAwareHybridMemoryLayer
    from config import _load_cfg

    cfg = _load_cfg(args.config.resolve(), use_cli_args=False)
    ext = cfg.external_skill_memory
    with tempfile.TemporaryDirectory(prefix="mlevolve-intent-") as temporary:
        snapshot = MemorySnapshotLoader(args.bundle_root.resolve()).load(
            current_path="CURRENT.json",
            session_overlay_path=Path(temporary) / "overlay",
            active_protocol_ref=(
                f"{cfg.evaluation_authority.active_protocol_id}@"
                f"{cfg.evaluation_authority.active_protocol_version}"
            ),
            authority_policy_version=str(
                cfg.evaluation_authority.policy_version
            ),
            verify_artifacts=False,
        )
        base = snapshot.base_bundle
        graph_path = base.path / "runforest" / "graph.json"
        index_path = base.path / "runforest" / "index.npz"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        nodes = {
            str(row["id"]): row
            for row in graph.get("nodes") or []
            if str(row.get("id") or "")
        }
        if args.best_id not in nodes:
            raise RuntimeError(f"same-task best is absent: {args.best_id}")

        layer = StageAwareHybridMemoryLayer(
            graph_path=str(graph_path),
            index_path=str(index_path),
            source_name=str(ext.source_name),
            mode=str(ext.mode),
            scoring_mode=str(ext.scoring_mode),
            top_k=int(ext.top_k),
            max_chars=int(ext.max_chars),
            retrieval_control="dynamic_hybrid",
            visibility_mode="off",
            excluded_run_ids=list(ext.excluded_run_ids),
            memory_snapshot=snapshot,
            experiment_r_enabled=True,
            experiment_r_candidate_limit=int(ext.experiment_r_candidate_limit),
            experiment_r_top_k=int(ext.experiment_r_top_k),
            experiment_r_prompt_token_budget=int(ext.experiment_r_prompt_token_budget),
            experiment_r_memory_pool_sha256=str(base.manifest_sha256),
            # The harness pin is deterministic and independent of the live LLM
            # choice.  The real Smoke keeps the Retrieval Agent enabled.
            experiment_r_agentic_retrieval_enabled=False,
        )
        prompt, refs = layer.retrieve_for_node(
            stage="draft",
            task_id=args.task_id,
            task_desc=args.task_description,
            query_parts=["start from the strongest historical same-task method"],
            draft_role="memory_transfer",
        )
        pack = layer.current_navigation_pack()
    same_task = pack["retrieval_agent"]["same_task_best_first"]
    pin = same_task["prompt_pin"]
    if same_task.get("best_runforest_id") != args.best_id:
        raise RuntimeError(
            "router best differs from requested best: "
            f"{same_task.get('best_runforest_id')}"
        )
    if not (
        pin.get("applied") is True
        and pin.get("prompt_visible") is True
        and args.best_id in refs
        and args.best_id in prompt
    ):
        raise RuntimeError("same-task best is not visible in the final Prompt")

    for path in (REPO / "mlevolve" / "run.py", ROOT / "run_assignment.py"):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    child = subprocess.run(
        [sys.executable, "-c", "print('candidate_subprocess_started')"],
        check=True,
        capture_output=True,
        text=True,
    )

    report = {
        "schema": "mlevolve_end2end_intent_confirmation_v1",
        "status": "ready",
        "checks": {
            "same_task_best_exists": True,
            "same_task_best_id": args.best_id,
            "same_task_best_is_final_prompt_visible": True,
            "prompt_visible_ids": list(pack["final_prompt_candidate_ids"]),
            "draft_slots": dict(pack["stage_route"]["requested_slots"]),
            "live_retrieval_agent_enabled_in_smoke": bool(
                ext.experiment_r_agentic_retrieval_enabled
            ),
            "solver_entrypoint_compiles": True,
            "candidate_subprocess_can_start": (
                child.stdout.strip() == "candidate_subprocess_started"
            ),
        },
    }
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
