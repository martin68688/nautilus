#!/usr/bin/env python3
"""Exercise the real Planner -> Strategy-owner return channel online."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from agents.atomic_actuation import run_atomic_staged_actuation_planner
from config import _load_cfg


def strategy_memo() -> dict:
    return {
        "decision": "propose",
        "candidate_compositions": [
            {
                "hypothesis_id": "descriptor-fusion-h1",
                "hypothesis": (
                    "Add target-derived plant shape descriptors through one bounded fusion "
                    "branch while preserving the pretrained EfficientNet backbone and OOF split."
                ),
                "source_memory_ids": ["online-smoke::descriptor-fusion"],
                "compatibility_checks": [
                    "descriptor width matches the fusion projection",
                    "fallback preserves baseline predictions and submission rows",
                ],
                "known_conflicts": ["broad backbone replacement is out of scope"],
                "estimated_compute_seconds": 120,
                "minimal_change_set": [
                    "derive target-only descriptors",
                    "project descriptors into the existing fusion interface",
                ],
                "forbidden_changes": [
                    "replace EfficientNet backbone",
                    "change OOF split",
                    "change submission schema",
                ],
                "expected_mechanism": (
                    "plant-shape descriptors complement convolutional texture features"
                ),
                "falsification_condition": (
                    "retain baseline predictions if submission-aligned OOF macro-F1 does not improve"
                ),
                "novelty_kind": "single_memory_actuation",
            }
        ],
        "recommended_hypothesis_id": "descriptor-fusion-h1",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    system_dir = Path(__file__).resolve().parent
    cfg = _load_cfg(system_dir / "intent_feedback.yaml", use_cli_args=False)
    agent = SimpleNamespace(cfg=cfg, acfg=cfg.agent)
    parent_code = """import numpy as np

def build_model():
    return {"backbone": "efficientnet_b3", "descriptor_width": 0}

def train_and_predict(model):
    return np.zeros((2, 12)), np.zeros((2, 12))

def write_submission(predictions):
    return predictions.shape
"""
    trace = run_atomic_staged_actuation_planner(
        agent,
        strategy_memo=strategy_memo(),
        parent_code=parent_code,
        budget={
            "total_search_seconds": 21600,
            "elapsed_search_seconds": 0,
            "remaining_search_seconds": 21600,
        },
        stage="improve",
    )
    receipt = {
        "schema": "mlevolve_v167_online_feedback_smoke_v1",
        "status": trace.get("status"),
        "planner_attempt_count": len(trace.get("contract_attempts") or []),
        "intent_alignment_enabled": trace.get("intent_alignment_enabled"),
        "intent_alignment_verified": trace.get("intent_alignment_verified"),
        "intent_alignment_statuses": [
            item.get("status") for item in trace.get("intent_alignment_attempts") or []
        ],
        "intent_alignment_reasons": [
            item.get("reason") for item in trace.get("intent_alignment_attempts") or []
        ],
        "plan_sha256": trace.get("plan_sha256"),
        "selected_hypothesis_id": (trace.get("plan") or {}).get("hypothesis_id"),
        "phase_count": len((trace.get("plan") or {}).get("phases") or []),
        "contract_attempts": trace.get("contract_attempts") or [],
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    if trace.get("status") != "accepted":
        return 1
    if trace.get("intent_alignment_verified") is not True:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
