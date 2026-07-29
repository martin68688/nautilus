#!/usr/bin/env python3
"""Online devpod smoke for formal prospective counterfactual recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from agents.coder import plan_and_code_query
from agents.memory.prospective_audit import ProspectiveAuditLogger
from config import _load_cfg
from utils.response import is_valid_python_script


PARENT_CODE = """import numpy as np


def evaluate(validation_rows, valid_mask, predictions):
    filtered = predictions[valid_mask]
    return np.sqrt(np.mean((filtered - validation_rows[\"fare\"]) ** 2))
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--incident-log", type=Path)
    parser.add_argument("--incident-journal", type=Path)
    parser.add_argument("--incident-parent-node", default="")
    parser.add_argument("--incident-failure-marker", default="")
    args = parser.parse_args()

    cfg = _load_cfg(args.config, use_cli_args=False)
    agent = SimpleNamespace(cfg=cfg, acfg=cfg.agent)
    incident_mode = bool(args.incident_log or args.incident_journal)
    if incident_mode:
        if not all(
            (
                args.incident_log,
                args.incident_journal,
                args.incident_parent_node,
                args.incident_failure_marker,
            )
        ):
            raise ValueError("incident replay requires log, journal, parent and marker")
        verbose_log = args.incident_log.read_text(encoding="utf-8")
        failure_position = verbose_log.index(args.incident_failure_marker)
        response_marker = "generate response: "
        response_position = verbose_log.rfind(
            response_marker,
            0,
            failure_position,
        )
        if response_position < 0:
            raise RuntimeError("incident log has no generation response before failure")
        completion = verbose_log[
            response_position + len(response_marker) : failure_position
        ]
        warning_marker = " WARNING: Code extraction failed after retries"
        warning_position = completion.rfind(warning_marker)
        if warning_position >= 0:
            log_line_start = completion.rfind("\n", 0, warning_position)
            completion = completion[:log_line_start].rstrip() + "\n"
        journal = json.loads(args.incident_journal.read_text(encoding="utf-8"))
        parents = [
            row
            for row in journal.get("nodes", [])
            if str(row.get("id")) == args.incident_parent_node
        ]
        if len(parents) != 1:
            raise RuntimeError("incident parent node is missing or ambiguous")
        parent_code = str(parents[0].get("code") or "")
        plan = ""
    else:
        prompt = (
            "You are producing an observer-only paired counterfactual for an ML "
            "debug decision. It will never be executed. Preserve the current "
            "decision context and output format.\n\n"
            "The current route requires complete SEARCH/REPLACE blocks. Repair the "
            "validation alignment bug using the suppressed memory below. Output "
            "only the patch.\n\n"
            "# Original code\n```python\n"
            + PARENT_CODE
            + "```\n\n# Frozen pre-Authority memory arm\n"
            "Validation predictions must preserve the original Host validation row "
            "count even when feature cleaning uses a mask.\n"
        )
        plan, completion = plan_and_code_query(
            agent,
            prompt,
            retries=1,
            generation_retries=2,
            request_timeout=180,
        )
        parent_code = PARENT_CODE
    node = SimpleNamespace(parent=SimpleNamespace(code=parent_code))
    recovered_plan, recovered_code, metadata = (
        ProspectiveAuditLogger._recover_counterfactual_completion(
            node,
            plan,
            completion,
        )
    )
    if not recovered_code.strip():
        raise RuntimeError("online counterfactual recovery produced no code")
    if not is_valid_python_script(recovered_code):
        raise RuntimeError("online counterfactual recovery produced invalid Python")

    result = {
        "schema": "mlevolve_counterfactual_recovery_smoke_v1",
        "status": "pass",
        "incident_replay": incident_mode,
        "requested_model": str(cfg.agent.code.model),
        "temperature": float(cfg.agent.code.temp),
        "counterfactual_generation_format": metadata.get(
            "counterfactual_generation_format", ""
        ),
        "counterfactual_action_hash": hashlib.sha256(
            recovered_plan.encode("utf-8")
        ).hexdigest(),
        "counterfactual_code_hash": hashlib.sha256(
            recovered_code.encode("utf-8")
        ).hexdigest(),
        **metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
