"""Live smoke for the generic Agent -> probe -> trace -> Agent loop.

This utility accepts arbitrary memory text and Python source. It contains no
task/model signatures and is intended for local or devpod debugging.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

from agents.adoption_verifier_agent import AdoptionVerifierAgent
from authority.actuation import ExperienceContract, Predicate
from authority.adoption_verification import sha256_text
from engine.executor import Interpreter


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", required=True)
    parser.add_argument("--code-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model",
        default=os.environ.get("ADOPTION_VERIFIER_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or "gpt-5.6-sol",
    )
    return parser.parse_args()


def main() -> int:
    args = _args()
    code_path = Path(args.code_file).resolve(strict=True)
    code = code_path.read_text(encoding="utf-8")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for the live verifier smoke")
    stage = SimpleNamespace(
        model=args.model,
        api_key=api_key,
        base_url=os.environ.get(
            "OPENAI_BASE_URL",
            "http://cliproxyapi-haoming.ecepxie.svc.cluster.local:8317/v1",
        ),
    )
    cfg = SimpleNamespace(
        log_dir=output,
        start_cpu_id=0,
        cpu_number=1,
        adoption_verifier=SimpleNamespace(
            enabled=True,
            mode="enforce",
            model=args.model,
            temperature=0.0,
            max_tokens=4096,
            max_contracts_per_call=8,
            max_code_chars=120000,
            require_signed_trace=False,
        ),
        evaluation_authority=SimpleNamespace(
            mode="shadow",
            protocol_runtime_mode="host_sdk_shadow",
            runtime_protocol_observer_enabled=False,
        ),
        agent=SimpleNamespace(
            code=stage,
            feedback=stage,
            search=SimpleNamespace(parallel_search_num=1, num_gpus=1),
            protocol_preflight=SimpleNamespace(enabled=False),
            candidate_execution_contract=SimpleNamespace(enabled=False),
        ),
    )
    agent = SimpleNamespace(
        cfg=cfg,
        acfg=cfg.agent,
        task_desc="Execute the supplied Python candidate and verify memory adoption.",
        evaluation_authority=SimpleNamespace(ledger=None),
    )
    candidate = SimpleNamespace(
        id=f"live-smoke-{sha256_text(code)[:16]}",
        code=code,
        adoption_verification_plan={},
        adoption_runtime_trace={},
        adoption_verifier_verdict={},
        adoption_verifier_mode="off",
    )
    clause_id = f"live-memory::{sha256_text(args.memory)[:24]}"
    contract = ExperienceContract(
        must_preserve=[],
        must_change=[Predicate(f"clause_applied::{clause_id}", True, args.memory)],
        must_not_use=[],
        expected_runtime_observations=[Predicate("target_path_executed", True)],
        clause_id=clause_id,
        sop_id=f"live-sop::{sha256_text(args.memory)[:24]}",
        active_protocol_ref="live-smoke@1#" + ("0" * 64),
        target_task_id="live-smoke",
        publication_class="certified",
    ).finalize()
    verifier = AdoptionVerifierAgent(agent)
    plan = verifier.prepare(candidate, [contract])
    (output / "workspace").mkdir(parents=True, exist_ok=True)
    interpreter = Interpreter(output / "workspace", timeout=120, cfg=cfg)
    result = interpreter.run(
        code,
        candidate.id,
        adoption_verification_plan=plan,
    )
    candidate.adoption_runtime_trace = dict(result.adoption_trace or {})
    verdict = verifier.finalize(candidate)
    summary = {
        "candidate_id": candidate.id,
        "execution_error": result.exc_type,
        "plan_hash": plan.get("plan_hash"),
        "trace_hash": candidate.adoption_runtime_trace.get("trace_hash"),
        "verdict": verdict.get("contract_results", []),
    }
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2))
    return 0 if result.exc_type is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
