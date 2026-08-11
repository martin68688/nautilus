#!/usr/bin/env python3
"""Offline causal Debug retrieval replay for the Leaf v7 memory release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
MLEVOLVE_ROOT = REPO_ROOT / "mlevolve"
if str(MLEVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE_ROOT))

from agents.memory.atomic_claim_memory import (  # noqa: E402
    AUTHORIZED_DEBUG_STATUS,
    structured_debug_relevance,
)


DINO_V45_TRANSITION = (
    "run::e2e-pilot-leaf-tiered-debug-router-v45__leaf-classification__"
    "dynamic_hybrid__seed-1::attempt-000::source-72bdeafd::transition::"
    "8e77146c99c9::c18a69664d3c"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def contains_all(*needles: str) -> Callable[[dict[str, Any]], bool]:
    lowered = tuple(value.lower() for value in needles)

    def predicate(claim: dict[str, Any]) -> bool:
        text = f"{claim.get('failure_text', '')}\n{claim.get('repair_action', '')}".lower()
        return all(value in text for value in lowered)

    return predicate


CASES: tuple[dict[str, Any], ...] = (
    {
        "name": "v76_dinov2_input_224_requires_518",
        "query": (
            "AssertionError: Input height (224) doesn't match model (518) for "
            "vit_small_patch14_dinov2.lvd142m"
        ),
        "expected": lambda claim: claim.get("source_transition_id")
        == DINO_V45_TRANSITION,
        "minimum_score": 0.90,
    },
    {
        "name": "dinov3_missing_hubconf_path",
        "query": (
            "FileNotFoundError: ./working/dinov3-main/hubconf.py not found "
            "while torch.hub.load loads DINOv3"
        ),
        "expected": contains_all("filenotfounderror", "dinov3", "hubconf.py"),
        "minimum_score": 0.40,
    },
    {
        "name": "odd_width_symmetry_broadcast_31_32",
        "query": (
            "ValueError: operands could not be broadcast together with shapes "
            "(891,31) (891,32) in create_hierarchical_features symmetry_2"
        ),
        "expected": contains_all("broadcast", "31", "32", "symmetr"),
        "minimum_score": 0.40,
    },
    {
        "name": "lightgbm_fit_verbose_api",
        "query": (
            "TypeError: LGBMClassifier.fit() got an unexpected keyword argument "
            "verbose"
        ),
        "expected": contains_all("lgbmclassifier.fit", "verbose"),
        "minimum_score": 0.50,
    },
    {
        "name": "nan_probability_log_loss",
        "query": (
            "ValueError: Input contains NaN when sklearn.metrics.log_loss "
            "evaluates probabilities"
        ),
        "expected": contains_all("input contains nan", "log_loss"),
        "minimum_score": 0.30,
    },
    {
        "name": "dataloader_shared_memory_bus_error",
        "query": (
            "RuntimeError: DataLoader worker exited unexpectedly with a bus "
            "error from insufficient shared memory and num_workers=4"
        ),
        "expected": contains_all("shared memory", "num_workers=4"),
        "minimum_score": 0.35,
    },
    {
        "name": "xgboost_early_stopping_api",
        "query": (
            "TypeError: XGBClassifier.fit() got an unexpected keyword argument "
            "early_stopping_rounds"
        ),
        "expected": contains_all("xgbclassifier.fit", "early_stopping_rounds"),
        "minimum_score": 0.35,
    },
)


def legacy_dino_observation(path: Path) -> dict[str, Any]:
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            candidate_id = str(
                value.get("candidate_id")
                or value.get("transition_id")
                or value.get("id")
                or ""
            )
            if candidate_id == DINO_V45_TRANSITION and "rank" in value:
                rows.append(dict(value))
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(checkpoint)
    if not rows:
        raise ValueError("v76 checkpoint has no legacy DINO transition observation")
    return rows[0]


def rank_case(
    claims: list[dict[str, Any]], case: dict[str, Any]
) -> dict[str, Any]:
    ranked: list[dict[str, Any]] = []
    for claim in claims:
        score, receipt = structured_debug_relevance(
            case["query"],
            claim.get("failure_text") or "",
            claim.get("repair_action") or "",
            claim,
        )
        ranked.append(
            {
                "claim_id": claim["id"],
                "source_transition_id": claim["source_transition_id"],
                "claim_type": claim["claim_type"],
                "score": score,
                "expected_match": bool(case["expected"](claim)),
                "rank_receipt": receipt,
                "failure_excerpt": str(claim.get("failure_text") or "")[:500],
                "repair_excerpt": str(claim.get("repair_action") or "")[:500],
            }
        )
    ranked.sort(key=lambda row: (-float(row["score"]), str(row["claim_id"])))
    expected_ranks = [
        index
        for index, row in enumerate(ranked, 1)
        if row["expected_match"]
    ]
    best_expected_rank = min(expected_ranks) if expected_ranks else None
    top = ranked[0]
    passed = bool(
        top["expected_match"]
        and float(top["score"]) >= float(case["minimum_score"])
        and best_expected_rank == 1
    )
    return {
        "query": case["query"],
        "minimum_score": case["minimum_score"],
        "best_expected_rank": best_expected_rank,
        "top_candidate": top,
        "top_five": ranked[:5],
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--v76-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    release_dir = args.release_dir.resolve(strict=True)
    claim_path = release_dir / "atomic_claims.json"
    bundle = json.loads(claim_path.read_text(encoding="utf-8"))
    claims = [
        dict(claim)
        for claim in bundle.get("claims") or []
        if claim.get("claim_status") == AUTHORIZED_DEBUG_STATUS
    ]
    results = {case["name"]: rank_case(claims, case) for case in CASES}
    legacy = legacy_dino_observation(args.v76_checkpoint.resolve(strict=True))

    # A novel NameError without an exact symbol in memory must not receive a
    # confident repair merely because the exception class is common.
    abstain_query = "NameError: name cleanup is not defined"
    abstain_rows = [
        structured_debug_relevance(
            abstain_query,
            claim.get("failure_text") or "",
            claim.get("repair_action") or "",
            claim,
        )[0]
        for claim in claims
    ]
    abstain_max = max(abstain_rows, default=0.0)
    report: dict[str, Any] = {
        "schema": "mlevolve_leaf_atomic_debug_retrieval_replay_v1",
        "release_dir": str(release_dir),
        "atomic_claims_file_sha256": sha256_file(claim_path),
        "authorized_debug_claim_count": len(claims),
        "legacy_v76_dino_observation": {
            "rank": legacy.get("rank"),
            "score": legacy.get("score"),
            "gate_reason": legacy.get("gate_reason"),
            "operation_authorized": legacy.get("operation_authorized"),
        },
        "v7_cases": results,
        "unseen_cleanup_nameerror_abstention": {
            "query": abstain_query,
            "maximum_score": abstain_max,
            "threshold": 0.50,
            "passed": abstain_max < 0.50,
        },
        "quality_gates": {
            "legacy_v76_dino_was_rank_41": legacy.get("rank") == 41,
            "v7_dino_is_rank_1": (
                results["v76_dinov2_input_224_requires_518"]["best_expected_rank"]
                == 1
            ),
            "all_debug_replays_pass": all(
                result["passed"] for result in results.values()
            ),
            "unseen_error_abstains": abstain_max < 0.50,
        },
        "report_sha256": "",
    }
    report["report_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in report.items() if key != "report_sha256"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    write_json(args.output, report)
    print(json.dumps(report["quality_gates"], indent=2, sort_keys=True))
    return 0 if all(report["quality_gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
