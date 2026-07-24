from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MLEVOLVE = REPO / "mlevolve"
if str(MLEVOLVE) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE))

from authority.rollout import (  # noqa: E402
    CanaryThresholds,
    evaluate_canary,
    load_shadow_records_from_ledger,
)


def _read_oracle(path: Path) -> dict[str, bool]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(
        payload.get("oracle_should_allow"), dict
    ):
        payload = payload["oracle_should_allow"]
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, bool)
        for key, value in payload.items()
    ):
        raise ValueError(
            "Canary oracle must be a decision_id -> boolean JSON object"
        )
    return dict(payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify an Authority ledger and evaluate only its staged-enforce "
            "decision records against an independently labeled canary oracle."
        )
    )
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--minimum-decisions", type=int, default=20)
    parser.add_argument(
        "--max-unauthorized-authority-allows", type=int, default=0
    )
    parser.add_argument("--max-false-denial-rate", type=float, default=0.05)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite canary report: {args.report}")
    report = evaluate_canary(
        load_shadow_records_from_ledger(args.ledger),
        oracle_should_allow=_read_oracle(args.oracle),
        thresholds=CanaryThresholds(
            minimum_decisions=args.minimum_decisions,
            max_unauthorized_authority_allows=(
                args.max_unauthorized_authority_allows
            ),
            max_false_denial_rate=args.max_false_denial_rate,
        ),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

