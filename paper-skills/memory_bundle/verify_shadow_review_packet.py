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
    load_shadow_records_from_ledger,
    verify_shadow_review_packet,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify completed shadow-review dispositions against the immutable "
            "decision evidence in an Authority ledger."
        )
    )
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite review report: {args.report}")
    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    if not isinstance(packet, dict):
        raise ValueError("Shadow review packet must be a JSON object")
    report = verify_shadow_review_packet(
        packet,
        load_shadow_records_from_ledger(args.ledger),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()

