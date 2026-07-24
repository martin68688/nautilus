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
    verify_canary_oracle_packet,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify independently labeled canary decisions against an "
            "immutable Authority ledger and emit an exact oracle map."
        )
    )
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--oracle", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    for output in (args.oracle, args.report):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite output: {output}")
    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    if not isinstance(packet, dict):
        raise ValueError("Canary oracle packet must be a JSON object")
    verified = verify_canary_oracle_packet(
        packet,
        load_shadow_records_from_ledger(args.ledger),
    )
    args.oracle.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.oracle.write_text(
        json.dumps(
            {"oracle_should_allow": verified["oracle_should_allow"]},
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    args.report.write_text(
        json.dumps(
            verified["review_report"],
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verified["review_report"], sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
