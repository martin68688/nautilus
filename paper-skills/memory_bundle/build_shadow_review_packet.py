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
    build_shadow_review_packet,
    load_shadow_records_from_ledger,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic, hash-bound disagreement packet for an "
            "independent human or supervising-agent shadow audit."
        )
    )
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--max-records", type=int, default=50)
    args = parser.parse_args()
    if args.packet.exists():
        raise FileExistsError(f"Refusing to overwrite review packet: {args.packet}")
    packet = build_shadow_review_packet(
        load_shadow_records_from_ledger(args.ledger),
        max_records=args.max_records,
    )
    args.packet.parent.mkdir(parents=True, exist_ok=True)
    args.packet.write_text(
        json.dumps(packet, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "population_count": packet["population_count"],
        "sample_count": packet["sample_count"],
        "evidence_hash": packet["evidence_hash"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()

