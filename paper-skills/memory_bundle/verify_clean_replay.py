from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MLEVOLVE = REPO / "mlevolve"
if str(MLEVOLVE) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE))

from authority.protocol_registry import ProtocolRegistry  # noqa: E402
from authority.replay_certifier import (  # noqa: E402
    ProtocolRepairSurface,
    verify_protocol_only_patch,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Classify a replay as method-preserved, successor-method, or "
            "human-review using the active ProtocolSpec repair surface."
        )
    )
    parser.add_argument("--source-code", required=True, type=Path)
    parser.add_argument("--replay-code", required=True, type=Path)
    parser.add_argument("--protocol-registry", required=True, type=Path)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--protocol-version", required=True)
    parser.add_argument("--source-artifact-id", required=True)
    parser.add_argument("--replay-artifact-id", required=True)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite replay report: {args.report}")
    registry = ProtocolRegistry(args.protocol_registry)
    spec = registry.get(args.protocol_id, args.protocol_version)
    surface = ProtocolRepairSurface.from_protocol_spec(spec)
    report = verify_protocol_only_patch(
        args.source_code.read_text(encoding="utf-8"),
        args.replay_code.read_text(encoding="utf-8"),
        surface,
        source_artifact_id=args.source_artifact_id,
        replay_artifact_id=args.replay_artifact_id,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report.as_dict(), sort_keys=True, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report.as_dict(), sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
