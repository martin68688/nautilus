from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MLEVOLVE = REPO / "mlevolve"
if str(MLEVOLVE) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE))

from authority.ledger import AuthorityLedger  # noqa: E402
from authority.rollout import BundleRollbackController  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Atomically repoint CURRENT.json to a verified prior immutable "
            "Bundle while retaining every Bundle and the hash-chained ledger."
        )
    )
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--target-bundle", required=True, type=Path)
    parser.add_argument("--expected-current-manifest-sha256", required=True)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite rollback report: {args.report}")
    report = BundleRollbackController(
        args.bundle_root,
        ledger=AuthorityLedger(args.ledger),
    ).rollback(
        target_bundle_path=args.target_bundle,
        expected_current_manifest_sha256=(
            args.expected_current_manifest_sha256
        ),
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()

