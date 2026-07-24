from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable


REPO = Path(__file__).resolve().parents[2]
MLEVOLVE = REPO / "mlevolve"
if str(MLEVOLVE) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE))

from authority.bundle_publisher import SleepTimePublisher  # noqa: E402
from authority.memory_snapshot import SessionOverlay  # noqa: E402


def load_pipeline(specification: str) -> Callable[..., Any]:
    module_name, separator, attribute = str(specification).partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("--pipeline must use module.path:callable syntax")
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError(f"Configured pipeline is not callable: {specification}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a validated sleep-time Bundle and atomically update CURRENT.json."
    )
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--session-overlay", required=True)
    parser.add_argument("--new-version", required=True)
    parser.add_argument(
        "--pipeline",
        required=False,
        help="Importable callable as module.path:callable. It receives parent, frozen overlay, and staging paths.",
    )
    parser.add_argument(
        "--positive-proposals",
        type=Path,
        help="Host-authored typed Positive Result/Adopted proposal JSON; selects the built-in strict writeback pipeline.",
    )
    parser.add_argument(
        "--protocol-registry",
        type=Path,
        help="Protocol registry directory for the built-in typed writeback pipeline.",
    )
    parser.add_argument("--policy-version", default="authority_v1")
    parser.add_argument("--collector-version", default="1")
    parser.add_argument("--expected-parent-manifest-sha256")
    parser.add_argument("--report")
    args = parser.parse_args()

    if bool(args.pipeline) == bool(args.positive_proposals):
        raise ValueError(
            "Provide exactly one of --pipeline or --positive-proposals"
        )
    if args.positive_proposals:
        if args.protocol_registry is None:
            raise ValueError(
                "--protocol-registry is required with --positive-proposals"
            )
        from positive_writeback_pipeline import make_positive_writeback_pipeline

        pipeline = make_positive_writeback_pipeline(
            new_version=args.new_version,
            proposals_path=args.positive_proposals,
            protocol_registry_path=args.protocol_registry,
            policy_version=args.policy_version,
            collector_version=args.collector_version,
        )
    else:
        pipeline = load_pipeline(args.pipeline)
    publisher = SleepTimePublisher(args.bundle_root)
    overlay = SessionOverlay(args.session_overlay)
    report = publisher.publish(
        new_version=args.new_version,
        overlay=overlay,
        pipeline=pipeline,
        expected_parent_manifest_sha256=args.expected_parent_manifest_sha256,
    )
    payload = report.as_dict()
    if args.report:
        target = Path(args.report).resolve()
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite publication report: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
