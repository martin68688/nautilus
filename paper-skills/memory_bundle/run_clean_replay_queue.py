from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MLEVOLVE = REPO / "mlevolve"
if str(MLEVOLVE) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE))

from authority.clean_replay_runner import run_replay_queue  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Execute an immutable Clean Replay queue with host-owned runtime "
            "protocol observation and emit certification material."
        )
    )
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--queue-manifest", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--protocol-id", default="mlevolve-default")
    parser.add_argument("--protocol-version", default="2")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--cpu-number", type=int, default=8)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--collector-id", default="wp6-clean-replay-host-v1"
    )
    args = parser.parse_args()
    if args.timeout <= 0 or args.cpu_number <= 0 or args.num_gpus <= 0:
        raise ValueError("Replay timeout/resources must be positive")
    report = run_replay_queue(
        bundle_path=args.bundle,
        queue_path=args.queue,
        queue_manifest_path=args.queue_manifest,
        data_root=args.data_root,
        output_dir=args.output,
        task_ids=args.task,
        protocol_id=args.protocol_id,
        protocol_version=args.protocol_version,
        timeout=args.timeout,
        cpu_number=args.cpu_number,
        num_gpus=args.num_gpus,
        collector_id=args.collector_id,
    )
    print(json.dumps(report, sort_keys=True, ensure_ascii=False))
    if report["status"] != "certification_material_ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
