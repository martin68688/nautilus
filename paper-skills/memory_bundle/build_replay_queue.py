from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
MLEVOLVE = REPO / "mlevolve"
if str(MLEVOLVE) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE))

from authority.clean_replay import build_replay_queue  # noqa: E402


def read_jsonl(path: str | Path) -> list[dict]:
    output = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Candidate row {line_number} is not an object")
        output.append(value)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic, method-diverse Clean Replay queue. Historical "
            "metrics order eligible candidates but are never emitted as evidence."
        )
    )
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--max-per-task", type=int, default=3)
    parser.add_argument("--created-at")
    args = parser.parse_args()
    for target in (args.queue, args.manifest):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite replay artifact: {target}")
    queue = build_replay_queue(
        read_jsonl(args.candidates),
        max_per_task=args.max_per_task,
        created_at=args.created_at,
    )
    queue.write(args.queue, args.manifest)
    print(json.dumps(queue.manifest_dict(), sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
