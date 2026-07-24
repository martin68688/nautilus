from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from schema import read_json, write_json_atomic


def source_fingerprint(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        stat = path.stat()
        records.append(
            (
                path.relative_to(root).as_posix(),
                stat.st_size,
                stat.st_mtime_ns,
            )
        )
    paths_blob = "\n".join(record[0] for record in records).encode()
    stat_blob = "\n".join(
        f"{path}\0{size}\0{mtime_ns}"
        for path, size, mtime_ns in records
    ).encode()
    return {
        "schema": "source_stat_fingerprint_v1",
        "root": str(root),
        "file_count": len(records),
        "paths_sha256": hashlib.sha256(paths_blob).hexdigest(),
        "stat_fingerprint_sha256": hashlib.sha256(stat_blob).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    current = source_fingerprint(args.root)
    if args.compare:
        expected = read_json(args.compare)
        for key in ("file_count", "paths_sha256", "stat_fingerprint_sha256"):
            if current[key] != expected.get(key):
                raise SystemExit(
                    f"source fingerprint mismatch for {key}: "
                    f"expected={expected.get(key)} current={current[key]}"
                )
    if args.output:
        write_json_atomic(args.output, current)
    print(json.dumps(current, indent=2))


if __name__ == "__main__":
    main()
