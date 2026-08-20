#!/usr/bin/env python3
"""Read-only verifier for the frozen v147-r7 runtime and UCI data staging."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


runtime = Path(sys.argv[1])
data_root = Path(sys.argv[2])
lock_path = runtime / "RELEASE_SOURCE_LOCK.json"
lock = json.loads(lock_path.read_text(encoding="utf-8"))
canonical = json.dumps(
    {**lock, "manifest_hash": ""},
    sort_keys=True,
    ensure_ascii=False,
    separators=(",", ":"),
).encode("utf-8")
assert hashlib.sha256(canonical).hexdigest() == lock["manifest_hash"]

expected = {row["path"]: row["sha256"] for row in lock["files"]}
actual: dict[str, str] = {}
for path in sorted(runtime.rglob("*")):
    if not path.is_file() or path.is_symlink() or path == lock_path:
        continue
    relative = path.relative_to(runtime).as_posix()
    if (
        relative.endswith((".pyc", ".pyo"))
        or "/__pycache__/" in f"/{relative}/"
        or Path(relative).name.startswith("._")
        or Path(relative).name == ".DS_Store"
    ):
        raise AssertionError(f"forbidden runtime artifact: {relative}")
    actual[relative] = sha256(path)
assert set(actual) == set(expected), {
    "missing": sorted(set(expected) - set(actual)),
    "extra": sorted(set(actual) - set(expected)),
}
bad = sorted(path for path, digest in actual.items() if digest != expected[path])
assert not bad, {"bad_hashes": bad}

expected_data = {
    "train.csv": "b683ca60b32a28e9998545314dd89157d841b4cc7397e0f14a478f45e7ab81de",
    "test.csv": "54f919e1d9d173b80a7c031920d9cb6815755ee339ce2d0216af543c2ce6cc7c",
    "sample_submission.csv": "435fd5c642576717c5d8e4c0504fbfcd52e7cc95da19aedd4a7da3cddeb55b4e",
}
for name, digest in expected_data.items():
    assert sha256(data_root / name) == digest, name
assert not (data_root / "target_labels_host_only.csv").exists()
image_count = sum(1 for path in data_root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
assert image_count == 1599, image_count

print(
    json.dumps(
        {
            "source_lock": lock["manifest_hash"],
            "locked_files": len(expected),
            "bad_hashes": 0,
            "extra_files": 0,
            "images": image_count,
            "host_labels_absent": True,
        },
        sort_keys=True,
    )
)
