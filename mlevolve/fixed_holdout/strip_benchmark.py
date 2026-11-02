"""Remove the benchmark predictions from a prepared Numerai split.

Rows and labels do not change, and the Host keeps the benchmark column in its
selection labels, so only the public trees and their digests do.  Run it on a
working copy of the split: files are replaced, so hard links to another copy
stay intact.  Lockbox labels live elsewhere, so the digest printed here is then
passed to the evaluator's copy with ``--public-tree-sha256``.

    python -m fixed_holdout.strip_benchmark --split <split_root>
    python -m fixed_holdout.strip_benchmark --split <evaluator_copy> \\
        --public-tree-sha256 <digest>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from fixed_holdout.common import tree_sha256
from fixed_holdout.numerai_source import HOST_OBJECTIVE, MEMORY_NOTE


def _replace_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def clean_description(text: str, resources: str | None = None) -> str:
    """Drop the benchmark's files from the task text and state the Host's objective.

    ``resources`` restates the compute budget for the machine that will run it.
    """
    text = text.replace(f"\n{HOST_OBJECTIVE}", "").replace(f"\n{MEMORY_NOTE}", "")
    text = text.replace("keep integer features as int8.", f"keep integer features as int8.\n{MEMORY_NOTE}")
    text = re.sub(r"- `benchmark/\*\.parquet`:.*?(?=\n  Neither folder)\n", "", text, flags=re.S)
    text = text.replace("  Neither folder contains labels; join them", "  They contain no labels; join them")
    text = re.sub(r"\son those rows with the metric above", "\non those rows", text)
    text = re.sub(r"\n`0\.5 \* CORR.*?```", "", text, flags=re.S)
    if resources:
        text, count = re.subn(
            r"Compute available to your program: .*?\. A program",
            f"Compute available to your program: {resources}. A program",
            text,
            flags=re.S,
        )
        if not count:
            raise ValueError("Description states no compute budget")
    if "benchmark" in text.lower():
        raise ValueError("Description still mentions the benchmark")
    return re.sub(
        r"(ranks\s+solutions during the search\.)", lambda m: f"{m.group(1)}\n{HOST_OBJECTIVE}", text
    )


def _update_manifest(path: Path, **fields: str) -> None:
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.update({key: value for key, value in fields.items() if key in manifest})
        _replace_text(path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def strip_benchmark(
    split_root: Path, public_tree_sha256: str | None = None, resources: str | None = None
) -> str:
    """Return the new public tree digest."""
    split_root = Path(split_root).resolve()
    inputs = [split_root / view / "input" for view in ("train_view", "refit_view")]
    for input_dir in inputs:
        if input_dir.exists():
            shutil.rmtree(input_dir / "benchmark", ignore_errors=True)
            description = input_dir / "description.md"
            _replace_text(
                description, clean_description(description.read_text(encoding="utf-8"), resources)
            )
    if inputs[0].exists():
        public_tree_sha256 = tree_sha256(inputs[0])
    if not public_tree_sha256:
        raise ValueError("No train view here; pass the digest printed for it")
    for view in ("train_view", "selection_view", "evaluator_view"):
        _update_manifest(split_root / view / "fixed_holdout_manifest.json", public_tree_sha256=public_tree_sha256)
    _update_manifest(split_root / "split_manifest.json", public_tree_sha256=public_tree_sha256)
    if inputs[1].exists():
        _update_manifest(
            split_root / "refit_view" / "fixed_holdout_manifest.json",
            refit_tree_sha256=tree_sha256(inputs[1]),
        )
    return public_tree_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--public-tree-sha256", help="Digest printed for the train view")
    parser.add_argument("--resources", help='Compute budget, e.g. "24 CPU threads, 52 GB RAM"')
    args = parser.parse_args()
    print(strip_benchmark(args.split, args.public_tree_sha256, args.resources))


if __name__ == "__main__":
    main()
