from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from fixed_holdout.formal_prepare import (
    build_aerial_holdout,
    build_birds_holdout,
    build_taxi_holdout,
)
from fixed_holdout.formal_verify import verify_formal_holdout
from tests.test_tier2_formal_holdout_builders import (
    _aerial_source,
    _birds_source,
    _taxi_source,
)


def test_all_three_formal_holdouts_pass_independent_verification(
    tmp_path: Path,
) -> None:
    roots = [
        build_aerial_holdout(_aerial_source(tmp_path), tmp_path / "aerial"),
        build_birds_holdout(
            _birds_source(tmp_path),
            tmp_path / "birds-output",
            split_revision="r2",
        ),
        build_taxi_holdout(
            _taxi_source(tmp_path),
            tmp_path / "taxi-output",
            minimum_train_rows=2,
            minimum_holdout_rows=2,
        ),
    ]
    reports = [
        verify_formal_holdout(root, verify_source_artifacts=True)
        for root in roots
    ]

    # The production Taxi floor is intentionally enforced by the verifier.
    assert reports[0]["valid"] is True
    assert reports[1]["valid"] is True
    assert reports[2]["valid"] is False
    assert "taxi_minimum_train_rows" in reports[2]["errors"]
    assert "taxi_minimum_holdout_rows" in reports[2]["errors"]
    for report in reports:
        unsigned = {key: value for key, value in report.items() if key != "report_hash"}
        assert report["report_hash"] == __import__("hashlib").sha256(
            json.dumps(
                unsigned,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()


def test_verifier_detects_evaluator_label_mutation(tmp_path: Path) -> None:
    root = build_aerial_holdout(_aerial_source(tmp_path), tmp_path / "formal")
    labels_path = root / "evaluator_view" / "labels.csv"
    labels = pd.read_csv(labels_path)
    labels.loc[0, "has_cactus"] = 1 - int(labels.loc[0, "has_cactus"])
    labels.to_csv(labels_path, index=False)

    report = verify_formal_holdout(root, verify_source_artifacts=False)

    assert report["valid"] is False
    assert "labels_hash" in report["errors"]


def test_verifier_detects_receipt_mutation(tmp_path: Path) -> None:
    root = build_aerial_holdout(_aerial_source(tmp_path), tmp_path / "formal")
    manifest_path = root / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split_receipt"]["overlap_count"] = 9
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = verify_formal_holdout(root, verify_source_artifacts=False)

    assert report["valid"] is False
    assert "split_manifest_hash" in report["errors"]
    assert "split_receipt_hash" in report["errors"]
    assert "aerial_overlap_receipt" in report["errors"]
