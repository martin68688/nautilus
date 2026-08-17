#!/usr/bin/env python3
"""Collect v123 preferred variants and score them through the Keychain service."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pandas as pd

from top10_v123_catalog import BY_ID, CANDIDATES, TOP10_SLOTS


COMPETITION = "leaf-classification"
PROXY_KEYS = (
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "https_proxy",
    "http_proxy",
    "all_proxy",
    "no_proxy",
)
REMOTE_PACK = r'''
import json
from pathlib import Path
import sys
import zipfile

bundle = Path(sys.argv[1])
expected_ids = json.loads(sys.argv[2])
roots = [Path(value) for value in sys.argv[3:]]
receipts = [
    path
    for root in roots
    for path in sorted(root.glob("*/attempt-000/REPRODUCTION_RECEIPT.json"))
]
passing = {}
for path in receipts:
    value = json.loads(path.read_text())
    if value.get("status") == "pass":
        passing.setdefault(value.get("candidate_id"), []).append(path)
selected = []
for candidate_id in expected_ids:
    matches = passing.get(candidate_id) or []
    if len(matches) != 1:
        raise SystemExit(
            f"expected one passing receipt for {candidate_id}, found {matches}"
        )
    selected.append(matches[0])
if bundle.exists():
    raise SystemExit(f"refusing to overwrite collection bundle: {bundle}")
bundle.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED) as archive:
    for receipt in selected:
        attempt = receipt.parent
        candidate_id = json.loads(receipt.read_text())["candidate_id"]
        archive.write(receipt, Path(candidate_id) / attempt.name / receipt.name)
        for submission in sorted((attempt / "submissions").glob("*.csv")):
            archive.write(
                submission,
                Path(candidate_id) / attempt.name / "submissions" / submission.name,
            )
print(bundle)
'''.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_env() -> dict[str, str]:
    env = dict(os.environ)
    for key in PROXY_KEYS:
        env.pop(key, None)
    return env


def run(argv: list[str], *, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        argv,
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"unsafe collection member: {member.filename}")
    archive.extractall(destination)


def validate_submission(path: Path, sample: pd.DataFrame) -> None:
    frame = pd.read_csv(path)
    if frame.shape != sample.shape:
        raise ValueError(f"submission shape mismatch: {path}: {frame.shape}")
    if list(frame.columns) != list(sample.columns):
        raise ValueError(f"submission columns mismatch: {path}")
    if frame["id"].tolist() != sample["id"].tolist():
        raise ValueError(f"submission IDs/order mismatch: {path}")
    values = frame.iloc[:, 1:].to_numpy(dtype=float)
    if not ((values >= 0.0).all() and (values <= 1.0).all()):
        raise ValueError(f"submission probability range mismatch: {path}")
    if not ((abs(values.sum(axis=1) - 1.0)) <= 1e-4).all():
        raise ValueError(f"submission row sums mismatch: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--namespace", default="ecepxie")
    parser.add_argument(
        "--remote-root",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--remote-bundle",
        default=(
            "/workspace/experiment-end2end-leaf-official-top10-v124/"
            "collection/top10-v123-preferred.zip"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sample-submission-zip",
        type=Path,
        default=Path(
            "coordination/kaggle_leaf_validation_20260810/"
            "sample_submission.csv.zip"
        ),
    )
    parser.add_argument(
        "--score-service",
        type=Path,
        default=Path.home() / ".local/bin/kaggle-score-service",
    )
    args = parser.parse_args()

    remote_roots = args.remote_root or [
        "/workspace/experiment-end2end-leaf-official-top10-v123/reproductions-v1",
        "/workspace/experiment-end2end-leaf-official-top10-v124/reproductions-v1",
    ]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    bundle = output_dir / "top10-v123-preferred.zip"
    extracted = output_dir / "artifacts"
    extracted.mkdir()
    sample_path = output_dir / "sample_submission.csv"

    kube_env = clean_env()
    run(
        [
            "kubectl",
            "exec",
            "-n",
            args.namespace,
            args.pod,
            "--",
            "python",
            "-c",
            REMOTE_PACK,
            args.remote_bundle,
            json.dumps([candidate["candidate_id"] for candidate in CANDIDATES]),
            *remote_roots,
        ],
        env=kube_env,
    )
    run(
        [
            "kubectl",
            "cp",
            "-n",
            args.namespace,
            f"{args.pod}:{args.remote_bundle}",
            str(bundle),
        ],
        env=kube_env,
    )
    with zipfile.ZipFile(bundle) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("collected reproduction bundle is corrupt")
        safe_extract(archive, extracted)
    with zipfile.ZipFile(args.sample_submission_zip.resolve(strict=True)) as archive:
        sample_path.write_bytes(archive.read("sample_submission.csv"))
    sample = pd.read_csv(sample_path)
    if sample.shape != (594, 100):
        raise ValueError(f"official sample shape mismatch: {sample.shape}")

    selected: list[dict[str, object]] = []
    for candidate in CANDIDATES:
        candidate_id = candidate["candidate_id"]
        attempt = extracted / candidate_id / "attempt-000"
        receipt_path = attempt / "REPRODUCTION_RECEIPT.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("status") != "pass":
            raise ValueError(f"candidate did not pass: {candidate_id}")
        if receipt.get("node_id") != candidate["node_id"]:
            raise ValueError(f"candidate node drift: {candidate_id}")
        if receipt.get("source_code_sha256") != candidate["expected_code_sha256"]:
            raise ValueError(f"candidate source drift: {candidate_id}")
        variant = candidate["official_submission_variant"]
        artifact = (receipt.get("variants") or {}).get(variant)
        if not isinstance(artifact, dict):
            raise ValueError(f"preferred variant absent: {candidate_id}:{variant}")
        path = attempt / "submissions" / f"{variant}.csv"
        if sha256_file(path) != artifact.get("canonical_sha256"):
            raise ValueError(f"preferred variant hash drift: {path}")
        validate_submission(path, sample)
        submit_copy = output_dir / "submissions" / f"{candidate_id}__{variant}.csv"
        submit_copy.parent.mkdir(exist_ok=True)
        shutil.copy2(path, submit_copy)
        selected.append(
            {
                "candidate_id": candidate_id,
                "node_id": candidate["node_id"],
                "variant": variant,
                "source_code_sha256": candidate["expected_code_sha256"],
                "submission_path": str(submit_copy),
                "submission_sha256": sha256_file(submit_copy),
                "reproduction_receipt_sha256": sha256_file(receipt_path),
            }
        )

    scores: list[dict[str, object]] = []
    for item in selected:
        message = (
            f"official-v123-top10 {item['candidate_id']} "
            f"variant={item['variant']} node={str(item['node_id'])[-12:]}"
        )
        output = run(
            [
                str(args.score_service.resolve(strict=True)),
                "submit",
                "--competition",
                COMPETITION,
                "--file",
                str(item["submission_path"]),
                "--message",
                message,
                "--expected-rows",
                "594",
                "--sample",
                str(sample_path),
                "--ledger",
                str(output_dir / "KAGGLE_SCORE_SERVICE_LEDGER.json"),
            ]
        )
        score = json.loads(output)
        scores.append({**item, "kaggle": score})

    score_by_candidate = {row["candidate_id"]: row for row in scores}
    ledger: dict[str, object] = {
        "schema": "leaf_official_top10_v123_score_ledger_v1",
        "competition": COMPETITION,
        "top10_slots": [
            {
                "rank": rank,
                "candidate_id": candidate_id,
                "unique_submission_sha256": score_by_candidate[candidate_id][
                    "submission_sha256"
                ],
            }
            for rank, candidate_id in enumerate(TOP10_SLOTS, start=1)
        ],
        "top10_slot_count": len(TOP10_SLOTS),
        "unique_candidate_count": len(scores),
        "reproduction_bundle_sha256": sha256_file(bundle),
        "submissions": scores,
    }
    ledger["ledger_sha256"] = hashlib.sha256(
        json.dumps(ledger, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    ledger_path = output_dir / "OFFICIAL_SCORE_LEDGER.json"
    ledger_path.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"ledger": str(ledger_path), **ledger}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
