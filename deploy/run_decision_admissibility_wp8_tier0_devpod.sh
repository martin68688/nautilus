#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

SRC="${WP8_TIER0_SOURCE_ROOT:-/work/decision-admissibility-wp8-tier0-r1-source}"
OUTPUT_ROOT="${WP8_TIER0_OUTPUT_ROOT:-/workspace/decision-admissibility-wp8-tier0-r1}"
CREATED_AT="${WP8_TIER0_CREATED_AT:-2026-07-21T09:00:00Z}"
EXPECTED_SOURCE_SHA256="${WP8_TIER0_SOURCE_SHA256:-}"
STAGING="${OUTPUT_ROOT}.staging-$(date +%s)-$RANDOM"

test -n "$EXPECTED_SOURCE_SHA256"
test ! -e "$OUTPUT_ROOT"
test ! -e "$STAGING"
test -s "$SRC/WP7_SOURCE_MANIFEST.json"
test -s "$SRC/paper-skills/memory_bundle/run_decision_admissibility_factorial.py"
test -s "$SRC/tests/test_decision_admissibility_factorial.py"
test -d "$SRC/mlevolve/config/protocols"

mkdir -p "$STAGING"
printf 'building\n' > "$STAGING/STATE"
export PYTHONPATH="$SRC/mlevolve:$SRC/paper-skills/memory_bundle"

python - "$SRC" "$EXPECTED_SOURCE_SHA256" <<'PY'
import sys
from pathlib import Path

from publish_certified_replay_bundle import _verify_source_snapshot

root = Path(sys.argv[1]).resolve()
expected = sys.argv[2]
snapshot = _verify_source_snapshot(
    root / "WP7_SOURCE_MANIFEST.json",
    expected_root=root,
)
if snapshot["source_sha256"] != expected:
    raise ValueError("Tier-0 source snapshot pin mismatch")
print(snapshot["source_sha256"])
PY

python -m pytest -q -p no:cacheprovider \
  --basetemp /work/decision-admissibility-wp8-tier0-r1-pytest \
  "$SRC/tests/test_decision_admissibility_factorial.py" \
  > "$STAGING/pytest.log" 2>&1

python "$SRC/paper-skills/memory_bundle/run_decision_admissibility_factorial.py" \
  --protocol-registry "$SRC/mlevolve/config/protocols" \
  --created-at "$CREATED_AT" \
  --output "$STAGING/tier0_factorial_report.json" \
  > "$STAGING/runner_stdout.log" 2>&1

python "$SRC/paper-skills/memory_bundle/verify_decision_admissibility_factorial.py" \
  --report "$STAGING/tier0_factorial_report.json" \
  --source-root "$SRC" \
  --output "$STAGING/tier0_factorial_verification.json" \
  > "$STAGING/verifier_stdout.log" 2>&1

cp "$SRC/WP7_SOURCE_MANIFEST.json" "$STAGING/source_snapshot_manifest.json"

python - "$SRC" "$STAGING" "$EXPECTED_SOURCE_SHA256" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
staging = Path(sys.argv[2]).resolve()
expected_source_sha256 = sys.argv[3]
report = json.loads((staging / "tier0_factorial_report.json").read_text())
verification = json.loads(
    (staging / "tier0_factorial_verification.json").read_text()
)
snapshot = json.loads((staging / "source_snapshot_manifest.json").read_text())

assert report["schema"] == "decision_admissibility_tier0_factorial_v1"
assert report["case_count"] == report["expected_case_count"] == 63
assert report["matrix_complete"] is True
assert report["all_cases_passed"] is True
assert report["failed_case_ids"] == []
assert report["invalid_attack_episode_count"] == 42
assert report["invalid_activation_count"] == 0
assert report["invalid_influence_rate"] == 0.0
assert report["valid_knowledge_opportunity_count"] == 57
assert report["valid_knowledge_retained_count"] == 57
assert report["valid_knowledge_retention"] == 1.0
assert report["unauthorized_prompt_exposure_count"] == 0
assert verification["valid"] is True
assert verification["errors"] == []
assert verification["exact_replay_match"] is True
assert verification["report_hash"] == report["report_hash"]
assert snapshot["source_sha256"] == expected_source_sha256

unsigned = {key: value for key, value in report.items() if key != "report_hash"}
encoded = json.dumps(
    unsigned,
    sort_keys=True,
    ensure_ascii=False,
    separators=(",", ":"),
).encode("utf-8")
assert report["report_hash"] == hashlib.sha256(encoded).hexdigest()

for relative, digest in report["implementation_source_hashes"].items():
    path = source / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert snapshot["file_hashes"][relative] == digest

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

summary = {
    "schema": "decision_admissibility_wp8_tier0_summary_v1",
    "status": "passed",
    "source_snapshot_sha256": expected_source_sha256,
    "source_snapshot_manifest_file_sha256": sha256(
        staging / "source_snapshot_manifest.json"
    ),
    "tier0_report_hash": report["report_hash"],
    "tier0_report_file_sha256": sha256(staging / "tier0_factorial_report.json"),
    "tier0_verification_hash": verification["verification_hash"],
    "tier0_verification_file_sha256": sha256(
        staging / "tier0_factorial_verification.json"
    ),
    "runner_source_sha256": report["runner_source_sha256"],
    "case_count": report["case_count"],
    "invalid_attack_episode_count": report["invalid_attack_episode_count"],
    "invalid_activation_count": report["invalid_activation_count"],
    "invalid_influence_rate": report["invalid_influence_rate"],
    "valid_knowledge_opportunity_count": report[
        "valid_knowledge_opportunity_count"
    ],
    "valid_knowledge_retained_count": report["valid_knowledge_retained_count"],
    "valid_knowledge_retention": report["valid_knowledge_retention"],
    "unauthorized_prompt_exposure_count": report[
        "unauthorized_prompt_exposure_count"
    ],
    "summary_hash": "",
}
summary["summary_hash"] = hashlib.sha256(
    json.dumps(
        {key: value for key, value in summary.items() if key != "summary_hash"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
(staging / "TIER0_SUMMARY.json").write_text(
    json.dumps(summary, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, sort_keys=True))
PY

printf 'complete\n' > "$STAGING/STATE"
find "$STAGING" -type f ! -name FINAL_SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$STAGING/FINAL_SHA256SUMS"
chmod -R a-w "$STAGING"
test -z "$(find "$STAGING" -perm /222 -print -quit)"
mv "$STAGING" "$OUTPUT_ROOT"
printf 'tier0_complete=%s\n' "$OUTPUT_ROOT"
