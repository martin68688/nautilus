#!/usr/bin/env bash
set -uo pipefail

ROOT=/workspace/experiment-end2end-leaf-official-top10-v123
RUNNER="$ROOT/bin-v1/run_top10_candidate_v123.py"
OUTPUT_ROOT="$ROOT/reproductions-v1"
CONTROLLER_LOG="$ROOT/CONTROLLER_EVENTS.log"
overall=0

for index in 0 1 2 3 4 5 6; do
  printf 'CONTROLLER_CANDIDATE_START index=%s time=%s\n' \
    "$index" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$CONTROLLER_LOG"
  if /usr/local/bin/python -u "$RUNNER" \
      --index "$index" \
      --attempt attempt-000 \
      --output-root "$OUTPUT_ROOT" \
      --timeout-seconds 21600; then
    status=pass
  else
    status=failed
    overall=1
  fi
  printf 'CONTROLLER_CANDIDATE_END index=%s status=%s time=%s\n' \
    "$index" "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    | tee -a "$CONTROLLER_LOG"
done

/usr/local/bin/python - "$OUTPUT_ROOT" "$ROOT/RUN_OUTCOME.json" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
output = Path(sys.argv[2])
receipts = []
for path in sorted(root.glob("*/attempt-000/REPRODUCTION_RECEIPT.json")):
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipts.append(
        {
            "candidate_id": receipt.get("candidate_id"),
            "node_id": receipt.get("node_id"),
            "source_code_sha256": receipt.get("source_code_sha256"),
            "status": receipt.get("status"),
            "error": receipt.get("error"),
            "runtime_seconds": receipt.get("runtime_seconds"),
            "variants": sorted((receipt.get("variants") or {}).keys()),
            "receipt_path": str(path),
            "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
payload = {
    "schema": "leaf_official_top10_v123_run_outcome_v1",
    "top10_slot_count": 10,
    "unique_candidate_count": 7,
    "completed_receipt_count": len(receipts),
    "status": (
        "pass"
        if len(receipts) == 7 and all(row["status"] == "pass" for row in receipts)
        else "failed"
    ),
    "candidates": receipts,
}
payload["outcome_sha256"] = hashlib.sha256(
    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
output.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, sort_keys=True))
PY

exit "$overall"
