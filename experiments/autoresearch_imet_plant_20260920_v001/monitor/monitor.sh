#!/bin/bash
set -u
TASK="$1"; ROOT="$2"; STATE="$3"
mkdir -p "$(dirname "$STATE")"
while true; do
  NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  UNITS=$(systemctl list-units --all --no-legend "autoresearch-${TASK}-r*-20260920*" 2>/dev/null | awk '{print $1":"$3}' | tr '\n' ' ')
  RUNS=$(find "$ROOT/runs" -maxdepth 2 -name RESULT.json -printf '%p\n' 2>/dev/null | sort | tail -20 | tr '\n' '|')
  GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | tr '\n' ';')
  DISK=$(df -P "$ROOT" | tail -1 | awk '{print $4" KB free"}')
  printf '{"timestamp":"%s","task":"%s","units":"%s","recent_results":"%s","gpu":"%s","disk":"%s"}\n' "$NOW" "$TASK" "$UNITS" "$RUNS" "$GPU" "$DISK" > "$STATE.tmp"
  mv "$STATE.tmp" "$STATE"
  sleep 60
done
