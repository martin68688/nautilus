#!/usr/bin/env bash
set -euo pipefail

root="${1:-/haoming/mlevolve-original-runs/be034ec/3h120}"
test -d "$root"

size_or_missing() {
  local path="$1"
  if [[ -f "$path" ]]; then
    stat -c '%s' "$path"
  else
    printf '%s' -1
  fi
}

printf 'run\ttask\tseed\tjournal\tfiltered_journal\tconfig\tbest_solution\n'
for run_dir in "$root"/20??????_??????_*; do
  [[ -d "$run_dir" ]] || continue
  run="${run_dir##*/}"
  task="${run#*_}"
  task="${task#*_}"
  logs="$run_dir/logs"
  seed='?'
  if [[ -f "$logs/config.yaml" ]]; then
    seed="$(awk '$1 == "seed:" {print $2; exit}' "$logs/config.yaml")"
    seed="${seed:-?}"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$run" \
    "$task" \
    "$seed" \
    "$(size_or_missing "$logs/journal.json")" \
    "$(size_or_missing "$logs/filtered_journal.json")" \
    "$(size_or_missing "$logs/config.yaml")" \
    "$(size_or_missing "$logs/best_solution.py")"
done
