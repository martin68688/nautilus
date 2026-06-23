#!/usr/bin/env bash
# Full Trace2Skill baseline pipeline (faithful 3-stage, creation-from-scratch).
#   Stage 1: render mlevolve journal.json nodes -> markdown trajectory logs
#   Stage 2: run Trace2Skill success + error analysts (their code, pointed at DeepSeek)
#   Stage 3: run Trace2Skill combined skill evolution (their code) over the parsed records
#
# Credentials: reads DEEPSEEK_API_KEY from env (or mlevolve/.env). Never echoes the key.
# Usage:
#   ./run_baseline.sh [run_dir ...]      # defaults to one spooky run (smoke test)
#   MAX_SUCCESS_PER_RUN=8 MAX_FAILED_PER_RUN=8 MAX_WORKERS=6 MODEL=deepseek-chat \
#     ./run_baseline.sh mlevolve/runs/2026*_spooky-author-identification
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
T2S="$REPO/third_party/Trace2Skill"
WS="$HERE/workspace"
mkdir -p "$WS/logs"

# ---- credentials (no echo) ----
if [ -z "${DEEPSEEK_API_KEY:-}" ] && [ -f "$REPO/mlevolve/.env" ]; then
  export DEEPSEEK_API_KEY="$(grep -E '^DEEPSEEK_API_KEY=' "$REPO/mlevolve/.env" | head -1 | cut -d= -f2- | tr -d "\"'")"
fi
: "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY not set (export it or put it in mlevolve/.env)}"
export OPENAI_API_KEY="$DEEPSEEK_API_KEY"
export OPENAI_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
MODEL="${MODEL:-deepseek-chat}"
GCFG="$HERE/deepseek_chat.json"
MAX_WORKERS="${MAX_WORKERS:-6}"
# Python env with openai+tqdm+diskcache (anaconda base on this machine). Override with PYTHON=...
PYTHON="${PYTHON:-/opt/anaconda3/bin/python}"
MAXS="${MAX_SUCCESS_PER_RUN:-8}"
MAXF="${MAX_FAILED_PER_RUN:-8}"

# ---- runs ----
if [ $# -gt 0 ]; then
  RUNS=("$@")
else
  RUNS=("$REPO/mlevolve/runs/20260509_185008_spooky-author-identification")
fi
# expand globs relative to repo
EXPANDED=()
for r in "${RUNS[@]}"; do
  if [[ "$r" != /* ]]; then r="$REPO/$r"; fi
  for g in $r; do EXPANDED+=("$g"); done
done

echo "=================================================================="
echo " Stage 1: render trajectories"
echo "=================================================================="
$PYTHON "$HERE/render_trajectories.py" \
  --runs "${EXPANDED[@]}" --out-dir "$WS/logs" \
  --max-success-per-run "$MAXS" --max-failed-per-run "$MAXF"
echo "rendered: $(ls "$WS/logs"/*.md 2>/dev/null | wc -l | tr -d ' ') logs"

echo "=================================================================="
echo " Stage 2a: success analyst (A+)"
echo "=================================================================="
( cd "$T2S" && $PYTHON analysis/run_success_analysis_llm.py \
    --logs_dir "$WS/logs" --output_dir "$WS/success_analysis" \
    --model "$MODEL" --base_url "$OPENAI_BASE_URL" --api_key "$OPENAI_API_KEY" \
    --generation_config "$GCFG" --max_workers "$MAX_WORKERS" )

echo "=================================================================="
echo " Stage 2b: error analyst (A-)"
echo "=================================================================="
( cd "$T2S" && $PYTHON analysis/run_error_analysis_llm.py \
    --logs_dir "$WS/logs" --output_dir "$WS/error_analysis" \
    --model "$MODEL" --base_url "$OPENAI_BASE_URL" --api_key "$OPENAI_API_KEY" \
    --generation_config "$GCFG" --max_workers "$MAX_WORKERS" )

echo "=================================================================="
echo " Stage 3: combined skill evolution (M merge operator)"
echo "=================================================================="
rm -rf "$WS/evolved_skill"; cp -r "$HERE/s0_skill" "$WS/evolved_skill"
# provide the skill-format checker the combined runner hard-requires (not vendored upstream)
mkdir -p "$T2S/skills/skill-creator/scripts"
cp -f "$HERE/quick_validate_stub.py" "$T2S/skills/skill-creator/scripts/quick_validate.py"
# P0 fix: --max-tokens was unset -> DeepSeek defaulted to 4096 -> large MERGE
#   outputs (avg 22KB) were truncated mid-JSON -> REDUCE stalled at 11 patches.
#   Default 8192 = DeepSeek-chat hard cap (env MAX_TOKENS overrides; GLM-5.2
#   supports up to 131072 — run_baseline_glm.sh sets 16384). --merge-batch-size 3
#   keeps each merge output under it; --max-merge-levels 10 gives headroom;
#   --temperature 0.2 / --max-workers 3 stabilize merges & ease rate limits;
#   --skip-translation drops one failure surface (json pipeline honors it).
( cd "$T2S" && $PYTHON -m skill_evolver.run_parallel_combined_skill_evolution \
    --error-json   "$WS/error_analysis/parsed_error_records.json" \
    --success-json "$WS/success_analysis/parsed_success_records.json" \
    --skill-dir    "$WS/evolved_skill" \
    --model "$MODEL" --base-url "$OPENAI_BASE_URL" --api-key "$OPENAI_API_KEY" \
    --generation-config "$GCFG" --patch-pipeline json \
    --max-tokens "${MAX_TOKENS:-8192}" \
    --batch-size 1 --merge-batch-size 3 --max-merge-levels 10 \
    --temperature 0.2 --max-workers 3 \
    --skip-translation \
    --save-intermediates --output-dir "$WS/evolved_out" --verbose || \
  echo "[evolver exited non-zero — check QUICK_VALIDATE_SCRIPT / parse_failures]" )

echo "=================================================================="
echo " DONE -> evolved skill: $WS/evolved_skill"
echo " evaluate: python3 $HERE/evaluate_recall.py --skill-dir $WS/evolved_skill"
echo "=================================================================="
