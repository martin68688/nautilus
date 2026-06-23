#!/usr/bin/env bash
# Run the Trace2Skill baseline on GLM-5.2 via a local OpenAI->Anthropic proxy.
#
# third_party/Trace2Skill is left byte-unchanged (it's the faithful baseline);
# glm_proxy.py translates its OpenAI Chat-Completions calls to GLM's Anthropic
# endpoint (Zhipu Coding Plan). This wrapper starts the proxy, points the
# baseline's OPENAI_BASE_URL at it, then execs run_baseline.sh with the same args.
#
# Usage:  ./run_baseline_glm.sh   [same args as run_baseline.sh]
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-/opt/anaconda3/bin/python}"
PORT="${GLM_PROXY_PORT:-18211}"

# The evolver's OpenAI client (httpx) honors HTTP_PROXY/HTTPS_PROXY. If a local
# proxy (Clash/V2Ray/etc.) is set, httpx would route the localhost request to it
# and fail with 502. Force localhost to be reached directly.
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"

# Start the proxy if it isn't already answering.
if ! curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "[glm] starting OpenAI->Anthropic proxy on port $PORT ..."
  "$PY" "$HERE/glm_proxy.py" >/tmp/glm_proxy.log 2>&1 &
  PROXY_PID=$!
  echo "[glm] proxy pid $PROXY_PID (log: /tmp/glm_proxy.log)"
  for _ in $(seq 1 20); do
    curl -s --max-time 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    sleep 0.5
  done
else
  PROXY_PID=""
  echo "[glm] proxy already up on port $PORT"
fi

# Trace2Skill reads OPENAI_* (analysts) and run_baseline.sh reads DEEPSEEK_*
# (which it re-exports as OPENAI_*). Point both at the proxy. The api_key value
# is irrelevant — the proxy uses GLM_API_KEY from paper-skills/.env.
export OPENAI_BASE_URL="http://127.0.0.1:$PORT/v1"
export OPENAI_API_KEY="glm-proxy"
export DEEPSEEK_API_KEY="glm-proxy"
export DEEPSEEK_BASE_URL="$OPENAI_BASE_URL"
export MODEL="${MODEL:-glm-5.2}"
# Cap analyst parallelism at 3 (the evolver is already --max-workers 3 in
# run_baseline.sh). The Coding Plan rate-limits; 3 concurrent long-output calls
# stays under the limit, 6-8 triggers 429 -> retry storms. Override: MAX_WORKERS=N.
export MAX_WORKERS="${MAX_WORKERS:-3}"

# GLM-5.2 supports up to 131072 output tokens (DeepSeek caps at 8192). 16384
# gives ~2.5x headroom over the ~6k-token merge outputs, 8x under the cap.
export MAX_TOKENS="${MAX_TOKENS:-16384}"

echo "[glm] MODEL=$MODEL  OPENAI_BASE_URL=$OPENAI_BASE_URL"
exec "$HERE/run_baseline.sh" "$@"
