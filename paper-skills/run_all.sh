#!/bin/bash
set -e

cd "$(dirname "$0")"

PYTHON=/opt/anaconda3/bin/python
VENUE=${1:-neurips}
YEAR=${2:-2024}

echo "=== Step 1: Fetch ${VENUE} ${YEAR} papers ==="
$PYTHON scripts/1_fetch.py --venue $VENUE --year $YEAR

echo "=== Step 2: Embed + cluster ==="
$PYTHON scripts/2_embed_cluster.py --venue $VENUE --year $YEAR

echo "=== Step 3: Classify papers with LLM ==="
$PYTHON scripts/3_classify.py --venue $VENUE --year $YEAR

echo "=== Step 4: Generate skills ==="
$PYTHON scripts/4_generate_skills.py --venue $VENUE --year $YEAR

echo "=== Done: output/${VENUE}-${YEAR}/ ==="
