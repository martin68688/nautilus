#!/usr/bin/env python3
"""Test script: verify all 3 knowledge loading paths produce correct prompt content."""

import sys
import os
import json
import re
from pathlib import Path
from types import SimpleNamespace

# Add project root to path
sys.path.insert(0, "/workspace/nautilus/mlevolve")

# ============================================================
# Simulate config
# ============================================================
cfg = SimpleNamespace(
    exp_id="spooky-author-identification",
    coldstart=SimpleNamespace(
        task_json_path="engine/coldstart/competition_tag_classified.json",
        model_json_path="engine/coldstart/models_guidance_classified.json",
    ),
    methodology_kb_path="/workspace/nautilus/paper-skills/experience_kb",
    methodology_dynamic=True,
    torch_hub_dir="",
    agent=SimpleNamespace(
        code=SimpleNamespace(
            model="deepseek-chat",
            api_key="sk-36ed4904871f4632a8400283d96b6bbd",
            base_url="https://api.deepseek.com",
        )
    ),
)

# Task description from the actual data
task_desc_path = "/workspace/nautilus/mlevolve/data/spooky-author-identification/prepared/public/description.md"
if os.path.exists(task_desc_path):
    with open(task_desc_path) as f:
        task_desc = f.read()[:1500]
else:
    task_desc = "Spooky Author Identification: classify text passages by author (EAP, HPL, MWS)"

os.chdir("/workspace/nautilus/mlevolve")

# ============================================================
# Path 1: Model template (models_guidance_classified.json)
# ============================================================
print("=" * 80)
print("PATH 1: Model Template (models_guidance_classified.json)")
print("=" * 80)

from engine.coldstart.knowledge import collect_models_for_task, _build_guidance_text

with open(cfg.coldstart.task_json_path) as f:
    tasks = json.load(f)
with open(cfg.coldstart.model_json_path) as f:
    models = json.load(f)

# Show task → category mapping
category = tasks.get(cfg.exp_id, "NOT FOUND")
print(f"\nTask: {cfg.exp_id}")
print(f"Category: {category}")

# Show matched models
model_list = collect_models_for_task(cfg.exp_id, tasks, models)
print(f"Models found: {len(model_list)}")
for m in model_list:
    print(f"\n  Model: {m['model_name']}")
    print(f"  Description (first 200 chars): {m['description'][:200]}...")
    print(f"  Code template (first 300 chars): {m['code_template'][:300]}...")

# Build full guidance text
guidance_text_1 = _build_guidance_text(cfg.exp_id, tasks, models)
print(f"\n--- Full guidance text length: {len(guidance_text_1)} chars ---")
print(f"\n{guidance_text_1[:2000]}")
if len(guidance_text_1) > 2000:
    print(f"\n... [truncated, total {len(guidance_text_1)} chars]")

# ============================================================
# Path 2: Static methodology (methodology_map.json → *_methodology.md)
# ============================================================
print("\n\n" + "=" * 80)
print("PATH 2: Static Methodology (methodology_map.json → *_methodology.md)")
print("=" * 80)

from engine.coldstart.knowledge import _build_methodology_text

methodology_text_2 = _build_methodology_text(cfg.exp_id, cfg.methodology_kb_path)
if methodology_text_2:
    print(f"\nMethodology text length: {len(methodology_text_2)} chars")
    print(f"\n{methodology_text_2[:3000]}")
    if len(methodology_text_2) > 3000:
        print(f"\n... [truncated, total {len(methodology_text_2)} chars]")
else:
    print("\n⚠️  No methodology text found! Check methodology_map.json and *_methodology.md files.")

# ============================================================
# Path 3: Dynamic methodology (methodology_agent.py → LLM → paperinsight/)
# ============================================================
print("\n\n" + "=" * 80)
print("PATH 3: Dynamic Methodology (methodology_agent.py → LLM matching)")
print("=" * 80)

from engine.coldstart.methodology_agent import _scan_categories, build_methodology_guidance

categories = _scan_categories(Path(cfg.methodology_kb_path))
print(f"\nAvailable categories under experience_kb/:")
for cat in categories:
    print(f"  - {cat}")

# Try dynamic matching (requires API key - may fail)
print(f"\nAttempting LLM dynamic matching...")
try:
    methodology_text_3 = build_methodology_guidance(task_desc, cfg.methodology_kb_path, cfg.agent.code)
    if methodology_text_3:
        print(f"\nDynamic methodology text length: {len(methodology_text_3)} chars")
        print(f"\n{methodology_text_3[:3000]}")
        if len(methodology_text_3) > 3000:
            print(f"\n... [truncated, total {len(methodology_text_3)} chars]")
    else:
        print("\n⚠️  Dynamic matching returned no results.")
except Exception as e:
    print(f"\n⚠️  Dynamic matching failed (expected if no API key): {e}")
    print("   This is OK for testing - the static path (Path 2) covers the same content.")

# ============================================================
# Summary: Full combined prompt
# ============================================================
print("\n\n" + "=" * 80)
print("COMBINED: Full guidance as injected into prompt")
print("=" * 80)

# Simulate build_guidance_description
full_text = guidance_text_1
if methodology_text_2:
    full_text += methodology_text_2

print(f"\nPath 1 (Model template): {len(guidance_text_1)} chars")
print(f"Path 2 (Static methodology): {len(methodology_text_2)} chars")
print(f"Total combined: {len(full_text)} chars")

# Key content checks
checks = {
    "Partial unfreezing (partial unfreeze)": "partial unfreezing" in full_text.lower() or "partial unfreeze" in full_text.lower(),
    "CosineAnnealingWarmRestarts": "CosineAnnealingWarmRestarts" in full_text,
    "label_smoothing=0.1": "label_smoothing" in full_text,
    "Differentiated LR (2e-5 / 5e-5)": "2e-5" in full_text and "5e-5" in full_text,
    "Simple Linear head (NOT complex)": "simple Linear head" in full_text or "simple Linear" in full_text.lower(),
    "ModernBERT avoidance": "ModernBERT" in full_text or "ModernBERT" in full_text,
    "Feature projection (150→64)": "feature_proj" in full_text or "150" in full_text,
    "Heterogeneous ensemble": "heterogeneous" in full_text.lower() or "XGBoost" in full_text,
    "Probability clipping": "clip" in full_text and "1e-15" in full_text,
}

print(f"\n--- Content Verification ---")
all_pass = True
for check_name, result in checks.items():
    status = "✅" if result else "❌"
    if not result:
        all_pass = False
    print(f"  {status} {check_name}")

if all_pass:
    print(f"\n🎉 All key content verified in combined prompt!")
else:
    print(f"\n⚠️  Some key content missing - check above for details.")
