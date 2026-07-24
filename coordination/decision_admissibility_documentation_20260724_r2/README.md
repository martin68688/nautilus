# Decision Admissibility final documentation (2026-07-24)

This versioned, append-only package closes the documentation items in §25.3 of
`coordination/decision_admissibility_complete_execution_plan_20260719.md`.
It does not replace or mutate any earlier WP report, Bundle, test receipt, or
formal result.

This corrected r2 package supersedes r1 only as the closeout documentation
authority. The sealed r1 package is retained as append-only failure evidence:
its ordinary file hashes were correct, but its self-hash used JavaScript number
serialization rather than the repository's Python canonical-JSON contract and
its presentation inventory undercounted Chinese alt-text objects. R2 uses the
canonical Python payload hash and binds the mechanically verified count of 117.

## Contents

- `implementation_report.md` — consolidated WP0–WP8 implementation and
  evidence boundary.
- `schema_api_reference.md` — canonical Stage, Claim, Operation, Protocol,
  Receipt, visibility, writeback, replay, and Bundle APIs.
- `migration_guide.md` — fail-closed migration from legacy whole-item
  validity / `PROMOTE` semantics to Decision Admissibility.
- `build_final_presentation.js` — reproducible bilingual presentation builder.
  English remains native editable PowerPoint text; Chinese is rendered with the
  system Hiragino Sans GB font into transparent high-resolution layers and is
  also stored as image alt text, so the verified LibreOffice path does not lose
  CJK glyphs.
- `manifest.json` and the sibling verification file — added at closeout after
  all referenced presentation and paper artifacts are frozen.

## Versioned final deliverables

- Bilingual PowerPoint (9 slides):
  `outputs/nautilus_decision_admissibility_wp8_final_bilingual_20260724_r3.pptx`
- Updated LaTeX source:
  `papers/runforest_iclr2025/main_wp8_final_20260724.tex`
- Updated paper PDF (21 pages):
  `papers/runforest_iclr2025/main_wp8_final_20260724.pdf`

Earlier PowerPoint and paper files remain untouched. The bilingual deck keeps
the frozen scientific boundary explicit in both languages: Full completion is
4/9 versus No Memory 6/9; all four estimable pairs are positive; raw
`p=0.0625`, Holm `p=0.25`; Taxi contributes zero pairs; Full superiority is
`rejected`; conditional utility is `diagnostic only`; and experience causality
is `pending_without_L4`.

## Reproduction anchors

Final engineering Gate:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 \
PYTHONPATH='mlevolve:paper-skills/memory_bundle' \
PYTEST_ADDOPTS='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
.venv/bin/python -B paper-skills/memory_bundle/verify_wp8_final_stop_gate.py \
  --stop-gate-root coordination/decision_admissibility_wp8_final_stop_gate_20260724_r1 \
  --repo-root . \
  --final-regression-receipt coordination/decision_admissibility_wp8_final_regression_20260724_r1/final_regression_receipt.json \
  --final-test-root coordination/decision_admissibility_wp8_final_tests_20260724_r4 \
  --host-test-receipt-root coordination/decision_admissibility_wp8_final_tests_20260724_r4
```

Expected: `verified=true`, 20 prerequisites, 6 kill gates, 47 acceptance
checks, and verification hash
`dcf2e62c47d9f22c03a7adb05d325f588367284aee8f7309d2d0831116718778`.

Versioned paper build:

```bash
cd papers/runforest_iclr2025
/opt/homebrew/bin/tectonic main_wp8_final_20260724.tex \
  --outdir . --keep-logs --keep-intermediates
```

Bilingual presentation build (write to a fresh path; do not overwrite the
frozen r3 deliverable):

```bash
NODE=/Users/haoming/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node
NODE_PATH=/Users/haoming/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules \
  "$NODE" coordination/decision_admissibility_documentation_20260724_r2/build_final_presentation.js \
  /tmp/nautilus_decision_admissibility_wp8_final_bilingual_rebuild.pptx
```

Verified rendering path:

```bash
mkdir -p /tmp/nautilus_bilingual_render
/Users/haoming/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/soffice \
  --headless --convert-to pdf --outdir /tmp/nautilus_bilingual_render \
  /tmp/nautilus_decision_admissibility_wp8_final_bilingual_rebuild.pptx
/Users/haoming/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm \
  -png -r 120 \
  /tmp/nautilus_bilingual_render/nautilus_decision_admissibility_wp8_final_bilingual_rebuild.pdf \
  /tmp/nautilus_bilingual_render/page
```

The frozen r3 presentation was rendered to a 9-page 16:9 PDF and every page
was visually inspected. Adding Chinese accessibility text did not change any
rendered page bytes relative to the visually approved r2 render.

The formal scientific boundary is fixed: WP8 engineering is complete; Full
superiority is rejected; conditional utility is diagnostic only; experience
causality is pending without formal L4 evidence.
