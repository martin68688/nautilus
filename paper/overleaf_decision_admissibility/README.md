# Decision Admissibility ICLR 2025 Paper Draft

This directory is an Overleaf-ready project built with the unmodified official
ICLR 2025 conference style.

## Build

```bash
../../.venv/bin/python make_figures.py
python3 make_dynamic_hybrid_figure.py
python3 evidence/verify_claims.py
tectonic main.tex
```

The paper is anonymous by default. Do not uncomment `\iclrfinalcopy` until the
camera-ready stage.

The current build uses Tectonic 0.15.0 and produces a 28-page letter-size PDF;
the main paper plus references occupies pages 1--21 and appendices begin on page
22. `main.tex` is the canonical Overleaf entry point and includes the latest
audited source, `main_wp8_final_20260724.tex`. See `evidence/build_report.md`
for hashes and verification details.

## Evidence policy

- `evidence/claims.md` is the claim ledger.
- `make_figures.py` reads the repository JSON reports directly.
- Leakage-blocked, protocol-biased, retrospective diagnostic, and pending
  results are labeled separately from downstream evidence.
- Historical values `0.2013` and `0.0494` are not valid paper results.
- The causal benchmark is retrospective, uses semantically aligned silver
  failure labels, and does not execute newly generated repairs.
- WP1--WP6 implementation and deterministic stop gates are represented
  separately from the still-pending WP7 online and WP8 causal/downstream gates.
- Current-node success, historical adoption, and historical causal influence
  are three separate writeback objects: `PROMOTE_RESULT` publishes a Result
  Fact from the node's own legal evidence; `PUBLISH_ADOPTION` requires
  contract-bound static/runtime actuation; `PUBLISH_CAUSAL` additionally
  requires a counterfactual pair.
- The sealed fixed-holdout scorer owns the only terminal Result Fact finalizer;
  it is idempotent and reports incomplete writeback explicitly.
- Positive Result SOPs and Positive Adopted SOPs have separate authority paths;
  missing historical actuation does not delete an independently valid result.
- The non-Spooky Full/seed-heldout/task-heldout Bundles are systems artifacts,
  not evidence of task generalization or final MLE improvement.
- Experiment A is a prospective gate-activity audit: 783/4,106 raw candidates
  are automatically suppressed over 171 natural decision opportunities. This
  19.07% value is not relabeled as Oracle-adjudicated invalid prevalence.
- The deprecated fixed-weight Tree variant is excluded from the production
  method and primary comparison but disclosed in an appendix ablation so its
  ranking-versus-fallback trade-off remains auditable.
- The current paper-focused Authority, routing, snapshot, publication,
  fixed-holdout, positive-distillation, and rollback regression suite reports
  257 passed tests.
- The latest broad WP7 local preflight reports 451 passed with the independently
  frozen composite benchmark excluded; that module remains 18 passed and one
  pre-existing detector-hash lock mismatch.

## Overleaf upload

Upload the complete source contents of this directory, including
`main_wp8_final_20260724.tex`, the official `.sty`, `.bst`, `natbib.sty`,
`fancyhdr.sty`, `math_commands.tex`, and `figures/`. Keep `main.tex` selected as
Overleaf's Main document.
