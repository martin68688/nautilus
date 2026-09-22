# Search figure revision contract

- Conclusion: the selected later rounds have lower fixed-search loss; the R4/R5 difference is small and not evidence of reproducible benefit.
- Evidence: panel a retains all five trajectories; panel b resolves the R4/R5 neighborhood. Quantitative hero plus subordinate detail, Python/matplotlib only.
- Style-only revision of the existing search-only figure. Original source, figure exports, CSV and summary remain unchanged. No newer experiment is inserted.
- Mapping: round is group, elapsed_minutes is independent wall-clock minutes, search_loss is host-certified fixed search log-loss, certified identifies available host results.
- Integrity: 35 node records, 18 certified measurements retained (2/2/10/1/3); 17 unscored nodes are not numeric scores. No certified values excluded. Each curve ends at its actual recorded stop, including service failures.
- Transform: positive finite losses displayed on a log y axis in panel a; panel b remains linear. No data transformation, smoothing, interpolation, confidence interval, or inferred trial score.
- Design: compact header, sequential one-row legend, aligned score ledger, smaller detail panel and two-line notes. Display-sized 13.2 × 7 inch canvas, PNG 300 dpi plus editable SVG/PDF. Not a journal submission size.
- Statistics: single adaptive search per displayed round, same fixed split and seed 0; no independent repeats or significance claim. Not an official Kaggle score.
- QA: source preflight passed with no failures. Display-width, 300 dpi and PNG-not-TIFF warnings accepted for screen delivery rather than journal submission. Log-guard warning is a static-detector limitation: all values have explicit finite and positive assertions. Visual inspection checked panel alignment, text, score ledger and data markers; R3 annotation raised to avoid point overlap. Source scores unchanged.
