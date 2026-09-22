# Search-only figure contract

Core conclusion: the five selected runs show lower host search losses over rounds, with the latest all-memory run only 0.19% below historical R4. This is descriptive, not causal or a stability claim.

Python/matplotlib, quantitative grid with one wide elapsed-time hero and one narrow R4/R5 zoom. Structurally adapted from the immediately preceding five-run figure. Report-size 16 by 7.8 inches; editable SVG/PDF and 300 dpi PNG. No journal submission claim.

Mapping: round -> group; elapsed_minutes -> independent time since run launch; search_loss -> host-certified fixed search log-loss; certified -> score eligibility. Read the same validated outer_search.csv and summary.json without edits. All 35 candidate records retained in source data; all 18 certified observations plotted (2/2/10/1/3); 17 unscored candidates have no numeric loss and are not fabricated as zero. Their wall time remains included. No tune data included or substituted. No resampling, smoothing, interpolation or new evaluation. Staircase is cumulative minimum after the first scored candidate, retained only until actual run end. Each point is an adaptive candidate observation, not an independent replicate. Same split, one seed per run, no confidence interval or significance test. Latest is labeled R5 only within the selected comparison, not the fifth chronological attempt. R4/R5 stop early due to model-service failure.

Source CSV/summary and underlying evidence hashes are provided with the preceding five-run figure. Original artifacts preserved. Markers and line patterns complement colors. R2's former tune-only display extension does not apply to this search-only plot.

QA: all 18 certified values and five endpoints reconcile to source summary. Static preflight: 11 PASS, 3 WARN, 0 FAIL. Accepted warnings concern report canvas width, 300 dpi preview, and absence of an unrequested TIFF submission raster. Final PNG visually inspected with readable labels, no clipping or overlap. SVG text and PDF fonttype 42 remain editable.
