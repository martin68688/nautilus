# Resumed search chart QA

User-requested display revision: R1 best-so-far is carried forward as a solid line from its true 93.7387-minute end to 120 minutes. True-end tick remains; no extra score markers or data rows. Footer explicitly identifies the display-only extension. All other curves unchanged. The earlier no-extension statement below describes the initial version; R5/R6 still have no artificial extension.

Claim: R5 and R6 best-so-far scores remain unchanged through actual continuation end.
Python structural adaptation: round maps to color/marker, elapsed_minutes maps to x, positive fixed search_loss maps to y; one fixed split and seed, no uncertainty or causal significance claims.
All 22 earlier measured points retained plus one newly host-verified R5 point; inherited node IDs deduplicated. R4 excluded at user's request. Unscored records remain in source CSV but do not appear as zero scores. The detail panel crops to .033–.054; the new .102981 score remains visible in the hero panel. Journal does not seal the last unfinished generated candidate as a completed measurement.
Actual runtime adds prior elapsed seconds to watchdog process duration, excludes offline intervals. R5 119.841191826 minutes, forced budget stop; R6 118.035403314 minutes, orderly budget admission stop. No artificial extension to 120 minutes. Empty circles are phase annotations, not measurements.
Export is a desktop comparison figure, not a specific journal submission: 13.2 × 7.3 inches, 300 dpi PNG and editable SVG/PDF. Width/DPI warnings are intentional. Log-axis positivity and finite guards execute explicitly for every plotted score; validator may not recognize generator-expression guards.
Visual inspection: no clipped text or overlapping endpoint labels; old layout and palette retained; model/hardware switch and incomplete R5 finalization stated in footer.
