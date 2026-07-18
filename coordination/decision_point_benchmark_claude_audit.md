# ClaudeAgent Independent Audit And Disposition

> Historical note: this audit reviewed the pre-v2 benchmark, where the method
> labelled `stage_hybrid_sop` was only the legacy field-aware gateway. The true
> production Stage Hybrid v2 implementation and its independent review are
> recorded in `stage_hybrid_v2_claude_audit.md`. The old 0.3543 result below is
> retained for provenance and must not be attributed to the production v2
> method.

## First Review Verdict

ClaudeAgent judged the original 60-query benchmark diagnostic-only and unsuitable
for a paper main table. It confirmed that the benchmark removed historical
parent-to-child recovery gold, but identified four blocking fairness defects:

1. `taxonomy_tfidf` received clean-evidence and abstraction-level filters that
   unfiltered controls did not receive.
2. Several different intents reused the same unordered gold set.
3. Some gold SOPs were not explicitly compatible with the query task family.
4. Every query reused the same globally selected blocked RunNodes.

It also flagged lexical candidate-pool construction, divergent safety predicates,
Tree task-identity advantage, environment-fragile MiniLM loading, underpowered
strata, and an approximate bootstrap p-value.

## Disposition

| Finding | Resolution |
|---|---|
| Privileged taxonomy filter | Removed. TF-IDF, BM25, LSA, and MiniLM now each have symmetric unfiltered and safety-filtered variants; no scorer receives target abstraction level. |
| Duplicate intent gold | Strict builder drops repeated unordered gold sets globally, including cross-family collisions. |
| Cross-task gold | Strict builder and validator require the explicit query task-family tag; `general` is not a wildcard. |
| Global blocked distractors | Each query receives five deterministic blocked RunNodes from its own task. |
| Lexical pool bias | Removed. Full 281-SOP inventory is used rather than lexical preselection. |
| Safety predicate divergence | Builder now requires the union of runtime positive-memory, execution validity, ranking, and numeric-metric conditions. Oracle confirms all retained gold is admissible. |
| Tree task bonus | Both no-task and task-aware Tree mappings are reported. |
| MiniLM reproducibility | Added an isolated benchmark requirements file with `numpy<2`; report records Python, NumPy, platform, model, and availability. |
| Approximate bootstrap p-value | Bootstrap is retained only for the 95% CI; p-values now use paired random sign flips with Holm correction. |
| Underpowered strata | Query count reduced honestly from 60 to 29; the uneven `10/5/5/3/3/3` retention and convenience-seed selection are disclosed; family/stage summaries are descriptive only. |

## Second Review And Final Corrections

ClaudeAgent's second review confirmed the symmetric scorer pairs, task-matched
blocked distractors, safety predicates, nDCG/AP implementations, paired
bootstrap intervals, random sign-flip p-values, and Holm correction. It kept
the verdict at appendix-only and identified four remaining concrete issues:

1. two cross-family queries still shared complete gold sets;
2. the production pool had 286 candidates while one test used 80;
3. Stage Hybrid was not compared directly with the strongest gated baselines;
4. strict filtering created an uneven convenience sample that was not fully disclosed.

The final revision globally deduplicates gold, removes the `general` task
escape hatch, tests the production 286-candidate configuration, reports micro
and macro-family summaries, adds all four Stage Hybrid versus gated-scorer
comparisons, and records the 29/60 retention design. Stage Hybrid scores
0.3543 nDCG@10 versus 0.4347 for MiniLM plus safety filtering. The paired
difference is -0.0805 with bootstrap 95% CI [-0.1678, 0.0043] and Holm-adjusted
p=0.4165; no superiority claim is supported.

## Remaining Claim Gate

The benchmark remains silver-label and diagnostic-only. Two independent blind
annotators and adjudication are still absent, so offline retrieval claims remain
disabled. Online adoption and downstream training claims are not measured.

## Final ClaudeAgent Verdict

The final read-only acceptance review found **no remaining engineering P0**.
It marked global gold deduplication, removal of the `general` escape hatch,
production 286-candidate test coverage, all four Stage Hybrid head-to-head
comparisons, sampling disclosure, and statistical/report consistency as
closed or confirmed.

The remaining blockers are claim gates rather than implementation defects:
two-person blind adjudication is incomplete; 29 uneven convenience points are
underpowered for superiority; and the sample is not representative. Minor
structural limitations remain because two decision-family labels share the
`leaf-classification` task domain and Tree-no-task still accesses graph
geometry. ClaudeAgent's disposition is: **appendix diagnostic with clean
engineering, not a paper main-table result**.
