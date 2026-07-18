# Stage Hybrid v2 Independent ClaudeAgent Audit

## Scope

ClaudeAgent independently reviewed the production implementation, benchmark
adapter, ablation, and focused tests in read-only mode. Review session:
`b3e9987c-51ac-4e8e-9307-771369a2af17`.

The review checked that the production method really uses both SOP and Tree
channels, stage-aware routing, task identity, geometry, weighted RRF, clean
evidence gates, and execution-to-SOP provenance. It also checked that the
legacy gateway remains an independent historical control.

## Findings And Disposition

| Review finding | Disposition |
|---|---|
| Legacy control might accidentally call the new scorer | Withdrawn after tracing the independent legacy implementation. |
| Tree candidates might bypass validity checks | Withdrawn after tracing explicit positive-transition and clean rank-eligible RunNode predicates. |
| Execution-to-SOP projection might launder dirty evidence | Withdrawn: projection requires the specific linked transition to be clean. |
| `sop_only` provenance might include filtered candidates | Withdrawn after tracing the returned ranked candidate set. |
| Raw metric-improvement bonus was not task-normalized | Fixed: positive improvements are converted to a within-task percentile before the bounded bonus is applied. |

## Final Verdict

ClaudeAgent's final focused re-review found no remaining P0 or P1 issue and no
new blocker after the task-local metric normalization. Its engineering verdict
was that Stage Hybrid v2 is acceptable for production use.

This is not a scientific superiority verdict. The benchmark contains 29
single-annotator silver decision points selected from 60 convenience seeds.
The v2 point estimate is 0.4522 nDCG@10 versus 0.3543 for the legacy gateway and
0.4347 for MiniLM plus the safety gate, but both relevant confidence intervals
cross zero. Blind annotation, adjudication, and concurrent online downstream
training remain required before a paper claim can be opened.
