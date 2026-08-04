# Experiment End2End Version and Selective-Port Audit

- Audit date: 2026-08-04
- Writable worktree: `/Users/haoming/Downloads/nautilus-exp-end2end`
- Branch: `codex/experiment-end2end-memory-systems`
- Initial HEAD: `742a24090adebba8780a6ddf4479db84c28ae311`
- Code baseline named by the plan: `f667132e1c67f0f53f26d7e22d0b4ef9b6dc671b`
- Initial writable-worktree status: clean
- Policy: every other Nautilus worktree was read-only throughout this audit.

## Reference inventory at audit time

| Worktree | Branch | HEAD | Dirty paths | Use |
|---|---|---|---:|---|
| End2End | `codex/experiment-end2end-memory-systems` | `742a2409…` | 0 initially | only writable tree |
| Main | `codex/dual-time-procedural-memory` | `f667132e…` | 608 | read-only runtime/Authority reference |
| Exp-R | `codex/experiment-r-dynamic-memory-routing` | `8d7e9a2c…` | 300 | read-only memory/binding/routing reference |
| Exp-B | `codex/experiment-b-claim-use-trace` | `3c05c7b2…` | 130 | read-only trace reference; no port selected |
| Exp-C | `codex/experiment-c-multitask-replication` | `f667132e…` | 103 | read-only task/evaluator asset reference |

The committed Main difference from End2End only deletes the two End2End
coordination documents. The committed Exp-R difference adds its experiment
scaffold and deletes those documents. The relevant runtime fixes live in dirty
reference trees, so no branch or worktree was merged wholesale.

Baseline targeted regression before edits:

```text
100 passed, 1 skipped in 18.61s
```

## Selected ports

| Change | Read-only source | Why required here | Local regression evidence |
|---|---|---|---|
| `excluded_run_ids` reaches `StageAwareHybridMemoryLayer` | Exp-R `engine/agent_search.py` / config | Frozen held-out exclusions otherwise existed only on paper | End2End layer/config tests |
| Collision-free Host artifact namespace | Exp-R `config/__init__.py` | 40 runs must not write the signed Host journals into the same binding root | unsafe/collision-free namespace test |
| Exact production Host SDK compatibility | Exp-R `protocol_runtime/activation.py`, `collector.py` | Production Host bindings pin SDK hash `1084155…`; runtime must match byte-for-byte semantics | exact SDK-hash gate and collector tests |
| Local Python/NumPy/Torch RNG state commitment | Exp-R `utils/seed.py`, `run.py` | Seed labels alone are not live RNG evidence | seed/run-identity tests; manifest explicitly denies provider determinism |
| Generic per-node routing trace before empty-ref return | adapted from Exp-R `agents/adoption.py` | No Memory must still retain raw pool, suppression, Bundle and route evidence | No-Memory empty-ref serialization test |
| Signed Host evaluator metric precedes LLM-parsed metric | Main `receipt_bridge.py`, `result_parse_agent.py` | avoids discarding a valid runtime score because of parser rounding/hallucination | full-runtime trusted-metric regression |
| Serialized Authority snapshot emission | Main `authority/adapters/mlevolve/runtime.py` | search threads share a four-file snapshot transaction | eight-thread serialization regression |
| Aerial row/lifecycle guidance | adapted from Main `protocol_runtime/activation.py` into Candidate prompt assembly | needed operational guidance, but changing the production-bound SDK would invalidate its hash | Aerial guidance order/field regression plus exact SDK-hash gate |

The End2End controller itself is new rather than copied from the 1,000+ line
Exp-R router. It reuses the existing Base Bundle loader, Authority visibility
gateway, Session Overlay, SOP/RunForest rankers, prospective audit logger and
adoption side channel, then applies one pure registered policy.

## Explicit non-ports

- No complete dirty worktree, router, historical replay anchor, parent
  checkpoint machinery or stage-only Exp-R harness was copied.
- No Exp-B implementation was required for the frozen routing/adoption fields.
- No Exp-C source tree, system configuration or analysis manifest is used.
  Only the release-bound task data and terminal evaluator assets are consumed.
- Main's historical replay-parent Authority relaxation was not selected because
  this Pilot does not enable historical replay bootstrap.
- Main's broad candidate process-memory guard and unrelated preflight AST
  changes were not selected. The workload instead freezes one search GPU,
  one parallel search, a per-index deadline and 256 GiB Host memory.
- Main's run-local SDK-root change was not selected because Exp-R's already
  tested immutable-binding namespace provides collision-free persistent roots
  while preserving the exact SDK hash required by the production bindings.

## Frozen dependency decisions

- Memory and Host bindings come from Exp-R production binding
  `47e0d38a…`; the four exact Base manifests, graphs, indexes, CURRENT files,
  Contracts and DataViews are separately pinned.
- Terminal evaluator assets come transitively from the Exp-C base release
  binding `668896c0…` under `/workspace/experiment-c-formal-releases-r3`.
  Launch-time PVC verification showed that `bee5a525…` is the Exp-C r9
  hardware-amended control binding, while the shared formal-release asset root
  is correctly restored to its original base binding. End2End does not reuse
  Exp-C hardware/control manifests, so the base asset binding is authoritative.
- The common container is
  `docker.io/haomingwang22/mlevolve@sha256:fe0b9c38…`.
- The common model binding is `deepseek-production-solver` at revision
  `sha256:6c728901…`.
- The Pilot uses seed 1 and remains exploratory. No significance claim is
  permitted by its manifest.
