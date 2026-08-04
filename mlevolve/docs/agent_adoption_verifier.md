# Agent-based memory adoption verification

The Agent verifier replaces task/model-specific adoption signatures with one
generic two-pass protocol:

1. Bind every prompt-exposed `ExperienceContract` to the reviewed candidate.
2. Ask an independent Verifier Agent for exact code evidence and executable
   source-line probes.
3. Validate all line ranges and bind the plan to the contract hashes and exact
   candidate code hash.
4. Let the Executor collect task-independent Python line events while the real
   candidate runs.
5. Seal the trace with the Host collector identity when one is active.
6. Ask the Verifier Agent for `adopted`, `partially_adopted`, `rejected`, or
   `uncertain` for every contract.
7. In `enforce` mode, publish an L3 adoption edge only when the final verdict is
   positive and at least one contract-bound probe actually executed.

No task name, benchmark name, model family, API signature, or hand-authored
score point appears in this path. The only initial runtime primitive is
`line_range_executed`; the Agent selects the ranges separately for every
memory/candidate pair.

## Configuration

```yaml
adoption_verifier:
  enabled: true
  mode: enforce
  model: ""                 # empty uses agent.feedback.model
  temperature: 0.0
  max_tokens: 4096
  max_contracts_per_call: 8
  max_code_chars: 120000
  require_signed_trace: true
```

`enforce` requires `evaluation_authority.protocol_runtime_mode` to be
`host_sdk_shadow` or `host_sdk_enforce`. Requiring signed traces additionally
requires Host Protocol Preflight so the signing key stays outside Candidate
state.

`shadow` records Agent plans and verdicts but retains the legacy adoption gate.
It is intended for parity measurement before enabling enforcement.

## Evidence artifacts

For each node, the run writes:

- `logs/adoption_verifier/<node_id>.plan.json`
- `logs/adoption_verifier/<node_id>.verdict.json`
- the Host-sealed runtime trace on the `SearchNode`
- hash-chained Authority ledger events for the plan and verdict

Positive Agent prose is never sufficient by itself. Invalid source ranges,
missing contracts, duplicate probes, missing traces, unexecuted probes, stale
hashes, and malformed Agent output all fail closed to `uncertain` or no
adoption edge.
