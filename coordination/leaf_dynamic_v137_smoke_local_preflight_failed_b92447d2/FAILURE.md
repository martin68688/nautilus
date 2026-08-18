Local-only release preflight failed before any Kubernetes resource was created.

- Release candidate: v137-smoke, source head `27bcfb0d`
- Failure: copied regression tests referenced test-only modules omitted from the compact runtime.
- Evidence: release/source-lock validation passed; pytest collection failed before assertions for `build_leaf_replay_gpt_v127_runtime` and `experiments.dynamic_memory_routing_injection_20260731`.
- Disposition: archived unchanged. No Pod, Job, PVC output, logical run, or attempt identity was created.
- Repair: source head `b92447d2` packages the narrow builder helper needed by the OpenAI-compatible regression tests. The broad design fixture remains a host-side regression test and is not required by the compact runtime preflight.
