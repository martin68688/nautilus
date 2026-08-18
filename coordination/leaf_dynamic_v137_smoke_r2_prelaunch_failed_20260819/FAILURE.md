# v137-smoke-r2 prelaunch failure (pre-Job)

- Staging Pod: `mlevolve-leaf-gpt56sol-v137-smoke-r2-stager`
- Pod UID: `40f26a4f-643b-48fc-b400-8ff02ab0fae4`
- Runtime: `/workspace/nautilus-exp-end2end-agent-v137-smoke-r2`
- Evaluator: `/workspace/experiment-end2end-leaf-official-evaluator-v137-smoke-r2`
- Source lock: `91d38279a9ec6b368888922394737b4ecdfb736c6965f90e7a7cb18a1c5bd710` (677 files)
- Container tests: 18/18 passed
- Source-lock audits before/after tests: 677 locked, 679 actual, 0 bad, 0 extra
- Evaluator: 990/594/594 rows; expected test/sample SHA256 values matched
- GPU Job created: no
- Logical run / attempt-000 created: no

The real `run_assignment.build_solver_command` prelaunch dry-run rejected the
release before Job submission. `systems.json.config_path` was written relative
to the runtime/repository root, while `run_assignment.py` resolves it relative
to the experiment directory. This duplicated the `experiments/...` path and
would have produced a `FileNotFoundError` at Job launch.

Commit `6e81e5f2` writes the config path relative to the experiment directory,
makes the validator resolve it with the same semantics as `run_assignment`,
removes mutable pytest caches, and rejects all untracked runtime extras. A
fresh `v137-smoke-r3` release is required because the r2 source lock and
execution manifest already bind the incorrect path.
