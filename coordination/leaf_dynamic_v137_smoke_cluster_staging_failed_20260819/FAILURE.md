# v137-smoke cluster staging failure (pre-Job)

- Staging Pod: `mlevolve-leaf-gpt56sol-v137-smoke-stager`
- Pod UID: `71d4e183-22d4-49d7-9048-ff957cd5d877`
- Runtime: `/workspace/nautilus-exp-end2end-agent-v137-smoke`
- Evaluator: `/workspace/experiment-end2end-leaf-official-evaluator-v137-smoke`
- Source lock: `eb144dc4c458d58af905265f145bba9a20c38ecea3d2e3cb318ca33a6c346e66` (687 files)
- GPU Job created: no
- Logical run / attempt-000 created: no

The runtime and fresh evaluator finished staging, and the evaluator passed the
990/594/594 row contract with test SHA256
`c70a539ec7a5e900af69307ed3f630ff2650497870170cb2d130b4158e27eeda`
and sample SHA256
`09c16c65e54876a01bafe82b140c0feeefd1b97ca81147f87b33c60879006e1a`.

The final fail-closed source-lock audit rejected the release before Job
submission for two packaging-only causes:

1. `__pycache__/*.pyc` files were locked and then rewritten by local runtime
   tests, so seven bytecode hashes differed in the container.
2. macOS tar provenance xattrs materialized as untracked `._*` AppleDouble
   files on Linux.

No Dynamic algorithm, controller, Replay, validation, or fusion behavior was
changed. Commit `7fbc6627` removes mutable bytecode from release runtimes and
supports a fresh `v137-smoke-r2` identity. r2 transfer archives are created
with `COPYFILE_DISABLE=1` and are accepted only after an exact source-lock
audit.
