# Claude Code Skills Index

This file indexes Claude Code skills available on this machine for the Nautilus project.
It is a capability map, not a copy of private skill state.

Last checked by Codex: 2026-07-01.

## Safety Rules

- Do not copy secrets, tokens, credentials, kubeconfigs, or full private transcripts into this repository.
- Do not read or print skill token files unless the user explicitly asks for a credential operation.
- If a skill references a local plaintext token, treat the path as sensitive and keep the token out of prompts, reports, commits, and logs.
- Prefer summaries and pointers over copying full skill bodies into `coordination/`.
- Re-check the source skill file before using operational details, because local skills can change outside git.

## Available Claude Code Skills

### `nrp-training`

Source:

- `~/.claude/skills/nrp-training/SKILL.md`
- `~/.claude/skills/nrp-training/job-experiment.yaml`
- `~/.claude/skills/nrp-training/pod-dev.yaml`

Use when:

- The user wants to run training, baselines, A/B experiments, or GPU jobs on NRP Nautilus.
- The user mentions NRP, Nautilus, GPU pod/job YAML, A6000, A100, H100, "进 pod", or "跑实验".

Core knowledge:

- Namespace: `ecepxie`.
- Persistent PVC is mounted at `/workspace`; container image state is ephemeral.
- Use dev Pods for interactive debugging and Jobs for formal unattended runs.
- Jobs must run the training process as PID 1 and exit when done. Do not append `sleep`, long-lived shells, tmux, code-server, or other keepalive processes to a Job.
- GPU requests must equal GPU limits.
- Match Job resource counts to `run.py` overrides:
  - `agent.search.num_gpus = GPU count`
  - `agent.search.parallel_search_num = GPU count`
  - `cpu_number = CPU count`
- Keep per-job overrides minimal; leave general experiment knobs in config unless intentionally changing run scale.
- Do not embed real API tokens in YAML.

Sensitive boundaries:

- Skill templates may refer to local env files, cluster auth, or model-provider credentials. Keep those outside `coordination/`.
- Before creating a real Job, re-read the source skill and template files.

### `pod-push-runs`

Source:

- `~/.claude/skills/pod-push-runs/SKILL.md`
- `~/.claude/skills/pod-push-runs/token` exists locally and is sensitive.

Use when:

- An `mlevolve` training Job has finished and the user asks to push run results, e.g. "push 一下 run", "进 pod push", or "帮我 push 结果".

Core knowledge:

- Intended flow: enter the live dev pod, identify the latest run under `/workspace/nautilus/mlevolve/runs/`, and push only the four core log files:
  - `journal.json`
  - `best_solution.py`
  - `config.yaml`
  - `filtered_journal.json`
- Confirm the target run before pushing.
- Report the resulting commit and branch after a successful push.

Sensitive boundaries:

- The GitHub PAT is stored in a local plaintext token file under the Claude skill directory.
- Do not read, copy, print, commit, or summarize the token value.
- Do not paste token-bearing remote URLs into `coordination/`, chat, logs, or committed files.
- If the dev pod is down, bring it up via the appropriate local pod YAML rather than moving secrets into the repository.

## Not Synchronized

- `~/.claude/tasks/`, `~/.claude/sessions/`, and plugin transcript/context caches are not indexed here; they are private operational history, not project skills.
- Claude HUD plugin files are not project-specific skills for this repository.
- Claude project memory has already been summarized separately in `shared_memory.md`; use that file for long-term project context.
