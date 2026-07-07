# Codex and Claude Code Coordination

This folder is the explicit shared workspace for Codex, Claude Code, and the user.
It is a working board, not a chat archive.

## Roles

- Codex acts as the research lead: problem framing, experiment design, review, risk checks, and synthesis.
- Claude Code acts as the implementation worker: code changes, command execution, experiments, and fixes.
- The user owns priorities, constraints, and final decisions.

## Files

- `current_task.md`: the active task specification. Usually overwritten for each new task.
- `claude_report.md`: Claude Code's implementation report for the active task. Usually overwritten.
- `codex_review.md`: Codex's review of Claude Code's result. Usually overwritten.
- `shared_memory.md`: durable shared memory for project goals, preferences, findings, and warnings.
- `decisions.md`: durable record of confirmed technical or research decisions.
- `claude_skills.md`: index of Claude Code's locally installed skills and safe usage boundaries.
- `archive/`: optional old task snapshots. Archive contents are ignored by git by default.

## Task Loop

1. Codex writes or updates `current_task.md`.
2. Claude Code reads this `README.md`, `shared_memory.md`, `decisions.md`, and `current_task.md`.
3. Claude Code performs the work, then updates `claude_report.md`.
4. Codex reviews the git diff, relevant outputs, and `claude_report.md`.
5. Codex updates `codex_review.md`.
6. If needed, Claude Code addresses the review and updates `claude_report.md` again.
7. Durable conclusions go into `shared_memory.md` or `decisions.md`.

## Memory Rules

- Keep `shared_memory.md` short and high signal.
- Do not paste full conversations, large logs, secrets, credentials, private tokens, or bulky experiment output.
- Store only information likely to matter across future tasks.
- Prefer updating existing bullets over appending duplicates.
- Record irreversible or important choices in `decisions.md`.

## Claude Code Start Prompt

Use this when starting Claude Code on a task:

```text
Please read coordination/README.md, coordination/shared_memory.md,
coordination/decisions.md, and coordination/current_task.md first.
Follow the coordination protocol. After finishing, update
coordination/claude_report.md with changed files, commands run, results,
open issues, and any proposed memory/decision updates.
```
