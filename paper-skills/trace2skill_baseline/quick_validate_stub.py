#!/usr/bin/env python3
"""Minimal skill-format checker.

Stand-in for Anthropic's skill-creator `quick_validate.py`, which the Trace2Skill
repo references (QUICK_VALIDATE_SCRIPT = skills/skill-creator/scripts/quick_validate.py)
but does not vendor. The combined skill-evolution runner hard-exits if this file is
absent (run_parallel_combined_skill_evolution.py:341-343), so a stand-in is required
to run the pipeline. This checks the load-bearing format invariants the evolver relies
on (SKILL.md present; YAML frontmatter with non-empty `name` and `description`) and is
otherwise permissive. It is infrastructure (a format guardrail), NOT part of the
Trace2Skill distillation/consolidation algorithm.

Usage: quick_validate.py <skill_dir>   -> exit 0 on pass, 1 on fail.
"""
import re
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: quick_validate.py <skill_dir>", file=sys.stderr)
        sys.exit(2)
    skill_dir = Path(sys.argv[1])
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        print(f"FAIL: SKILL.md missing in {skill_dir}", file=sys.stderr)
        sys.exit(1)
    text = skill_md.read_text(encoding="utf-8", errors="replace")
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not m:
        print("FAIL: SKILL.md missing YAML frontmatter (--- ... ---)", file=sys.stderr)
        sys.exit(1)
    fm = m.group(1)
    if not re.search(r"^name:\s*\S", fm, re.MULTILINE):
        print("FAIL: frontmatter missing non-empty 'name'", file=sys.stderr)
        sys.exit(1)
    if not re.search(r"^description:\s*\S", fm, re.MULTILINE):
        print("FAIL: frontmatter missing non-empty 'description'", file=sys.stderr)
        sys.exit(1)
    print(f"OK: SKILL.md valid ({len(text)} chars, {len(text.splitlines())} lines)")
    sys.exit(0)


if __name__ == "__main__":
    main()
