"""Plugin B: Analyze training run logs, extract wins/failures/hypotheses into experience_kb/"""
import os, json, argparse, subprocess
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

SYSTEM = """You are analyzing ML training run logs from an automated AI agent system.
Extract structured knowledge from judge evaluations and designer plans.
Be specific: every claim must cite the source (e.g. judge_3/node_15) and include a direct quote as evidence.
Return valid JSON only."""

WINS_PROMPT = """From these judge evaluations, extract techniques that had positive outcomes (high scores, beat baselines, worked as expected).

For each win:
- name: short descriptive name
- description: what the technique is
- condition: when it applies
- evidence: [{{source, quote, node_id, score}}]

Return JSON: {{"wins": [...]}}

Logs:
{logs}"""

FAILURES_PROMPT = """From these judge evaluations and designer plans, extract failures: blocked dependencies, installation errors, wrong assumptions, techniques that underperformed.

For each failure:
- name: short descriptive name
- description: what failed and why
- root_cause: underlying reason
- evidence: [{{source, quote, node_id}}]

Return JSON: {{"failures": [...]}}

Logs:
{logs}"""

HYPOTHESES_PROMPT = """From the designer plans and judge evaluations, extract hypotheses that were proposed and later confirmed or falsified.

For each hypothesis:
- hypothesis: the original claim
- status: "confirmed" | "falsified" | "untested"
- proposed_in: source file
- verdict_in: source file where outcome was determined (or null)
- evidence: [{{source, quote}}]

Return JSON: {{"hypotheses": [...]}}

Logs:
{logs}"""

DEDUP_PROMPT = """You are checking new entries against an existing knowledge base for duplicates.

EXISTING:
{existing}

NEW ENTRIES:
{new_entries}

For each new entry, decide:
- "skip": the core concept is already covered in existing (even if wording differs or new entry has more detail)
- "add": genuinely new concept not present in existing

Be strict: if the same technique/failure/hypothesis appears under any name, skip it.
Do NOT merge — either skip or add.

Return JSON: {{"results": [{{"action": "skip"|"add", "entry": {{...or null if skip}}}}]}}"""


def git_init_if_needed(repo_dir: Path):
    if not (repo_dir / ".git").exists():
        subprocess.run(["git", "init"], cwd=repo_dir, check=True)
        subprocess.run(["git", "commit", "--allow-empty", "-m", "init experience_kb"], cwd=repo_dir, check=True)


def git_commit(repo_dir: Path, message: str):
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    result = subprocess.run(["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True)
    if result.stdout.strip():
        subprocess.run(["git", "commit", "-m", message], cwd=repo_dir, check=True)
        print(f"  Git committed: {message}")


def load_logs(run_dir: Path) -> dict:
    steps = run_dir / "steps"
    logs = {"judges": [], "designers": []}

    for f in sorted(steps.glob("judge_*/output.json")):
        data = json.loads(f.read_text())
        data["_source"] = f.parent.name
        logs["judges"].append(data)

    for f in sorted(steps.glob("designer*/output.json")):
        text = f.read_text()[:8000]
        logs["designers"].append({"_source": f.parent.name, "content": text})

    return logs


def call_llm(prompt: str) -> dict:
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1])
    return json.loads(raw)


def dedup_entries(existing_md: str, new_entries: list) -> list:
    if not existing_md.strip() or not new_entries:
        return new_entries
    resp = call_llm(DEDUP_PROMPT.format(
        existing=existing_md[:4000],
        new_entries=json.dumps(new_entries, ensure_ascii=False)
    ))
    results = resp.get("results", [])
    deduped = []
    for i, r in enumerate(results):
        if r.get("action") == "skip":
            name = new_entries[i].get("name", new_entries[i].get("hypothesis", "?"))
            print(f"  Dedup: skipped '{name}'")
        elif r.get("entry"):
            deduped.append(r["entry"])
    return deduped


def render_wins(wins: list, header: bool = True, source_run: str = "") -> str:
    lines = ["# Wins — Effective Techniques\n"] if header else []
    for w in wins:
        lines.append(f"## {w['name']}")
        lines.append(f"{w['description']}\n")
        lines.append(f"**Condition**: {w.get('condition', 'N/A')}\n")
        if source_run:
            lines.append(f"**Source run**: `{source_run}`\n")
        lines.append("**Evidence**:")
        for e in w.get("evidence", []):
            lines.append(f"- `{e.get('source', '')}` node_{e.get('node_id', '?')}: \"{e.get('quote', '')}\"")
        lines.append("")
    return "\n".join(lines)


def render_failures(failures: list, header: bool = True, source_run: str = "") -> str:
    lines = ["# Failures — Errors & Blocked Dependencies\n"] if header else []
    for f in failures:
        lines.append(f"## {f['name']}")
        lines.append(f"{f['description']}\n")
        lines.append(f"**Root cause**: {f.get('root_cause', 'N/A')}\n")
        if source_run:
            lines.append(f"**Source run**: `{source_run}`\n")
        lines.append("**Evidence**:")
        for e in f.get("evidence", []):
            lines.append(f"- `{e.get('source', '')}` node_{e.get('node_id', '?')}: \"{e.get('quote', '')}\"")
        lines.append("")
    return "\n".join(lines)


def render_hypotheses(hypotheses: list, header: bool = True, source_run: str = "") -> str:
    lines = ["# Hypotheses — Confirmed, Falsified, Untested\n"] if header else []
    for h in hypotheses:
        status = h.get("status", "untested").upper()
        lines.append(f"## [{status}] {h['hypothesis']}")
        lines.append(f"**Proposed in**: `{h.get('proposed_in', '?')}`")
        lines.append(f"**Verdict in**: `{h.get('verdict_in', 'N/A')}`")
        if source_run:
            lines.append(f"**Source run**: `{source_run}`")
        lines.append("")
        lines.append("**Evidence**:")
        for e in h.get("evidence", []):
            lines.append(f"- `{e.get('source', '')}`: \"{e.get('quote', '')}\"")
        lines.append("")
    return "\n".join(lines)


def write_with_dedup(filepath: Path, new_entries: list, render_fn, existing_text: str, source_run: str = ""):
    deduped = dedup_entries(existing_text, new_entries)
    if not deduped:
        return
    if existing_text:
        filepath.write_text(existing_text + "\n" + render_fn(deduped, header=False, source_run=source_run))
    else:
        filepath.write_text(render_fn(deduped, source_run=source_run))


CONFLICT_PROMPT = """Two parallel runs produced conflicting conclusions about the same technique/failure/hypothesis.

CONFLICT in file: {filename}
TECHNIQUE/ENTRY: {entry_name}

RUN A ({run_a}):
{content_a}

RUN B ({run_b}):
{content_b}

RUN A original judge logs:
{trace_a}

RUN B original judge logs:
{trace_b}

Analyze:
1. What type of conflict is this? (direct_contradiction / condition_difference / environment_factor / sample_size)
2. What caused the difference? (different env, different data, different hyperparams, random variance?)
3. What is the correct conclusion? (keep_a / keep_b / conditional / merge)
4. Write the final resolved entry text (markdown format, same style as input)

Return JSON: {{"conflict_type": "...", "cause": "...", "resolution": "keep_a|keep_b|conditional|merge", "resolved_entry": "...", "reasoning": "..."}}"""


def resolve_git_conflicts(output_dir: Path, run_dir: Path, conflict_log_dir: Path):
    result = subprocess.run(["git", "diff", "--name-only", "--diff-filter=U"],
                            cwd=output_dir, capture_output=True, text=True)
    conflicted_files = result.stdout.strip().split("\n")
    conflicted_files = [f for f in conflicted_files if f]
    if not conflicted_files:
        return

    conflict_log_dir.mkdir(parents=True, exist_ok=True)
    judge_text = ""
    steps = run_dir / "steps"
    for f in sorted(steps.glob("judge_*/output.json")):
        judge_text += f.read_text()[:3000] + "\n"

    for filename in conflicted_files:
        filepath = output_dir / filename
        content = filepath.read_text()
        # parse conflict blocks
        import re
        blocks = re.findall(r'<<<<<<< .*?\n(.*?)=======\n(.*?)>>>>>>> .*?\n', content, re.DOTALL)
        resolved = content
        log_entries = []

        for content_a, content_b in blocks:
            name_match = re.search(r'^## (.+)', content_a.strip())
            entry_name = name_match.group(1) if name_match else "unknown"
            run_a_match = re.search(r'\*\*Source run\*\*: `([^`]+)`', content_a)
            run_b_match = re.search(r'\*\*Source run\*\*: `([^`]+)`', content_b)
            run_a = run_a_match.group(1) if run_a_match else "unknown"
            run_b = run_b_match.group(1) if run_b_match else "unknown"

            print(f"  Resolving conflict: {entry_name}")
            resp = call_llm(CONFLICT_PROMPT.format(
                filename=filename, entry_name=entry_name,
                run_a=run_a, content_a=content_a.strip(),
                run_b=run_b, content_b=content_b.strip(),
                trace_a=judge_text[:4000], trace_b=judge_text[:4000]
            ))

            resolved_entry = resp.get("resolved_entry", content_a.strip())
            conflict_block = f"<<<<<<< HEAD\n{content_a}=======\n{content_b}>>>>>>> "
            resolved = resolved.replace(content_a + "=======\n" + content_b, resolved_entry + "\n", 1)

            log_entries.append({
                "entry": entry_name, "file": filename,
                "run_a": run_a, "run_b": run_b,
                "conflict_type": resp.get("conflict_type"),
                "cause": resp.get("cause"),
                "resolution": resp.get("resolution"),
                "reasoning": resp.get("reasoning"),
            })

        # clean up remaining conflict markers
        resolved = re.sub(r'<<<<<<< .*?\n', '', resolved)
        resolved = re.sub(r'=======\n', '', resolved)
        resolved = re.sub(r'>>>>>>> .*?\n', '', resolved)
        filepath.write_text(resolved)

        # write conflict log
        from datetime import datetime
        log_file = conflict_log_dir / f"conflict_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename.replace('/', '_')}.md"
        log_lines = [f"# Conflict Resolution Log — {filename}\n"]
        for e in log_entries:
            log_lines += [
                f"## {e['entry']}",
                f"**Type**: {e['conflict_type']}",
                f"**Run A**: `{e['run_a']}`",
                f"**Run B**: `{e['run_b']}`",
                f"**Cause**: {e['cause']}",
                f"**Resolution**: {e['resolution']}",
                f"**Reasoning**: {e['reasoning']}\n",
            ]
        log_file.write_text("\n".join(log_lines))
        print(f"  Conflict log: {log_file.name}")

    subprocess.run(["git", "add", "."], cwd=output_dir, check=True)
    subprocess.run(["git", "commit", "-m", f"resolve conflicts from {run_dir.name}"], cwd=output_dir, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="path to run directory")
    parser.add_argument("--output_dir", default="experience_kb")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    conflict_log_dir = Path("cache/conflict_logs")
    output_dir.mkdir(parents=True, exist_ok=True)

    git_init_if_needed(output_dir)

    # pull latest and work in a new branch
    branch = f"run-{run_dir.name}"
    subprocess.run(["git", "checkout", "-b", branch], cwd=output_dir, capture_output=True)

    print("Loading logs...")
    logs = load_logs(run_dir)
    source_run = str(run_dir)

    judge_text = json.dumps(logs["judges"], ensure_ascii=False)[:12000]
    designer_text = json.dumps(logs["designers"], ensure_ascii=False)[:6000]
    combined = judge_text + "\n\nDESIGNER PLANS:\n" + designer_text

    wins_file = output_dir / "wins.md"
    failures_file = output_dir / "failures.md"
    hypotheses_file = output_dir / "hypotheses.md"

    print("Extracting wins...")
    wins_data = call_llm(WINS_PROMPT.format(logs=judge_text))
    write_with_dedup(wins_file, wins_data.get("wins", []), render_wins,
                     wins_file.read_text() if wins_file.exists() else "", source_run)

    print("Extracting failures...")
    failures_data = call_llm(FAILURES_PROMPT.format(logs=combined))
    write_with_dedup(failures_file, failures_data.get("failures", []), render_failures,
                     failures_file.read_text() if failures_file.exists() else "", source_run)

    print("Extracting hypotheses...")
    hypotheses_data = call_llm(HYPOTHESES_PROMPT.format(logs=combined))
    write_with_dedup(hypotheses_file, hypotheses_data.get("hypotheses", []), render_hypotheses,
                     hypotheses_file.read_text() if hypotheses_file.exists() else "", source_run)

    git_commit(output_dir, f"plugin-b: update from {run_dir.name}")

    # merge back to master
    merge_result = subprocess.run(
        ["git", "merge", branch], cwd=output_dir, capture_output=True, text=True
    )
    if merge_result.returncode != 0:
        print("  Merge conflict detected, resolving...")
        resolve_git_conflicts(output_dir, run_dir, conflict_log_dir)

    subprocess.run(["git", "checkout", "master"], cwd=output_dir, capture_output=True)
    print(f"Done. Output in {output_dir}/")
