"""Plugin C — Dreamer: sleep-time memory consolidation and forgetting."""
import os, re, json, argparse, subprocess
from pathlib import Path
from datetime import datetime
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

MAX_FORGET_PER_RUN = 3
FORGET_SEEN_THRESHOLD = 1
FORGET_DREAMER_RUNS = 10
ARCHIVE_DELETE_RUNS = 5

CONSOLIDATE_PROMPT = """You are consolidating a knowledge base built from ML training runs.

CURRENT KNOWLEDGE BASE:
{content}

Tasks:
1. Merge entries that describe the same core concept (even if worded differently). Keep the most informative version.
2. Update "Seen" counts: if two entries are merged, sum their seen counts.
3. Extract 2-5 meta-insights: higher-level patterns that emerge from multiple entries.

Return JSON:
{{
  "entries": [{{...same structure as input entries, with updated seen counts}}],
  "meta_insights": [{{"insight": "...", "supporting_entries": ["entry name 1", ...]}}]
}}"""

FORGET_PROMPT = """You are deciding whether to archive low-confidence entries from a knowledge base.

CANDIDATE ENTRIES FOR FORGETTING:
{candidates}

For each candidate, decide: "archive" or "keep".
Keep if: critical constraint, unique finding, or dangerous to forget.
Archive if: minor observation, easily re-discoverable, or superseded.

Return JSON: {{"decisions": [{{"name": "...", "action": "archive"|"keep", "reason": "..."}}]}}"""


def call_llm(prompt: str) -> dict:
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:-1])
    return json.loads(raw)


def git_commit(repo_dir: Path, message: str):
    subprocess.run(["git", "add", "."], cwd=repo_dir, check=True)
    result = subprocess.run(["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True)
    if result.stdout.strip():
        subprocess.run(["git", "commit", "-m", message], cwd=repo_dir, check=True)


def parse_entries(md: str) -> list[dict]:
    """Parse ## sections from markdown into list of dicts with name, body, seen, dreamer_runs, archived_runs."""
    entries = []
    blocks = re.split(r'\n(?=## )', md)
    for block in blocks:
        if not block.startswith("## "):
            continue
        lines = block.strip().split("\n")
        name = lines[0][3:].strip()
        body = "\n".join(lines[1:])
        seen_m = re.search(r'\*\*Seen\*\*:\s*(\d+)', body)
        dreamer_m = re.search(r'\*\*Dreamer runs\*\*:\s*(\d+)', body)
        archived_m = re.search(r'\*\*Archived runs\*\*:\s*(\d+)', body)
        seen = int(seen_m.group(1)) if seen_m else 1
        dreamer_runs = int(dreamer_m.group(1)) if dreamer_m else 0
        archived_runs = int(archived_m.group(1)) if archived_m else 0
        entries.append({"name": name, "body": body, "seen": seen, "dreamer_runs": dreamer_runs, "archived_runs": archived_runs})
    return entries


def render_entry(name: str, body: str, seen: int, dreamer_runs: int, archived_runs: int = 0) -> str:
    body = re.sub(r'\*\*Seen\*\*:\s*\d+', f'**Seen**: {seen}', body)
    body = re.sub(r'\*\*Dreamer runs\*\*:\s*\d+', f'**Dreamer runs**: {dreamer_runs}', body)
    if archived_runs > 0:
        if re.search(r'\*\*Archived runs\*\*:', body):
            body = re.sub(r'\*\*Archived runs\*\*:\s*\d+', f'**Archived runs**: {archived_runs}', body)
        else:
            body = body.rstrip() + f'\n**Archived runs**: {archived_runs}'
    if '**Seen**' not in body:
        body = body.rstrip() + f'\n\n**Seen**: {seen}\n**Dreamer runs**: {dreamer_runs}'
    return f"## {name}\n{body}\n"


def render_insights(insights: list) -> str:
    lines = ["# Meta-Insights\n"]
    for ins in insights:
        lines.append(f"## {ins['insight']}")
        lines.append(f"**Supporting**: {', '.join(ins.get('supporting_entries', []))}\n")
    return "\n".join(lines)


def task1_consolidate(output_dir: Path) -> dict:
    """Merge similar entries, extract meta-insights, update seen counts."""
    results = {}
    for fname in ["wins.md", "failures.md", "hypotheses.md"]:
        fpath = output_dir / fname
        if not fpath.exists():
            continue
        content = fpath.read_text()
        print(f"  Consolidating {fname}...")
        resp = call_llm(CONSOLIDATE_PROMPT.format(content=content[:8000]))
        entries = resp.get("entries", [])
        insights = resp.get("meta_insights", [])
        results[fname] = {"entries": entries, "insights": insights}
    return results


def task2_forget(output_dir: Path) -> list:
    """Archive low-confidence entries, delete long-archived ones."""
    archived = []
    archive_dir = output_dir / "archive"
    archive_dir.mkdir(exist_ok=True)

    for fname in ["wins.md", "failures.md", "hypotheses.md"]:
        fpath = output_dir / fname
        if not fpath.exists():
            continue
        entries = parse_entries(fpath.read_text())

        # increment dreamer_runs for all entries
        for e in entries:
            e["dreamer_runs"] += 1

        # find candidates for forgetting: threshold scales with seen count
        candidates = [e for e in entries
                      if e["dreamer_runs"] >= FORGET_DREAMER_RUNS + (e["seen"] - 1)][:MAX_FORGET_PER_RUN]

        if not candidates:
            # rewrite with updated dreamer_runs
            content = f"# {fname.replace('.md','').title()}\n\n"
            for e in entries:
                content += render_entry(e["name"], e["body"], e["seen"], e["dreamer_runs"]) + "\n"
            fpath.write_text(content)
            continue

        # LLM decides which to archive
        resp = call_llm(FORGET_PROMPT.format(
            candidates=json.dumps([{"name": e["name"], "body": e["body"][:500]} for e in candidates])
        ))
        to_archive = {d["name"] for d in resp.get("decisions", []) if d["action"] == "archive"}

        kept, archiving = [], []
        for e in entries:
            if e["name"] in to_archive:
                archiving.append(e)
                print(f"  Archiving: {e['name']}")
            else:
                kept.append(e)

        # write archive (archived_runs starts at 1)
        if archiving:
            archive_file = archive_dir / f"{fname.replace('.md', '')}_{datetime.now().strftime('%Y%m%d')}.md"
            with open(archive_file, "a") as af:
                for e in archiving:
                    af.write(render_entry(e["name"], e["body"], e["seen"], e["dreamer_runs"], archived_runs=1) + "\n")
            archived.extend([e["name"] for e in archiving])

        # rewrite main file
        content = f"# {fname.replace('.md','').title()}\n\n"
        for e in kept:
            content += render_entry(e["name"], e["body"], e["seen"], e["dreamer_runs"]) + "\n"
        fpath.write_text(content)

    # process existing archive files: increment archived_runs, delete entries >= ARCHIVE_DELETE_RUNS
    for archive_file in sorted(archive_dir.glob("*.md")):
        entries = parse_entries(archive_file.read_text())
        surviving = []
        for e in entries:
            e["archived_runs"] += 1
            if e["archived_runs"] >= ARCHIVE_DELETE_RUNS:
                print(f"  Permanently deleted: {e['name']} (archived_runs={e['archived_runs']})")
            else:
                surviving.append(e)
        if surviving:
            content = ""
            for e in surviving:
                content += render_entry(e["name"], e["body"], e["seen"], e["dreamer_runs"], e["archived_runs"]) + "\n"
            archive_file.write_text(content)
        else:
            archive_file.unlink()
            print(f"  Removed empty archive: {archive_file.name}")

    return archived


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="experience_kb")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("Task 1: Consolidating...")
    consolidated = task1_consolidate(output_dir)

    # write consolidated entries back + append insights to each file
    all_insights = []
    for fname, data in consolidated.items():
        fpath = output_dir / fname
        llm_entries = data["entries"]
        if llm_entries:
            # build lookup of original entries to preserve body content
            original_entries = {e["name"]: e for e in parse_entries(fpath.read_text())}
            header_title = fname.replace('.md','').capitalize()
            content = f"# {header_title}\n\n"
            for e in llm_entries:
                name = e.get("name", "")
                orig = original_entries.get(name, {})
                # use original body; only update seen count from LLM
                body = orig.get("body", e.get("body", ""))
                seen = e.get("seen", orig.get("seen", 1))
                dreamer_runs = orig.get("dreamer_runs", 0)
                content += render_entry(name, body, seen, dreamer_runs) + "\n"
            insights = data.get("insights", [])
            if insights:
                content += "\n## Meta-Insights\n\n"
                for ins in insights:
                    content += f"### {ins['insight']}\n"
                    content += f"**Supporting**: {', '.join(ins.get('supporting_entries', []))}\n\n"
            fpath.write_text(content)
        all_insights.extend(data.get("insights", []))

    print("Task 2: Forgetting...")
    archived = task2_forget(output_dir)
    if archived:
        print(f"  Archived {len(archived)} entries")

    git_commit(output_dir, f"dreamer: consolidate + forget ({datetime.now().strftime('%Y%m%d_%H%M')})")
    print("Done.")

