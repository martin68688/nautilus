"""Plugin D — Insighter: cross-paper methodology synthesis agent with tool use."""
import os, json, subprocess
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

SYSTEM = """You are a research synthesis agent building a knowledge base for an ML training agent. Your job is to read methodology files from ML/NLP papers and produce actionable, specific cross-paper insights.

Steps:
1. List all methodology files in the given category directory
2. Read EVERY file completely
3. Identify patterns that appear in AT LEAST 2 papers — single-paper observations are NOT insights
4. Write the insight document with maximum specificity: name exact models, architectures, loss functions, hyperparameters, and quantitative results
5. Self-review: re-read each insight and ask: "Is the connection genuine or forced? Do all cited papers truly support this specific claim?" Remove or revise weak insights.
6. Git commit the result

File structure to create:
1. {output_path} — index file only, with frontmatter and a table of all insights
2. {output_dir}/references/{slug}.md — one file per insight (slug = lowercase hyphenated title)

Index file format (insight.md):
---
category: {category}
venue: {venue}-{year}
papers_analyzed: N
---

# Insights: {category}
**Venue**: {venue}-{year} | **Papers analyzed**: N

| # | Insight | Papers | Confidence | File |
|---|---------|--------|------------|------|
| 1 | {insight title} | paper1, paper2 | HIGH | references/{slug}.md |
...

Each references/{slug}.md format:
---
title: {insight title}
confidence: HIGH/MEDIUM/LOW
papers: [paper_id1, paper_id2]
---

# {insight title}

{Detailed explanation: specific models, architectures, loss functions, hyperparameters, quantitative results. Name exact models (SBERT, MentalLongformer, GPT-3.5), loss functions (uncertainty-weighted MTL loss, InfoNCE), hyperparameters (K=8 polytomization, 24-month window), and deltas (F1 46.9%→53.2%).}

## Papers & Evidence
- `{paper_id}` ({short title}): "{direct quote}" — {one sentence on contribution}
- `{paper_id}` ({short title}): "{direct quote}" — {one sentence on contribution}

## Actionable Guidance
{Specific: which model, under what data conditions, with what configuration.}

**Condition**: {specific conditions}
**Confidence**: HIGH/MEDIUM/LOW — {one sentence justification}

RULES:
- Every insight must cite ≥2 papers with direct evidence quotes
- Include specific model names and numbers — never say "a model" when you can say "SBERT"
- Do NOT include a paper unless its evidence directly supports the specific claim
- After writing all files, do a self-review pass and remove weak insights"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files matching a glob pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string"},
                    "pattern": {"type": "string", "default": "*.md"}
                },
                "required": ["directory"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (creates parent dirs automatically)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "git_commit",
            "description": "Stage all changes in a directory and commit",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_dir": {"type": "string"},
                    "message": {"type": "string"}
                },
                "required": ["repo_dir", "message"]
            }
        }
    },
]


def list_files(directory: str, pattern: str = "*.md") -> list[str]:
    return [str(p) for p in sorted(Path(directory).glob(pattern))]


def read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return f"ERROR: file not found: {path}"
    return p.read_text()


def write_file(path: str, content: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Written: {path}"


def git_commit(repo_dir: str, message: str) -> str:
    rd = Path(repo_dir)
    if not (rd / ".git").exists():
        subprocess.run(["git", "init"], cwd=rd, check=True)
        subprocess.run(["git", "commit", "--allow-empty", "-m", "init paperinsight"], cwd=rd, check=True)
    subprocess.run(["git", "add", "."], cwd=rd, check=True)
    result = subprocess.run(["git", "status", "--porcelain"], cwd=rd, capture_output=True, text=True)
    if result.stdout.strip():
        subprocess.run(["git", "commit", "-m", message], cwd=rd, check=True)
        return f"Committed: {message}"
    return "Nothing to commit"


TOOL_FNS = {
    "list_files": lambda args: list_files(**args),
    "read_file": lambda args: read_file(**args),
    "write_file": lambda args: write_file(**args),
    "git_commit": lambda args: git_commit(**args),
}


def run_agent(category_dir: Path, output_path: Path, venue: str, year: int, category: str):
    insight_repo = output_path.parent.parent  # methodology_kb/paperinsight/

    user_msg = f"""Synthesize methodology insights for:
- Category directory: {category_dir}
- Index file: {output_path}
- References directory: {output_path.parent / 'references'}
- Venue: {venue}, Year: {year}, Category: {category}
- Git repo for committing: {insight_repo}

Complete the full task: list files, read all, synthesize, write index file + one references/{{slug}}.md per insight, git commit."""

    messages = [{"role": "user", "content": user_msg}]

    print(f"  Agent starting for {venue}-{year}/{category}...")
    while True:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": SYSTEM}] + messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        msg = resp.choices[0].message
        messages.append({"role": "assistant", "content": msg.content, "tool_calls": msg.tool_calls})

        if not msg.tool_calls:
            print(f"  Agent done.")
            break

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"    Tool: {fn_name}({list(args.keys())})")
            result = TOOL_FNS[fn_name](args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result)[:64000],
            })


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--venue", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--category", required=True)
    args = parser.parse_args()

    base = Path("methodology_kb")
    category_dir = base / f"{args.venue}-{args.year}" / args.category
    output_path = base / "paperinsight" / f"{args.venue}-{args.year}" / args.category / "insight.md"

    if not category_dir.exists():
        print(f"Category not found: {category_dir}")
        exit(1)

    run_agent(category_dir, output_path, args.venue, args.year, args.category)
    print(f"Done. Insight at {output_path}")
