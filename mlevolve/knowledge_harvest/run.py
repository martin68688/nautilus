"""Build a dated Numerai knowledge set: corpus -> items -> knowledge set.

Usage (from mlevolve/):
  python -m knowledge_harvest.run --out DIR --cutoff 2024-01-19
  python -m knowledge_harvest.run --out DIR --cutoff-from-train path/to/train.parquet

OPENAI_API_KEY / OPENAI_BASE_URL select the extraction model endpoint.
Every HTTP response and LLM answer is cached under DIR/cache, so a rerun with
the same inputs reproduces the same set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import consolidate, extract
from .forum import DiscourseForum, cutoff_datetime
from .http import CachedHttp
from .llm import JsonLLM
from .repos import repo_documents

FORUM_URL = "https://forum.numer.ai"
FORUM_CATEGORIES = {"announcements": 8, "data-science": 5, "tournament": 7, "council-of-elders": 12}
REPOS = ("numerai/example-scripts", "numerai/docs", "numerai/numerai-tools")
GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}


def cutoff_from_train(path: Path) -> str:
    """Day after the last training era: posts on the last era's day are kept."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    era = pq.read_table(path, columns=["era"]).column("era")
    last = datetime.fromisoformat(str(pc.max(era).as_py())[:10])
    return last.date().isoformat()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build_corpus(out: Path, cutoff: datetime, http: CachedHttp) -> tuple[list[dict], dict]:
    forum = DiscourseForum(FORUM_URL, http)
    topic_ids: list[tuple[int, str]] = []
    for slug, category_id in FORUM_CATEGORIES.items():
        topic_ids.extend((topic["id"], slug) for topic in forum.topics(slug, category_id, cutoff))
    documents = []
    for n, (topic_id, slug) in enumerate(sorted(set(topic_ids))):
        document = forum.topic_document(topic_id, cutoff, slug)
        if document:
            documents.append(document)
        if n % 50 == 0:
            print(f"forum topics {n}/{len(topic_ids)}", flush=True)
    commits = {}
    for repo in REPOS:
        commit, docs = repo_documents(http, repo, cutoff, git_dir=out / "cache" / "git")
        commits[repo] = commit
        documents.extend(docs)
    _write_jsonl(out / "documents.jsonl", documents)
    return documents, commits


def build_items(out: Path, documents: list[dict], llm: JsonLLM, workers: int) -> list[dict]:
    def one(document: dict) -> list[dict]:
        return extract.extract_document(document, llm)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, documents))
    items = [extract.item_record(item, n)
             for n, item in enumerate(item for per_doc in results for item in per_doc)]
    _write_jsonl(out / "items.jsonl", items)
    return items


def build_knowledge(items: list[dict], llm: JsonLLM, workers: int) -> list[dict]:
    by_category: dict[str, list[dict]] = {}
    for item in items:
        by_category.setdefault(item["category"], []).append(item)
    clusters = []
    dropped: dict[str, int] = {}
    for category in extract.CATEGORIES:
        raw = consolidate.cluster_category(by_category.get(category, []), llm)
        kinds = consolidate.label_kinds(raw, llm)
        for kind in kinds:
            if kind not in consolidate.KEEP_KINDS:
                dropped[kind] = dropped.get(kind, 0) + 1
        kept = [members for members, kind in zip(raw, kinds) if kind in consolidate.KEEP_KINDS]
        for members in consolidate.dedup_clusters(kept, llm):
            clusters.append((category, members))
    print(f"clusters kept {len(clusters)}, dropped {dropped}", flush=True)

    def one(entry: tuple[str, list[dict]]) -> dict:
        category, members = entry
        summary = consolidate.summarize(members, llm)
        evidence = [{**ev, "url": it["url"], "date": it["date"][:10], "authors": it.get("authors") or []}
                    for it in members for ev in it.get("evidence") or []]
        objections = [{**c, "url": it["url"], "date": it["date"][:10]}
                      for it in members for c in it.get("counter_evidence") or []]
        sources = sorted({(it["url"], it["doc_title"], it["date"][:10]) for it in members}, key=lambda s: s[2])
        return {
            "category": category,
            "title": summary.get("title") or members[0]["title"],
            "principle": summary.get("principle") or members[0]["principle"],
            "condition": summary.get("condition", ""),
            "caveats": summary.get("caveats") or [],
            "title_zh": summary.get("title_zh", ""),
            "summary_zh": summary.get("summary_zh", ""),
            **consolidate.grade(members),
            "evidence": evidence,
            "objections": objections,
            "sources": [{"url": u, "title": t, "date": d} for u, t, d in sources],
            "item_ids": [it["item_id"] for it in members],
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        entries = list(pool.map(one, clusters))
    entries.sort(key=lambda e: (GRADE_ORDER[e["grade"]], -e["n_sources"], e["category"], e["title"]))
    for n, entry in enumerate(entries, 1):
        entry["id"] = f"ks_{n:04d}"
    return entries


def render_markdown(entries: list[dict], manifest: dict) -> str:
    lines = [f"# Numerai knowledge set (sources before {manifest['cutoff']})", "",
             f"{len(entries)} methods from {manifest['n_documents']} documents.", ""]
    for e in entries:
        flag = " · disputed" if e["disputed"] else ""
        lines += [f"## {e['id']} [{e['grade']}{flag}] {e['title']}",
                  f"*{e['category']} · {e['stance']} · {e['n_sources']} sources · {e['n_authors']} authors"
                  f"{' · official' if e['official'] else ''}*", "", e["principle"], ""]
        if e["condition"]:
            lines += [f"When: {e['condition']}", ""]
        for ev in e["evidence"][:6]:
            lines.append(f"- {ev.get('claim', '')}: {ev.get('value', '')} ({ev.get('setting')}, {ev['date']}) — {ev['url']}")
        for ob in e["objections"][:3]:
            lines.append(f"- objection: {ob.get('claim', '')} — {ob['url']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cutoff", help="YYYY-MM-DD; sources dated on or after it are excluded")
    group.add_argument("--cutoff-from-train", type=Path)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stage", choices=["corpus", "items", "knowledge"], default="knowledge",
                        help="stop after this stage (earlier stages are reused when present)")
    args = parser.parse_args()

    cutoff = args.cutoff or cutoff_from_train(args.cutoff_from_train)
    cutoff_dt = cutoff_datetime(cutoff)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    http = CachedHttp(out / "cache" / "http")

    if (out / "documents.jsonl").exists() and (out / "commits.json").exists():
        documents = _read_jsonl(out / "documents.jsonl")
        commits = json.loads((out / "commits.json").read_text())
    else:
        documents, commits = build_corpus(out, cutoff_dt, http)
        (out / "commits.json").write_text(json.dumps(commits, indent=2))
    latest = max(p["created_at"] for d in documents for p in d["posts"])
    assert latest < cutoff_dt.isoformat(), f"source dated {latest} is not before cutoff {cutoff}"
    print(f"corpus: {len(documents)} documents, latest {latest}", flush=True)
    if args.stage == "corpus":
        return
    llm = JsonLLM(args.model, out / "cache" / "llm")

    items_path = out / "items.jsonl"
    items = _read_jsonl(items_path) if items_path.exists() else build_items(out, documents, llm, args.workers)
    print(f"items: {len(items)}", flush=True)
    if args.stage == "items":
        return

    entries = build_knowledge(items, llm, args.workers)
    manifest = {
        "schema": "mlevolve_knowledge_set_v1",
        "cutoff": cutoff,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": args.model,
        "prompts": [extract.PROMPT_VERSION, consolidate.PROMPT_VERSION],
        "forum": FORUM_URL,
        "forum_categories": FORUM_CATEGORIES,
        "repo_commits": commits,
        "n_documents": len(documents),
        "n_items": len(items),
        "n_entries": len(entries),
        "latest_source_time": latest,
        "documents_sha256": hashlib.sha256((out / "documents.jsonl").read_bytes()).hexdigest(),
    }
    (out / "knowledge_set.json").write_text(
        json.dumps({"manifest": manifest, "entries": entries}, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "knowledge_set.md").write_text(render_markdown(entries, manifest), encoding="utf-8")
    print(f"knowledge set: {len(entries)} entries -> {out / 'knowledge_set.json'}", flush=True)


if __name__ == "__main__":
    main()
