"""Per-document extraction of modelling methods and the evidence behind them.

The model sees one dated document at a time and may only report what that
document says. Every evidence quote is checked verbatim against the source;
quotes that are not found, and numbers not present in their quote, are
dropped, so outside or later knowledge from the extraction model cannot enter
the set as evidence.
"""

from __future__ import annotations

import re
from typing import Any, Callable

PROMPT_VERSION = "extract-v2"
MAX_CHUNK_CHARS = 30_000

CATEGORIES = (
    "data_and_eras", "targets", "features", "tree_models", "neural_networks",
    "ensembling", "neutralization_and_risk", "validation", "metrics_and_scoring",
    "compute_and_engineering", "other",
)
SETTINGS = ("live", "validation", "backtest", "in_sample", "unspecified")

SYSTEM = f"""You extract modelling knowledge for the Numerai stock-market tournament from ONE source document.
Use only what this document says. Never add outside knowledge, later developments, or your own opinions.

Return a JSON object: {{"relevant": bool, "items": [...]}}.
"relevant" is false when the document has no modelling content (payouts, staking mechanics, website issues, crypto, Numerai Signals only, social chat).
Each item is one concrete modelling method, practice or warning stated in the document:
{{
  "title": short imperative phrase, e.g. "Train separate models on auxiliary targets and average them",
  "category": one of {list(CATEGORIES)},
  "principle": 1-3 sentences: what to do and how, with the concrete parameters the document gives,
  "condition": when it applies (data version, target, compute limits) as stated; "" if not stated,
  "stance": "recommend" | "caution" (the document warns against it or reports it failing) | "neutral",
  "certainty": "measured" (the author reports their own experiment with numbers) | "experience" (the author reports
               their practice or observation without numbers) | "speculative" (the author guesses, asks, or hedges),
  "evidence": [{{"claim": what was measured, "metric": e.g. "mean corr", "value": number as written, "baseline_value": number or "",
                "setting": one of {list(SETTINGS)}, "data_version": as stated or "",
                "quote": verbatim excerpt (<= 300 chars) from the document that contains the number}}],
  "counter_evidence": [{{"claim": an objection or failed replication in this document, "author": username, "quote": verbatim excerpt}}],
  "post_numbers": [post numbers that state the item],
  "authors": [usernames]
}}
Rules: quotes must be copied exactly from the document text. Report numbers only when the document gives them.
Keep the author's own qualifiers ("not enough time to conclude", "just a guess", "AFAIK") in the principle; never turn a
hedge into a rule. "setting" is "live" only for tournament live results, "validation" only for a held-out period the
document names as validation; a random row split, an in-sample fit or a count of anything is not validation.
If a later post in this document revises or contradicts an item, put that post under counter_evidence.
Skip trivial items (installing packages, file paths). Merge repeated statements of the same method within this document.
"""


def render(document: dict) -> list[str]:
    """Render a document as text chunks of whole posts (a long post is split)."""
    header = f"Title: {document['title']}\nSource: {document['url']}\nDate: {document['created_at'][:10]}\n"
    blocks = []
    for post in document["posts"]:
        role = " (Numerai staff)" if post["staff"] else ""
        head = f"\n--- post #{post['post_number']} by {post['author']}{role}, {post['created_at'][:10]}, likes {post['likes']} ---\n"
        text = post["text"]
        for start in range(0, len(text), MAX_CHUNK_CHARS):
            blocks.append(head + text[start:start + MAX_CHUNK_CHARS])
    chunks, current = [], header
    for block in blocks:
        if len(current) + len(block) > MAX_CHUNK_CHARS and current != header:
            chunks.append(current)
            current = header + "(continued)\n"
        current += block
    chunks.append(current)
    return chunks


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _numbers(text: str) -> set[str]:
    return {m.lstrip("0") or "0" for m in re.findall(r"\d+(?:\.\d+)?", text or "")}


def verify_item(item: dict, source_text: str, posts: list[dict] | None = None) -> dict:
    """Keep only evidence whose quote is in the source and contains its value.

    Each kept quote is attributed to the post that contains it, so evidence
    counts its actual author, not everyone who took part in the thread.
    """
    source = _norm(source_text)
    normed_posts = [(_norm(p["text"]), p) for p in posts or []]
    kept, dropped = [], 0
    for ev in item.get("evidence") or []:
        quote = _norm(str(ev.get("quote", "")))
        value = str(ev.get("value", ""))
        if quote and quote in source and (_numbers(value) & _numbers(quote) or not _numbers(value)):
            if ev.get("setting") not in SETTINGS:
                ev["setting"] = "unspecified"
            post = next((p for text, p in normed_posts if quote in text), None)
            if post is not None:
                ev["author"] = post["author"]
                ev["post_date"] = post["created_at"][:10]
            kept.append(ev)
        else:
            dropped += 1
    counters = [c for c in item.get("counter_evidence") or []
                if _norm(str(c.get("quote", ""))) and _norm(str(c.get("quote", ""))) in source]
    item["evidence"] = kept
    item["counter_evidence"] = counters
    item["evidence_dropped_unverified"] = dropped
    if item.get("category") not in CATEGORIES:
        item["category"] = "other"
    if item.get("certainty") not in ("measured", "experience", "speculative"):
        item["certainty"] = "experience"
    return item


def extract_document(document: dict, llm: Callable[[str, str], dict]) -> list[dict]:
    items: list[dict] = []
    posts_by_number = {post["post_number"]: post for post in document["posts"]}
    for chunk_index, chunk in enumerate(render(document)):
        result = llm(SYSTEM, chunk)
        if not result.get("relevant", True):
            continue
        for item in result.get("items") or []:
            if not isinstance(item, dict) or not item.get("title") or not item.get("principle"):
                continue
            item = verify_item(item, chunk, document["posts"])
            numbers = [int(n) for n in item.get("post_numbers") or []
                       if str(n).isdigit() and int(n) in posts_by_number]
            item.update({
                "doc_id": document["doc_id"],
                "url": document["url"],
                "doc_title": document["title"],
                "source": document["source"],
                "chunk": chunk_index,
                "date": min((posts_by_number[n]["created_at"] for n in numbers), default=document["created_at"]),
                "staff": any(posts_by_number[n]["staff"] for n in numbers) or document["source"] == "repo",
                "likes": sum(posts_by_number[n]["likes"] for n in numbers),
            })
            items.append(item)
    return items


def item_record(item: dict, index: int) -> dict[str, Any]:
    return {"item_id": f"it{index:05d}", **item}
