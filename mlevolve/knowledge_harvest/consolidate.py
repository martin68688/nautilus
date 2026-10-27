"""Merge extracted items into methods and grade the evidence behind each.

Clustering and wording are done by the LLM from the extracted items only; the
grade is computed here from the verified evidence, never by the LLM.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Callable

PROMPT_VERSION = "consolidate-v2"
_BATCH = 70

CLUSTER_SYSTEM = """You group extracted Numerai modelling items that describe the same method.
Input: a JSON list of {"id", "title", "principle"}. Items describe the same method when following one
would mean doing the same thing (same technique, possibly different parameters).
Return {"clusters": [{"title": canonical short imperative title, "ids": [item ids]}]}.
Every input id must appear in exactly one cluster. Do not invent methods."""

MERGE_SYSTEM = """You merge groups of Numerai modelling methods that are the same method under different titles.
Input: a JSON list of {"key", "title"}. Return {"groups": [{"title": canonical title, "keys": [keys]}]}.
Every key must appear in exactly one group. Only merge true duplicates."""

KIND_SYSTEM = """You label candidate entries of a Numerai modelling knowledge base.
Input: a JSON list of {"key", "title", "principle"}. Label each key with one kind:
"modelling": how to build, train, combine or post-process models (data handling, targets, features, models, ensembles, neutralization);
"evaluation": how to validate or measure a model offline (validation design, metrics a modeller computes);
"mechanics": tournament payouts, staking, stake allocation, submissions, account or website matters, gaming the payout rules;
"generic": advice too vague to act on (e.g. "avoid overfitting", "use cross-validation") without any Numerai-specific detail.
Return a JSON object {"labels": [{"key": key, "kind": kind}]} covering every key."""

DEDUP_SYSTEM = """You merge duplicate entries of a Numerai modelling knowledge base.
Input: a JSON list of {"key", "title", "principle"}. Two entries are duplicates when they recommend the same
underlying technique (e.g. several entries about neutralizing predictions to features are one technique, even
with different proportions or feature subsets). Keep genuinely different techniques apart.
Return a JSON object {"groups": [{"title": canonical title, "keys": [keys]}]} covering every key exactly once."""

KEEP_KINDS = ("modelling", "evaluation")
_LABEL_BATCH = 60
_DEDUP_BATCH = 150


def _keyed(clusters: list[list[dict]]) -> list[dict]:
    return [{"key": f"c{n}", "title": members[0]["title"], "principle": members[0]["principle"][:300]}
            for n, members in enumerate(clusters)]


def label_kinds(clusters: list[list[dict]], llm: Callable[[str, str], dict]) -> list[str]:
    """Kind of each cluster; unlabeled clusters count as modelling (kept)."""
    keyed = _keyed(clusters)
    kinds = ["modelling"] * len(clusters)
    for start in range(0, len(keyed), _LABEL_BATCH):
        result = llm(KIND_SYSTEM, json.dumps(keyed[start:start + _LABEL_BATCH], ensure_ascii=False))
        for label in result.get("labels") or []:
            key = str(label.get("key", ""))
            if key[1:].isdigit() and int(key[1:]) < len(kinds):
                kinds[int(key[1:])] = label.get("kind") or "modelling"
    return kinds


def dedup_clusters(clusters: list[list[dict]], llm: Callable[[str, str], dict]) -> list[list[dict]]:
    """Merge clusters that describe the same technique (one category at a time)."""
    keyed = _keyed(clusters)
    merged: list[list[dict]] = []
    used: set[int] = set()
    for start in range(0, len(keyed), _DEDUP_BATCH):
        result = llm(DEDUP_SYSTEM, json.dumps(keyed[start:start + _DEDUP_BATCH], ensure_ascii=False))
        for group in result.get("groups") or []:
            members: list[dict] = []
            for key in group.get("keys") or []:
                key = str(key)
                if key[1:].isdigit() and int(key[1:]) < len(clusters) and int(key[1:]) not in used:
                    used.add(int(key[1:]))
                    members.extend(clusters[int(key[1:])])
            if members:
                merged.append(members)
    merged.extend(clusters[n] for n in range(len(clusters)) if n not in used)
    return merged


SUMMARY_SYSTEM = """You write one entry of a knowledge base about the Numerai tournament from the source items given.
Use only the items. Keep concrete parameters and conditions. State disagreements if items disagree.
Return a JSON object {"title": short imperative English title,
        "principle": 2-4 English sentences on what to do and how,
        "condition": English, when it applies or "",
        "caveats": [short English strings],
        "title_zh": Chinese title, "summary_zh": 2-3 Chinese sentences for a non-expert reader}."""


def _json_items(items: list[dict]) -> list[dict]:
    return [{"id": it["item_id"], "title": it["title"], "principle": it["principle"][:400]} for it in items]


def cluster_category(items: list[dict], llm: Callable[[str, str], dict]) -> list[list[dict]]:
    by_id = {it["item_id"]: it for it in items}
    groups: list[tuple[str, list[str]]] = []
    for start in range(0, len(items), _BATCH):
        batch = items[start:start + _BATCH]
        result = llm(CLUSTER_SYSTEM, json.dumps(_json_items(batch), ensure_ascii=False))
        assigned: set[str] = set()
        for cluster in result.get("clusters") or []:
            ids = [i for i in cluster.get("ids") or [] if i in by_id and i not in assigned]
            assigned.update(ids)
            if ids:
                groups.append((cluster.get("title") or by_id[ids[0]]["title"], ids))
        for it in batch:  # anything the model skipped becomes its own cluster
            if it["item_id"] not in assigned:
                groups.append((it["title"], [it["item_id"]]))
    if len(items) > _BATCH:
        keyed = [{"key": f"g{n}", "title": title} for n, (title, _) in enumerate(groups)]
        result = llm(MERGE_SYSTEM, json.dumps(keyed, ensure_ascii=False))
        merged, used = [], set()
        for group in result.get("groups") or []:
            ids = []
            for key in group.get("keys") or []:
                if key.startswith("g") and key[1:].isdigit() and int(key[1:]) < len(groups) and key not in used:
                    used.add(key)
                    ids.extend(groups[int(key[1:])][1])
            if ids:
                merged.append(ids)
        merged.extend(groups[n][1] for n in range(len(groups)) if f"g{n}" not in used)
        return [[by_id[i] for i in ids] for ids in merged]
    return [[by_id[i] for i in ids] for _, ids in groups]


def grade(members: list[dict]) -> dict:
    """Evidence grade from verified evidence.

    A: out-of-sample (live/validation/backtest) numbers from >= 2 independent authors
    B: out-of-sample numbers from one author
    C: no out-of-sample numbers, but stated by Numerai staff/official repos or >= 3 authors
    D: everything else (single unmeasured opinion, in-sample numbers only)
    """
    settings_by_author: dict[str, set[str]] = defaultdict(set)
    for it in members:
        for ev in it.get("evidence") or []:
            for author in it.get("authors") or ["?"]:
                settings_by_author[author].add(ev.get("setting", "unspecified"))
    authors = {a for it in members for a in it.get("authors") or []}
    oos_authors = {a for a, s in settings_by_author.items() if s & {"live", "validation", "backtest"}}
    staff = any(it.get("staff") for it in members)
    if len(oos_authors) >= 2:
        level = "A"
    elif oos_authors:
        level = "B"
    elif staff or len(authors) >= 3:
        level = "C"
    else:
        level = "D"
    stances = [it.get("stance") for it in members]
    return {
        "grade": level,
        "disputed": any(it.get("counter_evidence") for it in members)
        or ("caution" in stances and "recommend" in stances),
        "stance": max(set(stances), key=stances.count) if stances else "neutral",
        "n_sources": len({it["doc_id"] for it in members}),
        "n_authors": len(authors),
        "official": staff,
    }


def summarize(members: list[dict], llm: Callable[[str, str], dict]) -> dict:
    payload = [{
        "title": it["title"], "principle": it["principle"], "condition": it.get("condition", ""),
        "stance": it.get("stance"), "evidence": [ev.get("claim", "") + f" = {ev.get('value', '')} ({ev.get('setting')})"
                                                  for ev in it.get("evidence") or []],
        "objections": [c.get("claim", "") for c in it.get("counter_evidence") or []],
    } for it in sorted(members, key=lambda it: (-len(it.get("evidence") or []), -it.get("likes", 0)))[:40]]
    return llm(SUMMARY_SYSTEM, json.dumps(payload, ensure_ascii=False))
