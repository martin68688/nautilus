"""Verified public knowledge for draft and improve prompts.

Reads a knowledge set built by ``knowledge_harvest`` (sources dated before the
split's last training era) whose entries carry a ``check`` result from
``knowledge_harvest.verify``: the method was tried against a fixed baseline on
held-out training eras. Entries are offered as hypotheses with that result,
never as rules, and each shown entry id is logged for adoption tracking.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger("MLEvolve")

CATEGORY_DIRECTION = {
    "targets": "targets",
    "features": "features",
    "tree_models": "model",
    "neural_networks": "model",
    "ensembling": "ensemble",
    "neutralization_and_risk": "post_processing",
    "data_and_eras": "training",
}
_VERDICT_ORDER = {"helps": 0, "untested": 1, "no_effect": 2, "hurts": 3, "failed": 4}


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _check_line(entry: dict) -> str:
    check = entry.get("check") or {}
    verdict = check.get("verdict", "untested")
    if verdict == "untested":
        return "Training-period check: not run."
    if verdict == "failed":
        return "Training-period check: the test implementation failed; no result."
    return (
        f"Training-period check: {verdict.replace('_', ' ')} "
        f"(CORR {check.get('diff', 0):+.4f} vs baseline, t={check.get('t', 0):.1f}, "
        f"{check.get('eras', '?')} held-out training eras)."
    )


def select_entries(data: dict, direction: str | None, limit: int, include_failed_checks: bool = True) -> list[dict]:
    entries = [e for e in data.get("entries", []) if e.get("stance") != "caution"
               and CATEGORY_DIRECTION.get(e.get("category"))]
    if direction is not None:
        entries = [e for e in entries if CATEGORY_DIRECTION[e["category"]] == direction]
    def key(e: dict) -> tuple:
        verdict = (e.get("check") or {}).get("verdict", "untested")
        return (_VERDICT_ORDER.get(verdict, 5), "ABCD".index(e.get("grade", "D")), -e.get("n_sources", 0))
    entries.sort(key=key)
    chosen = [e for e in entries if (e.get("check") or {}).get("verdict", "untested") in ("helps", "untested")][:limit]
    if include_failed_checks:
        # One entry that did not pay off, so the agent does not re-test it blindly.
        tried = [e for e in entries if (e.get("check") or {}).get("verdict") in ("no_effect", "hurts")]
        if tried and len(chosen) < limit + 1:
            chosen.append(tried[0])
    return chosen


def render(data: dict, entries: list[dict]) -> str:
    cutoff = (data.get("manifest") or {}).get("cutoff", "?")
    lines = [
        f"## Public Numerai knowledge (forum and official docs dated before {cutoff})",
        "Hypotheses collected from public sources, each with the result of a quick check on held-out "
        "training eras against a fixed LightGBM baseline. A check result is evidence, not a guarantee; "
        "test any idea you adopt through the Host score.",
    ]
    for e in entries:
        lines.append(f"- [{e['id']}] {e['title']}: {e['principle']}")
        if e.get("condition"):
            lines.append(f"  When: {e['condition']}")
        lines.append(f"  {_check_line(e)}")
    return "\n".join(lines)


def knowledge_prompt(agent: Any, stage: str, direction: str | None) -> tuple[str, list[str]]:
    cfg = getattr(getattr(agent, "cfg", None), "knowledge", None)
    if not getattr(cfg, "enabled", False) or stage not in tuple(getattr(cfg, "stages", ("draft", "improve")) or ()):
        return "", []
    try:
        data = _load(str(cfg.path))
    except (OSError, ValueError) as error:
        logger.warning("Knowledge set unavailable (%s): %s", cfg.path, error)
        return "", []
    entries = select_entries(data, direction, int(getattr(cfg, "max_entries", 4)))
    if not entries:
        return "", []
    return render(data, entries), [e["id"] for e in entries]
