"""Hypothesis -> experiment -> conclusion loop over improvement attempts.

Every improvement child is one experiment in a research direction chosen by a
UCB bandit over earlier outcomes in this run (R&D-Agent style). The child must
state its direction and a falsifiable hypothesis; once the Host has scored it,
the attempt is recorded as confirmed (beat the node it started from), refuted
(scored no better) or failed (no working code even after repair). The ledger
is shown to later attempts across all branches and saved to the run logs.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

DEFAULT_DIRECTIONS = {
    "targets": "which labels to train on and how to combine them",
    "features": "which inputs to use and how to transform or construct them",
    "model": "model family, architecture or capacity",
    "training": "objective, hyperparameters, sample weighting and data selection",
    "ensemble": "combining several models or predictions",
    "post_processing": "transforming predictions after the model",
}

_DIRECTION_RE = re.compile(r"DIRECTION\s*[:=]\s*`?([A-Za-z_]+)", re.I)
_HYPOTHESIS_RE = re.compile(r"HYPOTHESIS\s*[:=]\s*(.+)", re.I)


def _origin(node: Any) -> Any:
    origin = node.parent
    while origin is not None and origin.is_buggy:
        origin = origin.parent
    return origin


def _attempt(node: Any) -> Any:
    """The improvement node that started this attempt (a repair inherits it)."""
    attempt = node
    while attempt is not None and attempt.stage == "debug":
        attempt = attempt.parent
    return attempt


def _score(node: Any) -> float | None:
    metric = getattr(node, "metric", None)
    return None if metric is None or metric.value is None else float(metric.value)


class ResearchLedger:
    def __init__(self, directions: dict[str, str], ucb_c: float = 1.0, shown: int = 10,
                 path: Path | None = None):
        self.directions = dict(directions)
        self.ucb_c = ucb_c
        self.shown = shown
        self.path = path
        self.records: dict[str, dict] = {}

    # ---- scheduling ------------------------------------------------------
    def choose(self) -> str:
        """UCB1 over directions; reward = attempt confirmed. Untried first."""
        counts = {d: 0 for d in self.directions}
        wins = {d: 0 for d in self.directions}
        for rec in self.records.values():
            if rec["direction"] in counts and rec["verdict"] != "pending":
                counts[rec["direction"]] += 1
                wins[rec["direction"]] += rec["verdict"] == "confirmed"
        untried = [d for d in self.directions if counts[d] == 0]
        if untried:
            return untried[0]
        total = sum(counts.values())
        return max(self.directions, key=lambda d: wins[d] / counts[d]
                   + self.ucb_c * math.sqrt(math.log(total) / counts[d]))

    # ---- prompt ----------------------------------------------------------
    def prompt_text(self, direction: str) -> str:
        names = ", ".join(self.directions)
        lines = [
            "## Research loop",
            f"This child is one controlled experiment in the research direction `{direction}` "
            f"({self.directions[direction]}). The direction was chosen from the outcomes of earlier experiments.",
            "Begin your plan with exactly these two lines:",
            f"DIRECTION: {direction}   (or another of: {names}, only if this direction cannot be tested on the current solution)",
            "HYPOTHESIS: one falsifiable sentence: the single change you make and the Host score change you expect",
            "Change only what the hypothesis needs, so the Host score tells whether it holds.",
        ]
        done = [r for r in self.records.values() if r["verdict"] != "pending"]
        if done:
            lines.append("")
            lines.append("Experiment ledger of this run (all branches): direction tried / confirmed")
            for name in self.directions:
                tried = [r for r in done if r["direction"] == name]
                if tried:
                    ok = sum(r["verdict"] == "confirmed" for r in tried)
                    lines.append(f"- {name}: {len(tried)} / {ok}")
            lines.append("Latest conclusions:")
            for rec in done[-self.shown:]:
                delta = "" if rec["delta"] is None else f" (Host score {rec['delta']:+.5f} vs its start)"
                lines.append(f"- [{rec['verdict']}] {rec['direction']}: {rec['hypothesis']}{delta}")
        return "\n".join(lines)

    # ---- recording -------------------------------------------------------
    def register(self, node: Any, scheduled: str) -> None:
        plan = getattr(node, "plan", "") or ""
        found = _DIRECTION_RE.search(plan)
        direction = found.group(1).lower() if found and found.group(1).lower() in self.directions else scheduled
        hyp = _HYPOTHESIS_RE.search(plan)
        hypothesis = (hyp.group(1).strip() if hyp else plan.strip().splitlines()[0] if plan.strip() else "")[:240]
        node.research_direction = direction
        self.records[node.id] = {
            "node_id": node.id, "direction": direction, "scheduled": scheduled,
            "hypothesis": hypothesis, "verdict": "pending", "delta": None,
            "origin_id": getattr(_origin(node), "id", None),
        }
        self._save()

    def conclude(self, node: Any, threshold: float, maximize: bool = True) -> None:
        """Record the outcome of a scored attempt or of its repair."""
        attempt = _attempt(node)
        rec = self.records.get(getattr(attempt, "id", None))
        if rec is None:
            return
        if node.is_buggy:
            # A failed attempt may still be repaired; it stays "failed" unless a repair works.
            if rec["verdict"] == "pending":
                rec["verdict"] = "failed"
        else:
            origin = _origin(node)
            start, end = (_score(origin) if origin is not None else None), _score(node)
            if start is not None and end is not None:
                delta = end - start if maximize else start - end
                rec["delta"] = delta
                rec["verdict"] = "confirmed" if delta > threshold else "refuted"
            rec["result_node_id"] = node.id
        self._save()

    def _save(self) -> None:
        if self.path is None:
            return
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(list(self.records.values()), indent=1), encoding="utf-8")
        tmp.replace(self.path)


def ledger_for(agent: Any) -> ResearchLedger | None:
    scfg = getattr(agent, "scfg", None)
    if not getattr(scfg, "research_loop", False):
        return None
    ledger = getattr(agent, "research_ledger", None)
    if ledger is None:
        directions = dict(getattr(scfg, "research_directions", None) or DEFAULT_DIRECTIONS)
        log_dir = getattr(getattr(agent, "cfg", None), "log_dir", None)
        ledger = ResearchLedger(
            directions,
            ucb_c=float(getattr(scfg, "research_ucb_c", 1.0)),
            shown=int(getattr(scfg, "research_ledger_shown", 10)),
            path=Path(log_dir) / "research_ledger.json" if log_dir else None,
        )
        agent.research_ledger = ledger
    return ledger
