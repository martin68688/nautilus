"""Method-family tagging and a selection quota that keeps families alive.

Greedy, metric-only selection lets a weaker-but-different family (for example
a neural net next to tree ensembles) disappear after one bad attempt.  The
quota occasionally routes expansion to the best node of a family that has
dropped out of the current top-k, so it keeps receiving improve/repair steps.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any

logger = logging.getLogger("MLEvolve")

_FAMILY_PATTERNS = {
    # A network must be defined or instantiated; an unused ``import torch`` is not enough.
    "neural_net": [r"\(\s*(?:torch\.)?nn\.Module\s*\)", r"\bnn\.Sequential\(", r"\bkeras\.(?:Sequential|Model)\(",
                   r"\btf\.keras\.(?:Sequential|Model)\(", r"\bflax\.linen\b", r"\bMLP(?:Regressor|Classifier)\("],
    "gbdt": [r"\blightgbm\b", r"\bxgboost\b", r"\bcatboost\b", r"\bHistGradientBoosting\w*\b",
             r"\bGradientBoosting(?:Regressor|Classifier)\b"],
    "random_forest": [r"\bRandomForest(?:Regressor|Classifier)\b", r"\bExtraTrees(?:Regressor|Classifier)\b"],
    "linear": [r"\b(?:Ridge|Lasso|ElasticNet|LinearRegression|LogisticRegression|SGDRegressor|SGDClassifier)\b"],
    "kernel": [r"\bSV[RC]\b", r"\bKernelRidge\b"],
    "neighbors": [r"\bKNeighbors(?:Regressor|Classifier)\b"],
}
# A family whose component is printed as removed or broken is not "alive".
_REMOVED = re.compile(r"\b(removed|dropped|disabled|excluded)\b", re.IGNORECASE)
_NEURAL_NAMES = re.compile(r"\b\w*(MLP|Net|Transformer|LSTM|GRU|CNN)\w*\b")


def classify(code: str, term_out: str = "") -> list[str]:
    """Families a program actually trains, based on its source and its log."""
    found = [
        family for family, patterns in _FAMILY_PATTERNS.items()
        if any(re.search(p, code or "") for p in patterns)
    ]
    if "neural_net" in found:
        removed = any(
            _REMOVED.search(line) and _NEURAL_NAMES.search(line)
            for line in str(term_out or "").splitlines()
        )
        if removed:
            found.remove("neural_net")
    return found


def _cfg(agent: Any):
    return getattr(getattr(agent, "cfg", None), "family_quota", None)


def pick_underrepresented(agent: Any, top_k_nodes: list[Any]) -> Any | None:
    """Return the best expandable node of a family missing from the top-k, if any."""
    cfg = _cfg(agent)
    if not bool(getattr(cfg, "enabled", False)):
        return None
    if random.random() >= float(getattr(cfg, "probability", 0.3)):
        return None
    top_families = {f for node in top_k_nodes for f in (node.method_families or [])}
    maximize = agent.metric_maximize if agent.metric_maximize is not None else True
    for family in getattr(cfg, "families", []) or []:
        if family in top_families:
            continue
        pool = [
            node for nodes in agent.branch_all_nodes.values() for node in nodes
            if node.is_buggy is False
            and node.metric is not None and node.metric.value is not None
            and family in (node.method_families or [])
            and not node.is_terminal
            and not node.reached_child_limit(agent.scfg, for_topk=True)
        ]
        if not pool:
            continue
        chosen = sorted(pool, key=lambda n: n.metric.value, reverse=maximize)[0]
        logger.info(
            "🧬 Family quota: '%s' missing from top-%d; expanding its best node %s (metric=%.4f)",
            family, len(top_k_nodes), chosen.id, chosen.metric.value,
        )
        return chosen
    return None
