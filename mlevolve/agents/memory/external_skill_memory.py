"""Runtime external SkillGraph memory for MLEvolve.

This layer is intentionally separate from GlobalMemoryLayer:
- GlobalMemoryLayer stores records produced inside the current run.
- ExternalSkillMemoryLayer reads a pre-built, run-independent skill graph.

The layer returns two things for every node-generation prompt:
1. human-readable SOP guidance to inject into the prompt;
2. side-channel node ids for adoption tracking.
"""

from __future__ import annotations

import collections
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("MLEvolve")


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _mlevolve_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_graph_path(graph_path: str | Path) -> Path:
    """Resolve graph paths robustly from repo root, mlevolve root, or cwd."""
    p = Path(graph_path)
    if p.is_absolute():
        return p

    candidates = [
        Path.cwd() / p,
        _mlevolve_root() / p,
        _repo_root() / p,
        _repo_root() / "mlevolve" / p,
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return (_mlevolve_root() / p).resolve()


def read_skillgraph_node_text(ref_id: str, graph_path: str | Path) -> str:
    """Fetch one graph node as text for post-run adoption analysis."""
    path = resolve_graph_path(graph_path)
    if not path.exists():
        return ""
    try:
        graph = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    for node in graph.get("nodes", []):
        if node.get("id") == ref_id:
            return _node_to_text(node, include_stats=False)
    return ""


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")}


def _node_text_for_scoring(node: dict[str, Any]) -> str:
    return " ".join(
        str(node.get(k, ""))
        for k in ("title", "principle", "condition", "category", "scope")
    )


def _node_to_text(node: dict[str, Any], include_stats: bool = True) -> str:
    parts = [
        f"Title: {node.get('title', '')}",
        f"When: {node.get('condition', '')}",
        f"How: {node.get('principle', '')}",
        f"Type: {node.get('category', '')} / {node.get('scope', '')}",
    ]
    if include_stats:
        parts.append(
            "Evidence: "
            f"n_use={node.get('n_use', 0)}, "
            f"n_succ={node.get('n_succ', 0)}, "
            f"p_hat={node.get('p_hat', 0.0)}"
        )
    return "\n".join(parts)


class ExternalSkillMemoryLayer:
    """Query-time retriever over a pre-built SkillGraph JSON file."""

    def __init__(
        self,
        graph_path: str,
        source_name: str = "skillgraph",
        top_k: int = 8,
        depth: int = 2,
        beam_width: int = 3,
        general_cap: int = 2,
        task_seed_limit: int = 6,
        max_chars: int = 5000,
        include_draft: bool = True,
        include_improve: bool = True,
        include_evolution: bool = True,
        include_debug: bool = True,
        include_fusion: bool = True,
    ) -> None:
        self.graph_path = resolve_graph_path(graph_path)
        self.source_name = source_name or "skillgraph"
        self.top_k = top_k
        self.depth = depth
        self.beam_width = beam_width
        self.general_cap = general_cap
        self.task_seed_limit = task_seed_limit
        self.max_chars = max_chars
        self.enabled_stages = {
            "draft": include_draft,
            "improve": include_improve,
            "evolution": include_evolution,
            "debug": include_debug,
            "fusion": include_fusion,
            "multi_fusion": include_fusion,
            "fusion_draft": include_fusion,
            "aggregation": include_fusion,
        }

        self.graph: dict[str, Any] = {}
        self.nodes: dict[str, dict[str, Any]] = {}
        self.out_edges: dict[str, list[tuple[str, str, float]]] = collections.defaultdict(list)
        self.in_edges: dict[str, list[tuple[str, str, float]]] = collections.defaultdict(list)
        self._node_tokens: dict[str, set[str]] = {}
        self._load()

    def _load(self) -> None:
        if not self.graph_path.exists():
            raise FileNotFoundError(f"SkillGraph file not found: {self.graph_path}")
        self.graph = json.loads(self.graph_path.read_text(encoding="utf-8"))
        self.nodes = {n["id"]: n for n in self.graph.get("nodes", [])}
        for edge in self.graph.get("edges", []):
            src, dst = edge.get("src"), edge.get("dst")
            if src not in self.nodes or dst not in self.nodes:
                continue
            kind = edge.get("kind", "")
            weight = float(edge.get("weight", 1.0))
            self.out_edges[src].append((dst, kind, weight))
            self.in_edges[dst].append((src, kind, weight))
        self._node_tokens = {
            nid: _tokenize(_node_text_for_scoring(node))
            for nid, node in self.nodes.items()
        }
        logger.info(
            "[ExternalSkillMemory] loaded %s nodes / %s edges from %s",
            len(self.nodes),
            len(self.graph.get("edges", [])),
            self.graph_path,
        )

    def stage_enabled(self, stage: str) -> bool:
        return bool(self.enabled_stages.get(stage, False))

    def retrieve_for_node(
        self,
        *,
        stage: str,
        task_id: str,
        task_desc: str,
        query_parts: list[str] | None = None,
    ) -> tuple[str, list[str]]:
        if not self.stage_enabled(stage):
            return "", []
        if not self.nodes:
            return "", []

        task_type = self._resolve_task_type(task_id=task_id, task_desc=task_desc)
        if not task_type:
            logger.info("[ExternalSkillMemory] no task category matched graph; skipping")
            return "", []

        query_text = "\n".join([task_desc or "", *(query_parts or [])])
        result_ids = self._retrieve_ids(task_type=task_type, query_text=query_text)
        if not result_ids:
            return "", []

        text = self._format_context(result_ids, task_type=task_type, stage=stage)
        if self.max_chars > 0 and len(text) > self.max_chars:
            text = text[: self.max_chars].rstrip() + "\n... (external memory truncated)"
        logger.info(
            "[ExternalSkillMemory] stage=%s task=%s retrieved=%s ids=%s",
            stage,
            task_type,
            len(result_ids),
            ",".join(result_ids),
        )
        return text, result_ids

    def _resolve_task_type(self, *, task_id: str, task_desc: str) -> str:
        categories = sorted(
            {n.get("category", "") for n in self.nodes.values() if n.get("category") != "general"}
        )
        if not categories:
            return ""
        if task_id in categories:
            return task_id

        lowered = (task_desc or "").lower()
        for cat in categories:
            if cat and cat.lower() in lowered:
                return cat

        task_tokens = _tokenize((task_id or "") + " " + (task_desc or ""))
        best_cat, best_score = "", 0.0
        for cat in categories:
            cat_tokens = _tokenize(cat.replace("-", " "))
            if not cat_tokens:
                continue
            score = len(task_tokens & cat_tokens) / math.sqrt(len(cat_tokens))
            if score > best_score:
                best_cat, best_score = cat, score
        return best_cat if best_score > 0 else ""

    def _node_score(self, nid: str, query_tokens: set[str]) -> float:
        node = self.nodes[nid]
        toks = self._node_tokens.get(nid, set())
        if not toks:
            lexical = 0.0
        else:
            lexical = len(query_tokens & toks) / math.sqrt(len(toks))
        title_tokens = _tokenize(str(node.get("title", "")))
        title_boost = 0.25 * len(query_tokens & title_tokens)
        stats = (
            0.15 * float(node.get("p_hat", 0.0))
            + 0.03 * math.log1p(float(node.get("n_use", 0)))
            - 0.02 * float(node.get("level", 0))
        )
        return lexical + title_boost + stats

    def _seed_select(self, task_type: str, query_text: str) -> list[str]:
        generals = sorted(
            (nid for nid, n in self.nodes.items() if n.get("category") == "general"),
            key=lambda nid: (
                -float(self.nodes[nid].get("p_hat", 0.0)),
                -float(self.nodes[nid].get("n_use", 0)),
                self.nodes[nid].get("title", ""),
            ),
        )
        if self.general_cap >= 0:
            generals = generals[: self.general_cap]

        task_nodes = [nid for nid, n in self.nodes.items() if n.get("category") == task_type]
        query_tokens = _tokenize(query_text)
        task_nodes = sorted(
            task_nodes,
            key=lambda nid: (
                -self._node_score(nid, query_tokens),
                -float(self.nodes[nid].get("p_hat", 0.0)),
                -float(self.nodes[nid].get("n_use", 0)),
                self.nodes[nid].get("level", 0),
                self.nodes[nid].get("title", ""),
            ),
        )
        if self.task_seed_limit > 0:
            task_nodes = task_nodes[: self.task_seed_limit]
        return generals + task_nodes

    def _backward_bfs(self, seeds: list[str]) -> set[str]:
        found: set[str] = set()
        frontier = list(seeds)
        for _ in range(max(0, self.depth)):
            nxt: list[str] = []
            for dst in frontier:
                for src, kind, _weight in self.in_edges.get(dst, []):
                    if kind == "prereq" and src not in seeds and src not in found:
                        found.add(src)
                        nxt.append(src)
            if not nxt:
                break
            frontier = nxt
        return found

    def _forward_beam(self, seeds: list[str], task_type: str) -> tuple[set[str], dict[str, float]]:
        sigma = {s: 1.0 for s in seeds}
        found: set[str] = set()
        frontier = list(seeds)
        for _ in range(3):
            candidates: list[str] = []
            for src in frontier:
                for dst, kind, weight in self.out_edges.get(src, []):
                    cat = self.nodes[dst].get("category")
                    if cat not in ("general", task_type):
                        continue
                    prop = sigma.get(src, 0.0) * weight
                    if prop > sigma.get(dst, 0.0):
                        sigma[dst] = prop
                    if dst not in seeds and dst not in found:
                        candidates.append(dst)
            if not candidates:
                break
            top = sorted(set(candidates), key=lambda nid: -sigma.get(nid, 0.0))[: self.beam_width]
            found.update(top)
            frontier = top
        return found, sigma

    def _retrieve_ids(self, task_type: str, query_text: str) -> list[str]:
        seeds = self._seed_select(task_type, query_text)
        if not seeds:
            return []
        bfs = self._backward_bfs(seeds)
        beam, sigma = self._forward_beam(seeds, task_type)
        sigma.update({s: 1.0 for s in seeds})
        union = set(seeds) | bfs | beam

        generals = sorted(
            [nid for nid in union if self.nodes[nid].get("category") == "general"],
            key=lambda nid: -sigma.get(nid, 0.0),
        )
        tasks = sorted(
            [nid for nid in union if self.nodes[nid].get("category") != "general"],
            key=lambda nid: (
                self.nodes[nid].get("level", 0),
                -self._node_score(nid, _tokenize(query_text)),
                -sigma.get(nid, 0.0),
                self.nodes[nid].get("title", ""),
            ),
        )
        if self.general_cap >= 0:
            selected = generals[: self.general_cap] + tasks
        else:
            selected = generals + tasks
        return selected[: self.top_k]

    def _format_context(self, ids: list[str], *, task_type: str, stage: str) -> str:
        lines = [
            "These SOP memories come from a persistent external skill graph, not from this run.",
            "Use them only when their WHEN condition matches the current code, data, and error state.",
            "Prefer concrete applicable SOPs; keep conflicting or mismatched SOPs as warnings.",
            f"Matched task type: {task_type}; generation stage: {stage}.",
            "",
        ]
        for idx, nid in enumerate(ids, 1):
            node = self.nodes[nid]
            lines.append(f"Memory {idx}:")
            lines.append(_node_to_text(node, include_stats=True))
            lines.append("")
        return "\n".join(lines).strip()


def fetch_external_skill_memory(agent: Any, stage: str, **kwargs: Any) -> tuple[str, list[str], str]:
    """Small agent-facing helper.

    Returns (prompt_text, ref_ids, source_name).
    """
    layer = getattr(agent, "external_skill_memory", None)
    if layer is None:
        return "", [], "skillgraph"
    try:
        text, ref_ids = layer.retrieve_for_node(
            stage=stage,
            task_id=getattr(agent.cfg, "exp_id", "") or "",
            task_desc=getattr(agent, "task_desc", "") or "",
            query_parts=[str(v) for v in kwargs.values() if v],
        )
        return text, ref_ids, layer.source_name
    except Exception as exc:
        logger.warning("[ExternalSkillMemory] retrieval failed: %s", exc)
        return "", [], getattr(layer, "source_name", "skillgraph")
