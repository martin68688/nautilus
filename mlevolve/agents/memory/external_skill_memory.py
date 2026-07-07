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
import hashlib
import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger("MLEvolve")


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}")
_OPPOSING_TERMS = (
    ("large", "small"),
    ("larger", "smaller"),
    ("complex", "simple"),
    ("full", "partial"),
    ("increase", "reduce"),
    ("more", "less"),
    ("enable", "disable"),
    ("use", "avoid"),
    ("freeze", "unfreeze"),
)


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


def resolve_memory_path(path_value: str | Path, *, base_dir: Path | None = None) -> Path:
    """Resolve auxiliary memory paths relative to graph dir, cwd, mlevolve root, or repo root."""
    p = Path(path_value)
    if p.is_absolute():
        return p
    candidates = []
    if base_dir is not None:
        candidates.append(base_dir / p)
    candidates += [
        Path.cwd() / p,
        _mlevolve_root() / p,
        _repo_root() / p,
        _repo_root() / "mlevolve" / p,
    ]
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return ((base_dir or _mlevolve_root()) / p).resolve()


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
        for k in (
            "title", "principle", "action", "sop", "condition", "applies_when",
            "conditions", "prevents", "failure_modes", "failure_mode", "category", "scope",
        )
    )


def _node_to_text(node: dict[str, Any], include_stats: bool = True) -> str:
    when = node.get("condition") or "; ".join(_as_list(node.get("applies_when")) + _as_list(node.get("conditions")))
    how = node.get("principle") or node.get("action") or node.get("sop") or ""
    prevents = "; ".join(_as_list(node.get("prevents")) + _as_list(node.get("failure_modes")) + _as_list(node.get("failure_mode")))
    parts = [
        f"Title: {node.get('title', '')}",
        f"When: {when}",
        f"How: {how}",
        f"Type: {node.get('category', '')} / {node.get('scope', '')}",
    ]
    if prevents:
        parts.append(f"Prevents/Risks: {prevents}")
    if include_stats:
        parts.append(
            "Evidence: "
            f"n_use={node.get('n_use', 0)}, "
            f"n_succ={node.get('n_succ', 0)}, "
            f"p_hat={node.get('p_hat', 0.0)}"
        )
    return "\n".join(parts)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, tuple):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    return [text] if text else []


def _edge_kind(edge: dict[str, Any]) -> str:
    return str(edge.get("kind") or edge.get("type") or "")


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def poincare_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Distance in the Poincare ball with curvature -1."""
    diff2 = float(np.sum((u - v) ** 2))
    un2 = min(float(np.sum(u * u)), 1.0 - 1e-7)
    vn2 = min(float(np.sum(v * v)), 1.0 - 1e-7)
    denom = max(1e-12, (1.0 - un2) * (1.0 - vn2))
    arg = 1.0 + (2.0 * diff2 / denom)
    return float(math.acosh(max(1.0, arg)))


def euclidean_distance(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.linalg.norm(u - v))


class ExternalSkillMemoryLayer:
    """Query-time retriever over a pre-built SkillGraph JSON file."""

    def __init__(
        self,
        graph_path: str,
        source_name: str = "skillgraph",
        mode: str = "skillgraph",
        index_path: str = "",
        text_model_path: str = "",
        scoring_mode: str = "lexical",
        enable_agentic: bool = False,
        navigator_max_steps: int = 3,
        navigator_reference_budget: int = 1200,
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
        cfg: Any | None = None,
    ) -> None:
        self.graph_path = resolve_graph_path(graph_path)
        self.source_name = source_name or "skillgraph"
        self.mode = mode or "skillgraph"
        self.scoring_mode = (scoring_mode or "lexical").lower()
        if self.mode == "agentic_hyperbolic" and self.scoring_mode == "lexical":
            self.scoring_mode = "poincare"
        if self.mode == "flat_twin_agentic" and self.scoring_mode == "lexical":
            self.scoring_mode = "flat_twin"
        if self.scoring_mode not in {"lexical", "poincare", "flat_twin"}:
            raise ValueError(f"Unsupported external skill memory scoring_mode: {self.scoring_mode}")
        self.agentic_enabled = bool(
            enable_agentic
            or self.mode in {"agentic_hyperbolic", "flat_twin_agentic"}
            or "agentic" in self.source_name
        )
        if self.agentic_enabled and self.source_name == "skillgraph":
            self.source_name = "flat_twin_agentic_memory" if self.mode == "flat_twin_agentic" else "hyperbolic_agentic_memory"
        self.navigator_max_steps = max(1, min(3, int(navigator_max_steps or 3)))
        self.navigator_reference_budget = max(200, int(navigator_reference_budget or 1200))
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
        self.cfg = cfg

        self.graph: dict[str, Any] = {}
        self.nodes: dict[str, dict[str, Any]] = {}
        self.out_edges: dict[str, list[tuple[str, str, float]]] = collections.defaultdict(list)
        self.in_edges: dict[str, list[tuple[str, str, float]]] = collections.defaultdict(list)
        self._node_tokens: dict[str, set[str]] = {}
        self.index_path = resolve_memory_path(index_path, base_dir=self.graph_path.parent) if index_path else self.graph_path.parent / "hyper_index.npz"
        self.text_model_path = resolve_memory_path(text_model_path, base_dir=self.graph_path.parent) if text_model_path else self.graph_path.parent / "hyper_text_model.joblib"
        self._index_node_ids: list[str] = []
        self._poincare_coords: dict[str, np.ndarray] = {}
        self._flat_twin_coords: dict[str, np.ndarray] = {}
        self._radius_by_id: dict[str, float] = {}
        self._text_model: dict[str, Any] | None = None
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
            kind = _edge_kind(edge)
            weight = float(edge.get("weight", 1.0))
            self.out_edges[src].append((dst, kind, weight))
            self.in_edges[dst].append((src, kind, weight))
        self._node_tokens = {
            nid: _tokenize(_node_text_for_scoring(node))
            for nid, node in self.nodes.items()
        }
        if self.scoring_mode in {"poincare", "flat_twin"}:
            self._load_geometric_index()
        logger.info(
            "[ExternalSkillMemory] loaded %s nodes / %s edges from %s (mode=%s scoring=%s)",
            len(self.nodes),
            len(self.graph.get("edges", [])),
            self.graph_path,
            self.mode,
            self.scoring_mode,
        )

    def _load_geometric_index(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(f"Hyperbolic index required for {self.scoring_mode} scoring: {self.index_path}")
        if not self.text_model_path.exists():
            raise FileNotFoundError(f"Hyperbolic text model required for {self.scoring_mode} scoring: {self.text_model_path}")
        data = np.load(self.index_path)
        node_ids = [str(x) for x in data["node_ids"].tolist()]
        poincare = np.asarray(data["poincare"], dtype=np.float32)
        flat_twin = np.asarray(data["flat_twin"] if "flat_twin" in data.files else poincare, dtype=np.float32)
        if poincare.shape != flat_twin.shape or not np.array_equal(poincare, flat_twin):
            raise ValueError("Flat-Twin main control must use the exact same coordinates as Poincare")
        self._index_node_ids = node_ids
        self._poincare_coords = {nid: poincare[i] for i, nid in enumerate(node_ids) if nid in self.nodes}
        self._flat_twin_coords = {nid: flat_twin[i] for i, nid in enumerate(node_ids) if nid in self.nodes}
        if "radius" in data.files:
            radii = np.asarray(data["radius"], dtype=np.float32)
            self._radius_by_id = {nid: float(radii[i]) for i, nid in enumerate(node_ids) if nid in self.nodes}
        self._text_model = joblib.load(self.text_model_path)
        if not self._poincare_coords:
            raise ValueError(f"No graph SOP ids matched hyperbolic index ids in {self.index_path}")

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
        if self.agentic_enabled:
            try:
                text, ref_ids = self._retrieve_agentic(
                    task_type=task_type,
                    stage=stage,
                    task_desc=task_desc,
                    query_text=query_text,
                )
                if text and ref_ids:
                    if self.max_chars > 0 and len(text) > self.max_chars:
                        text = text[: self.max_chars].rstrip() + "\n... (agentic external memory truncated)"
                    logger.info(
                        "[ExternalSkillMemory] agentic stage=%s task=%s retrieved=%s ids=%s",
                        stage,
                        task_type,
                        len(ref_ids),
                        ",".join(ref_ids),
                    )
                    return text, ref_ids
            except Exception as exc:
                logger.warning("[ExternalSkillMemory] agentic navigator failed, falling back: %s", exc)

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

    def _project_query_direction(self, text: str) -> np.ndarray:
        if not self._text_model:
            raise RuntimeError("hyperbolic text model is not loaded")
        vectorizer = self._text_model.get("vectorizer")
        svd = self._text_model.get("svd")
        dims = int(self._text_model.get("dims") or 0)
        if vectorizer is None or dims <= 0:
            raise RuntimeError("invalid hyperbolic text model")
        tfidf = vectorizer.transform([text or ""])
        dense = svd.transform(tfidf) if svd is not None else tfidf.toarray()[:, :1]
        if dense.shape[1] < dims:
            dense = np.pad(dense, ((0, 0), (0, dims - dense.shape[1])), constant_values=0.0)
        direction = dense[0].astype(np.float32)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(direction).all() or norm < 1e-8:
            seed = int(hashlib.md5((text or "empty").encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            direction = rng.normal(size=dims).astype(np.float32)
            norm = float(np.linalg.norm(direction))
        return direction / max(norm, 1e-8)

    def _query_radius(self, radius_band: str = "") -> float:
        radii = list(self._radius_by_id.values())
        if not radii:
            return 0.5
        bands = {b.strip().lower() for b in (radius_band or "").split(",") if b.strip()}
        if bands:
            selected = [
                r for nid, r in self._radius_by_id.items()
                if self._radius_band(self.nodes.get(nid, {})).lower() in bands
            ]
            if selected:
                return float(np.median(selected))
        return float(np.median(radii))

    def _geometry_distance(self, query_point: np.ndarray, nid: str) -> float:
        if self.scoring_mode == "poincare":
            coord = self._poincare_coords.get(nid)
            if coord is None:
                return float("inf")
            return poincare_distance(query_point, coord)
        if self.scoring_mode == "flat_twin":
            coord = self._flat_twin_coords.get(nid)
            if coord is None:
                return float("inf")
            return euclidean_distance(query_point, coord)
        return float("inf")

    def _shared_constraint_bonus(
        self,
        *,
        node: dict[str, Any],
        condition_tokens: set[str],
        failure_tokens: set[str],
    ) -> float:
        cond_tokens = _tokenize(self._node_condition_text(node))
        fail_tokens = _tokenize(self._node_failure_text(node))
        cond_bonus = 0.40 * len(condition_tokens & cond_tokens) if condition_tokens else 0.0
        fail_bonus = 0.45 * len(failure_tokens & fail_tokens) if failure_tokens else 0.0
        if condition_tokens and not (condition_tokens & (cond_tokens | self._node_tokens.get(str(node.get("id", "")), set()))):
            cond_bonus -= 0.15
        evidence = (
            0.18 * float(node.get("p_hat", 0.0) or 0.0)
            + 0.04 * math.log1p(float(node.get("n_use", 0.0) or 0.0))
        )
        return cond_bonus + fail_bonus + evidence

    def _sort_known_sops(self, ids: list[str], query_text: str) -> list[str]:
        q_tokens = _tokenize(query_text)
        if self.scoring_mode in {"poincare", "flat_twin"}:
            query_direction = self._project_query_direction(query_text)
            query_point = query_direction * self._query_radius("")
            return sorted(
                [nid for nid in ids if nid in self.nodes],
                key=lambda nid: (
                    self._geometry_distance(query_point, nid),
                    -self._node_score(nid, q_tokens),
                    self.nodes[nid].get("title", ""),
                ),
            )
        return sorted(
            [nid for nid in ids if nid in self.nodes],
            key=lambda nid: (
                -self._node_score(nid, q_tokens),
                -float(self.nodes[nid].get("p_hat", 0.0) or 0.0),
                self.nodes[nid].get("title", ""),
            ),
        )

    def _seed_select(self, task_type: str, query_text: str) -> list[str]:
        generals = sorted(
            (nid for nid, n in self.nodes.items() if n.get("category") == "general" and self._is_sop_node(n)),
            key=lambda nid: (
                -float(self.nodes[nid].get("p_hat", 0.0)),
                -float(self.nodes[nid].get("n_use", 0)),
                self.nodes[nid].get("title", ""),
            ),
        )
        if self.general_cap >= 0:
            generals = generals[: self.general_cap]

        task_nodes = [nid for nid, n in self.nodes.items() if n.get("category") == task_type and self._is_sop_node(n)]
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
                    if not self._is_sop_node(self.nodes[dst]):
                        continue
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
            [nid for nid in union if self.nodes[nid].get("category") == "general" and self._is_sop_node(self.nodes[nid])],
            key=lambda nid: -sigma.get(nid, 0.0),
        )
        tasks = sorted(
            [nid for nid in union if self.nodes[nid].get("category") != "general" and self._is_sop_node(self.nodes[nid])],
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

    # ------------------------------------------------------------------
    # Agentic hyperbolic-map navigation tools
    # ------------------------------------------------------------------

    def _is_sop_node(self, node: dict[str, Any]) -> bool:
        """Return True for executable SOP-like nodes.

        Future hyper_graph artifacts carry type=SOP. Current SkillGraph compact
        cards have no type field, so they are treated as SOP candidates.
        """
        node_type = str(node.get("type", "")).lower()
        if not node_type:
            return True
        return node_type in {"sop", "skill_unit", "procedure", "memory"}

    def _node_region(self, node: dict[str, Any]) -> str:
        return str(
            node.get("skill_id")
            or node.get("skill")
            or node.get("region")
            or node.get("category")
            or node.get("scope")
            or "unknown"
        )

    def _node_condition_text(self, node: dict[str, Any]) -> str:
        values = []
        values += _as_list(node.get("applies_when"))
        values += _as_list(node.get("conditions"))
        values += _as_list(node.get("condition"))
        return " ".join(values)

    def _node_failure_text(self, node: dict[str, Any]) -> str:
        values = []
        values += _as_list(node.get("prevents"))
        values += _as_list(node.get("failure_modes"))
        values += _as_list(node.get("failure_mode"))
        return " ".join(values)

    def _node_reference_ids(self, node: dict[str, Any]) -> list[str]:
        refs = []
        refs += _as_list(node.get("reference_ids"))
        refs += _as_list(node.get("references"))
        refs += _as_list(node.get("ref"))
        refs += _as_list(node.get("implementation_ids"))
        refs += _as_list(node.get("evidence_ids"))
        return refs

    def _radius_band(self, node: dict[str, Any]) -> str:
        """Approximate core/mid/edge if no true hyperbolic radius is present."""
        radius = node.get("radius")
        if radius is not None:
            try:
                r = float(radius)
                if r <= 0.35:
                    return "core"
                if r <= 0.70:
                    return "middle"
                return "edge"
            except Exception:
                pass
        p_hat = float(node.get("p_hat", 0.0) or 0.0)
        n_use = float(node.get("n_use", 0.0) or 0.0)
        confidence = str(node.get("confidence", "")).lower()
        if confidence == "high" or p_hat >= 0.67 or n_use >= 5:
            return "core"
        if confidence == "medium" or p_hat >= 0.34 or n_use >= 2:
            return "middle"
        return "edge"

    def _node_summary(self, nid: str, *, include_neighbors: bool = False) -> dict[str, Any]:
        node = self.nodes[nid]
        summary = {
            "id": nid,
            "title": node.get("title", ""),
            "region": self._node_region(node),
            "type": node.get("type", "SOP"),
            "radius_band": self._radius_band(node),
            "when": self._node_condition_text(node) or node.get("condition", ""),
            "action": node.get("action") or node.get("principle", ""),
            "prevents": self._node_failure_text(node),
            "evidence": {
                "n_use": node.get("n_use", 0),
                "n_succ": node.get("n_succ", 0),
                "p_hat": node.get("p_hat", 0.0),
                "metric": node.get("metric", {}),
            },
            "references": self._node_reference_ids(node)[:3],
        }
        if include_neighbors:
            neighbors = []
            for dst, kind, weight in self.out_edges.get(nid, [])[:8]:
                neighbors.append({"id": dst, "kind": kind, "weight": weight, "title": self.nodes[dst].get("title", "")})
            for src, kind, weight in self.in_edges.get(nid, [])[:8]:
                neighbors.append({"id": src, "kind": f"in:{kind}", "weight": weight, "title": self.nodes[src].get("title", "")})
            summary["neighbors"] = neighbors
        return summary

    def _rank_sop_candidates(
        self,
        *,
        task_type: str,
        query_text: str,
        region: str = "",
        condition: str = "",
        failure_mode: str = "",
        radius_band: str = "",
        top_k: int | None = None,
    ) -> list[str]:
        q_tokens = _tokenize(" ".join([query_text, region, condition, failure_mode, radius_band]))
        region_l = (region or "").lower()
        condition_tokens = _tokenize(condition)
        failure_tokens = _tokenize(failure_mode)
        bands = {b.strip().lower() for b in (radius_band or "").split(",") if b.strip()}
        geometric = self.scoring_mode in {"poincare", "flat_twin"}
        query_point: np.ndarray | None = None
        if geometric:
            query_direction = self._project_query_direction(" ".join([query_text, region, condition, failure_mode]))
            query_point = query_direction * self._query_radius(radius_band)

        scored: list[tuple[float, str]] = []
        for nid, node in self.nodes.items():
            if not self._is_sop_node(node):
                continue
            cat = node.get("category")
            if cat not in ("general", task_type) and task_type:
                continue
            if region_l and region_l not in self._node_region(node).lower() and region_l not in str(cat).lower():
                continue
            if bands and self._radius_band(node).lower() not in bands:
                continue

            text = _node_text_for_scoring(node)
            cond_text = self._node_condition_text(node)
            fail_text = self._node_failure_text(node)
            text_tokens = _tokenize(text)
            cond_tokens = _tokenize(cond_text)
            fail_tokens = _tokenize(fail_text)
            semantic = len(q_tokens & text_tokens) / math.sqrt(max(1, len(text_tokens)))
            cond_bonus = 0.40 * len(condition_tokens & cond_tokens) if condition_tokens else 0.0
            fail_bonus = 0.45 * len(failure_tokens & fail_tokens) if failure_tokens else 0.0
            if condition_tokens and not (condition_tokens & (cond_tokens | text_tokens)):
                cond_bonus -= 0.15
            evidence = (
                0.18 * float(node.get("p_hat", 0.0) or 0.0)
                + 0.04 * math.log1p(float(node.get("n_use", 0.0) or 0.0))
            )
            # Edge-band preservation: low-frequency, condition-matched SOPs should
            # remain reachable rather than disappearing behind high-support tips.
            rare_bonus = 0.10 if self._radius_band(node) == "edge" and condition_tokens & (cond_tokens | text_tokens) else 0.0
            if geometric and query_point is not None:
                distance = self._geometry_distance(query_point, nid)
                if not math.isfinite(distance):
                    continue
                # The distance is the only geometry-specific term. Condition,
                # failure, evidence, and rare bonuses are shared constraints
                # used identically by Poincare and same-coordinate Flat-Twin.
                score = -distance + 0.05 * (cond_bonus + fail_bonus + evidence + rare_bonus)
            else:
                score = semantic + cond_bonus + fail_bonus + evidence + rare_bonus
            scored.append((score, nid))

        scored.sort(key=lambda x: (-x[0], self.nodes[x[1]].get("title", "")))
        return [nid for _, nid in scored[: (top_k or self.top_k)]]

    def inspect_map(self, *, task_type: str, query_text: str, context: str = "") -> dict[str, Any]:
        q_tokens = _tokenize(query_text + " " + context)
        by_region: dict[str, list[str]] = collections.defaultdict(list)
        by_band: dict[str, list[str]] = collections.defaultdict(list)
        cond_counter: collections.Counter[str] = collections.Counter()
        fail_counter: collections.Counter[str] = collections.Counter()

        for nid, node in self.nodes.items():
            if not self._is_sop_node(node):
                continue
            if node.get("category") not in ("general", task_type):
                continue
            by_region[self._node_region(node)].append(nid)
            by_band[self._radius_band(node)].append(nid)
            for tok in _tokenize(self._node_condition_text(node)):
                if tok in q_tokens:
                    cond_counter[tok] += 1
            for tok in _tokenize(self._node_failure_text(node)):
                if tok in q_tokens:
                    fail_counter[tok] += 1

        ranked_regions = sorted(
            by_region,
            key=lambda r: (
                -sum(self._node_score(nid, q_tokens) for nid in by_region[r]) / max(1, len(by_region[r])),
                r,
            ),
        )[:6]
        region_cards = []
        for region in ranked_regions:
            ids = sorted(by_region[region], key=lambda nid: -self._node_score(nid, q_tokens))[:3]
            region_cards.append({
                "region": region,
                "n_sops": len(by_region[region]),
                "examples": [self._node_summary(nid) for nid in ids],
            })

        return {
            "tool": "inspect_map",
            "matched_task_type": task_type,
            "scoring_mode": self.scoring_mode,
            "regions": region_cards,
            "radius_bands": {band: len(ids) for band, ids in sorted(by_band.items())},
            "condition_hotspots": cond_counter.most_common(8),
            "failure_hotspots": fail_counter.most_common(8),
            "navigation_suggestions": [
                "navigate to the most task-matched region first",
                "use condition/failure filters when the current node has a clear symptom",
                "expand via conflicts_with/refines/prevents if a selected SOP is risky or too narrow",
            ],
        }

    def navigate(
        self,
        *,
        task_type: str,
        query_text: str,
        region: str = "",
        condition: str = "",
        failure_mode: str = "",
        radius_band: str = "",
        top_k: int = 5,
    ) -> dict[str, Any]:
        ids = self._rank_sop_candidates(
            task_type=task_type,
            query_text=query_text,
            region=region,
            condition=condition,
            failure_mode=failure_mode,
            radius_band=radius_band,
            top_k=max(1, min(10, int(top_k or 5))),
        )
        return {
            "tool": "navigate",
            "scoring_mode": self.scoring_mode,
            "filters": {
                "region": region,
                "condition": condition,
                "failure_mode": failure_mode,
                "radius_band": radius_band,
            },
            "sops": [self._node_summary(nid) for nid in ids],
        }

    def expand(
        self,
        *,
        node_id: str,
        edge_types: list[str] | None = None,
        hops: int = 1,
        top_k: int = 8,
    ) -> dict[str, Any]:
        if node_id not in self.nodes:
            return {"tool": "expand", "error": f"unknown node_id: {node_id}", "sops": []}
        allowed = {e.lower() for e in (edge_types or []) if e}
        if not allowed:
            allowed = {"prevents", "applies_when", "refines", "conflicts_with", "prereq", "co_occur", "enhance"}
        found: set[str] = set()
        frontier = {node_id}
        trace = []
        for _ in range(max(1, min(3, int(hops or 1)))):
            nxt: set[str] = set()
            for src in frontier:
                for dst, kind, weight in self.out_edges.get(src, []):
                    if kind.lower() in allowed and dst not in found and dst != node_id:
                        found.add(dst)
                        nxt.add(dst)
                        trace.append({"src": src, "dst": dst, "kind": kind, "weight": weight})
                for dst, kind, weight in self.in_edges.get(src, []):
                    base_kind = kind.lower().replace("in:", "")
                    if base_kind in allowed and dst not in found and dst != node_id:
                        found.add(dst)
                        nxt.add(dst)
                        trace.append({"src": dst, "dst": src, "kind": f"in:{kind}", "weight": weight})
            frontier = nxt
            if not frontier:
                break
        sop_ids = [nid for nid in found if self._is_sop_node(self.nodes[nid])][:top_k]
        return {
            "tool": "expand",
            "anchor": self._node_summary(node_id),
            "edge_types": sorted(allowed),
            "path_trace": trace[:12],
            "sops": [self._node_summary(nid) for nid in sop_ids],
        }

    def inspect_sop(self, *, sop_id: str) -> dict[str, Any]:
        if sop_id not in self.nodes:
            return {"tool": "inspect_sop", "error": f"unknown sop_id: {sop_id}"}
        return {"tool": "inspect_sop", "sop": self._node_summary(sop_id, include_neighbors=True)}

    def _has_opposition(self, left: str, right: str) -> bool:
        ltoks, rtoks = _tokenize(left), _tokenize(right)
        for a, b in _OPPOSING_TERMS:
            if (a in ltoks and b in rtoks) or (b in ltoks and a in rtoks):
                return True
        return False

    def check_conflicts(self, *, sop_ids: list[str], context: str = "") -> dict[str, Any]:
        ids = [nid for nid in sop_ids if nid in self.nodes and self._is_sop_node(self.nodes[nid])]
        warnings = []
        relations = []
        explicit_conflicts = set()
        for src in ids:
            for dst, kind, _weight in self.out_edges.get(src, []):
                if dst in ids and kind in {"conflicts_with", "prevents"}:
                    explicit_conflicts.add(tuple(sorted((src, dst))))
        for i, left_id in enumerate(ids):
            for right_id in ids[i + 1:]:
                left, right = self.nodes[left_id], self.nodes[right_id]
                ltext = _node_text_for_scoring(left)
                rtext = _node_text_for_scoring(right)
                lcond = _tokenize(self._node_condition_text(left))
                rcond = _tokenize(self._node_condition_text(right))
                lfail = _tokenize(self._node_failure_text(left))
                rfail = _tokenize(self._node_failure_text(right))
                same_condition = bool(lcond & rcond)
                same_failure = bool(lfail & rfail)
                opposed = self._has_opposition(ltext, rtext)
                explicit = tuple(sorted((left_id, right_id))) in explicit_conflicts
                if explicit or (same_condition and opposed):
                    label = "true_conflict"
                    warning = "same condition with opposing actions; treat one side as a risk warning unless evidence strongly favors it"
                elif opposed:
                    label = "condition_branch"
                    warning = "opposing actions appear condition-dependent; choose only the branch matching the current constraints"
                elif same_failure:
                    label = "complementary"
                    warning = "both SOPs address a shared failure mode and may form a method set"
                else:
                    label = "risk_warning"
                    warning = "no direct conflict found; verify conditions before adoption"
                relations.append({
                    "left": left_id,
                    "right": right_id,
                    "label": label,
                    "reason": warning,
                })
                if label in {"true_conflict", "condition_branch", "risk_warning"}:
                    warnings.append(f"{left.get('title', left_id)} vs {right.get('title', right_id)}: {warning}")
        return {"tool": "check_conflicts", "relations": relations, "risk_warnings": warnings[:6]}

    def open_reference(self, *, ref_id: str, budget: int | None = None) -> dict[str, Any]:
        budget = max(200, min(self.navigator_reference_budget, int(budget or self.navigator_reference_budget)))
        candidates: list[Path] = []
        raw = str(ref_id or "").strip()
        if not raw:
            return {"tool": "open_reference", "error": "empty ref_id", "text": ""}
        p = Path(raw)
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates += [
                self.graph_path.parent / raw,
                self.graph_path.parent / "references" / raw,
                self.graph_path.parent / "references" / f"{raw}.md",
                self.graph_path.parent / "references" / f"{_slugify(raw)}.md",
                _repo_root() / raw,
            ]
        for cand in candidates:
            if cand.exists() and cand.is_file():
                text = cand.read_text(encoding="utf-8", errors="ignore")
                return {
                    "tool": "open_reference",
                    "ref_id": raw,
                    "path": str(cand),
                    "text": text[:budget],
                    "truncated": len(text) > budget,
                }
        if raw in self.nodes:
            return {
                "tool": "open_reference",
                "ref_id": raw,
                "text": _node_to_text(self.nodes[raw], include_stats=True)[:budget],
                "truncated": False,
            }
        return {"tool": "open_reference", "ref_id": raw, "text": "", "error": "reference not found"}

    def _navigation_action_spec(self):
        from llm import FunctionSpec

        return FunctionSpec(
            name="choose_memory_navigation_action",
            description="Choose the next read-only tool call for navigating the hyperbolic SOP memory map.",
            json_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["navigate", "expand", "inspect_sop", "check_conflicts", "open_reference", "finish"],
                    },
                    "reason": {"type": "string"},
                    "region": {"type": "string"},
                    "condition": {"type": "string"},
                    "failure_mode": {"type": "string"},
                    "radius_band": {"type": "string"},
                    "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
                    "node_id": {"type": "string"},
                    "sop_id": {"type": "string"},
                    "sop_ids": {"type": "array", "items": {"type": "string"}},
                    "edge_types": {"type": "array", "items": {"type": "string"}},
                    "hops": {"type": "integer", "minimum": 1, "maximum": 3},
                    "ref_id": {"type": "string"},
                    "selected_sops": {"type": "array", "items": {"type": "string"}},
                    "risk_warnings": {"type": "array", "items": {"type": "string"}},
                    "rejected_sops": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "id": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                            "required": ["id", "reason"],
                        },
                    },
                },
                "required": ["action", "reason"],
            },
        )

    def _execute_navigation_action(
        self,
        action: dict[str, Any],
        *,
        task_type: str,
        query_text: str,
    ) -> dict[str, Any]:
        name = str(action.get("action", "finish"))
        if name == "navigate":
            return self.navigate(
                task_type=task_type,
                query_text=query_text,
                region=str(action.get("region", "")),
                condition=str(action.get("condition", "")),
                failure_mode=str(action.get("failure_mode", "")),
                radius_band=str(action.get("radius_band", "")),
                top_k=int(action.get("top_k") or 5),
            )
        if name == "expand":
            return self.expand(
                node_id=str(action.get("node_id") or action.get("sop_id") or ""),
                edge_types=action.get("edge_types") or [],
                hops=int(action.get("hops") or 1),
                top_k=int(action.get("top_k") or 8),
            )
        if name == "inspect_sop":
            return self.inspect_sop(sop_id=str(action.get("sop_id") or action.get("node_id") or ""))
        if name == "check_conflicts":
            return self.check_conflicts(sop_ids=action.get("sop_ids") or action.get("selected_sops") or [], context=query_text)
        if name == "open_reference":
            return self.open_reference(ref_id=str(action.get("ref_id", "")))
        return {"tool": "finish", "selected_sops": action.get("selected_sops") or [], "risk_warnings": action.get("risk_warnings") or []}

    def _collect_sop_ids(self, observation: Any) -> list[str]:
        ids: list[str] = []
        if isinstance(observation, dict):
            if "sops" in observation:
                for item in observation.get("sops") or []:
                    if isinstance(item, dict) and item.get("id") in self.nodes:
                        ids.append(item["id"])
            if "sop" in observation and isinstance(observation["sop"], dict):
                nid = observation["sop"].get("id")
                if nid in self.nodes:
                    ids.append(nid)
            if "selected_sops" in observation:
                ids += [nid for nid in observation.get("selected_sops") or [] if nid in self.nodes]
        return ids

    def _llm_choose_navigation_action(
        self,
        *,
        stage: str,
        task_type: str,
        task_desc: str,
        query_text: str,
        observations: list[dict[str, Any]],
        known_ids: list[str],
    ) -> dict[str, Any]:
        if self.cfg is None:
            raise RuntimeError("cfg is required for LLM navigation")
        from llm import query

        obs_text = json.dumps(observations[-3:], ensure_ascii=False, indent=2)[:6000]
        known_cards = [self._node_summary(nid) for nid in known_ids[:10] if nid in self.nodes]
        system = (
            "You are MemoryNavigator, a cautious agent navigating a persistent hyperbolic SOP map "
            "for an ML-engineering solver. Choose exactly one read-only map tool call. "
            "Prefer applicable SOPs over generic tips. Use conflicts as warnings, not commands. "
            "Finish only when you have enough SOPs for the next code-generation step."
        )
        user = {
            "Stage": stage,
            "Task type": task_type,
            "Task description": task_desc[:2000],
            "Current query/context": query_text[-5000:],
            "Previous map observations": obs_text,
            "Known candidate SOPs": known_cards,
            "Tool policy": [
                "If no candidate SOP has been inspected yet, call navigate.",
                "If candidates are risky or opposed, call check_conflicts.",
                "If one SOP needs code evidence, call open_reference.",
                "Otherwise finish with selected_sops and concise risk_warnings.",
            ],
        }
        model = getattr(self.cfg.agent.feedback, "model", None) or getattr(self.cfg.agent.code, "model", "")
        return query(
            system_message=system,
            user_message=user,
            model=model,
            temperature=0.2,
            max_tokens=1200,
            func_spec=self._navigation_action_spec(),
            cfg=self.cfg,
        )

    def _deterministic_agentic_pack(self, *, task_type: str, stage: str, task_desc: str, query_text: str) -> tuple[dict[str, Any], list[str]]:
        trace = []
        map_obs = self.inspect_map(task_type=task_type, query_text=query_text, context=task_desc)
        trace.append("inspect_map(context)")
        nav_obs = self.navigate(task_type=task_type, query_text=query_text, top_k=self.top_k)
        trace.append("navigate(region=auto, condition=auto, failure_mode=auto)")
        ids = self._collect_sop_ids(nav_obs)[: self.top_k]
        conflicts = self.check_conflicts(sop_ids=ids[:4], context=query_text)
        trace.append("check_conflicts(selected_sops)")
        pack = {
            "selected_sops": ids,
            "risk_warnings": conflicts.get("risk_warnings", []),
            "implementation_hints": [],
            "navigation_trace": trace,
            "rejected_sops": [],
            "map_summary": map_obs,
            "mode": "deterministic_fallback",
            "stage": stage,
        }
        return pack, ids

    def _retrieve_agentic(
        self,
        *,
        task_type: str,
        stage: str,
        task_desc: str,
        query_text: str,
    ) -> tuple[str, list[str]]:
        if self.cfg is None:
            pack, ids = self._deterministic_agentic_pack(
                task_type=task_type, stage=stage, task_desc=task_desc, query_text=query_text
            )
            return self._format_agentic_context(pack, ids, task_type=task_type), ids

        started = time.time()
        observations: list[dict[str, Any]] = []
        trace: list[str] = []
        known_ids: list[str] = []
        risk_warnings: list[str] = []
        implementation_hints: list[str] = []
        rejected_sops: list[dict[str, str]] = []

        first_obs = self.inspect_map(task_type=task_type, query_text=query_text, context=task_desc)
        observations.append(first_obs)
        trace.append("inspect_map(context)")

        for _step in range(max(0, self.navigator_max_steps - 1)):
            action = self._llm_choose_navigation_action(
                stage=stage,
                task_type=task_type,
                task_desc=task_desc,
                query_text=query_text,
                observations=observations,
                known_ids=known_ids,
            )
            action_name = str(action.get("action", "finish"))
            trace.append(f"{action_name}({action.get('reason', '')[:120]})")
            if action_name == "finish":
                for nid in action.get("selected_sops") or []:
                    if nid in self.nodes and nid not in known_ids:
                        known_ids.append(nid)
                risk_warnings += [str(x) for x in action.get("risk_warnings") or []]
                rejected_sops += [
                    {"id": str(x.get("id", "")), "reason": str(x.get("reason", ""))}
                    for x in action.get("rejected_sops") or []
                    if isinstance(x, dict)
                ]
                break

            obs = self._execute_navigation_action(action, task_type=task_type, query_text=query_text)
            observations.append(obs)
            for nid in self._collect_sop_ids(obs):
                if nid not in known_ids:
                    known_ids.append(nid)
            if obs.get("tool") == "check_conflicts":
                risk_warnings += [str(x) for x in obs.get("risk_warnings") or []]
            if obs.get("tool") == "open_reference" and obs.get("text"):
                implementation_hints.append(str(obs.get("text", ""))[:700])

        if not known_ids:
            pack, ids = self._deterministic_agentic_pack(
                task_type=task_type, stage=stage, task_desc=task_desc, query_text=query_text
            )
            pack["mode"] = "llm_empty_fallback"
            return self._format_agentic_context(pack, ids, task_type=task_type), ids

        q_tokens = _tokenize(query_text)
        selected = self._sort_known_sops(known_ids, query_text)[: self.top_k]
        conflicts = self.check_conflicts(sop_ids=selected[:4], context=query_text)
        risk_warnings += [str(x) for x in conflicts.get("risk_warnings") or []]
        pack = {
            "selected_sops": selected,
            "risk_warnings": list(dict.fromkeys(risk_warnings))[:6],
            "implementation_hints": implementation_hints[:2],
            "navigation_trace": trace,
            "rejected_sops": rejected_sops[:6],
            "map_summary": observations[0],
            "mode": self.mode,
            "stage": stage,
            "elapsed_sec": round(time.time() - started, 3),
        }
        return self._format_agentic_context(pack, selected, task_type=task_type), selected

    def _format_agentic_context(self, pack: dict[str, Any], ids: list[str], *, task_type: str) -> str:
        lines = [
            "## Agentic Hyperbolic Memory Navigation",
            "A MemoryNavigator agent inspected the persistent SOP map before this generation step.",
            "Treat selected SOPs as candidate actions only when their WHEN condition matches the current code, data, and error state.",
            "Risk warnings constrain the plan; do not blindly implement warning-only SOPs.",
            f"Matched task type: {task_type}; mode: {pack.get('mode', self.mode)}; stage: {pack.get('stage', '')}.",
            "",
            "### Navigator Trace",
        ]
        for item in pack.get("navigation_trace", []):
            lines.append(f"- {item}")

        map_summary = pack.get("map_summary") or {}
        regions = map_summary.get("regions") or []
        if regions:
            lines += ["", "### Map Landmarks"]
            for region in regions[:4]:
                titles = ", ".join(
                    ex.get("title", "")
                    for ex in region.get("examples", [])[:2]
                    if ex.get("title")
                )
                lines.append(f"- {region.get('region')}: {region.get('n_sops')} SOPs; examples: {titles}")

        lines += ["", "### Selected SOPs"]
        for idx, nid in enumerate(ids, 1):
            if nid not in self.nodes:
                continue
            node = self.nodes[nid]
            lines.append(f"Memory {idx} [{nid}]:")
            lines.append(_node_to_text(node, include_stats=True))
            refs = self._node_reference_ids(node)
            if refs:
                lines.append(f"References: {', '.join(refs[:3])}")
            lines.append("")

        if pack.get("risk_warnings"):
            lines += ["### Conflict / Risk Warnings"]
            for warning in pack.get("risk_warnings", [])[:6]:
                lines.append(f"- {warning}")
            lines.append("")

        if pack.get("rejected_sops"):
            lines += ["### Rejected SOPs"]
            for item in pack.get("rejected_sops", [])[:6]:
                lines.append(f"- {item.get('id')}: {item.get('reason')}")
            lines.append("")

        if pack.get("implementation_hints"):
            lines += ["### Reference-backed Implementation Hints"]
            for hint in pack.get("implementation_hints", [])[:2]:
                lines.append(hint.strip())
                lines.append("")
        return "\n".join(lines).strip()

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
