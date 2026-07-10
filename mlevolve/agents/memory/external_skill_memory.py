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

from agents.leakage_audit import structural_sha256

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


def external_memory_section_title(source_name: str) -> str:
    source = (source_name or "").lower()
    if "run_forest" in source:
        return "Agentic Run-Forest Memory Navigation"
    if "agentic" in source:
        return "Agentic Hyperbolic Memory Navigation"
    return "External Skill Memory"


def external_memory_section_intro(source_name: str, context: str) -> str:
    source = (source_name or "").lower()
    if "run_forest" in source:
        return (
            f"Below is a read-only run-forest map path pack retrieved before {context}: "
            "use matched paths, selected transitions, attached SOP signposts, risk warnings, "
            "and evidence refs only when they match the current code/data/error state."
        )
    if "agentic" in source:
        return f"Below is the navigator-selected SOP memory pack retrieved before {context}:"
    return f"Below are persistent SOP memories retrieved before {context}:"


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
    node_type = str(node.get("type", ""))
    if node_type in {"Run", "RunNode", "Transition", "Evidence"}:
        fields = [
            ("Type", node_type),
            ("Task", node.get("task")),
            ("Stage", node.get("stage") or node.get("stage_pair")),
            ("Outcome", node.get("outcome")),
            ("Metric", node.get("metric")),
            ("Metric delta", node.get("metric_delta")),
            ("Metric improvement", node.get("metric_improvement")),
            ("Parent buggy", node.get("parent_buggy")),
            ("Child buggy", node.get("child_buggy")),
            ("Plan", node.get("plan")),
            ("Code summary", node.get("code_summary")),
            ("Analysis", node.get("analysis")),
            ("Terminal/Error", node.get("terminal_excerpt") or node.get("error") or node.get("traceback")),
            ("Evidence", node.get("text")),
        ]
        lines = [f"{label}: {value}" for label, value in fields if value not in (None, "", [])]
        if node.get("attached_sop_ids"):
            lines.append(f"Attached SOPs: {', '.join(_as_list(node.get('attached_sop_ids'))[:8])}")
        return "\n".join(lines)

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
        geometry_distance_weight: float = 0.30,
        geometry_semantic_weight: float = 0.20,
        geometry_constraint_weight: float = 0.05,
        geometry_condition_weight: float = 0.18,
        geometry_failure_weight: float = 0.14,
        geometry_evidence_weight: float = 0.08,
        geometry_reliability_weight: float = 0.08,
        geometry_conflict_weight: float = 0.10,
        geometry_distance_norm: str = "none",
        geometry_query_radius_quantile: float = 0.5,
        geometry_query_radius_mode: str = "predicted_distribution",
        geometry_query_radius_bands: list[str] | tuple[str, ...] | str | None = None,
        geometry_query_radius_top_bands: int = 2,
        geometry_radius_fusion: str = "weighted_max",
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
        if self.mode == "agentic_euclidean" and self.scoring_mode == "lexical":
            self.scoring_mode = "euclidean"
        if self.scoring_mode not in {"lexical", "poincare", "flat_twin", "euclidean"}:
            raise ValueError(f"Unsupported external skill memory scoring_mode: {self.scoring_mode}")
        self.geometry_distance_weight = float(geometry_distance_weight)
        self.geometry_semantic_weight = float(geometry_semantic_weight)
        self.geometry_constraint_weight = float(geometry_constraint_weight)
        self.geometry_condition_weight = float(geometry_condition_weight)
        self.geometry_failure_weight = float(geometry_failure_weight)
        self.geometry_evidence_weight = float(geometry_evidence_weight)
        self.geometry_reliability_weight = float(geometry_reliability_weight)
        self.geometry_conflict_weight = float(geometry_conflict_weight)
        self.geometry_distance_norm = (geometry_distance_norm or "none").lower()
        if self.geometry_distance_norm not in {"none", "minmax", "zscore"}:
            raise ValueError(f"Unsupported geometry_distance_norm: {self.geometry_distance_norm}")
        self.geometry_query_radius_quantile = min(0.95, max(0.05, float(geometry_query_radius_quantile)))
        self.geometry_query_radius_mode = (geometry_query_radius_mode or "predicted_distribution").lower()
        if self.geometry_query_radius_mode not in {"quantile", "predicted_distribution"}:
            raise ValueError(f"Unsupported geometry_query_radius_mode: {self.geometry_query_radius_mode}")
        if geometry_query_radius_bands is None:
            self.geometry_query_radius_bands = ["core", "middle", "edge"]
        elif isinstance(geometry_query_radius_bands, str):
            self.geometry_query_radius_bands = [b.strip().lower() for b in geometry_query_radius_bands.split(",") if b.strip()]
        else:
            self.geometry_query_radius_bands = [str(b).strip().lower() for b in geometry_query_radius_bands if str(b).strip()]
        self.geometry_query_radius_bands = [b for b in self.geometry_query_radius_bands if b in {"core", "middle", "edge"}] or ["core", "middle", "edge"]
        self.geometry_query_radius_top_bands = max(1, min(3, int(geometry_query_radius_top_bands or 2)))
        self.geometry_radius_fusion = (geometry_radius_fusion or "weighted_max").lower()
        if self.geometry_radius_fusion not in {"weighted_max", "weighted_mean"}:
            raise ValueError(f"Unsupported geometry_radius_fusion: {self.geometry_radius_fusion}")
        self.agentic_enabled = bool(
            enable_agentic
            or self.mode in {"agentic_hyperbolic", "flat_twin_agentic", "agentic_euclidean"}
            or "agentic" in self.source_name
        )
        if self.agentic_enabled and self.source_name == "skillgraph":
            if self.mode == "flat_twin_agentic":
                self.source_name = "flat_twin_agentic_memory"
            elif self.mode == "agentic_euclidean":
                self.source_name = "euclidean_agentic_memory"
            else:
                self.source_name = "hyperbolic_agentic_memory"
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
        self._euclidean_coords: dict[str, np.ndarray] = {}
        self._radius_by_id: dict[str, float] = {}
        self._reliability_by_id: dict[str, float] = {}
        self._sentence_embedder: Any | None = None
        self._text_model: dict[str, Any] | None = None
        self._last_agentic_pack: dict[str, Any] = {}
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
        if self.scoring_mode in {"poincare", "flat_twin", "euclidean"}:
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
        if "euclidean" in data.files:
            euclidean = np.asarray(data["euclidean"], dtype=np.float32)
        elif "direction" in data.files:
            euclidean = np.asarray(data["direction"], dtype=np.float32)
        else:
            raise ValueError("Agentic Euclidean Memory requires euclidean or direction coordinates in the index")
        if poincare.shape != flat_twin.shape or not np.array_equal(poincare, flat_twin):
            raise ValueError("Flat-Twin main control must use the exact same coordinates as Poincare")
        if euclidean.shape != poincare.shape:
            raise ValueError("Euclidean memory coordinates must have the same dimensionality as Poincare coordinates")
        if np.array_equal(euclidean, poincare):
            raise ValueError("Agentic Euclidean Memory must use independent flat coordinates, not Poincare coordinates")
        self._index_node_ids = node_ids
        self._poincare_coords = {nid: poincare[i] for i, nid in enumerate(node_ids) if nid in self.nodes}
        self._flat_twin_coords = {nid: flat_twin[i] for i, nid in enumerate(node_ids) if nid in self.nodes}
        self._euclidean_coords = {nid: euclidean[i] for i, nid in enumerate(node_ids) if nid in self.nodes}
        if "radius" in data.files:
            radii = np.asarray(data["radius"], dtype=np.float32)
            self._radius_by_id = {nid: float(radii[i]) for i, nid in enumerate(node_ids) if nid in self.nodes}
        if "reliability_score" in data.files:
            reliability = np.asarray(data["reliability_score"], dtype=np.float32)
            self._reliability_by_id = {nid: float(reliability[i]) for i, nid in enumerate(node_ids) if nid in self.nodes}
        else:
            self._reliability_by_id = {
                nid: self._node_reliability_score(self.nodes[nid])
                for nid in node_ids
                if nid in self.nodes
            }
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

    def _node_reliability_score(self, node: dict[str, Any]) -> float:
        if node.get("reliability_score") is not None:
            try:
                return max(0.0, min(1.0, float(node.get("reliability_score"))))
            except Exception:
                pass
        metric = node.get("metric") if isinstance(node.get("metric"), dict) else {}
        p_hat = float(node.get("p_hat", metric.get("p_hat", 0.0)) or 0.0)
        n_use = float(node.get("n_use", metric.get("n_use", 0.0)) or 0.0)
        support = math.log1p(max(0.0, n_use)) / math.log1p(10.0)
        has_source = bool(node.get("source_branches") or node.get("evidence_turns") or node.get("reference_ids") or node.get("evidence_ids"))
        return max(0.0, min(1.0, 0.65 * p_hat + 0.25 * support + 0.10 * float(has_source)))

    def _node_evidence_score(self, node: dict[str, Any]) -> float:
        if node.get("evidence_confidence") == "high":
            return 1.0
        if node.get("evidence_confidence") == "medium":
            return 0.65
        if node.get("source_branches") or node.get("evidence_turns") or node.get("reference_ids") or node.get("evidence_ids") or node.get("metric"):
            return 0.45
        return 0.0

    def _project_query_direction(self, text: str) -> np.ndarray:
        if not self._text_model:
            raise RuntimeError("hyperbolic text model is not loaded")
        method = str(self._text_model.get("method") or "tfidf_truncated_svd")
        dims = int(self._text_model.get("dims") or 0)
        if dims <= 0:
            raise RuntimeError("invalid hyperbolic text model")
        if method in {"sentence_embedding_svd", "contrastive_projection"}:
            projection = self._text_model.get("projection")
            model_name = str(self._text_model.get("embedding_model") or "")
            if projection is None or not model_name:
                raise RuntimeError("invalid sentence embedding hyperbolic text model")
            if self._sentence_embedder is None:
                from sentence_transformers import SentenceTransformer

                self._sentence_embedder = SentenceTransformer(model_name, local_files_only=True)
            embedding = np.asarray(
                self._sentence_embedder.encode([text or ""], normalize_embeddings=False, show_progress_bar=False),
                dtype=np.float32,
            )
            dense = projection.transform(embedding)
        else:
            vectorizer = self._text_model.get("vectorizer")
            svd = self._text_model.get("svd")
            if vectorizer is None:
                raise RuntimeError("invalid TF-IDF hyperbolic text model")
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

    def _band_center_radius(self, band: str) -> float:
        selected = [
            r for nid, r in self._radius_by_id.items()
            if self._radius_band(self.nodes.get(nid, {})).lower() == band
        ]
        if selected:
            return float(np.quantile(selected, self.geometry_query_radius_quantile))
        return {"core": 0.24, "middle": 0.52, "edge": 0.82}.get(band, 0.52)

    def _normalize_radius_weights(self, weights: dict[str, float]) -> list[tuple[str, float]]:
        filtered = {b: max(0.0, float(weights.get(b, 0.0))) for b in self.geometry_query_radius_bands}
        total = sum(filtered.values())
        if total <= 0:
            filtered = {"middle": 1.0}
            total = 1.0
        ranked = sorted(filtered.items(), key=lambda item: (-item[1], item[0]))[: self.geometry_query_radius_top_bands]
        subtotal = sum(v for _, v in ranked) or 1.0
        return [(band, weight / subtotal) for band, weight in ranked if weight > 0]

    def _predict_query_radius_distribution(self, query_text: str, radius_band: str = "") -> list[dict[str, float | str]]:
        if self.geometry_query_radius_mode == "quantile":
            return [{
                "band": radius_band or "quantile",
                "radius": self._query_radius(radius_band),
                "weight": 1.0,
                "source": "quantile",
            }]

        explicit_bands = [b.strip().lower() for b in (radius_band or "").split(",") if b.strip()]
        explicit_bands = [b for b in explicit_bands if b in {"core", "middle", "edge"}]
        if explicit_bands:
            weights = {b: 1.0 for b in explicit_bands}
            source = "radius_band_hint"
        else:
            text = (query_text or "").lower()
            weights = {"core": 0.20, "middle": 0.50, "edge": 0.30}
            source = "deterministic_query_radius_v1"
            if "minimal_context" in text or "minimal context" in text:
                weights = {"core": 0.10, "middle": 0.35, "edge": 0.55}
            elif "abstract_failure" in text or "abstract failure" in text:
                weights = {"core": 0.10, "middle": 0.40, "edge": 0.50}
            elif "rare_partial_clue" in text or "rare partial clue" in text or "partial clue" in text:
                weights = {"core": 0.10, "middle": 0.30, "edge": 0.60}
            elif "debug" in text or "failure" in text or "traceback" in text or "error" in text:
                weights = {"core": 0.10, "middle": 0.25, "edge": 0.65}
            elif "method_set" in text or "method set" in text or "draft" in text or "broad" in text:
                weights = {"core": 0.30, "middle": 0.55, "edge": 0.15}

            edge_needles = (
                "api", "exception", "traceback", "error", "shape mismatch", "shape", "dimension",
                "dtype", "cuda", "device", "path", "file not found", "checkpoint", "attribute",
                "unexpected keyword", "base_estimator", "estimator", "version", "undefined",
                "column", "sample_submission",
            )
            if any(n in text for n in edge_needles) or re.search(r"\b[A-Za-z]+Error\b", query_text or ""):
                weights["edge"] += 0.25
                weights["middle"] += 0.05
                weights["core"] = max(0.05, weights["core"] - 0.15)

            broad_needles = ("strategy", "model family", "ensemble", "feature pipeline", "pipeline", "cross validation", "cv", "regularization")
            if any(n in text for n in broad_needles):
                weights["middle"] += 0.20
                weights["core"] += 0.10
                weights["edge"] = max(0.05, weights["edge"] - 0.15)

        return [
            {
                "band": band,
                "radius": self._band_center_radius(band),
                "weight": weight,
                "source": source,
            }
            for band, weight in self._normalize_radius_weights(weights)
        ]

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
                return float(np.quantile(selected, self.geometry_query_radius_quantile))
        return float(np.quantile(radii, self.geometry_query_radius_quantile))

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
        if self.scoring_mode == "euclidean":
            coord = self._euclidean_coords.get(nid)
            if coord is None:
                return float("inf")
            return euclidean_distance(query_point, coord)
        return float("inf")

    def _geometry_similarity(
        self,
        *,
        query_direction: np.ndarray,
        nid: str,
        query_text: str,
        radius_band: str = "",
    ) -> tuple[float, list[dict[str, float | str]]]:
        if self.scoring_mode == "euclidean":
            distance = self._geometry_distance(query_direction, nid)
            if not math.isfinite(distance):
                return 0.0, []
            return 1.0 / (1.0 + distance), [{"band": "euclidean_direction", "distance": distance, "weight": 1.0}]

        distribution = self._predict_query_radius_distribution(query_text, radius_band)
        pieces: list[tuple[float, dict[str, float | str]]] = []
        for item in distribution:
            radius = float(item["radius"])
            weight = float(item["weight"])
            query_point = query_direction * radius
            distance = self._geometry_distance(query_point, nid)
            if not math.isfinite(distance):
                continue
            similarity = 1.0 / (1.0 + distance)
            pieces.append((weight * similarity, {
                "band": str(item["band"]),
                "radius": radius,
                "weight": weight,
                "distance": distance,
            }))
        if not pieces:
            return 0.0, []
        if self.geometry_radius_fusion == "weighted_mean":
            return float(sum(score for score, _ in pieces)), [detail for _, detail in pieces]
        best_score, _best_detail = max(pieces, key=lambda x: x[0])
        return float(best_score), [detail for _, detail in pieces]

    def _token_overlap_score(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return min(1.0, len(left & right) / math.sqrt(max(1, min(len(left), len(right)))))

    def _combined_candidate_score(
        self,
        *,
        nid: str,
        query_tokens: set[str],
        condition_tokens: set[str],
        failure_tokens: set[str],
        geometry_similarity: float | None = None,
    ) -> float:
        node = self.nodes[nid]
        text_tokens = self._node_tokens.get(nid, set())
        cond_tokens = _tokenize(self._node_condition_text(node))
        fail_tokens = _tokenize(self._node_failure_text(node))
        semantic = self._token_overlap_score(query_tokens, text_tokens)
        condition = self._token_overlap_score(condition_tokens, cond_tokens | text_tokens) if condition_tokens else 0.0
        failure = self._token_overlap_score(failure_tokens, fail_tokens | text_tokens) if failure_tokens else 0.0
        evidence = self._node_evidence_score(node)
        reliability = self._reliability_by_id.get(nid, self._node_reliability_score(node))
        if geometry_similarity is None:
            return semantic + condition + failure + 0.25 * reliability + 0.10 * evidence
        score = (
            self.geometry_distance_weight * geometry_similarity
            + self.geometry_semantic_weight * semantic
            + self.geometry_condition_weight * condition
            + self.geometry_failure_weight * failure
            + self.geometry_evidence_weight * evidence
            + self.geometry_reliability_weight * reliability
        )
        if condition_tokens and condition <= 0.0:
            score -= 0.15 * self.geometry_conflict_weight
        return score

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

    def _sort_known_sops(self, ids: list[str], query_text: str, radius_band: str = "") -> list[str]:
        q_tokens = _tokenize(query_text)
        if self.scoring_mode in {"poincare", "flat_twin", "euclidean"}:
            query_direction = self._project_query_direction(query_text)
            return sorted(
                [nid for nid in ids if nid in self.nodes],
                key=lambda nid: (
                    -self._combined_candidate_score(
                        nid=nid,
                        query_tokens=q_tokens,
                        condition_tokens=set(),
                        failure_tokens=set(),
                        geometry_similarity=self._geometry_similarity(
                            query_direction=query_direction,
                            nid=nid,
                            query_text=query_text,
                            radius_band=radius_band,
                        )[0],
                    ),
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
                if r <= 0.60:
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
        geometric = self.scoring_mode in {"poincare", "flat_twin", "euclidean"}
        query_direction: np.ndarray | None = None
        if geometric:
            query_direction = self._project_query_direction(" ".join([query_text, region, condition, failure_mode]))

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

            geometry_similarity = None
            if geometric and query_direction is not None:
                geometry_similarity = self._geometry_similarity(
                    query_direction=query_direction,
                    nid=nid,
                    query_text=" ".join([query_text, region, condition, failure_mode]),
                    radius_band=radius_band,
                )[0]
                if geometry_similarity <= 0.0:
                    continue
            score = self._combined_candidate_score(
                nid=nid,
                query_tokens=q_tokens,
                condition_tokens=condition_tokens,
                failure_tokens=failure_tokens,
                geometry_similarity=geometry_similarity,
            )
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
            "query_radius_distribution": self._predict_query_radius_distribution(query_text + " " + context),
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
            "query_radius_distribution": self._predict_query_radius_distribution(
                " ".join([query_text, region, condition, failure_mode]),
                radius_band,
            ),
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
            "Known candidate SOPs": json.dumps(known_cards, ensure_ascii=False, indent=2),
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
            "llm_tool_calls": 0,
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
            self._last_agentic_pack = pack
            return self._format_agentic_context(pack, ids, task_type=task_type), ids

        started = time.time()
        observations: list[dict[str, Any]] = []
        trace: list[str] = []
        known_ids: list[str] = []
        risk_warnings: list[str] = []
        implementation_hints: list[str] = []
        rejected_sops: list[dict[str, str]] = []
        llm_tool_calls = 0
        last_radius_band = ""

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
            llm_tool_calls += 1
            action_name = str(action.get("action", "finish"))
            trace.append(f"{action_name}({action.get('reason', '')[:120]})")
            if action_name == "navigate":
                last_radius_band = str(action.get("radius_band", "") or last_radius_band)
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
            pack["llm_tool_calls"] = llm_tool_calls
            self._last_agentic_pack = pack
            return self._format_agentic_context(pack, ids, task_type=task_type), ids

        q_tokens = _tokenize(query_text)
        selected = self._sort_known_sops(known_ids, query_text, radius_band=last_radius_band)[: self.top_k]
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
            "llm_tool_calls": llm_tool_calls,
        }
        self._last_agentic_pack = pack
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


class RunForestMemoryLayer:
    """Read-only navigator over Hyperbolic Run-Forest Memory.

    The layer turns run/journal history into a "map path pack":
    matched run paths, selected transitions, attached SOP signposts, risk
    warnings, and evidence refs. It intentionally keeps the artifact read-only.
    """

    def __init__(
        self,
        graph_path: str,
        source_name: str = "run_forest_agentic_memory",
        mode: str = "run_forest_agentic",
        index_path: str = "",
        scoring_mode: str = "poincare",
        enable_agentic: bool = False,
        navigator_max_steps: int = 3,
        navigator_reference_budget: int = 1200,
        top_k: int = 6,
        max_chars: int = 6000,
        include_draft: bool = True,
        include_improve: bool = True,
        include_evolution: bool = True,
        include_debug: bool = True,
        include_fusion: bool = True,
        cfg: Any | None = None,
        **_: Any,
    ) -> None:
        self.graph_path = resolve_graph_path(graph_path)
        self.index_path = resolve_memory_path(index_path, base_dir=self.graph_path.parent) if index_path else self.graph_path.parent / "run_forest_index.npz"
        self.source_name = source_name or "run_forest_agentic_memory"
        self.mode = mode or "run_forest_agentic"
        if self.mode == "run_forest_flat_twin" and scoring_mode == "lexical":
            scoring_mode = "flat_twin"
        elif self.mode == "run_forest_euclidean" and scoring_mode == "lexical":
            scoring_mode = "euclidean"
        elif self.mode.startswith("run_forest") and scoring_mode == "lexical":
            scoring_mode = "poincare"
        self.scoring_mode = (scoring_mode or "poincare").lower()
        if self.scoring_mode not in {"poincare", "flat_twin", "euclidean"}:
            raise ValueError(f"Unsupported run-forest scoring_mode: {self.scoring_mode}")
        self.agentic_enabled = bool(enable_agentic or "agentic" in self.mode or "agentic" in self.source_name)
        if "run_forest" not in self.source_name:
            self.source_name = "run_forest_agentic_memory" if self.agentic_enabled else "run_forest_memory"
        self.navigator_max_steps = max(1, min(3, int(navigator_max_steps or 3)))
        self.navigator_reference_budget = max(200, int(navigator_reference_budget or 1200))
        self.top_k = max(1, int(top_k or 6))
        self.max_chars = max_chars
        self.cfg = cfg
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
        self.out_edges: dict[str, list[tuple[str, str, float, dict[str, Any]]]] = collections.defaultdict(list)
        self.in_edges: dict[str, list[tuple[str, str, float, dict[str, Any]]]] = collections.defaultdict(list)
        self._node_tokens: dict[str, set[str]] = {}
        self._index_node_ids: list[str] = []
        self._poincare_coords: dict[str, np.ndarray] = {}
        self._euclidean_coords: dict[str, np.ndarray] = {}
        self._run_nodes: list[str] = []
        self._transitions: list[str] = []
        self._sops: list[str] = []
        self._evidence: list[str] = []
        self._failure_patterns: list[str] = []
        self._failure_patterns_by_source: dict[str, list[str]] = collections.defaultdict(list)
        self._run_nodes_by_run: dict[str, list[str]] = collections.defaultdict(list)
        self._children_by_node: dict[str, list[str]] = collections.defaultdict(list)
        self._transitions_by_parent: dict[str, list[str]] = collections.defaultdict(list)
        self._transitions_by_child: dict[str, list[str]] = collections.defaultdict(list)
        self._evidence_by_transition: dict[str, list[str]] = collections.defaultdict(list)
        self._last_agentic_pack: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.graph_path.exists():
            raise FileNotFoundError(f"Run-forest graph not found: {self.graph_path}")
        if not self.index_path.exists():
            raise FileNotFoundError(f"Run-forest index not found: {self.index_path}")
        self.graph = json.loads(self.graph_path.read_text(encoding="utf-8"))
        meta = self.graph.get("meta") or {}
        if meta.get("schema") != "hyperbolic_run_forest_memory_v1":
            raise ValueError(f"Not a run-forest memory graph: {self.graph_path}")
        if (
            meta.get("leak_verified") is not True
            or meta.get("paper_grade") is not True
            or meta.get("leak_audited") is not True
            or meta.get("positive_admission_enforced") is not True
        ):
            raise ValueError(
                "Run-Forest memory graph is not source-allowlisted and code-audited; rebuild with "
                "build_run_forest_memory.py --allowlist ... --require-clean-provenance"
            )
        self.nodes = {str(n["id"]): n for n in self.graph.get("nodes", []) if n.get("id")}
        for edge in self.graph.get("edges", []):
            src, dst = str(edge.get("src")), str(edge.get("dst"))
            if src not in self.nodes or dst not in self.nodes:
                continue
            kind = _edge_kind(edge)
            weight = float(edge.get("weight", 1.0))
            self.out_edges[src].append((dst, kind, weight, edge))
            self.in_edges[dst].append((src, kind, weight, edge))
            if kind == "has_failure_pattern":
                self._failure_patterns_by_source[src].append(dst)

        data = np.load(self.index_path, allow_pickle=True)
        node_ids = [str(x) for x in data["node_ids"].tolist()]
        poincare = np.asarray(data["poincare"], dtype=np.float32)
        flat_twin = np.asarray(data["flat_twin"] if "flat_twin" in data.files else poincare, dtype=np.float32)
        euclidean = np.asarray(data["euclidean"], dtype=np.float32)
        if not np.array_equal(poincare, flat_twin):
            raise ValueError("Run-Forest Flat-Twin must use exact same coordinates as Poincare")
        self._index_node_ids = node_ids
        self._poincare_coords = {nid: poincare[i] for i, nid in enumerate(node_ids) if nid in self.nodes}
        self._euclidean_coords = {nid: euclidean[i] for i, nid in enumerate(node_ids) if nid in self.nodes}

        for nid, node in self.nodes.items():
            node_type = str(node.get("type", ""))
            self._node_tokens[nid] = _tokenize(self._node_text(node))
            if node_type == "RunNode":
                self._run_nodes.append(nid)
                self._run_nodes_by_run[str(node.get("run_id"))].append(nid)
                parent_id = node.get("parent_id")
                if parent_id:
                    self._children_by_node[str(parent_id)].append(nid)
            elif node_type == "Transition":
                self._transitions.append(nid)
                self._transitions_by_parent[str(node.get("parent_node_id"))].append(nid)
                self._transitions_by_child[str(node.get("child_node_id"))].append(nid)
            elif node_type == "SOP":
                self._sops.append(nid)
            elif node_type == "Evidence":
                self._evidence.append(nid)
                self._evidence_by_transition[str(node.get("transition_id"))].append(nid)
            elif node_type == "FailurePattern":
                self._failure_patterns.append(nid)
        for values in self._run_nodes_by_run.values():
            values.sort(key=lambda nid: (self.nodes[nid].get("step") or 0, nid))
        for values in self._children_by_node.values():
            values.sort(key=lambda nid: (self.nodes[nid].get("step") or 0, nid))
        logger.info(
            "[RunForestMemory] loaded %s nodes / %s edges from %s (scoring=%s agentic=%s)",
            len(self.nodes),
            len(self.graph.get("edges", [])),
            self.graph_path,
            self.scoring_mode,
            self.agentic_enabled,
        )

    def stage_enabled(self, stage: str) -> bool:
        return bool(self.enabled_stages.get(stage, False))

    def _node_text(self, node: dict[str, Any]) -> str:
        fields = [
            node.get("task"),
            node.get("stage"),
            node.get("stage_pair"),
            node.get("outcome"),
            node.get("plan"),
            node.get("code_summary"),
            node.get("analysis"),
            node.get("terminal_excerpt"),
            node.get("title"),
            node.get("action"),
            node.get("text"),
        ]
        return " ".join(str(v) for v in fields if v)

    def _token_overlap(self, left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / math.sqrt(max(1, min(len(left), len(right))))

    def _coords(self) -> dict[str, np.ndarray]:
        return self._euclidean_coords if self.scoring_mode == "euclidean" else self._poincare_coords

    def _distance(self, query: np.ndarray, candidate: np.ndarray) -> float:
        if self.scoring_mode == "poincare":
            return poincare_distance(query, candidate)
        return euclidean_distance(query, candidate)

    def _task_score(self, node: dict[str, Any], task_id: str, task_desc: str) -> float:
        task = str(node.get("task", "")).lower()
        text = f"{task_id} {task_desc}".lower()
        if task and task in text:
            return 0.35
        task_tokens = _tokenize(task.replace("-", " "))
        query_tokens = _tokenize(text)
        return 0.18 * self._token_overlap(task_tokens, query_tokens)

    def _query_anchor(self, query_text: str, candidate_ids: list[str]) -> np.ndarray | None:
        coords = self._coords()
        q_tokens = _tokenize(query_text)
        scored = []
        for nid in candidate_ids:
            if nid not in coords:
                continue
            lexical = self._token_overlap(q_tokens, self._node_tokens.get(nid, set()))
            if lexical > 0:
                scored.append((lexical, nid))
        if not scored:
            candidate_ids = [nid for nid in candidate_ids if nid in coords]
            if not candidate_ids:
                return None
            return np.mean(np.vstack([coords[nid] for nid in candidate_ids[: min(8, len(candidate_ids))]]), axis=0)
        scored.sort(reverse=True)
        top = scored[:8]
        weights = np.asarray([score for score, _nid in top], dtype=np.float32)
        weights = weights / max(float(weights.sum()), 1e-8)
        anchor = np.sum(np.vstack([coords[nid] for _score, nid in top]) * weights[:, None], axis=0)
        if self.scoring_mode in {"poincare", "flat_twin"}:
            norm = float(np.linalg.norm(anchor))
            if norm >= 0.985:
                anchor = anchor / max(norm, 1e-8) * 0.985
        return anchor.astype(np.float32)

    def _rank(
        self,
        *,
        query_text: str,
        candidate_ids: list[str],
        task_id: str,
        task_desc: str,
        top_k: int,
        stage_bonus: dict[str, float] | None = None,
        outcome_bonus: dict[str, float] | None = None,
    ) -> list[str]:
        coords = self._coords()
        anchor = self._query_anchor(query_text, candidate_ids)
        q_tokens = _tokenize(query_text)
        scored: list[tuple[float, str]] = []
        for nid in candidate_ids:
            if nid not in self.nodes:
                continue
            node = self.nodes[nid]
            lexical = self._token_overlap(q_tokens, self._node_tokens.get(nid, set()))
            task = self._task_score(node, task_id, task_desc)
            stage = str(node.get("stage") or node.get("stage_pair") or "")
            outcome = str(node.get("outcome") or "")
            bonus = (stage_bonus or {}).get(stage, 0.0) + (outcome_bonus or {}).get(outcome, 0.0)
            metric_improvement = node.get("metric_improvement")
            if isinstance(metric_improvement, (int, float)) and metric_improvement > 0:
                bonus += 0.08
            geometry = 0.0
            if anchor is not None and nid in coords:
                geometry = 1.0 / (1.0 + self._distance(anchor, coords[nid]))
            score = 0.50 * geometry + 0.32 * lexical + task + bonus
            scored.append((score, nid))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [nid for _score, nid in scored[:top_k]]

    def _ancestor_path(self, node_id: str, max_hops: int = 4) -> list[str]:
        path = [node_id]
        cur = node_id
        while len(path) < max_hops:
            parent = self.nodes.get(cur, {}).get("parent_id")
            if not parent or parent not in self.nodes:
                break
            path.append(str(parent))
            cur = str(parent)
        return list(reversed(path))

    def _path_label(self, path: list[str]) -> str:
        labels = []
        for nid in path:
            node = self.nodes[nid]
            stage = node.get("stage", node.get("type", "node"))
            step = node.get("step")
            labels.append(f"{node.get('run_short_id', node.get('run_id', 'run'))}/T{step}:{stage}:{nid.split('::')[-1][:8]}")
        return " -> ".join(labels)

    def _transition_card(self, transition_id: str) -> dict[str, Any]:
        node = self.nodes[transition_id]
        return {
            "id": transition_id,
            "stage_pair": node.get("stage_pair"),
            "outcome": node.get("outcome"),
            "metric_delta": node.get("metric_delta"),
            "metric_improvement": node.get("metric_improvement"),
            "parent_buggy": node.get("parent_buggy"),
            "child_buggy": node.get("child_buggy"),
            "attached_sop_ids": node.get("attached_sop_ids") or [],
            "evidence_refs": self._evidence_by_transition.get(transition_id, [])[:3],
        }

    def _positive_memory_eligible(self, node: dict[str, Any]) -> bool:
        audit = node.get("leakage_audit")
        if isinstance(audit, dict) and audit:
            return (
                audit.get("status") == "clean"
                and audit.get("memory_disposition") == "positive_eligible"
                and audit.get("paper_grade_eligible") is True
            )
        return False

    def structural_failure_patterns(self, code: str) -> list[dict[str, Any]]:
        digest = structural_sha256(code)
        if not digest:
            return []
        return [
            node for node in self.nodes.values()
            if node.get("type") == "FailurePattern"
            and node.get("structural_sha256") == digest
        ]

    def _failure_pattern_card(self, pattern_id: str) -> dict[str, Any]:
        node = self.nodes.get(pattern_id, {})
        return {
            "id": pattern_id,
            "issue_code": node.get("issue_code"),
            "category": node.get("category"),
            "severity": node.get("severity"),
            "evidence": node.get("evidence"),
            "remediation": node.get("remediation"),
            "code_sha256": node.get("code_sha256"),
        }

    def inspect_forest(self, *, task_id: str, task_desc: str, query_text: str) -> dict[str, Any]:
        q_tokens = _tokenize(query_text)
        task_counts = collections.Counter(str(n.get("task", "unknown")) for n in self.nodes.values() if n.get("type") == "RunNode")
        stage_counts = collections.Counter(str(n.get("stage", "unknown")) for n in self.nodes.values() if n.get("type") == "RunNode")
        outcome_counts = collections.Counter(str(n.get("outcome", "unknown")) for n in self.nodes.values() if n.get("type") == "Transition")
        candidate_runs = collections.Counter()
        for nid in self._run_nodes:
            node = self.nodes[nid]
            score = self._token_overlap(q_tokens, self._node_tokens.get(nid, set())) + self._task_score(node, task_id, task_desc)
            if score > 0:
                candidate_runs[str(node.get("run_id"))] += score
        return {
            "tool": "inspect_forest",
            "scoring_mode": self.scoring_mode,
            "node_counts": collections.Counter(str(n.get("type", "unknown")) for n in self.nodes.values()),
            "task_counts_top": task_counts.most_common(8),
            "stage_counts": dict(stage_counts),
            "transition_outcomes": dict(outcome_counts),
            "matched_runs": candidate_runs.most_common(6),
            "navigation_suggestions": {
                "draft": "look for task-level successful branches and reusable SOP signposts",
                "improve": "look for similar local-best lineage and metric-improving transitions",
                "debug": "look for similar failed nodes, then expand explicit child transitions for fixes",
            },
        }

    def _pack_for_draft(self, *, task_id: str, task_desc: str, query_text: str, strategy: str) -> dict[str, Any]:
        candidates = [
            nid for nid in self._run_nodes
            if self.nodes[nid].get("stage") in {"draft", "improve", "evolution"}
            and self._positive_memory_eligible(self.nodes[nid])
        ]
        selected_nodes = self._rank(
            query_text=query_text,
            candidate_ids=candidates,
            task_id=task_id,
            task_desc=task_desc,
            top_k=self.top_k,
            stage_bonus={"draft": 0.08, "improve": 0.04, "evolution": 0.04},
        )
        transitions = []
        for nid in selected_nodes:
            transitions += self._transitions_by_child.get(nid, [])
        return self._build_pack("draft_task_successful_branches", selected_nodes, transitions[: self.top_k], strategy)

    def _pack_for_improve(self, *, task_id: str, task_desc: str, query_text: str, strategy: str) -> dict[str, Any]:
        candidates = [
            nid for nid in self._run_nodes
            if self._positive_memory_eligible(self.nodes[nid])
            and (
                self.nodes[nid].get("local_best_node_id")
                or self.nodes[nid].get("metric_improvement") is not None
            )
        ]
        selected_nodes = self._rank(
            query_text=query_text,
            candidate_ids=candidates,
            task_id=task_id,
            task_desc=task_desc,
            top_k=self.top_k,
            stage_bonus={"improve": 0.10, "evolution": 0.06},
        )
        transitions = []
        for nid in selected_nodes:
            transitions += self._transitions_by_child.get(nid, [])
            best = self.nodes[nid].get("local_best_node_id")
            if best:
                transitions += self._transitions_by_child.get(str(best), [])
        return self._build_pack("improve_local_best_lineage", selected_nodes, transitions[: self.top_k], strategy)

    def _pack_for_debug(self, *, task_id: str, task_desc: str, query_text: str, strategy: str) -> dict[str, Any]:
        failed = [
            nid for nid in self._run_nodes
            if self.nodes[nid].get("is_buggy") is True
            or not self._positive_memory_eligible(self.nodes[nid])
            or "error" in self._node_text(self.nodes[nid]).lower()
            or "traceback" in self._node_text(self.nodes[nid]).lower()
        ]
        selected_failed = self._rank(
            query_text=query_text,
            candidate_ids=failed,
            task_id=task_id,
            task_desc=task_desc,
            top_k=max(self.top_k, 8),
            stage_bonus={"debug": 0.10, "improve": 0.04},
        )
        repair_transitions: list[str] = []
        for nid in selected_failed:
            for tid in self._transitions_by_parent.get(nid, []):
                t = self.nodes[tid]
                child = self.nodes.get(str(t.get("child_node_id")), {})
                if t.get("outcome") == "debug_fixed" or child.get("is_buggy") is False:
                    repair_transitions.append(tid)
            repair_transitions += self._transitions_by_child.get(nid, [])
        return self._build_pack("debug_failure_recovery_paths", selected_failed[: self.top_k], repair_transitions[: self.top_k], strategy)

    def _build_pack(self, pack_type: str, selected_nodes: list[str], selected_transitions: list[str], strategy: str) -> dict[str, Any]:
        selected_transitions = list(dict.fromkeys([tid for tid in selected_transitions if tid in self.nodes]))[: self.top_k]
        attached_sops: list[str] = []
        evidence_refs: list[str] = []
        failure_pattern_ids: list[str] = []
        for tid in selected_transitions:
            attached_sops += [sid for sid in self.nodes[tid].get("attached_sop_ids") or [] if sid in self.nodes]
            evidence_refs += self._evidence_by_transition.get(tid, [])
            transition = self.nodes[tid]
            for source_id in [transition.get("parent_node_id"), transition.get("child_node_id")]:
                failure_pattern_ids += self._failure_patterns_by_source.get(str(source_id), [])
        attached_sops = list(dict.fromkeys(attached_sops))[: self.top_k]
        evidence_refs = list(dict.fromkeys(evidence_refs))[: self.top_k]

        matched_paths = []
        for nid in selected_nodes[: self.top_k]:
            matched_paths.append(self._path_label(self._ancestor_path(nid, max_hops=4)))

        warnings = []
        for nid in selected_nodes[: self.top_k]:
            parent = self.nodes[nid].get("parent_id")
            failure_pattern_ids += self._failure_patterns_by_source.get(nid, [])
            if parent:
                siblings = [sid for sid in self._children_by_node.get(str(parent), []) if sid != nid]
                bad_siblings = [sid for sid in siblings if not self._positive_memory_eligible(self.nodes[sid])]
                if bad_siblings:
                    warnings.append(
                        f"{self.nodes[nid].get('run_short_id', '')}/T{self.nodes[nid].get('step')} has {len(bad_siblings)} blocked or protocol-biased sibling attempts under the same parent; inspect the failure patterns before copying."
                    )
                    for sibling_id in bad_siblings:
                        failure_pattern_ids += self._failure_patterns_by_source.get(sibling_id, [])
        failure_pattern_ids = list(dict.fromkeys(fid for fid in failure_pattern_ids if fid in self.nodes))[:6]
        failure_cards = [self._failure_pattern_card(fid) for fid in failure_pattern_ids]
        for card in failure_cards:
            warnings.append(
                f"[{card.get('issue_code')}] {card.get('evidence')} Fix: {card.get('remediation')}"
            )
        transition_cards = [self._transition_card(tid) for tid in selected_transitions]
        return {
            "pack_type": pack_type,
            "strategy": strategy,
            "scoring_mode": self.scoring_mode,
            "matched_run_paths": matched_paths,
            "selected_nodes": selected_nodes[: self.top_k],
            "selected_transitions": selected_transitions,
            "transition_cards": transition_cards,
            "attached_sops": attached_sops,
            "risk_warnings": list(dict.fromkeys(warnings))[:6],
            "evidence_refs": evidence_refs,
            "failure_pattern_ids": failure_pattern_ids,
            "failure_patterns": failure_cards,
            "navigation_trace": ["inspect_forest(context)", f"{pack_type}(strategy={strategy})"],
        }

    def _strategy_action_spec(self):
        from llm import FunctionSpec

        return FunctionSpec(
            name="choose_run_forest_navigation_strategy",
            description="Choose a read-only navigation strategy for Hyperbolic Run-Forest Memory.",
            json_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": ["draft_successful_branches", "improve_local_best_lineage", "debug_failure_recovery", "finish"],
                    },
                    "reason": {"type": "string"},
                    "focus": {"type": "string"},
                    "risk_keywords": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["strategy", "reason"],
            },
        )

    def _llm_choose_strategy(self, *, stage: str, task_id: str, task_desc: str, query_text: str, forest_obs: dict[str, Any]) -> dict[str, Any]:
        if self.cfg is None:
            raise RuntimeError("cfg is required for LLM run-forest navigation")
        from llm import query

        system = (
            "You are a read-only Memory Navigator for an ML coding agent. "
            "You inspect a hyperbolic run forest and choose one navigation strategy. "
            "Use DeepSeek/OpenAI-compatible function calling when configured. "
            "Do not invent memories; choose a strategy for deterministic map tools to execute."
        )
        user_payload = {
            "stage": stage,
            "task_id": task_id,
            "task_desc": task_desc[:1800],
            "current_context": query_text[-5000:],
            "forest_observation": forest_obs,
            "policy": {
                "debug": "prefer debug_failure_recovery",
                "improve": "prefer improve_local_best_lineage",
                "draft": "prefer draft_successful_branches",
            },
        }
        user = json.dumps(user_payload, ensure_ascii=False, indent=2)
        model = getattr(self.cfg.agent.feedback, "model", None) or getattr(self.cfg.agent.code, "model", "")
        return query(
            system_message=system,
            user_message=user,
            model=model,
            temperature=0.1,
            max_tokens=900,
            func_spec=self._strategy_action_spec(),
            cfg=self.cfg,
        )

    def _retrieve_pack(self, *, stage: str, task_id: str, task_desc: str, query_text: str) -> dict[str, Any]:
        forest_obs = self.inspect_forest(task_id=task_id, task_desc=task_desc, query_text=query_text)
        llm_tool_calls = 0
        strategy = {
            "draft": "draft_successful_branches",
            "improve": "improve_local_best_lineage",
            "evolution": "improve_local_best_lineage",
            "debug": "debug_failure_recovery",
        }.get(stage, "draft_successful_branches")
        llm_reason = "deterministic stage policy"
        if self.agentic_enabled and self.cfg is not None:
            try:
                action = self._llm_choose_strategy(
                    stage=stage,
                    task_id=task_id,
                    task_desc=task_desc,
                    query_text=query_text,
                    forest_obs=forest_obs,
                )
                llm_tool_calls += 1
                chosen = str(action.get("strategy", "") or "")
                if chosen == "draft_successful_branches":
                    strategy = "draft_successful_branches"
                elif chosen == "improve_local_best_lineage":
                    strategy = "improve_local_best_lineage"
                elif chosen == "debug_failure_recovery":
                    strategy = "debug_failure_recovery"
                llm_reason = str(action.get("reason", ""))
            except Exception as exc:
                logger.warning("[RunForestMemory] LLM navigator failed; deterministic fallback: %s", exc)
                llm_reason = f"llm_failed_fallback: {exc}"

        if strategy == "debug_failure_recovery":
            pack = self._pack_for_debug(task_id=task_id, task_desc=task_desc, query_text=query_text, strategy=strategy)
        elif strategy == "improve_local_best_lineage":
            pack = self._pack_for_improve(task_id=task_id, task_desc=task_desc, query_text=query_text, strategy=strategy)
        else:
            pack = self._pack_for_draft(task_id=task_id, task_desc=task_desc, query_text=query_text, strategy=strategy)
        pack["forest_observation"] = forest_obs
        pack["llm_tool_calls"] = llm_tool_calls
        pack["llm_reason"] = llm_reason
        pack["agentic_enabled"] = self.agentic_enabled
        return pack

    def _format_sop(self, sop_id: str) -> str:
        node = self.nodes.get(sop_id, {})
        when = "; ".join(_as_list(node.get("applies_when"))) or str(node.get("condition", ""))
        action = node.get("action") or node.get("principle") or node.get("title") or ""
        return f"{sop_id}: {node.get('title', '')}\n  When: {when}\n  Action: {action}"

    def _format_pack(self, pack: dict[str, Any]) -> str:
        prompt_pack = {
            "matched_run_paths": pack.get("matched_run_paths", [])[:4],
            "selected_transitions": pack.get("selected_transitions", [])[:6],
            "attached_sops": pack.get("attached_sops", [])[:6],
            "risk_warnings": pack.get("risk_warnings", [])[:6],
            "evidence_refs": pack.get("evidence_refs", [])[:6],
            "failure_patterns": pack.get("failure_patterns", [])[:6],
        }
        lines = [
            "## Agentic Run-Forest Memory Navigation",
            "A read-only Memory Navigator inspected the historical run forest before this generation step.",
            "Use this as a map path pack: follow path evidence and SOP signposts only when they match the current code/data/error state.",
            f"Mode: {self.mode}; scoring: {self.scoring_mode}; strategy: {pack.get('strategy')}; llm_tool_calls: {pack.get('llm_tool_calls', 0)}.",
            f"Navigator reason: {pack.get('llm_reason', '')}",
            "",
            "### Map Path Pack JSON",
            json.dumps(prompt_pack, ensure_ascii=False, indent=2),
            "",
            "### Matched Run Paths",
        ]
        for item in pack.get("matched_run_paths", [])[:4]:
            lines.append(f"- {item}")
        lines += ["", "### Selected Transitions"]
        for card in pack.get("transition_cards", [])[:6]:
            lines.append(
                f"- {card.get('id')}: {card.get('stage_pair')} outcome={card.get('outcome')} "
                f"metric_delta={card.get('metric_delta')} metric_improvement={card.get('metric_improvement')}"
            )
        if pack.get("attached_sops"):
            lines += ["", "### Attached SOP Signposts"]
            for sop_id in pack.get("attached_sops", [])[:6]:
                lines.append(self._format_sop(sop_id))
        if pack.get("risk_warnings"):
            lines += ["", "### Risk Warnings"]
            for warning in pack.get("risk_warnings", [])[:6]:
                lines.append(f"- {warning}")
        if pack.get("failure_patterns"):
            lines += ["", "### Leakage / Evaluation Failure Patterns"]
            for card in pack.get("failure_patterns", [])[:6]:
                lines.append(
                    f"- [{card.get('issue_code')}] severity={card.get('severity')}: {card.get('evidence')}"
                )
                lines.append(f"  Required fix: {card.get('remediation')}")
        if pack.get("evidence_refs"):
            lines += ["", "### Evidence Refs"]
            for evidence_id in pack.get("evidence_refs", [])[:6]:
                evidence = self.nodes.get(evidence_id, {})
                lines.append(f"- {evidence_id}: {str(evidence.get('text', ''))[:500]}")
        return "\n".join(lines).strip()

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
        query_text = "\n".join([task_desc or "", *(query_parts or [])])
        pack = self._retrieve_pack(stage=stage, task_id=task_id, task_desc=task_desc, query_text=query_text)
        self._last_agentic_pack = pack
        ref_ids = list(dict.fromkeys(
            list(pack.get("selected_transitions", []))
            + list(pack.get("attached_sops", []))
            + list(pack.get("evidence_refs", []))
            + list(pack.get("selected_nodes", []))
            + list(pack.get("failure_pattern_ids", []))
        ))
        text = self._format_pack(pack)
        if self.max_chars > 0 and len(text) > self.max_chars:
            text = text[: self.max_chars].rstrip() + "\n... (run-forest memory truncated)"
        logger.info(
            "[RunForestMemory] stage=%s strategy=%s refs=%s",
            stage,
            pack.get("strategy"),
            ",".join(ref_ids[:10]),
        )
        return text, ref_ids[: max(self.top_k * 3, 12)]


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
