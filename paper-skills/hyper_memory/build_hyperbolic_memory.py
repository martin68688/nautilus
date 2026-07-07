"""
Build a bootstrap hyperbolic SOP memory from the adapted SkillGraph-C artifact.

This is the v0 bridge requested for the hyperbolic structural-memory branch:
  1. Read graph_skillgraph_c_trace_prereq.json.
  2. Treat compact-card SkillGraph nodes as SOP-like nodes.
  3. Compute initial Poincare/Lorentz coordinates from text direction + evidence radius.
  4. Run one GraphBuilderAgent pass to add Condition / FailureMode / Evidence nodes and
     applies_when / prevents / refines / conflicts_with edges.
  5. Validate patches and write hyper_graph.json, hyper_index.npz, graph_builder_report.json.

The GraphBuilderAgent is deterministic by default so the artifact is reproducible offline.
It is deliberately shaped as patch proposal + programmatic validation: future LLM-backed
builders can emit the same patch schema without being allowed to mutate the graph directly.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO / "paper-skills" / "distillation" / "graph_build" / "graph_skillgraph_c_trace_prereq.json"
DEFAULT_OUT = REPO / "paper-skills" / "hyper_memory"

ALLOWED_EDGE_KINDS = {
    "contains",
    "applies_when",
    "prevents",
    "supported_by",
    "implemented_by",
    "refines",
    "conflicts_with",
    "co_occur",
    "enhance",
    "prereq",
}

FAILURE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("overfitting", ("overfit", "overfitting", "validation loss", "small dataset", "small data", "regularization")),
    ("poor calibration", ("calibration", "overconfident", "label smoothing", "focal loss", "log loss")),
    ("data leakage", ("leak", "leakage", "train fold", "validation fold", "fit on train", "train-only")),
    ("out of memory", ("oom", "out of memory", "memory", "gpu memory", "cuda memory")),
    ("timeout or slow execution", ("timeout", "slow", "speed", "fast", "num_workers", "cache", "runtime")),
    ("training instability", ("unstable", "stabilize", "gradient", "clipping", "nan", "overflow", "amp")),
    ("underfitting or low capacity", ("underfit", "capacity", "smaller model", "small model", "distilbert")),
    ("syntax or generated-code artifact", ("syntax", "merge marker", "artifact", "undefined", "duplicate", "script-order")),
    ("api or checkpoint mismatch", ("attribute", "api", "checkpoint", "load", "path", "file not found", "pretrained")),
    ("submission format mismatch", ("submission", "sample_submission", "column", "csv", "format")),
    ("noisy or dead-code features", ("dead-code", "unused", "noisy", "handcrafted", "feature not connected")),
]

OPPOSING_TERMS: tuple[tuple[str, str], ...] = (
    ("large", "small"),
    ("larger", "smaller"),
    ("complex", "simple"),
    ("full", "partial"),
    ("increase", "reduce"),
    ("enable", "disable"),
    ("freeze", "unfreeze"),
)


def slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:max_len] or "empty"


def stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{2,}", (text or "").lower()))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except Exception:
        return str(path)


def radius_band(radius: float) -> str:
    if radius <= 0.35:
        return "core"
    if radius <= 0.70:
        return "middle"
    return "edge"


def poincare_to_lorentz(points: np.ndarray) -> np.ndarray:
    norms2 = np.sum(points * points, axis=1, keepdims=True)
    norms2 = np.clip(norms2, 0.0, 0.98 ** 2)
    denom = np.maximum(1e-9, 1.0 - norms2)
    time = (1.0 + norms2) / denom
    space = (2.0 * points) / denom
    return np.concatenate([time, space], axis=1)


def direction_to_angles(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return spherical angles for the first three direction dimensions.

    theta is the azimuth around the x/y plane in [0, 2*pi). phi is the polar
    angle from the positive z axis in [0, pi]. The direction vector remains the
    canonical coordinate carrier; these angles make the "theta/radius" map view
    explicit in JSON and npz artifacts.
    """
    xyz = directions[:, :3]
    if xyz.shape[1] < 3:
        xyz = np.pad(xyz, ((0, 0), (0, 3 - xyz.shape[1])), constant_values=0.0)
    norms = np.maximum(np.linalg.norm(xyz, axis=1), 1e-9)
    theta = np.mod(np.arctan2(xyz[:, 1], xyz[:, 0]), 2.0 * math.pi)
    phi = np.arccos(np.clip(xyz[:, 2] / norms, -1.0, 1.0))
    return theta.astype(np.float32), phi.astype(np.float32)


def node_text_for_direction(node: dict[str, Any]) -> str:
    return " ".join(
        clean_text(node.get(k, ""))
        for k in ("title", "principle", "condition", "category", "scope")
    )


def infer_failure_modes(node: dict[str, Any]) -> list[str]:
    text = " ".join(clean_text(node.get(k, "")) for k in ("title", "principle", "condition")).lower()
    modes = [label for label, needles in FAILURE_RULES if any(n in text for n in needles)]
    if not modes:
        if node.get("category") == "general":
            modes = ["general execution failure"]
        else:
            modes = [f"{node.get('category', 'task')} method failure"]
    return list(dict.fromkeys(modes))[:3]


def build_text_directions(nodes: list[dict[str, Any]], dims: int = 16) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    corpus = [
        node_text_for_direction(n)
        for n in nodes
    ]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=4096)
    tfidf = vectorizer.fit_transform(corpus)
    n_components = min(dims, max(1, min(tfidf.shape) - 1))
    svd = None
    if n_components >= 1 and tfidf.shape[1] > 1:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        dense = svd.fit_transform(tfidf)
        explained = [float(x) for x in getattr(svd, "explained_variance_ratio_", [])]
    else:
        dense = tfidf.toarray()[:, :1]
        explained = []
    if dense.shape[1] < dims:
        dense = np.pad(dense, ((0, 0), (0, dims - dense.shape[1])), constant_values=0.0)

    directions = normalize(dense, norm="l2")
    for i, row in enumerate(directions):
        if not np.isfinite(row).all() or np.linalg.norm(row) < 1e-8:
            seed = int(hashlib.md5(nodes[i]["id"].encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            fallback = rng.normal(size=dims)
            directions[i] = fallback / np.linalg.norm(fallback)
    meta = {
        "method": "tfidf_truncated_svd",
        "dims": dims,
        "actual_components": int(n_components),
        "n_terms": int(tfidf.shape[1]),
        "explained_variance_ratio": explained,
    }
    text_model = {
        "version": "hyper_text_model_v1",
        "method": "tfidf_truncated_svd",
        "fields": ["title", "principle", "condition", "category", "scope"],
        "dims": dims,
        "vectorizer": vectorizer,
        "svd": svd,
    }
    return directions.astype(np.float32), meta, text_model


def coordinate_quality_report(
    *,
    nodes: list[dict[str, Any]],
    directions: np.ndarray,
    theta: np.ndarray,
    n_bins: int = 16,
    rng_seed: int = 42,
) -> dict[str, Any]:
    centered = directions - directions.mean(axis=0, keepdims=True)
    try:
        singular = np.linalg.svd(centered, compute_uv=False)
    except Exception:
        singular = np.zeros((0,), dtype=np.float32)
    if singular.size and float(np.sum(singular ** 2)) > 0:
        probs = (singular ** 2) / np.sum(singular ** 2)
        effective_rank = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))
    else:
        effective_rank = 0.0

    bins = np.floor((theta % (2.0 * math.pi)) / (2.0 * math.pi) * n_bins).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    bin_counts = collections.Counter(int(b) for b in bins)
    top2_mass = sum(c for _, c in bin_counts.most_common(2)) / max(1, len(nodes))

    labels = []
    for node in nodes:
        category = clean_text(node.get("category"))
        failures = set(infer_failure_modes(node))
        labels.append((category, failures))

    sims = directions @ directions.T
    np.fill_diagonal(sims, -np.inf)
    k = min(10, max(1, len(nodes) - 1))
    nn_hits = []
    for i, (_category, failures) in enumerate(labels):
        nn = np.argsort(-sims[i])[:k]
        hits = 0
        for j in nn:
            same_category = labels[j][0] == labels[i][0] and bool(labels[i][0])
            shared_failure = bool(failures & labels[j][1])
            hits += int(same_category or shared_failure)
        nn_hits.append(hits / k)
    nn_coherence = float(np.mean(nn_hits)) if nn_hits else 0.0

    rng = np.random.default_rng(rng_seed)
    random_hits = []
    all_idx = np.arange(len(nodes))
    for i, (_category, failures) in enumerate(labels):
        pool = np.delete(all_idx, i)
        if pool.size == 0:
            continue
        sample = rng.choice(pool, size=min(k, pool.size), replace=False)
        hits = 0
        for j in sample:
            same_category = labels[j][0] == labels[i][0] and bool(labels[i][0])
            shared_failure = bool(failures & labels[j][1])
            hits += int(same_category or shared_failure)
        random_hits.append(hits / max(1, len(sample)))
    random_coherence = float(np.mean(random_hits)) if random_hits else 0.0
    coherence_lift = nn_coherence - random_coherence

    thresholds = {
        "effective_rank_min": 6.0,
        "top2_bin_mass_max": 0.55,
        "neighbor_coherence_lift_min": 0.10,
    }
    passed = (
        effective_rank >= thresholds["effective_rank_min"]
        and top2_mass <= thresholds["top2_bin_mass_max"]
        and coherence_lift >= thresholds["neighbor_coherence_lift_min"]
    )
    return {
        "status": "passed" if passed else "coordinate_quality_null",
        "passed": passed,
        "thresholds": thresholds,
        "direction_effective_rank": effective_rank,
        "singular_values": [float(x) for x in singular[: min(16, len(singular))]],
        "theta_bin_count": n_bins,
        "theta_bin_counts": {str(k): int(v) for k, v in sorted(bin_counts.items())},
        "theta_top2_bin_mass": top2_mass,
        "neighbor_top_k": k,
        "neighbor_category_or_failure_coherence": nn_coherence,
        "random_category_or_failure_coherence": random_coherence,
        "neighbor_coherence_lift": coherence_lift,
        "note": "Geometry ablation is not claim-grade unless this report passes.",
    }


def compute_radii(nodes: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    max_use = max(float(n.get("n_use", 0) or 0) for n in nodes) or 1.0
    max_level = max(float(n.get("level", 0) or 0) for n in nodes) or 1.0
    radii = []
    core_scores = []
    for n in nodes:
        p_hat = clamp(float(n.get("p_hat", 0.0) or 0.0), 0.0, 1.0)
        support = math.log1p(float(n.get("n_use", 0) or 0.0)) / math.log1p(max_use)
        level_norm = float(n.get("level", 0) or 0.0) / max_level
        general_bonus = 0.05 if n.get("category") == "general" or n.get("scope") == "universal_general" else 0.0
        core = clamp(0.65 * p_hat + 0.25 * support + 0.10 * (1.0 - level_norm) + general_bonus, 0.0, 1.0)
        radius = clamp(0.08 + 0.84 * (1.0 - core), 0.08, 0.92)
        core_scores.append(core)
        radii.append(radius)
    meta = {
        "formula": "radius=clamp(0.08+0.84*(1-(0.65*p_hat+0.25*support+0.10*(1-level_norm)+general_bonus)))",
        "max_n_use": max_use,
        "max_level": max_level,
    }
    return np.asarray(radii, dtype=np.float32), meta


@dataclass
class GraphPatch:
    op: str
    kind: str
    src: str | None = None
    dst: str | None = None
    node: dict[str, Any] | None = None
    weight: float = 1.0
    reason: str = ""
    evidence: str = ""
    attrs: dict[str, Any] = field(default_factory=dict)


class GraphBuilderAgent:
    """One-pass patch builder over SOP-like compact cards.

    The class is intentionally named as an agent: it inspects SOP cards, proposes
    graph patches, and never mutates the graph directly. Programmatic validation
    decides which patches are allowed to land.
    """

    def __init__(self, nodes: list[dict[str, Any]], base_edges: list[dict[str, Any]]) -> None:
        self.nodes = nodes
        self.node_by_id = {n["id"]: n for n in nodes}
        self.base_edges = base_edges
        self.condition_for_sop: dict[str, str] = {}
        self.failures_for_sop: dict[str, list[str]] = {}
        self.failure_label_by_id: dict[str, str] = {}

    def propose(self) -> list[GraphPatch]:
        patches: list[GraphPatch] = []
        patches.extend(self._propose_skill_containment())
        patches.extend(self._propose_condition_failure_evidence())
        patches.extend(self._propose_sop_relations())
        return patches

    def _propose_skill_containment(self) -> list[GraphPatch]:
        patches = []
        seen_skills = set()
        for node in self.nodes:
            region = node.get("category") or "unknown"
            scope = node.get("scope") or ""
            skill_key = f"{region}:{scope}" if scope and region == "general" else region
            skill_id = f"skill_{slugify(skill_key, 72)}"
            if skill_id not in seen_skills:
                seen_skills.add(skill_id)
                patches.append(GraphPatch(
                    op="add_node",
                    kind="Skill",
                    node={
                        "id": skill_id,
                        "type": "Skill",
                        "title": skill_key,
                        "category": region,
                        "scope": scope,
                        "description": f"Skill region for {skill_key}",
                    },
                    reason="Skill node anchors SOP-like compact cards into map regions.",
                    evidence=skill_key,
                ))
            patches.append(GraphPatch(
                op="add_edge",
                kind="contains",
                src=skill_id,
                dst=node["id"],
                weight=1.0,
                reason="Every SOP-like card belongs to a Skill region.",
                evidence=f"{node['id']} category={region} scope={scope}",
            ))
        return patches

    def _condition_phrase(self, node: dict[str, Any]) -> str:
        cond = clean_text(node.get("condition", ""))
        if not cond:
            return f"general {node.get('category', 'task')} context"
        return cond[:220]

    def _failure_modes(self, node: dict[str, Any]) -> list[str]:
        return infer_failure_modes(node)

    def _propose_condition_failure_evidence(self) -> list[GraphPatch]:
        patches = []
        for node in self.nodes:
            sid = node["id"]
            cond = self._condition_phrase(node)
            cond_id = stable_id("cond", cond.lower())
            self.condition_for_sop[sid] = cond_id
            patches.append(GraphPatch(
                op="add_node",
                kind="Condition",
                node={"id": cond_id, "type": "Condition", "title": cond, "text": cond},
                reason="Condition extracted from compact-card condition field.",
                evidence=cond,
            ))
            patches.append(GraphPatch(
                op="add_edge",
                kind="applies_when",
                src=sid,
                dst=cond_id,
                weight=0.85,
                reason="SOP applies under its distilled condition.",
                evidence=cond,
            ))

            failure_modes = self._failure_modes(node)
            fail_ids = []
            for failure in failure_modes:
                fail_id = stable_id("fail", failure.lower())
                self.failure_label_by_id[fail_id] = failure
                fail_ids.append(fail_id)
                patches.append(GraphPatch(
                    op="add_node",
                    kind="FailureMode",
                    node={"id": fail_id, "type": "FailureMode", "title": failure, "text": failure},
                    reason="FailureMode inferred from SOP title/principle/condition.",
                    evidence=" | ".join([clean_text(node.get("title")), clean_text(node.get("principle")), clean_text(node.get("condition"))])[:500],
                ))
                patches.append(GraphPatch(
                    op="add_edge",
                    kind="prevents",
                    src=sid,
                    dst=fail_id,
                    weight=0.70,
                    reason=f"SOP is relevant to preventing or mitigating {failure}.",
                    evidence=clean_text(node.get("principle") or node.get("title")),
                ))
            self.failures_for_sop[sid] = fail_ids

            ev_id = f"ev_{sid}"
            patches.append(GraphPatch(
                op="add_node",
                kind="Evidence",
                node={
                    "id": ev_id,
                    "type": "Evidence",
                    "title": f"SkillGraph-C evidence for {sid}",
                    "source": "graph_skillgraph_c_trace_prereq",
                    "n_use": node.get("n_use", 0),
                    "n_succ": node.get("n_succ", 0),
                    "p_hat": node.get("p_hat", 0.0),
                    "level": node.get("level", 0),
                },
                reason="Trace-proxy stats are attached as explicit evidence nodes.",
                evidence=f"n_use={node.get('n_use',0)} n_succ={node.get('n_succ',0)} p_hat={node.get('p_hat',0)}",
            ))
            patches.append(GraphPatch(
                op="add_edge",
                kind="supported_by",
                src=sid,
                dst=ev_id,
                weight=0.75,
                reason="SOP is supported by SkillGraph-C trace-proxy evidence.",
                evidence=f"n_use={node.get('n_use',0)} n_succ={node.get('n_succ',0)} p_hat={node.get('p_hat',0)}",
            ))
        return patches

    def _opposition_terms(self, a: dict[str, Any], b: dict[str, Any]) -> list[str]:
        a_tokens = tokenize(" ".join([clean_text(a.get("title")), clean_text(a.get("principle"))]))
        b_tokens = tokenize(" ".join([clean_text(b.get("title")), clean_text(b.get("principle"))]))
        terms = []
        for x, y in OPPOSING_TERMS:
            if (x in a_tokens and y in b_tokens) or (y in a_tokens and x in b_tokens):
                terms.append(f"{x}/{y}")
        return terms

    def _non_generic_failures(self, failure_ids: set[str]) -> set[str]:
        out = set()
        for fid in failure_ids:
            label = self.failure_label_by_id.get(fid, "")
            if label == "general execution failure":
                continue
            if label.endswith(" method failure"):
                continue
            out.add(fid)
        return out

    def _token_jaccard(self, a: dict[str, Any], b: dict[str, Any]) -> float:
        at = tokenize(" ".join([clean_text(a.get("title")), clean_text(a.get("principle"))]))
        bt = tokenize(" ".join([clean_text(b.get("title")), clean_text(b.get("principle"))]))
        return len(at & bt) / len(at | bt) if (at | bt) else 0.0

    def _propose_sop_relations(self) -> list[GraphPatch]:
        patches = []
        by_category = collections.defaultdict(list)
        for node in self.nodes:
            by_category[node.get("category", "unknown")].append(node)

        refine_out_count: collections.Counter[str] = collections.Counter()
        for category, group in by_category.items():
            for i, a in enumerate(group):
                for b in group[i + 1:]:
                    a_fail = set(self.failures_for_sop.get(a["id"], []))
                    b_fail = set(self.failures_for_sop.get(b["id"], []))
                    shared_failures = a_fail & b_fail
                    strong_shared_failures = self._non_generic_failures(shared_failures)
                    same_failure = bool(strong_shared_failures)
                    same_condition = self.condition_for_sop.get(a["id"]) == self.condition_for_sop.get(b["id"])
                    if not (same_failure or same_condition):
                        continue
                    opposition_terms = self._opposition_terms(a, b)
                    if opposition_terms and (same_failure or same_condition):
                        patches.append(GraphPatch(
                            op="add_edge",
                            kind="conflicts_with",
                            src=a["id"],
                            dst=b["id"],
                            weight=0.60,
                            reason="GraphBuilderAgent found same condition/failure context with opposing action words.",
                            evidence=f"{a.get('title')} || {b.get('title')}",
                            attrs={
                                "undirected": True,
                                "category": category,
                                "same_condition": same_condition,
                                "shared_failure": sorted(strong_shared_failures),
                                "opposition_terms": opposition_terms,
                            },
                        ))
                    elif same_failure and not same_condition:
                        # Keep this sparse: only connect nearby evidence levels to avoid a dense clique.
                        if (
                            abs(int(a.get("level", 0) or 0) - int(b.get("level", 0) or 0)) <= 2
                            and self._token_jaccard(a, b) >= 0.12
                        ):
                            src, dst = (a, b) if int(a.get("level", 0) or 0) <= int(b.get("level", 0) or 0) else (b, a)
                            if refine_out_count[src["id"]] >= 3:
                                continue
                            refine_out_count[src["id"]] += 1
                            patches.append(GraphPatch(
                                op="add_edge",
                                kind="refines",
                                src=src["id"],
                                dst=dst["id"],
                                weight=0.45,
                                reason="SOPs share a FailureMode but differ in condition, suggesting a condition-specific refinement.",
                                evidence=f"{src.get('title')} -> {dst.get('title')}",
                                attrs={"shared_failure": sorted(a_fail & b_fail)},
                            ))
        return patches


def build_sop_node(
    node: dict[str, Any],
    radius: float,
    direction: np.ndarray,
    theta: float,
    phi: float,
    poincare: np.ndarray,
    lorentz: np.ndarray,
) -> dict[str, Any]:
    return {
        "id": node["id"],
        "type": "SOP",
        "title": clean_text(node.get("title")),
        "action": clean_text(node.get("principle")),
        "principle": clean_text(node.get("principle")),
        "applies_when": [clean_text(node.get("condition"))] if clean_text(node.get("condition")) else [],
        "condition": clean_text(node.get("condition")),
        "category": node.get("category"),
        "scope": node.get("scope", ""),
        "skill_id": f"skill_{slugify((node.get('category') or 'unknown') + ((':' + node.get('scope')) if node.get('scope') and node.get('category') == 'general' else ''), 72)}",
        "source_node_id": node["id"],
        "metric": {
            "p_hat": node.get("p_hat", 0.0),
            "n_use": node.get("n_use", 0),
            "n_succ": node.get("n_succ", 0),
            "level": node.get("level", 0),
            "signal": "skillgraph_trace_proxy",
        },
        "confidence": "high" if float(node.get("p_hat", 0) or 0) >= 0.67 else ("medium" if float(node.get("p_hat", 0) or 0) >= 0.34 else "low"),
        "radius": round(float(radius), 6),
        "radius_band": radius_band(float(radius)),
        "angle_theta": round(float(theta), 6),
        "angle_phi": round(float(phi), 6),
        "angle_direction": [round(float(x), 6) for x in direction.tolist()],
        "poincare": [round(float(x), 6) for x in poincare.tolist()],
        "lorentz": [round(float(x), 6) for x in lorentz.tolist()],
    }


def patch_to_edge(patch: GraphPatch) -> dict[str, Any]:
    edge = {
        "src": patch.src,
        "dst": patch.dst,
        "kind": patch.kind,
        "weight": round(float(patch.weight), 6),
        "provenance": "graph_builder_agent",
        "reason": patch.reason,
        "evidence": patch.evidence,
    }
    edge.update(patch.attrs)
    return edge


def validate_and_apply_patches(
    *,
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
    patches: list[GraphPatch],
) -> dict[str, Any]:
    applied = []
    rejected = []
    skipped = []
    seen_edges = {(e.get("src"), e.get("dst"), e.get("kind")) for e in edges}

    def reject(patch: GraphPatch, reason: str) -> None:
        rejected.append({"op": patch.op, "kind": patch.kind, "src": patch.src, "dst": patch.dst, "reason": reason})

    def skip(patch: GraphPatch, reason: str) -> None:
        skipped.append({"op": patch.op, "kind": patch.kind, "src": patch.src, "dst": patch.dst, "reason": reason})

    for patch in patches:
        if patch.kind not in ALLOWED_EDGE_KINDS and patch.op == "add_edge":
            reject(patch, "illegal edge kind")
            continue
        if not patch.evidence:
            reject(patch, "missing text evidence")
            continue
        if patch.op == "add_node":
            if not patch.node or not patch.node.get("id") or not patch.node.get("type"):
                reject(patch, "invalid node patch")
                continue
            nid = patch.node["id"]
            if nid not in nodes:
                nodes[nid] = patch.node
                applied.append({"op": "add_node", "kind": patch.kind, "id": nid})
            else:
                skip(patch, "duplicate node id")
            continue
        if patch.op != "add_edge":
            reject(patch, "unknown op")
            continue
        if patch.src not in nodes or patch.dst not in nodes:
            reject(patch, "src or dst node does not exist")
            continue
        if patch.kind == "conflicts_with":
            if not patch.reason or not patch.evidence or not (patch.attrs.get("opposition_terms") or "opposing" in patch.reason.lower()):
                reject(patch, "conflicts_with requires explicit opposition evidence")
                continue
            if not (patch.attrs.get("same_condition") or patch.attrs.get("shared_failure")):
                reject(patch, "conflicts_with requires same condition or shared failure context")
                continue
        key = (patch.src, patch.dst, patch.kind)
        if key in seen_edges:
            skip(patch, "duplicate edge")
            continue
        seen_edges.add(key)
        edge = patch_to_edge(patch)
        edges.append(edge)
        applied.append({"op": "add_edge", "kind": patch.kind, "src": patch.src, "dst": patch.dst})

    return {
        "applied": applied,
        "rejected": rejected,
        "skipped": skipped,
        "applied_by_kind": dict(collections.Counter(x["kind"] for x in applied)),
        "rejected_by_reason": dict(collections.Counter(x["reason"] for x in rejected)),
        "skipped_by_reason": dict(collections.Counter(x["reason"] for x in skipped)),
    }


def convert_base_edges(base_edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for edge in base_edges:
        kind = edge.get("kind") or edge.get("type")
        if kind not in {"co_occur", "enhance", "prereq"}:
            continue
        converted.append({
            "src": edge.get("src"),
            "dst": edge.get("dst"),
            "kind": kind,
            "weight": float(edge.get("weight", 1.0)),
            "provenance": "skillgraph_c_base",
        })
    return converted


def provenance_report(graph: dict[str, Any], source_nodes: list[dict[str, Any]]) -> dict[str, Any]:
    meta = graph.get("meta", {}) or {}
    required_meta = ["source_runs", "allowlist", "leak_verified"]
    missing_meta = [k for k in required_meta if k not in meta]
    nodes_with_source = [
        n["id"]
        for n in source_nodes
        if n.get("source_branches") or n.get("evidence_turns")
    ]
    clean = not missing_meta and len(nodes_with_source) == len(source_nodes) and bool(meta.get("leak_verified"))
    return {
        "status": "clean_certified" if clean else "uncertified_bootstrap",
        "paper_grade": clean,
        "missing_meta_fields": missing_meta,
        "nodes_with_source_evidence": len(nodes_with_source),
        "nodes_total": len(source_nodes),
        "source_runs": meta.get("source_runs", []),
        "allowlist": meta.get("allowlist", []),
        "leak_verified": meta.get("leak_verified", False),
        "note": "Current artifact is paper-grade only when every SOP has source evidence and the graph meta records clean-run provenance.",
    }


def build(input_path: Path, output_dir: Path, dims: int = 16, require_clean_provenance: bool = False) -> dict[str, Any]:
    graph = json.loads(input_path.read_text(encoding="utf-8"))
    source_nodes = graph.get("nodes", [])
    source_edges = graph.get("edges", [])
    if not source_nodes:
        raise ValueError(f"No nodes found in {input_path}")

    provenance = provenance_report(graph, source_nodes)
    if require_clean_provenance and not provenance["paper_grade"]:
        raise ValueError(f"Input graph lacks clean provenance: {provenance}")

    directions, direction_meta, text_model = build_text_directions(source_nodes, dims=dims)
    radii, radius_meta = compute_radii(source_nodes)
    theta, phi = direction_to_angles(directions)
    quality = coordinate_quality_report(nodes=source_nodes, directions=directions, theta=theta)
    poincare = directions * radii[:, None]
    lorentz = poincare_to_lorentz(poincare)
    flat_twin = poincare.copy()

    hyper_nodes: dict[str, dict[str, Any]] = {}
    node_ids = []
    for i, node in enumerate(source_nodes):
        sop = build_sop_node(node, radii[i], directions[i], theta[i], phi[i], poincare[i], lorentz[i])
        hyper_nodes[sop["id"]] = sop
        node_ids.append(sop["id"])

    hyper_edges = convert_base_edges(source_edges)
    agent = GraphBuilderAgent(source_nodes, source_edges)
    patches = agent.propose()
    patch_result = validate_and_apply_patches(nodes=hyper_nodes, edges=hyper_edges, patches=patches)

    node_type_counts = dict(collections.Counter(n.get("type", "unknown") for n in hyper_nodes.values()))
    edge_kind_counts = dict(collections.Counter(e.get("kind", "unknown") for e in hyper_edges))
    hyper_graph = {
        "meta": {
            "schema": "hyperbolic-sop-memory-v0",
            "source_graph": display_path(input_path),
            "source_schema": graph.get("meta", {}).get("schema"),
            "provenance_status": provenance["status"],
            "paper_grade": provenance["paper_grade"],
            "builder": "GraphBuilderAgent(heuristic_patch_v1)+programmatic_validation",
            "coordinate_model": f"Poincare ball + Lorentz hyperboloid, curvature=1, dims={dims}",
            "angle_model": "theta/phi spherical angles over TF-IDF-SVD direction; radius from p_hat/n_use/level",
            "flat_twin_model": "same coordinates as poincare; Flat-Twin swaps only the distance function at runtime",
            "node_count": len(hyper_nodes),
            "edge_count": len(hyper_edges),
            "note": "Bootstrap artifact from SkillGraph-C compact cards; metric is p_hat/n_use proxy, not transition-level metric_delta.",
        },
        "nodes": list(hyper_nodes.values()),
        "edges": hyper_edges,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    graph_path = output_dir / "hyper_graph.json"
    index_path = output_dir / "hyper_index.npz"
    text_model_path = output_dir / "hyper_text_model.joblib"
    report_path = output_dir / "graph_builder_report.json"
    quality_path = output_dir / "coordinate_quality_report.json"

    graph_path.write_text(json.dumps(hyper_graph, ensure_ascii=False, indent=2), encoding="utf-8")
    joblib.dump(text_model, text_model_path)
    np.savez_compressed(
        index_path,
        node_ids=np.asarray(node_ids, dtype=str),
        poincare=poincare.astype(np.float32),
        lorentz=lorentz.astype(np.float32),
        flat_twin=flat_twin.astype(np.float32),
        radius=radii.astype(np.float32),
        theta=theta.astype(np.float32),
        phi=phi.astype(np.float32),
        angle=np.stack([theta, phi], axis=1).astype(np.float32),
        direction=directions.astype(np.float32),
    )

    radius_values = [float(r) for r in radii]
    source_sop_ids = {n["id"] for n in source_nodes}
    all_node_ids = set(hyper_nodes)
    patch_edges = [e for e in hyper_edges if e.get("provenance") == "graph_builder_agent"]
    conflicts = [e for e in hyper_edges if e.get("kind") == "conflicts_with"]
    edge_src_by_kind = collections.defaultdict(set)
    edge_dst_by_kind = collections.defaultdict(set)
    for edge in hyper_edges:
        edge_src_by_kind[edge.get("kind")].add(edge.get("src"))
        edge_dst_by_kind[edge.get("kind")].add(edge.get("dst"))
    report = {
        "input": display_path(input_path),
        "outputs": {
            "hyper_graph": display_path(graph_path),
            "hyper_index": display_path(index_path),
            "hyper_text_model": display_path(text_model_path),
            "graph_builder_report": display_path(report_path),
            "coordinate_quality_report": display_path(quality_path),
        },
        "provenance": provenance,
        "source": {
            "nodes": len(source_nodes),
            "edges": len(source_edges),
            "edge_kinds": dict(collections.Counter(e.get("kind") or e.get("type") for e in source_edges)),
        },
        "hyper_graph": {
            "nodes": len(hyper_nodes),
            "edges": len(hyper_edges),
            "node_type_counts": node_type_counts,
            "edge_kind_counts": edge_kind_counts,
        },
        "coordinates": {
            "direction": direction_meta,
            "angle": {
                "theta": "azimuth atan2(y, x), normalized to [0, 2*pi)",
                "phi": "polar angle arccos(z / ||direction||)",
            },
            "radius": radius_meta,
            "radius_min": min(radius_values),
            "radius_max": max(radius_values),
            "radius_mean": sum(radius_values) / len(radius_values),
            "radius_bands": dict(collections.Counter(radius_band(r) for r in radius_values)),
            "poincare_max_norm": float(np.linalg.norm(poincare, axis=1).max()),
            "lorentz_minkowski_error_max": float(np.max(np.abs(-(lorentz[:, 0] ** 2) + np.sum(lorentz[:, 1:] ** 2, axis=1) + 1.0))),
            "flat_twin_identity": bool(np.array_equal(flat_twin.astype(np.float32), poincare.astype(np.float32))),
            "quality_report": quality,
        },
        "graph_builder_agent": {
            "patches_proposed": len(patches),
            "patches_applied": len(patch_result["applied"]),
            "patches_rejected": len(patch_result["rejected"]),
            "patches_skipped": len(patch_result["skipped"]),
            "applied_by_kind": patch_result["applied_by_kind"],
            "rejected_by_reason": patch_result["rejected_by_reason"],
            "skipped_by_reason": patch_result["skipped_by_reason"],
            "sample_rejections": patch_result["rejected"][:10],
        },
        "validation": {
            "source_sop_ids_preserved": all(
                sid in hyper_nodes and hyper_nodes[sid].get("type") == "SOP"
                for sid in source_sop_ids
            ),
            "all_node_ids_unique": len(hyper_nodes) == len(all_node_ids),
            "all_edge_endpoints_exist": all(
                e.get("src") in all_node_ids and e.get("dst") in all_node_ids
                for e in hyper_edges
            ),
            "all_base_edge_endpoints_exist": all(e.get("src") in hyper_nodes and e.get("dst") in hyper_nodes for e in convert_base_edges(source_edges)),
            "all_edge_kinds_allowed": all(e.get("kind") in ALLOWED_EDGE_KINDS for e in hyper_edges),
            "patch_edges_have_reason_or_evidence": all(
                e.get("reason") or e.get("evidence")
                for e in patch_edges
            ),
            "every_sop_has_skill_region": source_sop_ids <= edge_dst_by_kind["contains"],
            "every_sop_has_condition_edge": source_sop_ids <= edge_src_by_kind["applies_when"],
            "every_sop_has_failure_edge": source_sop_ids <= edge_src_by_kind["prevents"],
            "every_sop_has_evidence_edge": source_sop_ids <= edge_src_by_kind["supported_by"],
            "conflicts_have_reason_and_evidence": all(
                e.get("kind") != "conflicts_with" or (e.get("reason") and e.get("evidence"))
                for e in hyper_edges
            ),
            "conflicts_have_opposition_terms": all(
                e.get("opposition_terms")
                for e in conflicts
            ),
            "conflicts_have_condition_or_failure_context": all(
                e.get("kind") != "conflicts_with" or (e.get("same_condition") or e.get("shared_failure"))
                for e in hyper_edges
            ),
            "flat_twin_same_coordinates_as_poincare": bool(np.array_equal(flat_twin.astype(np.float32), poincare.astype(np.float32))),
            "coordinate_quality_gate_passed": quality["passed"],
            "paper_grade_provenance": provenance["paper_grade"],
        },
    }
    quality_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build bootstrap hyperbolic SOP memory from SkillGraph-C.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dims", type=int, default=16)
    parser.add_argument("--require-clean-provenance", action="store_true")
    args = parser.parse_args()

    report = build(args.input, args.output_dir, dims=args.dims, require_clean_provenance=args.require_clean_provenance)
    print(f"hyper_graph: {report['outputs']['hyper_graph']}")
    print(f"hyper_index: {report['outputs']['hyper_index']}")
    print(f"hyper_text_model: {report['outputs']['hyper_text_model']}")
    print(f"coordinate_quality_report: {report['outputs']['coordinate_quality_report']}")
    print(f"report: {report['outputs']['graph_builder_report']}")
    print(f"nodes: {report['hyper_graph']['nodes']}  edges: {report['hyper_graph']['edges']}")
    print(f"node types: {report['hyper_graph']['node_type_counts']}")
    print(f"edge kinds: {report['hyper_graph']['edge_kind_counts']}")
    print(f"provenance: {report['provenance']['status']}")
    print(f"coordinate quality: {report['coordinates']['quality_report']['status']}")


if __name__ == "__main__":
    main()
