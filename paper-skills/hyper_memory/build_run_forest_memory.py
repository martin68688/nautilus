"""Build Hyperbolic Run-Forest Memory with SOP distillation attachments.

This builder materializes the design:

    Run/journal memory = primary hyperbolic forest
    SOP memory = distilled signposts attached to transitions/subtrees
    Evidence = code/metric/error references opened on demand

It consumes MLEvolve journal.json files plus the existing SOP hyper_graph.json.
The resulting artifact is intentionally read-only and offline-buildable.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


REPO = Path(__file__).resolve().parents[2]
MLEVOLVE_ROOT = REPO / "mlevolve"
if str(MLEVOLVE_ROOT) not in sys.path:
    sys.path.insert(0, str(MLEVOLVE_ROOT))

from agents.leakage_audit import audit_code, merge_audits


DEFAULT_RUNS_DIR = REPO / "mlevolve" / "runs"
DEFAULT_SOP_GRAPH = REPO / "paper-skills" / "hyper_memory" / "hyper_graph.json"
DEFAULT_OUT_DIR = REPO / "paper-skills" / "hyper_memory"
DEFAULT_ALLOWLIST = REPO / "paper-skills" / "eval_skill_memory" / "clean_run_allowlist.json"


def read_json(path: Path) -> dict[str, Any] | list[Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def short_text(value: Any, limit: int = 900) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def stable_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path, *, base: Path = REPO) -> str:
    """Return a stable readable path even when artifacts live outside this clone."""
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def run_id_matches_prefix(run_id: str, prefix: str) -> bool:
    return run_id == prefix or run_id.startswith(f"{prefix}_") or run_id.startswith(f"{prefix}-")


def load_clean_allowlist(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Clean run allowlist not found: {path}")
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Clean run allowlist must be a JSON object: {path}")
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"Clean run allowlist missing entries: {path}")

    allowed_run_ids: set[str] = set()
    allowed_paths: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not entry.get("allowed"):
            continue
        run_id = str(entry.get("run_id") or "").strip()
        if not run_id:
            continue
        allowed_run_ids.add(run_id)
        if entry.get("path"):
            allowed_paths[run_id] = str(entry["path"])
    if not allowed_run_ids:
        raise ValueError(f"Clean run allowlist has no allowed run ids: {path}")

    blocked_prefixes = [
        str(item.get("run_id")).strip()
        for item in data.get("blocked_runs", [])
        if isinstance(item, dict) and str(item.get("run_id", "")).strip()
    ]
    return {
        "path": path,
        "schema": data.get("schema"),
        "status": data.get("status"),
        "hash": file_sha256(path),
        "allowed_run_ids": sorted(allowed_run_ids),
        "allowed_paths": allowed_paths,
        "blocked_prefixes": sorted(set(blocked_prefixes)),
    }


def classify_run_source(run_id: str, provenance: dict[str, Any] | None) -> str:
    if provenance is None:
        return "unrestricted"
    run_id = str(run_id)
    short_id = run_short_id(run_id)
    for prefix in provenance.get("blocked_prefixes", []):
        if run_id_matches_prefix(run_id, prefix) or run_id_matches_prefix(short_id, prefix):
            return "blocked"
    allowed = set(provenance.get("allowed_run_ids", []))
    if run_id in allowed or short_id in allowed:
        return "allowed"
    if any(run_id_matches_prefix(run_id, prefix) for prefix in allowed):
        return "allowed"
    return "not_allowlisted"


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "when",
    "then",
    "than",
    "before",
    "after",
    "using",
    "use",
    "uses",
    "used",
    "set",
    "make",
    "check",
    "ensure",
    "data",
    "model",
    "code",
}


def tokens(text: str) -> set[str]:
    return {
        item
        for item in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", text.lower())
        if item not in STOPWORDS
    }


def token_overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, min(len(left), len(right)))


def evidence_turn_map(sop: dict[str, Any]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = collections.defaultdict(set)
    for item in sop.get("evidence_turns") or []:
        match = re.match(r"B(\d+)\.T(\d+)$", str(item).strip())
        if match:
            out[match.group(1)].add(int(match.group(2)))
    return dict(out)


def run_short_id(run_id: str) -> str:
    parts = run_id.split("_")
    if len(parts) >= 2 and parts[0].isdigit():
        return "_".join(parts[:2])
    return run_id


def task_from_run_id(run_id: str) -> str:
    prefix = run_short_id(run_id)
    task = run_id[len(prefix) :].lstrip("_")
    return task or "unknown"


def metric_value(node: dict[str, Any]) -> float | None:
    metric = node.get("metric")
    if not isinstance(metric, dict):
        return None
    value = metric.get("value")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def metric_maximize(child: dict[str, Any], parent: dict[str, Any] | None = None) -> bool | None:
    for node in (child, parent):
        if not isinstance(node, dict):
            continue
        metric = node.get("metric")
        if isinstance(metric, dict) and isinstance(metric.get("maximize"), bool):
            return bool(metric["maximize"])
    return None


def signed_metric_improvement(child: dict[str, Any], parent: dict[str, Any]) -> float | None:
    child_value = metric_value(child)
    parent_value = metric_value(parent)
    if child_value is None or parent_value is None:
        return None
    raw = child_value - parent_value
    maximize = metric_maximize(child, parent)
    if maximize is False:
        return -raw
    return raw


def node_text(node: dict[str, Any]) -> str:
    return "\n".join(
        [
            short_text(node.get("plan"), 600),
            short_text(node.get("code_summary"), 600),
            short_text(node.get("analysis"), 600),
            short_text("".join(node.get("_term_out") or []), 600),
        ]
    ).strip()


def load_journals(
    runs_dir: Path,
    provenance: dict[str, Any] | None = None,
) -> tuple[list[tuple[str, Path, dict[str, Any]]], dict[str, Any]]:
    rows: list[tuple[str, Path, dict[str, Any]]] = []
    report: dict[str, Any] = {
        "runs_dir": str(runs_dir),
        "discovered_journal_count": 0,
        "included_journal_count": 0,
        "included_source_runs": [],
        "excluded_runs": [],
        "excluded_by_reason": {},
        "missing_allowed_runs": [],
    }
    seen_allowed: set[str] = set()
    allowed = set(provenance.get("allowed_run_ids", [])) if provenance else set()
    for path in sorted(runs_dir.glob("*/logs/journal.json")):
        report["discovered_journal_count"] += 1
        try:
            data = read_json(path)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        nodes = data.get("nodes")
        if not isinstance(nodes, list) or len(nodes) < 2:
            continue
        run_id = path.parents[1].name
        source_status = classify_run_source(run_id, provenance)
        if provenance is not None and source_status != "allowed":
            reason = "blocked_run" if source_status == "blocked" else "not_allowlisted"
            report["excluded_runs"].append(
                {
                    "run_id": run_id,
                    "run_short_id": run_short_id(run_id),
                    "journal_path": str(path),
                    "reason": reason,
                }
            )
            counter = collections.Counter(report["excluded_by_reason"])
            counter[reason] += 1
            report["excluded_by_reason"] = dict(sorted(counter.items()))
            continue
        seen_allowed.add(run_short_id(run_id))
        rows.append((run_id, path, data))
    report["included_journal_count"] = len(rows)
    report["included_source_runs"] = sorted({run_short_id(run_id) for run_id, _path, _data in rows})
    if provenance is not None:
        report["missing_allowed_runs"] = sorted(allowed - seen_allowed)
    return rows, report


def journal_graph(journal: dict[str, Any]) -> dict[str, Any]:
    raw_nodes = [n for n in journal.get("nodes", []) if isinstance(n, dict) and n.get("id")]
    nodes = {str(n["id"]): n for n in raw_nodes}
    parents: dict[str, str] = {}
    for child, parent in (journal.get("node2parent") or {}).items():
        child_id = str(child)
        parent_id = str(parent)
        if child_id in nodes and parent_id in nodes and child_id != parent_id:
            parents[child_id] = parent_id
    children: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for child, parent in parents.items():
        children[parent].append(child)
    for child_list in children.values():
        child_list.sort(key=lambda node_id: (nodes[node_id].get("step") or 0, node_id))
    roots = sorted(
        [node_id for node_id in nodes if node_id not in parents],
        key=lambda node_id: (nodes[node_id].get("step") or 0, node_id),
    )
    depths: dict[str, int] = {}
    queue = collections.deque((root, 0) for root in roots)
    while queue:
        node_id, depth = queue.popleft()
        if node_id in depths and depths[node_id] <= depth:
            continue
        depths[node_id] = depth
        for child in children.get(node_id, []):
            queue.append((child, depth + 1))
    for node_id in nodes:
        depths.setdefault(node_id, 0)
    return {"nodes": nodes, "parents": parents, "children": children, "roots": roots, "depths": depths}


def descendant_leaf_counts(children: dict[str, list[str]], roots: list[str]) -> dict[str, int]:
    memo: dict[str, int] = {}

    def rec(node_id: str) -> int:
        if node_id in memo:
            return memo[node_id]
        child_list = children.get(node_id, [])
        memo[node_id] = 1 if not child_list else sum(rec(child) for child in child_list)
        return memo[node_id]

    for root in roots:
        rec(root)
    return memo


def assign_run_coords(
    graph: dict[str, Any],
    lo: float,
    hi: float,
    edge_len: float = 0.82,
) -> dict[str, np.ndarray]:
    children: dict[str, list[str]] = graph["children"]
    roots: list[str] = graph["roots"]
    depths: dict[str, int] = graph["depths"]
    leaf_counts = descendant_leaf_counts(children, roots)
    coords: dict[str, np.ndarray] = {}

    def assign(node_id: str, span_lo: float, span_hi: float, depth_offset: int = 1) -> None:
        theta = (span_lo + span_hi) / 2.0
        level = depths.get(node_id, 0) + depth_offset
        hyperbolic_radius = min(level * edge_len, 5.1)
        radius = min(math.tanh(hyperbolic_radius / 2.0), 0.988)
        coords[node_id] = np.asarray([radius * math.cos(theta), radius * math.sin(theta)], dtype=np.float64)
        child_list = children.get(node_id, [])
        if not child_list:
            return
        total = sum(leaf_counts.get(child, 1) for child in child_list)
        cur = span_lo
        for child in child_list:
            width = (span_hi - span_lo) * leaf_counts.get(child, 1) / max(1, total)
            assign(child, cur, cur + width, depth_offset=depth_offset)
            cur += width

    total = sum(leaf_counts.get(root, 1) for root in roots)
    cur = lo
    for root in roots:
        width = (hi - lo) * leaf_counts.get(root, 1) / max(1, total)
        assign(root, cur, cur + width)
        cur += width
    return coords


def sop_provenance_status(sop: dict[str, Any], provenance: dict[str, Any] | None) -> str:
    if provenance is None:
        return "unrestricted"
    branches = sop.get("source_branches") or []
    if not branches:
        return "missing_source_branches"
    has_allowed = False
    for branch in branches:
        if not isinstance(branch, list | tuple) or not branch:
            return "malformed_source_branch"
        status = classify_run_source(str(branch[0]), provenance)
        if status == "blocked":
            return "blocked"
        if status != "allowed":
            return "not_allowlisted"
        has_allowed = True
    return "allowed" if has_allowed else "missing_source_branches"


def load_sops(
    graph_path: Path,
    provenance: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], list[str]], dict[str, Any]]:
    if not graph_path.exists():
        return [], {}, {"total_sops": 0, "included_sops": 0, "excluded_sops": []}
    graph = read_json(graph_path)
    if not isinstance(graph, dict):
        return [], {}, {"total_sops": 0, "included_sops": 0, "excluded_sops": []}
    raw_sops = [n for n in graph.get("nodes", []) if isinstance(n, dict) and n.get("type") == "SOP"]
    sops: list[dict[str, Any]] = []
    excluded_sops: list[dict[str, Any]] = []
    for sop in raw_sops:
        status = sop_provenance_status(sop, provenance)
        if provenance is not None and status != "allowed":
            excluded_sops.append(
                {
                    "sop_id": str(sop.get("id")),
                    "title": short_text(sop.get("title"), 160),
                    "reason": status,
                    "source_branches": sop.get("source_branches") or [],
                }
            )
            continue
        sops.append(sop)
    branch_to_sops: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for sop in sops:
        sop_id = str(sop.get("id"))
        for branch in sop.get("source_branches") or []:
            if not isinstance(branch, list | tuple) or len(branch) < 2:
                continue
            branch_to_sops[(str(branch[0]), str(branch[1]))].append(sop_id)
    for values in branch_to_sops.values():
        values.sort()
    report = {
        "total_sops": len(raw_sops),
        "included_sops": len(sops),
        "excluded_sops": excluded_sops,
        "excluded_sop_count": len(excluded_sops),
        "excluded_sops_by_reason": dict(sorted(collections.Counter(item["reason"] for item in excluded_sops).items())),
    }
    return sops, dict(branch_to_sops), report


def transition_outcome(child: dict[str, Any], parent: dict[str, Any]) -> str:
    improvement = signed_metric_improvement(child, parent)
    if parent.get("is_buggy") is True and child.get("is_buggy") is False:
        return "debug_fixed"
    if child.get("is_buggy") is True:
        return "buggy"
    if improvement is None:
        return "unknown"
    if improvement > 1e-12:
        return "metric_improved"
    if improvement < -1e-12:
        return "metric_worsened"
    return "metric_flat"


def ball_midpoint(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    point = (left + right) / 2.0
    norm = float(np.linalg.norm(point))
    if norm >= 0.985:
        point = point / max(norm, 1e-12) * 0.985
    return point.astype(np.float64)


def attract_toward(left: np.ndarray, right: np.ndarray, weight: float) -> np.ndarray:
    point = (1.0 - weight) * left + weight * right
    norm = float(np.linalg.norm(point))
    if norm >= 0.985:
        point = point / max(norm, 1e-12) * 0.985
    return point.astype(np.float64)


def fallback_sop_coord(sop: dict[str, Any]) -> np.ndarray:
    text = " ".join([str(sop.get("skill_id", "")), str(sop.get("title", "")), str(sop.get("condition", ""))])
    seed = int(stable_hash(text, 8), 16)
    theta = (seed % 1000003) / 1000003.0 * 2.0 * math.pi
    band = str(sop.get("radius_band", "middle"))
    radius = {"core": 0.28, "middle": 0.62, "edge": 0.88}.get(band, 0.62)
    return np.asarray([radius * math.cos(theta), radius * math.sin(theta)], dtype=np.float64)


def lorentz_from_poincare(coords: np.ndarray) -> np.ndarray:
    norms = np.sum(coords * coords, axis=1)
    denom = np.maximum(1.0 - norms, 1e-12)
    time = (1.0 + norms) / denom
    spatial = 2.0 * coords / denom[:, None]
    return np.column_stack([time, spatial]).astype(np.float64)


def euclidean_text_coords(nodes: list[dict[str, Any]], dims: int = 16) -> np.ndarray:
    texts = [str(node.get("text", "") or node.get("title", "") or node.get("id", "")) for node in nodes]
    if len(texts) <= 1:
        return np.zeros((len(texts), dims), dtype=np.float64)
    vectorizer = TfidfVectorizer(min_df=1, max_features=12000, ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    target_dims = min(dims, max(1, min(matrix.shape) - 1))
    if target_dims <= 0:
        dense = matrix.toarray().astype(np.float64)
    else:
        svd = TruncatedSVD(n_components=target_dims, random_state=42)
        dense = svd.fit_transform(matrix).astype(np.float64)
    dense = normalize(dense, norm="l2", axis=1)
    if dense.shape[1] < dims:
        dense = np.pad(dense, ((0, 0), (0, dims - dense.shape[1])), mode="constant")
    return dense[:, :dims].astype(np.float64)


def build_artifact(
    runs_dir: Path,
    sop_graph_path: Path,
    allowlist_path: Path | None = None,
    require_clean_provenance: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    runs_dir = Path(runs_dir).resolve()
    sop_graph_path = Path(sop_graph_path).resolve()
    allowlist_path = Path(allowlist_path).resolve() if allowlist_path is not None else None
    if require_clean_provenance and allowlist_path is None:
        raise ValueError("--require-clean-provenance requires --allowlist")
    provenance = load_clean_allowlist(allowlist_path) if allowlist_path is not None else None
    journals, journal_report = load_journals(runs_dir, provenance)
    sops, branch_to_sops, sop_report = load_sops(sop_graph_path, provenance)
    if require_clean_provenance:
        if not journals:
            raise ValueError("Clean provenance requested but no allowlisted journals were included")
        if journal_report.get("missing_allowed_runs"):
            raise ValueError(
                "Clean provenance requested but allowlisted runs are missing from runs-dir: "
                f"{journal_report['missing_allowed_runs']}"
            )
        included = {run_short_id(run_id) for run_id, _path, _data in journals}
        allowed = set(provenance.get("allowed_run_ids", [])) if provenance else set()
        if not included.issubset(allowed):
            raise ValueError(f"Clean provenance violation: included non-allowlisted runs {sorted(included - allowed)}")
        bad_sops = [item for item in sop_report.get("excluded_sops", []) if item.get("reason") == "blocked"]
        if bad_sops:
            raise ValueError(f"Clean provenance violation: SOP graph contains blocked source SOPs: {bad_sops[:3]}")
        if not sops:
            raise ValueError("Clean provenance requested but no clean SOPs were included")
    sop_by_id = {str(sop.get("id")): sop for sop in sops}
    sop_turns_by_id = {sop_id: evidence_turn_map(sop) for sop_id, sop in sop_by_id.items()}
    sop_tokens_by_id = {
        sop_id: tokens(
            "\n".join(
                [
                    str(sop.get("title", "")),
                    str(sop.get("action") or sop.get("principle") or ""),
                    str(sop.get("condition") or ""),
                    " ".join(str(x) for x in (sop.get("prevents") or [])),
                ]
            )
        )
        for sop_id, sop in sop_by_id.items()
    }
    run_graphs = [(run_id, path, journal, journal_graph(journal)) for run_id, path, journal in journals]

    run_leaf_counts = []
    for run_id, _path, _journal, graph in run_graphs:
        leaf_counts = descendant_leaf_counts(graph["children"], graph["roots"])
        run_leaf_counts.append((run_id, max(1, sum(leaf_counts.get(root, 1) for root in graph["roots"]))))
    total_leaves = sum(count for _run_id, count in run_leaf_counts)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    coord_by_node: dict[str, np.ndarray] = {}
    transition_coords_for_sop: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    transition_attach_count = 0
    attachment_quality_counts: collections.Counter[str] = collections.Counter()
    evidence_count = 0
    failure_patterns: dict[str, dict[str, Any]] = {}
    audited_code_node_count = 0
    audit_status_counts: collections.Counter[str] = collections.Counter()
    audit_issue_counts: collections.Counter[str] = collections.Counter()
    total_local_best_attractions = 0
    local_best_attraction_weight = 0.0

    cur_angle = 0.0
    for (run_id, path, journal, graph), (_rid, run_leaf_count) in zip(run_graphs, run_leaf_counts):
        width = 2.0 * math.pi * run_leaf_count / max(1, total_leaves)
        lo, hi = cur_angle, cur_angle + width
        cur_angle += width
        task = task_from_run_id(run_id)
        run_node_id = f"run::{run_id}"
        run_coord = np.asarray([0.08 * math.cos((lo + hi) / 2), 0.08 * math.sin((lo + hi) / 2)], dtype=np.float64)
        coord_by_node[run_node_id] = run_coord
        nodes.append(
            {
                "id": run_node_id,
                "type": "Run",
                "run_id": run_id,
                "run_short_id": run_short_id(run_id),
                "task": task,
                "journal_path": display_path(path),
                "node_count": len(graph["nodes"]),
                "edge_count": len(graph["parents"]),
                "root_count": len(graph["roots"]),
                "angle_span": [lo, hi],
                "text": f"{task} {run_id}",
            }
        )

        local_best_map = {str(k): str(v) for k, v in (journal.get("node2best_local_node") or {}).items()}
        raw_coords = assign_run_coords(graph, lo, hi)
        local_best_attractions = 0
        if local_best_attraction_weight > 0.0:
            for raw_id, best_raw in local_best_map.items():
                if raw_id == best_raw or raw_id not in raw_coords or best_raw not in raw_coords:
                    continue
                # Keep this disabled by default. Local-best is an explicit search
                # side-channel; forcing it into coordinates hurt the tree geometry
                # ablation. Runtime follows the local_best edge after retrieval.
                raw_coords[raw_id] = attract_toward(raw_coords[raw_id], raw_coords[best_raw], local_best_attraction_weight)
                local_best_attractions += 1
        total_local_best_attractions += local_best_attractions
        for raw_id, raw_node in sorted(graph["nodes"].items(), key=lambda item: (item[1].get("step") or 0, item[0])):
            node_id = f"run::{run_id}::node::{raw_id}"
            parent_raw = graph["parents"].get(raw_id)
            parent_node = graph["nodes"].get(parent_raw) if parent_raw else None
            parent_metric = metric_value(parent_node) if parent_node else None
            current_metric = metric_value(raw_node)
            improvement = signed_metric_improvement(raw_node, parent_node) if parent_node else None
            local_best_raw = local_best_map.get(raw_id)
            raw_code = str(raw_node.get("code") or "")
            static_audit = audit_code(raw_code)
            stored_audit = raw_node.get("leakage_audit") if isinstance(raw_node.get("leakage_audit"), dict) else None
            leakage_audit = merge_audits(raw_code, static_audit, stored_audit)
            if raw_code:
                audited_code_node_count += 1
                audit_status_counts[str(leakage_audit.get("status", "unknown"))] += 1
                for issue in leakage_audit.get("issues", []):
                    audit_issue_counts[str(issue.get("issue_code", "unknown"))] += 1
            audit_text = " ".join(
                f"{item.get('issue_code')}: {item.get('evidence')} Fix: {item.get('remediation')}"
                for item in leakage_audit.get("issues", [])
            )
            node_record = {
                "id": node_id,
                "type": "RunNode",
                "run_id": run_id,
                "run_short_id": run_short_id(run_id),
                "task": task,
                "original_node_id": raw_id,
                "parent_id": f"run::{run_id}::node::{parent_raw}" if parent_raw else None,
                "depth": graph["depths"].get(raw_id, 0),
                "stage": raw_node.get("stage"),
                "step": raw_node.get("step"),
                "branch_id": raw_node.get("branch_id"),
                "from_topk": raw_node.get("from_topk"),
                "is_leaf": not bool(graph["children"].get(raw_id)),
                "num_children": len(graph["children"].get(raw_id, [])),
                "metric": current_metric,
                "parent_metric": parent_metric,
                "metric_delta": None if current_metric is None or parent_metric is None else current_metric - parent_metric,
                "metric_improvement": improvement,
                "maximize": metric_maximize(raw_node, parent_node),
                "is_buggy": raw_node.get("is_buggy"),
                "is_valid": raw_node.get("is_valid"),
                "is_debug_success": raw_node.get("is_debug_success"),
                "local_best_node_id": (
                    f"run::{run_id}::node::{local_best_raw}" if local_best_raw in graph["nodes"] else None
                ),
                "plan": short_text(raw_node.get("plan"), 900),
                "code_summary": short_text(raw_node.get("code_summary"), 900),
                "code_sha256": hashlib.sha256(raw_code.encode("utf-8")).hexdigest() if raw_code else "",
                "code_length": len(raw_code),
                "analysis": short_text(raw_node.get("analysis"), 900),
                "leakage_audit": leakage_audit,
                "audit_status": leakage_audit.get("status"),
                "paper_grade_eligible": leakage_audit.get("paper_grade_eligible"),
                "metric_disposition": leakage_audit.get("metric_disposition"),
                "memory_disposition": leakage_audit.get("memory_disposition"),
                "leakage_issue_codes": [
                    str(item.get("issue_code")) for item in leakage_audit.get("issues", [])
                ],
                "terminal_excerpt": short_text("".join(raw_node.get("_term_out") or []), 700),
                "text": "\n".join(part for part in [node_text(raw_node), audit_text] if part),
            }
            nodes.append(node_record)
            coord_by_node[node_id] = raw_coords[raw_id]
            edges.append({"src": run_node_id, "dst": node_id, "kind": "contains", "weight": 1.0})
            if node_record["local_best_node_id"]:
                edges.append({"src": node_id, "dst": node_record["local_best_node_id"], "kind": "points_to_local_best", "weight": 0.6})

            for issue in leakage_audit.get("issues", []):
                issue_code = str(issue.get("issue_code") or "UNKNOWN_LEAKAGE_ISSUE")
                pattern_key = f"{leakage_audit.get('code_sha256')}::{issue_code}"
                pattern_id = f"failure::leakage::{stable_hash(pattern_key)}"
                if pattern_id not in failure_patterns:
                    pattern_record = {
                        "id": pattern_id,
                        "type": "FailurePattern",
                        "failure_kind": "leakage_audit",
                        "issue_code": issue_code,
                        "category": issue.get("category"),
                        "severity": issue.get("severity"),
                        "execution_disposition": issue.get("execution_disposition"),
                        "code_sha256": leakage_audit.get("code_sha256"),
                        "evidence": issue.get("evidence"),
                        "remediation": issue.get("remediation"),
                        "source_node_ids": [node_id],
                        "task": task,
                        "text": " ".join(
                            str(value)
                            for value in [
                                issue_code,
                                issue.get("category"),
                                issue.get("evidence"),
                                issue.get("remediation"),
                            ]
                            if value
                        ),
                    }
                    failure_patterns[pattern_id] = pattern_record
                    nodes.append(pattern_record)
                    coord_by_node[pattern_id] = raw_coords[raw_id] * 0.995
                else:
                    sources = failure_patterns[pattern_id]["source_node_ids"]
                    if node_id not in sources:
                        sources.append(node_id)
                edges.append({"src": node_id, "dst": pattern_id, "kind": "has_failure_pattern", "weight": 1.0})
                edges.append({"src": pattern_id, "dst": node_id, "kind": "blocks_adoption_of", "weight": 1.0})

        for child_raw, parent_raw in sorted(graph["parents"].items(), key=lambda item: (graph["nodes"][item[0]].get("step") or 0, item[0])):
            child = graph["nodes"][child_raw]
            parent = graph["nodes"][parent_raw]
            child_node_id = f"run::{run_id}::node::{child_raw}"
            parent_node_id = f"run::{run_id}::node::{parent_raw}"
            transition_id = f"run::{run_id}::transition::{parent_raw[:10]}::{child_raw[:10]}"
            branch_key = (run_short_id(run_id), str(child.get("branch_id")))
            candidate_sops = list(branch_to_sops.get(branch_key, []))
            transition_text = "\n".join([node_text(parent), node_text(child)]).strip()
            transition_tokens = tokens(transition_text)
            attached_pairs: list[tuple[str, str, float]] = []
            child_step = child.get("step")
            parent_step = parent.get("step")
            branch_id_text = str(child.get("branch_id"))
            for sop_id in candidate_sops:
                turns = sop_turns_by_id.get(sop_id, {}).get(branch_id_text, set())
                if (
                    isinstance(child_step, int)
                    and child_step in turns
                    or isinstance(parent_step, int)
                    and parent_step in turns
                ):
                    attached_pairs.append((sop_id, "evidence_turn_match", 1.0))
            exact_ids = {sop_id for sop_id, _quality, _score in attached_pairs}
            lexical_candidates: list[tuple[float, str]] = []
            for sop_id in candidate_sops:
                if sop_id in exact_ids:
                    continue
                score = token_overlap_score(sop_tokens_by_id.get(sop_id, set()), transition_tokens)
                if score >= 0.08:
                    lexical_candidates.append((score, sop_id))
            for score, sop_id in sorted(lexical_candidates, reverse=True)[:2]:
                attached_pairs.append((sop_id, "branch_lexical_match", score))
            attached_sops = [sop_id for sop_id, _quality, _score in attached_pairs[:6]]
            transition_attach_count += int(bool(attached_sops))
            transition_coord = ball_midpoint(coord_by_node[parent_node_id], coord_by_node[child_node_id])
            coord_by_node[transition_id] = transition_coord
            for sop_id in attached_sops:
                transition_coords_for_sop[sop_id].append(transition_coord)

            transition_record = {
                "id": transition_id,
                "type": "Transition",
                "run_id": run_id,
                "run_short_id": run_short_id(run_id),
                "task": task,
                "parent_node_id": parent_node_id,
                "child_node_id": child_node_id,
                "parent_original_node_id": parent_raw,
                "child_original_node_id": child_raw,
                "branch_id": child.get("branch_id"),
                "depth": graph["depths"].get(child_raw, 0),
                "stage_pair": f"{parent.get('stage')}->{child.get('stage')}",
                "outcome": transition_outcome(child, parent),
                "parent_metric": metric_value(parent),
                "child_metric": metric_value(child),
                "metric_delta": (
                    None
                    if metric_value(parent) is None or metric_value(child) is None
                    else metric_value(child) - metric_value(parent)
                ),
                "metric_improvement": signed_metric_improvement(child, parent),
                "parent_buggy": parent.get("is_buggy"),
                "child_buggy": child.get("is_buggy"),
                "attached_sop_ids": [f"sop::{sop_id}" for sop_id in attached_sops],
                "attachment_quality": [
                    {"sop_id": f"sop::{sop_id}", "quality": quality, "score": score}
                    for sop_id, quality, score in attached_pairs[:6]
                ],
                "text": transition_text,
            }
            nodes.append(transition_record)
            edges.append({"src": parent_node_id, "dst": child_node_id, "kind": "parent_of", "weight": 1.0})
            edges.append({"src": parent_node_id, "dst": transition_id, "kind": "has_transition", "weight": 1.0})
            edges.append({"src": transition_id, "dst": child_node_id, "kind": "transition_to", "weight": 1.0})

            if transition_record["parent_metric"] is not None or transition_record["child_metric"] is not None or parent.get("is_buggy") is True:
                evidence_id = f"evidence::{stable_hash(transition_id)}"
                evidence_count += 1
                evidence_text = (
                    f"{transition_record['stage_pair']} outcome={transition_record['outcome']} "
                    f"parent_metric={transition_record['parent_metric']} child_metric={transition_record['child_metric']} "
                    f"parent_error={short_text(''.join(parent.get('_term_out') or []), 500)}"
                )
                nodes.append(
                    {
                        "id": evidence_id,
                        "type": "Evidence",
                        "run_id": run_id,
                        "run_short_id": run_short_id(run_id),
                        "transition_id": transition_id,
                        "parent_node_id": parent_node_id,
                        "child_node_id": child_node_id,
                        "evidence_kind": "transition_metric_error",
                        "text": evidence_text,
                    }
                )
                coord_by_node[evidence_id] = transition_coord * 0.995
                edges.append({"src": transition_id, "dst": evidence_id, "kind": "supported_by", "weight": 0.8})
            for sop_id, quality, score in attached_pairs[:6]:
                attachment_quality_counts[quality] += 1
                edges.append(
                    {
                        "src": transition_id,
                        "dst": f"sop::{sop_id}",
                        "kind": "distills_to",
                        "weight": 0.9 if quality == "evidence_turn_match" else 0.55,
                        "quality": quality,
                        "score": score,
                    }
                )

    for sop in sops:
        sop_id = str(sop.get("id"))
        node_id = f"sop::{sop_id}"
        attached = transition_coords_for_sop.get(sop_id, [])
        if attached:
            centroid = np.mean(np.vstack(attached), axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm < 0.12:
                centroid = fallback_sop_coord(sop)
            elif norm >= 0.975:
                centroid = centroid / norm * 0.975
            coord = centroid.astype(np.float64)
        else:
            coord = fallback_sop_coord(sop)
        coord_by_node[node_id] = coord
        nodes.append(
            {
                "id": node_id,
                "type": "SOP",
                "sop_id": sop_id,
                "title": sop.get("title"),
                "action": sop.get("action") or sop.get("principle"),
                "applies_when": sop.get("applies_when") or [sop.get("condition")],
                "prevents": sop.get("prevents"),
                "skill_id": sop.get("skill_id"),
                "radius_band": sop.get("radius_band"),
                "source_branches": sop.get("source_branches") or [],
                "evidence_turns": sop.get("evidence_turns") or [],
                "attached_transition_count": len(attached),
                "text": "\n".join(
                    [
                        str(sop.get("title", "")),
                        str(sop.get("action") or sop.get("principle") or ""),
                        str(sop.get("condition") or ""),
                        " ".join(str(x) for x in (sop.get("prevents") or [])),
                    ]
                ),
            }
        )

    node_ids = [node["id"] for node in nodes]
    poincare = np.vstack([coord_by_node[node_id] for node_id in node_ids]).astype(np.float64)
    flat_twin = poincare.copy()
    lorentz = lorentz_from_poincare(poincare)
    euclidean = euclidean_text_coords(nodes, dims=16)

    graph = {
        "meta": {
            "schema": "hyperbolic_run_forest_memory_v1",
            "builder": "build_run_forest_memory.py",
            "runs_dir": str(runs_dir.relative_to(REPO)) if runs_dir.is_relative_to(REPO) else str(runs_dir),
            "sop_graph": str(sop_graph_path.relative_to(REPO)) if sop_graph_path.exists() and sop_graph_path.is_relative_to(REPO) else str(sop_graph_path),
            "journal_count": len(journals),
            "source_runs": sorted({run_short_id(run_id) for run_id, _path, _journal in journals}),
            "allowlist": provenance.get("allowed_run_ids", []) if provenance else [],
            "allowlist_hash": provenance.get("hash", "") if provenance else "",
            "allowlist_path": (
                str(allowlist_path.relative_to(REPO))
                if allowlist_path is not None and allowlist_path.exists() and allowlist_path.is_relative_to(REPO)
                else (str(allowlist_path) if allowlist_path is not None else "")
            ),
            "blocked_run_prefixes": provenance.get("blocked_prefixes", []) if provenance else [],
            "source_membership_verified": bool(provenance and require_clean_provenance),
            "leak_audited": audited_code_node_count > 0,
            "positive_admission_enforced": True,
            "leak_verified": bool(provenance and require_clean_provenance and audited_code_node_count > 0),
            "paper_grade": bool(provenance and require_clean_provenance and audited_code_node_count > 0),
            "provenance_status": (
                "source_allowlisted_and_code_audited"
                if provenance and require_clean_provenance and audited_code_node_count > 0
                else "uncertified_bootstrap"
            ),
            "audit_status_counts": dict(sorted(audit_status_counts.items())),
            "audit_issue_counts": dict(sorted(audit_issue_counts.items())),
            "failure_pattern_count": len(failure_patterns),
            "paper_grade_definition": (
                "Source membership is allowlisted, every code-bearing node receives deterministic audit metadata, "
                "and runtime positive retrieval admits only memory_disposition=positive_eligible. Blocked, biased, "
                "warning, and unavailable nodes remain in the graph solely as negative/debug evidence."
            ),
            "coordinate_model": "global circular run forest layout; radius grows with run-tree depth",
            "flat_twin_model": "same coordinates as poincare; distance function changes only",
            "euclidean_model": "independent TF-IDF-SVD text coordinates over RunNode/Transition/SOP/Evidence text",
            "memory_thesis": "Run/journal memory is the primary hyperbolic forest; SOPs are signposts attached to transitions; references/evidence are leaves.",
        },
        "nodes": nodes,
        "edges": edges,
    }
    index = {
        "node_ids": np.asarray(node_ids, dtype=object),
        "node_types": np.asarray([node.get("type", "") for node in nodes], dtype=object),
        "poincare": poincare,
        "flat_twin": flat_twin,
        "lorentz": lorentz,
        "euclidean": euclidean,
    }
    report = {
        "schema": "run_forest_builder_report_v1",
        "provenance_status": graph["meta"]["provenance_status"],
        "source_membership_verified": graph["meta"]["source_membership_verified"],
        "leak_audited": graph["meta"]["leak_audited"],
        "positive_admission_enforced": graph["meta"]["positive_admission_enforced"],
        "paper_grade_provenance": graph["meta"]["paper_grade"],
        "leak_verified": graph["meta"]["leak_verified"],
        "source_runs": sorted({run_short_id(run_id) for run_id, _path, _journal in journals}),
        "allowlist_path": graph["meta"]["allowlist_path"],
        "allowlist_hash": graph["meta"]["allowlist_hash"],
        "allowlist": graph["meta"]["allowlist"],
        "blocked_run_prefixes": graph["meta"]["blocked_run_prefixes"],
        "journal_filter_report": journal_report,
        "sop_filter_report": sop_report,
        "journal_count": len(journals),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_type_counts": dict(sorted(collections.Counter(node["type"] for node in nodes).items())),
        "edge_kind_counts": dict(sorted(collections.Counter(edge["kind"] for edge in edges).items())),
        "sop_count": len(sops),
        "sops_with_attached_transitions": sum(1 for coords in transition_coords_for_sop.values() if coords),
        "transition_count": sum(1 for node in nodes if node.get("type") == "Transition"),
        "transitions_with_sop_attachments": transition_attach_count,
        "attachment_quality_counts": dict(sorted(attachment_quality_counts.items())),
        "evidence_count": evidence_count,
        "audited_code_node_count": audited_code_node_count,
        "audit_status_counts": dict(sorted(audit_status_counts.items())),
        "audit_issue_counts": dict(sorted(audit_issue_counts.items())),
        "failure_pattern_count": len(failure_patterns),
        "local_best_coordinate_attractions": total_local_best_attractions,
        "local_best_attraction_weight": local_best_attraction_weight,
        "poincare_max_norm": float(np.max(np.linalg.norm(poincare, axis=1))) if len(poincare) else 0.0,
        "flat_twin_same_as_poincare": bool(np.array_equal(flat_twin, poincare)),
        "euclidean_shape": list(euclidean.shape),
        "run_node_topology_preserved": True,
        "sop_attachment_rule": "match SOP source_branches [run_short, branch_id], prefer evidence_turn B{branch}.T{step} alignment, fallback to at most two branch-local lexical matches",
    }
    return graph, index, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--sop-graph", type=Path, default=DEFAULT_SOP_GRAPH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--allowlist", type=Path, default=None)
    parser.add_argument("--require-clean-provenance", action="store_true")
    args = parser.parse_args()

    graph, index, report = build_artifact(
        args.runs_dir,
        args.sop_graph,
        allowlist_path=args.allowlist,
        require_clean_provenance=args.require_clean_provenance,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = args.out_dir / "run_forest_graph.json"
    index_path = args.out_dir / "run_forest_index.npz"
    report_path = args.out_dir / "run_forest_builder_report.json"
    graph_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(index_path, **index)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {graph_path}")
    print(f"Wrote {index_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
