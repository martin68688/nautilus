"""
build_edges_levels.py — build the SkillGraph G=(V,E) from distilled nodes (faithful static build).

Implements the paper's INIT state (Algorithm 1 lines 4-5), NOT online evolution:
  - InitGraph (Appendix A.2):
      * enhance  (w=0.2): every general -> every task-specific
      * co_occur (w=0.3): task-specific same-category skills that CO-OCCUR (share >=1 source branch)
      * prereq : NONE at init (emerge only via evolution — excluded)
  - Levels (Appendix A.1): BFS over incoming {prereq, enhance}; co_occur EXCLUDED; level-0 = no dep parent
  - Stats (Appendix A.3, offline proxy): n_use=distinct source branches,
      n_succ=branches with best_metric below that run's median, p_hat=n_succ/n_use

NOT done (online evolution, excluded for the static baseline = paper's "w/o Graph Evolution" ablation):
  Merge/Split/Insert/Deprecate, path-reinforcement, decay/prune, progressive-unlocking, GRPO.
  NOTE on Merge: paper's neighbor-Jaccard Merge is an evolution op needing differentiated neighborhoods;
  on the static init graph enhance-all makes all generals share identical neighborhoods (Jaccard~1.0),
  so Merge would collapse them — hence correctly excluded here. Batch-pool + prompt-dedup handle dupes.

Output: graph_build/graph.json  {meta, nodes[+level/stats], edges[src,dst,kind,weight]}.
"""
import json, glob, pathlib, statistics, re
from collections import Counter, defaultdict

REPO = pathlib.Path(__file__).resolve().parents[2]
RUNS = REPO / "mlevolve" / "runs"
GB = REPO / "paper-skills" / "distillation" / "graph_build"
IN = GB / "merged_nodes.json" if (GB / "merged_nodes.json").exists() else GB / "raw_nodes.json"
OUT = GB / "graph.json"

_W_ENHANCE, _W_COOCCUR, _W_PREREQ = 0.2, 0.3, 0.4
_EVID_RE = re.compile(r"B(\d+)\.T(\d+)", re.I)

_JCACHE = {}
def load_journal(run_ts):
    if run_ts not in _JCACHE:
        fs = glob.glob(str(RUNS / f"{run_ts}_*" / "logs" / "journal.json"))
        J = json.load(open(fs[0])) if fs else {"nodes": []}
        _JCACHE[run_ts] = J["nodes"] if isinstance(J, dict) and "nodes" in J else J
    return _JCACHE[run_ts]

def run_maximize(run_ts):
    for n in load_journal(run_ts):
        m = n.get("metric")
        if isinstance(m, dict) and m.get("maximize") is not None:
            return bool(m["maximize"])
    return False

def branch_best(run_ts, bid):
    ns = [n for n in load_journal(run_ts) if str(n.get("branch_id")) == str(bid)]
    def mv(n):
        m = n.get("metric"); return m.get("value") if isinstance(m, dict) else m
    vals = [mv(n) for n in ns if mv(n) is not None and not n.get("is_buggy")]
    if not vals:
        return None
    return max(vals) if run_maximize(run_ts) else min(vals)

def run_median_best(run_ts):
    bids = sorted({str(n.get("branch_id")) for n in load_journal(run_ts) if n.get("branch_id") is not None}, key=int)
    vals = [b for b in (branch_best(run_ts, b) for b in bids) if b is not None]
    return statistics.median(vals) if vals else None


def build_init_edges(nodes, selective=False):
    """InitGraph (A.2). If selective, only universal_general nodes get broad enhance (optimized
    baseline); demoted api_warning/implementation_note do NOT enhance every task."""
    edges = []
    all_generals = [n for n in nodes if n["category"] == "general"]
    generals = [g for g in all_generals if (not selective) or g.get("scope") == "universal_general"]
    by_cat = {}
    for n in nodes:
        if n["category"] != "general":
            by_cat.setdefault(n["category"], []).append(n)
    # enhance: (universal) general -> every task-specific (A.2; selective = universal only)
    for g in generals:
        for cat, ts in by_cat.items():
            for t in ts:
                edges.append({"src": g["id"], "dst": t["id"], "kind": "enhance", "weight": _W_ENHANCE})
    # co_occur: task-specific same-category that share >=1 source branch (A.2 "within same category")
    seen = set()
    for cat, ts in by_cat.items():
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                a, b = ts[i], ts[j]
                sa = {tuple(x) for x in a["source_branches"]}
                sb = {tuple(x) for x in b["source_branches"]}
                if sa & sb:  # co-occur in at least one shared source branch
                    key = (a["id"], b["id"])
                    if key not in seen:
                        seen.add(key)
                        edges.append({"src": a["id"], "dst": b["id"], "kind": "co_occur", "weight": _W_COOCCUR})
    return edges


def evidence_events(n):
    """Return approximate (run_ts, branch_id, turn) evidence events.

    The distilled labels only keep B*.T* while source_branches keeps run+branch, so when a
    merged node has the same branch id from multiple runs we conservatively attach that turn to
    each matching run. This is only used for the adapted trace-order baseline.
    """
    branch_to_runs = defaultdict(list)
    for run_ts, bid in n.get("source_branches", []):
        branch_to_runs[str(bid)].append(str(run_ts))
    events = set()
    for ev in n.get("evidence_turns", []):
        m = _EVID_RE.search(str(ev))
        if not m:
            continue
        bid, turn = m.group(1), int(m.group(2))
        for run_ts in branch_to_runs.get(bid, []):
            events.add((run_ts, bid, turn))
    return sorted(events)


def build_trace_order_prereq_edges(nodes, window=3, min_support=1, max_per_dst=3):
    """Adapted SkillGraph-C edge builder.

    Adds prereq edges from earlier evidenced nodes to later evidenced nodes within the same
    run branch and task category. This is NOT the paper-faithful static InitGraph; it is an
    MLE-trace adaptation so backward-BFS has execution-order structure to follow.
    """
    by_id = {n["id"]: n for n in nodes}
    events_by_node = {n["id"]: evidence_events(n) for n in nodes if n["category"] != "general"}
    mean_turn = {}
    for nid, events in events_by_node.items():
        turns = [t for _, _, t in events]
        if turns:
            mean_turn[nid] = statistics.mean(turns)

    branch_events = defaultdict(list)
    for nid, events in events_by_node.items():
        for run_ts, bid, turn in events:
            branch_events[(run_ts, bid, by_id[nid]["category"])].append((turn, nid))

    support = Counter()
    gaps = defaultdict(list)
    for items in branch_events.values():
        ordered = sorted(set(items), key=lambda x: (x[0], x[1]))
        for i, (dst_turn, dst) in enumerate(ordered):
            for src_turn, src in ordered[max(0, i - window):i]:
                if src == dst or src_turn >= dst_turn:
                    continue
                support[(src, dst)] += 1
                gaps[(src, dst)].append(dst_turn - src_turn)

    by_dst = defaultdict(list)
    for (src, dst), sup in support.items():
        if sup < min_support:
            continue
        if support.get((dst, src), 0) >= sup:
            continue
        if mean_turn.get(src, 10**9) >= mean_turn.get(dst, -1):
            continue
        avg_gap = round(statistics.mean(gaps[(src, dst)]), 3)
        by_dst[dst].append((src, sup, avg_gap))

    edges = []
    for dst, candidates in by_dst.items():
        candidates.sort(key=lambda x: (-x[1], x[2], mean_turn.get(x[0], 10**9), x[0]))
        for src, sup, avg_gap in candidates[:max_per_dst]:
            edges.append({"src": src, "dst": dst, "kind": "prereq", "weight": _W_PREREQ,
                          "support": sup, "avg_turn_gap": avg_gap, "source": "trace_order"})
    return edges


def compute_levels(nodes, edges):
    parents = {n["id"]: [] for n in nodes}
    for e in edges:
        if e["kind"] in ("prereq", "enhance"):
            parents[e["dst"]].append(e["src"])
    level = {n["id"]: 0 for n in nodes}
    for _ in range(len(nodes) + 1):
        changed = False
        for n in nodes:
            lv = 0
            for p in parents[n["id"]]:
                lv = max(lv, level[p] + 1)
            if lv != level[n["id"]]:
                level[n["id"]] = lv; changed = True
        if not changed:
            break
    return level


def compute_stats(nodes):
    run_med = {}
    for n in nodes:
        for r, _ in n["source_branches"]:
            run_med.setdefault(r, run_median_best(r))
    for n in nodes:
        sbr = {(r, b) for r, b in n["source_branches"]}
        n_use = len(sbr)
        n_succ = 0
        for r, b in sbr:
            bm = branch_best(r, b)
            med = run_med.get(r)
            if bm is not None and med is not None:
                if (bm > med) if run_maximize(r) else (bm < med):
                    n_succ += 1
        n["n_use"] = n_use
        n["n_succ"] = n_succ
        n["p_hat"] = round(n_succ / n_use, 3) if n_use else 0.0


def main():
    import sys
    args = sys.argv[1:]
    def int_arg(name, default):
        return int(args[args.index(name) + 1]) if name in args else default
    in_path = args[args.index("--input") + 1] if "--input" in args else str(IN)
    out_path = args[args.index("--output") + 1] if "--output" in args else str(OUT)
    selective = "--selective-general-enhance" in args
    trace_prereq = "--trace-order-prereq" in args
    prereq_window = int_arg("--trace-prereq-window", 3)
    prereq_min_support = int_arg("--trace-prereq-min-support", 1)
    prereq_max_per_dst = int_arg("--trace-prereq-max-per-dst", 3)
    data = json.load(open(in_path))
    nodes = data["nodes"]
    edges = build_init_edges(nodes, selective)
    prereq_edges = []
    if trace_prereq:
        prereq_edges = build_trace_order_prereq_edges(
            nodes, window=prereq_window, min_support=prereq_min_support,
            max_per_dst=prereq_max_per_dst)
        edges.extend(prereq_edges)
    level = compute_levels(nodes, edges)
    for n in nodes:
        n["level"] = level[n["id"]]
    compute_stats(nodes)

    # order nodes by level then id for readability
    nodes.sort(key=lambda n: (n["level"], n["id"]))
    # Paper-faithful compact-card nodes: drop our non-baseline extensions
    # (evidence_turns, source_branches, references). Paper node = {title, principle,
    # condition, category} + framework {level, n_use, n_succ, p_hat}.
    clean = [{"id": n["id"], "title": n["title"], "principle": n.get("principle", ""),
              "condition": n.get("condition", ""), "category": n["category"],
              "scope": n.get("scope", "universal_general" if n["category"] == "general" else "task_specific"),
              "level": n["level"], "n_use": n["n_use"], "n_succ": n["n_succ"],
              "p_hat": n["p_hat"]} for n in nodes]
    graph = {
        "meta": {"schema": "skillgraph-static-v1", "teacher": data.get("meta", {}).get("teacher"),
                 "n_nodes": len(clean), "n_edges": len(edges), "selective_general_enhance": selective,
                 "trace_order_prereq": trace_prereq, "n_trace_order_prereq": len(prereq_edges),
                 "trace_prereq_config": {"window": prereq_window, "min_support": prereq_min_support,
                                         "max_per_dst": prereq_max_per_dst},
                 "note": ("OPTIMIZED baseline: only universal_general enhance broadly; "
                          "demoted api_warning/implementation_note are task-scoped. " if selective else
                         "paper-faithful static init graph (w/o Graph Evolution). ")
                         + ("ADAPTED SkillGraph-C: trace-order prereq edges added. " if trace_prereq else "")
                         + "compact-card nodes; stats are trace-evidence proxy (no RL rollout)"},
        "nodes": clean,
        "edges": edges,
    }
    pathlib.Path(out_path).write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- sanity report ----
    g = sum(1 for n in nodes if n["category"] == "general")
    ek = Counter(e["kind"] for e in edges)
    lv = Counter(n["level"] for n in nodes)
    cats = Counter(n["category"] for n in nodes)
    print(f"=== graph -> {out_path} (selective={selective}) ===")
    print(f"nodes: {len(nodes)}  (general={g}, task-specific={len(nodes)-g})")
    print(f"edges: {len(edges)}  by kind: {dict(ek)}")
    print(f"level distribution: {dict(sorted(lv.items()))}")
    print(f"categories: {dict(cats)}")
    print(f"top-5 by p_hat: {[(n['title'][:40], n['p_hat'], n['n_use']) for n in sorted(nodes, key=lambda x:-x['p_hat'])[:5]]}")


def _slug(title):
    import re
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]


if __name__ == "__main__":
    main()
