"""
skillgraph_retrieve.py — SkillGraph §3.2 graph-aware retrieval (paper-faithful, static).

4 steps, pure graph traversal, ZERO LLM:
  1. Seed select (Eq.1): R_seed = general ∪ task-type-matched, from V_active (= full graph;
     progressive-unlocking excluded).
  2. Backward BFS (depth D=2): follow INCOMING prerequisite edges from seeds -> R_BFS
     (foundational prereqs). NOTE: our init graph has 0 prereq edges, so R_BFS is empty —
     faithful to "prereq emerge via evolution" (excluded).
  3. Forward beam (width B=3): follow OUTGOING edges, σ(v)=max_{u∈parents} σ(u)·w(u,v),
     seeds σ=1 -> R_beam.
  4. Topological ordering (Eq.2): TopoSort(R_seed ∪ R_BFS ∪ R_beam), cap K_max=8.

Hyperparams (paper Table 5): K_max=8, D=2, B=3.

Demo (--demo): for each task_type in the graph, run retrieval and show that the SEED is a
focused subset (multi-task filtering) — the key behavior single-task lacked.

Usage:
  python skillgraph_retrieve.py graph.json <task_type> [--K 8] [--D 2] [--B 3]
  python skillgraph_retrieve.py graph.json --demo
"""
import json, sys, collections

def build_adj(graph):
    nodes = {n["id"]: n for n in graph["nodes"]}
    out_edges = collections.defaultdict(list)   # src -> [(dst, kind, w)]
    in_edges = collections.defaultdict(list)    # dst -> [(src, kind, w)]
    for e in graph["edges"]:
        out_edges[e["src"]].append((e["dst"], e["kind"], e.get("weight", 1.0)))
        in_edges[e["dst"]].append((e["src"], e["kind"], e.get("weight", 1.0)))
    return nodes, out_edges, in_edges

def seed_select(nodes, task_type):
    return sorted(nid for nid, n in nodes.items()
                  if n["category"] == "general" or n["category"] == task_type)

def backward_bfs(seeds, in_edges, D=2):
    """Follow INCOMING prereq edges backward from seeds, depth D. Returns non-seed prereq ancestors."""
    R, frontier = set(), list(seeds)
    for _ in range(D):
        nxt = []
        for v in frontier:
            for src, kind, w in in_edges.get(v, []):
                if kind == "prereq" and src not in seeds and src not in R:
                    R.add(src); nxt.append(src)
        if not nxt:
            break
        frontier = nxt
    return R

def forward_beam(seeds, out_edges, B=3, max_steps=3):
    """Beam over OUTGOING edges. σ(v)=max_parent σ(u)·w; seeds σ=1. Returns non-seed reached + σ map."""
    sigma = {s: 1.0 for s in seeds}
    R, frontier = set(), list(seeds)
    for _ in range(max_steps):
        candidates = []
        for u in frontier:
            for dst, kind, w in out_edges.get(u, []):
                prop = sigma.get(u, 0.0) * w
                if prop > sigma.get(dst, 0.0):
                    sigma[dst] = prop
                if dst not in seeds and dst not in R:
                    candidates.append(dst)
        if not candidates:
            break
        top = sorted(set(candidates), key=lambda x: -sigma.get(x, 0.0))[:B]
        R.update(top)
        frontier = top
    return R, sigma

def topo_cap(union, nodes, sigma, K=8, general_cap=None):
    """Dependency-ordered (generals level-0 before task-specific), capped at K.
    general_cap=None -> FAITHFUL (all generals first; with >K generals returns K generals).
    general_cap=N   -> NON-PAPER pragmatic variant: at most N generals, rest task-specific
                       (useful because we over-produce generals; the paper has few)."""
    gens = sorted([n for n in union if nodes[n].get("category") == "general"],
                  key=lambda nid: -sigma.get(nid, 0.0))
    tasks = sorted([n for n in union if nodes[n].get("category") != "general"],
                   key=lambda nid: (nodes[nid].get("level", 0), -sigma.get(nid, 0.0)))
    if general_cap is None:
        return (gens + tasks)[:K]
    ng = min(general_cap, len(gens))
    return gens[:ng] + tasks[:max(0, K - ng)]

def retrieve(graph, task_type, K=8, D=2, B=3, general_cap=None):
    nodes, out_edges, in_edges = build_adj(graph)
    seeds = seed_select(nodes, task_type)
    r_bfs = backward_bfs(seeds, in_edges, D)
    r_beam, sigma = forward_beam(seeds, out_edges, B)
    sigma.update({s: 1.0 for s in seeds})
    union = set(seeds) | r_bfs | r_beam
    chain = topo_cap(union, nodes, sigma, K, general_cap)
    return {"task": task_type, "n_graph": len(nodes), "n_seed": len(seeds),
            "n_bfs": len(r_bfs), "n_beam": len(r_beam), "n_union": len(union),
            "chain": [(nid, nodes[nid]["category"], nodes[nid]["title"]) for nid in chain]}

def main():
    graph = json.load(open(sys.argv[1]))
    K = int(sys.argv[sys.argv.index("--K")+1]) if "--K" in sys.argv else 8
    D = int(sys.argv[sys.argv.index("--D")+1]) if "--D" in sys.argv else 2
    B = int(sys.argv[sys.argv.index("--B")+1]) if "--B" in sys.argv else 3
    gcap = int(sys.argv[sys.argv.index("--general-cap")+1]) if "--general-cap" in sys.argv else None
    task_types = sorted({n["category"] for n in graph["nodes"] if n["category"] != "general"})
    targets = task_types if "--demo" in sys.argv else [sys.argv[2]]
    for t in targets:
        r = retrieve(graph, t, K, D, B, general_cap=gcap)
        print(f"\n=== retrieve(task={t})  general_cap={gcap} ===")
        print(f"  graph={r['n_graph']}  seed={r['n_seed']}  backward_bfs={r['n_bfs']}  "
              f"forward_beam={r['n_beam']}  union={r['n_union']}  -> cap {K}")
        g_in_chain = sum(1 for _, c, _ in r["chain"] if c == "general")
        print(f"  chain: {g_in_chain} general + {len(r['chain'])-g_in_chain} task-specific")
        for nid, cat, title in r["chain"]:
            print(f"    [{cat[:24]:24s}] {title[:60]}")

if __name__ == "__main__":
    main()
