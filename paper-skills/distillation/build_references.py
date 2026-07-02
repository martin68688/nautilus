"""
build_references.py — generate the classic-skill REFERENCE body for each skill node.

Reads graph_build/graph.json (the built graph; falls back to raw_nodes.json). For every node, maps
its evidence_turns labels ("B<branch>.T<turn>") -> real journal nodes and pulls the ACTUAL code_summary
/ code snippet / analysis / metric. Writes references/<slug>.md. Nothing hallucinated.

Usage:
  python build_references.py [--show <substr>]   # --show prints one full reference
"""
import re, sys, json, glob, pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
RUNS = REPO / "mlevolve" / "runs"
GB = REPO / "paper-skills" / "distillation" / "graph_build"
IN = GB / "graph.json"
OUT = GB / "references"

_JCACHE = {}
def load_journal(run_ts):
    if run_ts not in _JCACHE:
        fs = glob.glob(str(RUNS / f"{run_ts}_*" / "logs" / "journal.json"))
        J = json.load(open(fs[0])) if fs else {"nodes": []}
        _JCACHE[run_ts] = J["nodes"] if isinstance(J, dict) and "nodes" in J else J
    return _JCACHE[run_ts]

def branch_nodes(run_ts, bid):
    return sorted([n for n in load_journal(run_ts) if str(n.get("branch_id")) == str(bid)],
                  key=lambda n: n.get("step", 0))

def metric_val(n):
    m = n.get("metric"); return (m.get("value") if isinstance(m, dict) else m)

_STOP = set("use for in the a an to when on with and of by from as is are be it that this set get".split())
def keywords(title):
    return [t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", title.lower())
            if t not in _STOP and len(t) > 2]

def extract_snippet(code, kws, max_lines=22):
    if not code or not kws: return ""
    lines = code.splitlines()
    for i, ln in enumerate(lines):                       # 1) def/class whose name matches
        m = re.match(r"\s*(def|class)\s+(\w+)", ln)
        if m and any(k in m.group(2).lower() for k in kws):
            j = i + 1
            while j < len(lines) and j - i < max_lines:
                if re.match(r"(def|class)\s+\w+", lines[j]): break
                j += 1
            return "\n".join(lines[i:j])
    hits = [i for i, ln in enumerate(lines)             # 2) grep, skip docstring/comment
            if any(k in ln.lower() for k in kws) and not ln.strip().startswith(("#", '"""', "'''"))]
    if not hits: return ""
    return "\n".join(lines[max(0, hits[0]-2):hits[0]+10][:max_lines])

def slugify(title):
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60]

def ev_nodes(node):
    """Yield (run_ts, bid, turn, journal_node) for each valid B{bid}.T{turn} evidence label."""
    src = {(str(b), r) for r, b in node.get("source_branches", [])}
    for lab in node.get("evidence_turns", []):
        m = re.match(r"B(\d+)\.T(\d+)", lab, flags=re.I)
        if not m: continue
        bid, turn = m.group(1), int(m.group(2))
        run_ts = next((r for (b, r) in src if b == bid), None)
        if not run_ts: continue
        bns = branch_nodes(run_ts, bid)
        if 1 <= turn <= len(bns):
            yield run_ts, bid, turn, bns[turn - 1]

def build_reference(node):
    L = [f"# {node['title']}\n",
         f"**category**: {node['category']}  |  **level**: {node.get('level','?')}  |  "
         f"p_hat: {node.get('p_hat','?')} (n_use={node.get('n_use','?')})\n",
         "## When / why", node.get("condition", "").strip() or "_(no condition)_", "",
         "## Principle", node.get("principle", "").strip(), ""]
    kws = keywords(node["title"]); seen_summ, snips, evid = set(), [], []
    for run_ts, bid, turn, n in ev_nodes(node):
        summ = (n.get("code_summary") or "").strip()
        key = summ[:80]
        if summ and key not in seen_summ:
            seen_summ.add(key)
            evid.append((run_ts, bid, turn, n))
        snip = extract_snippet(n.get("code") or "", kws)
        if snip and snip not in snips: snips.append(snip)
    if evid:
        L += ["## What the run actually did (code_summary)"]
        for run_ts, bid, turn, n in evid:
            L.append(f"- **run {run_ts} / branch {bid} / Turn {turn}** (stage={n.get('stage')}, "
                     f"metric={metric_val(n)}, buggy={n.get('is_buggy')}):")
            L.append(f"  > {(n.get('code_summary') or '_(no summary)_').strip()[:400]}")
        L.append("")
    if snips:
        L += ["## Real code (from the evidence turn's solution)", "```python", snips[0], "```", ""]
    L += ["## Evidence"]
    if evid:
        for run_ts, bid, turn, n in evid:
            L.append(f"- run {run_ts} / branch {bid} / Turn {turn}: stage={n.get('stage')}, "
                     f"val_log_loss={metric_val(n)}, buggy={n.get('is_buggy')}")
            an = (n.get("analysis") or "").strip()
            if an: L.append(f"    analysis: {an[:240]}")
    else:
        L.append("_(no cited turns resolved)_")
    return "\n".join(L)

def main():
    show = None
    args = sys.argv[1:]
    if args and args[0] == "--show": show = (args[1].lower() if len(args) > 1 else None)
    src = IN if IN.exists() else (GB / "raw_nodes.json")
    data = json.load(open(src))
    nodes = data.get("nodes", data) if isinstance(data, dict) else data
    OUT.mkdir(parents=True, exist_ok=True)
    n_evid = 0
    for node in nodes:
        (OUT / f"{slugify(node['title'])}.md").write_text(build_reference(node), encoding="utf-8")
        if any(True for _ in ev_nodes(node)): n_evid += 1
        if show and show in node["title"].lower():
            print(build_reference(node)); print("\n" + "=" * 60)
    print(f"生成 {len(nodes)} 个 reference -> {OUT}/  ({n_evid} 个解析到真实证据 turn)")

if __name__ == "__main__":
    main()
