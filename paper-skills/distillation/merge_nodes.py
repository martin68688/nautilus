"""
merge_nodes.py — build-time one-pass Merge to collapse near-duplicate skill nodes.

WHY TEXT SIMILARITY, NOT neighbor-Jaccard: the paper's Merge (§3.3.1) uses neighbor-set Jaccard,
which is DEGENERATE on our single-task init graph — enhance-all (every general->every task-specific)
collapses all generals to identical neighborhoods AND inflates task-specific Jaccard to a ~0.83
baseline (49 shared general-neighbors dominate). So we DETECT candidates by text similarity
(token-Jaccard on title+principle) — a non-paper pragmatic choice that actually catches semantic
near-dupes — and MERGE them with the teacher LLM (the paper's operation: synthesize one concise skill,
Appendix E merge prompt). Clusters form WITHIN category so a process skill never merges with a technique.

Pipeline: distill -> [this] merge_nodes -> build_edges_levels -> build_references.
"""
import os, re, json, sys, pathlib, textwrap, time

REPO = pathlib.Path(__file__).resolve().parents[2]
GB = REPO / "paper-skills" / "distillation" / "graph_build"
IN = GB / "raw_nodes.json"
OUT = GB / "merged_nodes.json"
ENV = REPO / "mlevolve" / ".env"

def load_env():
    if ENV.exists():
        for line in ENV.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                k, v = s.split("=", 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
load_env()
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
TAU = 0.50  # text-Jaccard threshold on title+principle
STOP = set("use for in the a an to when on with and of by from as is are be it that this via into your".split())

def toks(s): return set(re.findall(r"[a-z0-9]+", (s or "").lower())) - STOP
def jac(a, b): return len(a & b) / len(a | b) if (a | b) else 0.0

def cluster(nodes):
    clusters = []  # (rep_toks, [nodes]); greedy first-match
    for n in nodes:
        tk = toks(n["title"] + " " + n.get("principle", ""))
        for c in clusters:
            if jac(tk, c[0]) >= TAU:
                c[1].append(n); break
        else:
            clusters.append((tk, [n]))
    return [c[1] for c in clusters]

MERGE_SYS = textwrap.dedent("""\
    You merge redundant ML-skill nodes (same idea, different wording) into ONE concise skill.
    Synthesize a single unified skill that preserves the most CONCRETE details (parameters,
    conditions, mechanisms) from all inputs. Keep `category` identical to the inputs.
    Output ONLY a JSON object {"title","principle","condition","category","scope"}.
      - title: short imperative, <=12 words.
      - principle: HOW — merge the most specific params/mechanisms from all inputs.
      - condition: WHEN — union of the inputs' applicability.
      - scope: preserve the narrowest useful scope from the inputs.""")

def merge_call(skills):
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120, max_retries=2)
    blob = "\n\n".join(f"({i+1}) title: {s['title']}\n    principle: {s.get('principle','')}\n    "
                       f"condition: {s.get('condition','')}\n    category: {s.get('category','')}\n    "
                       f"scope: {s.get('scope','')}"
                       for i, s in enumerate(skills))
    resp = None
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=MODEL, temperature=0.2, max_tokens=600, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": MERGE_SYS},
                          {"role": "user", "content": f"Skills to merge:\n{blob}\n\nOutput the merged skill JSON now."}])
            break
        except Exception as e:
            if attempt == 4:
                raise
            wait = min(45, 5 * (attempt + 1))
            print(f"    !! merge API error: {type(e).__name__}: {e}; retrying in {wait}s", flush=True)
            time.sleep(wait)
    t = resp.choices[0].message.content or ""
    for c in re.findall(r"\{.*\}", t, flags=re.DOTALL) + [t]:
        try:
            o = json.loads(c)
            if isinstance(o, dict) and o.get("title"):
                return o
        except Exception:
            pass
    return None

def union_meta(cluster):
    sb, ev, seen_sb, seen_ev = [], [], set(), set()
    for n in cluster:
        for r, b in n.get("source_branches", []):
            if (r, b) not in seen_sb: seen_sb.add((r, b)); sb.append([r, b])
        for e in n.get("evidence_turns", []):
            if e not in seen_ev: seen_ev.add(e); ev.append(e)
    return sb, ev


def default_scope(category):
    return "universal_general" if category == "general" else "task_specific"


def merged_scope(cluster, category):
    """Keep merged scope deterministic; never let merging widen a warning into a global SOP."""
    if category == "general":
        return "universal_general"
    scopes = [n.get("scope") or default_scope(category) for n in cluster]
    for s in ("api_warning", "implementation_note", "task_specific"):
        if s in scopes:
            return s
    return default_scope(category)


def main():
    if not API_KEY:
        print("ERROR: DEEPSEEK_API_KEY not set"); sys.exit(1)
    data = json.load(open(IN))
    nodes = data["nodes"]
    for n in nodes:
        n.setdefault("scope", default_scope(n.get("category", "")))
    by_cat = {}
    for n in nodes:
        by_cat.setdefault(n["category"], []).append(n)
    merged = []
    for cat in sorted(by_cat):
        grp = by_cat[cat]
        clusters = cluster(grp)
        n_merged = sum(1 for c in clusters if len(c) > 1)
        print(f"  [{cat}] {len(grp)} -> {len(clusters)} nodes ({n_merged} clusters merged)")
        for c in clusters:
            if len(c) == 1:
                merged.append(c[0]); continue
            o = merge_call(c)
            if o:
                sb, ev = union_meta(c)
                mcat = (o.get("category") or cat).strip() or cat
                merged.append({"id": "", "title": o["title"].strip(),
                               "principle": (o.get("principle") or "").strip(),
                               "condition": (o.get("condition") or "").strip(),
                               "category": mcat,
                               "scope": merged_scope(c, mcat),
                               "evidence_turns": ev, "source_branches": sb})
            else:  # fallback: keep richest, union metadata
                rich = max(c, key=lambda n: len(n.get("evidence_turns", [])))
                sb, ev = union_meta(c)
                rich["evidence_turns"], rich["source_branches"] = ev, sb
                rich["scope"] = merged_scope(c, rich.get("category", cat))
                merged.append(rich)
    for i, n in enumerate(merged, 1):
        n["id"] = f"sg_{i:04d}"
    OUT.write_text(json.dumps({"meta": {"teacher": MODEL, "tau": TAU, "n_before": len(nodes),
                                        "n_after": len(merged)}, "nodes": merged},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== {len(nodes)} -> {len(merged)} nodes -> {OUT} ===")

if __name__ == "__main__":
    main()
