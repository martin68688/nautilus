"""
normalize_general_nodes.py — Codex Part 2+3: post-hoc normalization of general nodes.

Deterministic (no LLM). Reads merged_nodes.json and produces a cleaner general layer:
  1. Infer each general node's source tasks (from source_branches -> run config.yaml).
  2. Classify each general:
       - canonical universal SOP cluster (8 curated clusters) -> merge into that cluster node.
       - API/model/library/framework warning -> DEMOTE to its dominant source task (scope=api_warning).
       - ambiguous -> keep general ONLY if it appears across >=2 task types; else demote (scope=implementation_note).
  3. Merge each canonical cluster's members into ONE node (canonical title; union source_branches,
     evidence_turns, absorbed ids/titles). Demoted nodes become task-specific.

Outputs:
  merged_nodes_general_normalized.json   (the normalized node set)
  general_normalization_report.md        (human-readable counts + per-cluster + case studies)
  general_normalization_report.json      (machine-readable)

Acceptance target: active category=general <= 10 (preferably <= 6 so faithful K=8 retrieval
returns >= 2 task-specific per task).
"""
import json, re, glob, pathlib
from collections import Counter, defaultdict

REPO = pathlib.Path(__file__).resolve().parents[2]
GB = REPO / "paper-skills" / "distillation" / "graph_build"
RUNS = REPO / "mlevolve" / "runs"
IN = GB / "merged_nodes.json"
OUT = GB / "merged_nodes_general_normalized.json"
REP_MD = GB / "general_normalization_report.md"
REP_JSON = GB / "general_normalization_report.json"

# 8 canonical universal SOP clusters (Codex): canonical title + absorb regex (title+principle)
CANONICAL = [
    ("Validate definitions, order, and data-flow before execution",
     r"define|before us|before call|before first|before the loop|variable.*defined|function.*before|"
     r"class.*before|init.*before|order of (definition|definition)|declare.*before|scope.*before|"
     r"\bshape\b|row count|column.*match|\.npy|checkpoint|intermediate|file.*before load|"
     r"dimension.*match|path.*exist|length.*match|index.*align|concat.*shape|feature.*matrix"),
    ("Clean merged or generated code before execution",
     r"merge conflict|merge marker|stray|duplicate.*def|duplicate.*function|duplicate.*block|"
     r"markdown|html in|syntax error|remove.*comment|orphan|leftover|artifact.*before"),
    ("Fit transformations only on training data to avoid leakage",
     r"fit.*train|train fold|train only|train data only|avoid leak|scaler.*train|vectorizer.*train|"
     r"PCA.*train|selector.*train|group stat.*train|transductive|normalize.*train|do not fit.*full"),
    ("Run small smoke tests before full expensive runs",
     r"smoke|one fold|few rows|one batch|test.*before full|isolat.*fold|minimal repro|"
     r"sanity.*check|quick test|tiny.*batch|debug.*before"),
    ("Check model and tensor interface consistency",
     r"\bforward\b|attribute|input dim|output.*match|device|dtype|tensor.*shape|scalar loss|"
     r"signature|logits|hidden state|backbone.*output|head.*input"),
    ("Check GPU or resource budget before loading large models",
     r"OOM|out.of.memory|GPU memory|memory.*budget|resource|reduce.*batch|model.*size.*reduce|"
     r"memory.*footprint|fit.*memory"),
    ("Change one major component at a time",
     r"one component|one change at|incremental|isolat.*change|single variable|one variable|ablation"),
]
DEMOTABLE = (r"\bAPI\b|parameter name|import path|correct.*import|correct.*parameter|correct.*API|"
             r"timm|torchvision|huggingface|transformers\.|allow_pickle|weights_only|RandAugment|"
             r"SentenceTransformer|CalibratedClassifierCV|num_workers|version compat|deprecat|"
             r"correct.*method name|correct.*argument|correct.*flag|correct.*keyword")


def run_task(run_ts, _cache={}):
    if run_ts not in _cache:
        fs = glob.glob(str(RUNS / f"{run_ts}_*" / "logs" / "config.yaml"))
        t = None
        if fs:
            for line in pathlib.Path(fs[0]).read_text(encoding="utf-8", errors="ignore").splitlines():
                if "exp_id" in line and ":" in line:
                    t = line.split(":", 1)[1].strip().strip('"').strip("'")
                    break
        _cache[run_ts] = t
    return _cache[run_ts]


def node_tasks(n):
    return [t for t in (run_task(r) for r, _ in n.get("source_branches", [])) if t]


def classify(n):
    blob = (n["title"] + " " + (n.get("principle") or "")).lower()
    for canon, pat in CANONICAL:
        if re.search(pat, blob):
            return ("cluster", canon)
    if re.search(DEMOTABLE, blob):
        return ("demote_api", None)
    return ("ambiguous", None)


def dedup_seq(pairs):
    seen, out = set(), []
    for x in pairs:
        k = tuple(x) if isinstance(x, list) else x
        if k not in seen:
            seen.add(k); out.append(x)
    return out


def main():
    data = json.load(open(IN, encoding="utf-8"))
    nodes = data["nodes"]
    generals = [n for n in nodes if n["category"] == "general"]
    task_nodes = [n for n in nodes if n["category"] != "general"]

    clusters = defaultdict(list)   # canon title -> [members]
    demote_api = []
    ambiguous = []
    for n in generals:
        kind, canon = classify(n)
        if kind == "cluster":
            clusters[canon].append(n)
        elif kind == "demote_api":
            demote_api.append(n)
        else:
            ambiguous.append(n)

    # ambiguous: keep general ONLY if >=2 task types (truly universal); else demote
    amb_kept, amb_demoted = [], []
    for n in ambiguous:
        if len(set(node_tasks(n))) >= 2:
            amb_kept.append(n)
        else:
            amb_demoted.append(n)

    # build canonical cluster nodes (deterministic merge)
    canon_nodes = []
    case_studies = {}
    for canon, members in clusters.items():
        if not members:
            continue
        src = dedup_seq([sb for m in members for sb in m.get("source_branches", [])])
        ev = dedup_seq([e for m in members for e in m.get("evidence_turns", [])])
        rep = max(members, key=lambda m: len(m.get("principle", "")))
        node = {
            "title": canon,
            "principle": rep.get("principle", ""),
            "condition": "Applies across all ML tasks (universal process SOP).",
            "category": "general",
            "scope": "universal_general",
            "evidence_turns": ev[:60],
            "source_branches": src,
            "absorbed_node_ids": [m["id"] for m in members],
            "absorbed_titles": [m["title"] for m in members],
            "general_normalization_action": "canonical_cluster_merge",
        }
        canon_nodes.append(node)
        case_studies[canon] = {"members": len(members),
                               "sample_absorbed": [m["title"] for m in members[:4]]}

    # ambiguous kept -> universal_general
    for n in amb_kept:
        n["scope"] = "universal_general"; n["was_general"] = True
        n["general_normalization_action"] = "kept_cross_task"
        n["kept_reason"] = f"appears across {len(set(node_tasks(n)))} task types"

    # demoted -> task-specific
    demoted = []
    for n in demote_api + amb_demoted:
        tasks = node_tasks(n)
        dom = Counter(tasks).most_common(1)[0][0] if tasks else "spooky-author-identification"
        is_api = n in demote_api
        n["category"] = dom
        n["scope"] = "api_warning" if is_api else "implementation_note"
        n["was_general"] = True
        n["general_normalization_action"] = "demoted_to_task"
        n["demotion_reason"] = ("API/model/library/framework warning" if is_api
                                else "single-task evidence, not universal")
        demoted.append(n)

    # task_nodes: ensure scope set
    for n in task_nodes:
        n.setdefault("scope", "task_specific")

    out_nodes = canon_nodes + amb_kept + demoted + task_nodes
    for i, n in enumerate(out_nodes, 1):
        n["id"] = f"sg_{i:04d}"

    n_gen_before = len(generals)
    n_gen_after = len(canon_nodes) + len(amb_kept)
    meta = {"n_before": len(nodes), "n_after": len(out_nodes),
            "general_before": n_gen_before, "general_after": n_gen_after,
            "canonical_clusters": len(canon_nodes), "ambiguous_kept": len(amb_kept),
            "demoted": len(demoted), "demoted_api": len(demote_api),
            "demoted_ambiguous_single_task": len(amb_demoted),
            "task_specific_unchanged": len(task_nodes)}
    json.dump({"meta": meta, "nodes": out_nodes}, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # reports
    json.dump({"meta": meta, "case_studies": case_studies,
               "ambiguous_kept": [{"title": n["title"], "tasks": list(set(node_tasks(n)))} for n in amb_kept]},
              open(REP_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    md = ["# General Normalization Report", "",
          f"- nodes: {meta['n_before']} -> {meta['n_after']}",
          f"- **active general: {meta['general_before']} -> {meta['general_after']}** "
          f"(canonical clusters: {meta['canonical_clusters']}, ambiguous cross-task kept: {meta['ambiguous_kept']})",
          f"- demoted to task-specific: {meta['demoted']} (api_warning {meta['demoted_api']}, "
          f"single-task ambiguous {meta['demoted_ambiguous_single_task']})",
          "", "## Canonical clusters", ""]
    for canon, cs in case_studies.items():
        md.append(f"### {canon}  ({cs['members']} absorbed)")
        for t in cs["sample_absorbed"]:
            md.append(f"  - {t}")
        md.append("")
    md += ["## Ambiguous kept (cross >=2 task types)", ""]
    for n in amb_kept:
        md.append(f"- {n['title']}  tasks={list(set(node_tasks(n)))}")
    open(REP_MD, "w", encoding="utf-8").write("\n".join(md))

    print(f"=== normalize: {meta['n_before']} -> {meta['n_after']} nodes ===")
    print(f"  general: {meta['general_before']} -> {meta['general_after']} "
          f"(canonical={meta['canonical_clusters']}, kept_ambiguous={meta['ambiguous_kept']})")
    print(f"  demoted: {meta['demoted']}  (api={meta['demoted_api']}, single_task={meta['demoted_ambiguous_single_task']})")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
