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

# Canonical universal SOP clusters. Each entry uses a fixed generic principle so model/API details
# from absorbed nodes cannot leak back into the active universal layer.
CANONICAL = [
    {
        "title": "Run a script-order sanity check before execution",
        "principle": ("Before running a long script, scan it top-to-bottom and ensure every variable, "
                      "function, class, helper, fold index, and loader is defined before first use."),
        "condition": "Long scripts, merged snippets, multi-stage pipelines, or refactors.",
        "pattern": (r"define|before use|before first use|before first call|first usage|variable.*defined|"
                    r"function.*before|class.*before|helper.*before|script order|definition order|"
                    r"fold indices.*before|dataloader objects before|device variable.*before"),
    },
    {
        "title": "Clean merged or generated code before execution",
        "principle": ("Before execution, remove merge markers, stray prose, duplicated definitions, "
                      "Markdown/HTML fragments, and other non-Python artifacts."),
        "condition": "After patching, merging, copy-pasting, or generating code.",
        "pattern": (r"merge conflict|merge marker|stray|duplicate.*def|duplicate.*function|"
                    r"duplicate.*block|markdown|html|syntax error|orphan|artifact.*before"),
    },
    {
        "title": "Validate data-flow contracts between pipeline stages",
        "principle": ("Before a downstream stage loads, concatenates, or transforms artifacts, verify "
                      "that files exist and sample counts, row order, columns, and feature dimensions match."),
        "condition": "Pipelines with separate preprocessing, feature extraction, training, and inference stages.",
        "pattern": (r"feature extraction.*return|row count|same number of rows|number of samples|"
                    r"consistent.*row|feature matrix|feature arrays|path.*exist|file.*before load|"
                    r"intermediate|\.npy|length.*match|index.*align|concat.*feature|all required columns"),
    },
    {
        "title": "Fit transformations only on training data to avoid leakage",
        "principle": ("Fit scalers, vectorizers, PCA, feature selectors, and group statistics only on the "
                      "training split or fold, then transform validation and test data with the fitted object."),
        "condition": "Any cross-validation or train/validation pipeline with learned preprocessing.",
        "pattern": (r"fit.*train|train fold|train only|train data only|avoid leak|scaler.*train|"
                    r"vectorizer.*train|PCA.*train|selector.*train|group stat.*train|transductive|"
                    r"normalize.*train|do not fit.*full"),
    },
    {
        "title": "Run small smoke tests before full expensive runs",
        "principle": ("Run one fold, a few rows, or one batch first to validate shapes, memory use, "
                      "numerical stability, and file outputs before launching the full run."),
        "condition": "New training loops, expensive cross-validation, large models, or uncertain data pipelines.",
        "pattern": (r"smoke|one fold|few rows|one batch|test.*before full|isolat.*fold|minimal repro|"
                    r"sanity.*check|quick test|tiny.*batch|debug.*before"),
    },
    {
        "title": "Check model and tensor interface consistency",
        "principle": ("Inspect model inputs and outputs before wiring components together; verify forward "
                      "return structure, head dimensions, tensor device, dtype, shape, and scalar loss."),
        "condition": "Custom heads, wrapped models, custom losses, or multi-tensor training code.",
        "pattern": (r"\bforward\b|model.*attribute|attribute names|input dim|output.*structure|"
                    r"output.*match|tensor.*shape|scalar loss|signature|logits|hidden state|"
                    r"backbone.*output|head.*input|device placement|same device"),
    },
    {
        "title": "Check GPU or resource budget before loading large models",
        "principle": ("Check available memory and expected footprint before loading large models or starting "
                      "expensive training; reduce model size, batch size, or sequence length when needed."),
        "condition": "Large models, limited GPU memory, container limits, or previous OOM failures.",
        "pattern": (r"OOM|out.of.memory|GPU memory|memory.*budget|reduce.*batch|model.*size.*reduce|"
                    r"memory.*footprint|fit.*memory|large models"),
    },
    {
        "title": "Change one major component at a time",
        "principle": ("When improving a pipeline, change one major component at a time and compare against "
                      "the previous best so regressions can be attributed."),
        "condition": "Iterative model, data, feature, or training-strategy changes.",
        "pattern": r"one component|one change at|incremental|isolat.*change|single variable|one variable|ablation",
    },
]

DEMOTABLE = (
    r"\bAPI\b|parameter name|import path|correct (?:import|parameter|API|method|argument|flag|keyword)|"
    r"timm|torchvision|huggingface|transformers|sentence ?transformer|sentence-bert|sbert|"
    r"calibratedclassifiercv|"
    r"xgboost|lightgbm|catboost|scikit-learn|sklearn|nltk|randaugment|"
    r"pandas|dataframe|pd\.|\.loc|\.iloc|boolean indexing|str\.count|regex|regular expression|"
    r"numpy|np\.|sparse matrix|dense array|pd\.cut|bin edges|"
    r"stylometric|tf-idf|tfidf|cuda|leaf classification|tabular feature engineering|"
    r"multi-branch mlp|cross-attention|margin/shape/texture|margin|texture|"
    r"allow_pickle|weights_only|dataloader|dataset|collate|iterator|num_workers|pin_memory|drop_last|batchnorm|"
    r"adamw|optimizer|learning rate|lr schedule|warmup|cosine|"
    r"amp|mixed precision|gradient accumulation|gradient checkpoint|label smoothing|"
    r"bcewithlogitsloss|loss function|log loss|one-hot|probabilit|"
    r"deberta|modernbert|transformer model|transformer models|truncation|pytorch|torch\.|"
    r"model checkpoint|state dict|swa state|checkpoint filenames|checkpoint.*cpu"
)


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


def node_branch_count(n):
    return len({tuple(x) for x in n.get("source_branches", [])})


def source_summary(nodes):
    branches = {tuple(sb) for n in nodes for sb in n.get("source_branches", [])}
    tasks = {t for n in nodes for t in node_tasks(n)}
    return branches, tasks


def classify(n):
    blob = (n["title"] + " " + (n.get("principle") or "")).lower()
    scope = (n.get("scope") or "").strip()
    if scope in ("api_warning", "implementation_note") or re.search(DEMOTABLE, blob, flags=re.I):
        return ("demote_api", None)
    for sop in CANONICAL:
        if re.search(sop["pattern"], blob, flags=re.I):
            return ("cluster", sop["title"])
    return ("ambiguous", None)


def canonical_by_title(title):
    return next(s for s in CANONICAL if s["title"] == title)


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
    low_evidence_canonical = []
    case_studies = {}
    for canon, members in clusters.items():
        if not members:
            continue
        branches, tasks = source_summary(members)
        if len(branches) < 2 and len(tasks) < 2:
            low_evidence_canonical.extend(members)
            continue
        sop = canonical_by_title(canon)
        src = dedup_seq([sb for m in members for sb in m.get("source_branches", [])])
        ev = dedup_seq([e for m in members for e in m.get("evidence_turns", [])])
        node = {
            "title": canon,
            "principle": sop["principle"],
            "condition": sop["condition"],
            "category": "general",
            "scope": "universal_general",
            "evidence_turns": ev[:60],
            "source_branches": src,
            "absorbed_node_ids": [m["id"] for m in members],
            "absorbed_titles": [m["title"] for m in members],
            "source_task_types": sorted(tasks),
            "general_normalization_action": "canonical_cluster_merge",
        }
        canon_nodes.append(node)
        case_studies[canon] = {"members": len(members),
                               "source_branches": len(branches),
                               "source_task_types": sorted(tasks),
                               "sample_absorbed": [m["title"] for m in members[:4]]}

    # ambiguous kept -> universal_general
    for n in amb_kept:
        n["scope"] = "universal_general"; n["was_general"] = True
        n["general_normalization_action"] = "kept_cross_task"
        n["kept_reason"] = f"appears across {len(set(node_tasks(n)))} task types"

    # demoted -> task-specific
    demoted = []
    demote_api_ids = {id(n) for n in demote_api}
    for n in demote_api + amb_demoted + low_evidence_canonical:
        tasks = node_tasks(n)
        dom = Counter(tasks).most_common(1)[0][0] if tasks else "spooky-author-identification"
        is_api = id(n) in demote_api_ids
        n["category"] = dom
        n["scope"] = "api_warning" if is_api else "implementation_note"
        n["was_general"] = True
        n["general_normalization_action"] = "demoted_to_task"
        n["demotion_reason"] = ("API/model/library/framework warning" if is_api
                                else "single-source evidence, not universal")
        demoted.append(n)

    # task_nodes: ensure scope set
    for n in task_nodes:
        n.setdefault("scope", "task_specific")

    out_nodes = canon_nodes + amb_kept + demoted + task_nodes
    for i, n in enumerate(out_nodes, 1):
        n["id"] = f"sg_{i:04d}"

    bad_universal = []
    for n in canon_nodes + amb_kept:
        titles = [n["title"]] + n.get("absorbed_titles", [])
        for title in titles:
            if re.search(DEMOTABLE, title, flags=re.I):
                bad_universal.append((n["title"], title))
    if bad_universal:
        msg = "\n".join(f"{canon} <- {title}" for canon, title in bad_universal[:20])
        raise RuntimeError("API/model/library-specific nodes leaked into universal_general:\n" + msg)

    n_gen_before = len(generals)
    n_gen_after = len(canon_nodes) + len(amb_kept)
    meta = {"n_before": len(nodes), "n_after": len(out_nodes),
            "general_before": n_gen_before, "general_after": n_gen_after,
            "canonical_clusters": len(canon_nodes), "ambiguous_kept": len(amb_kept),
            "demoted": len(demoted), "demoted_api": len(demote_api),
            "demoted_ambiguous_single_task": len(amb_demoted),
            "demoted_low_evidence_canonical": len(low_evidence_canonical),
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
          f"single-task ambiguous {meta['demoted_ambiguous_single_task']}, "
          f"low-evidence canonical {meta['demoted_low_evidence_canonical']})",
          "", "## Canonical clusters", ""]
    for canon, cs in case_studies.items():
        md.append(f"### {canon}  ({cs['members']} absorbed; "
                  f"{cs['source_branches']} source branches; "
                  f"tasks={cs['source_task_types']})")
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
    print(f"  demoted: {meta['demoted']}  (api={meta['demoted_api']}, "
          f"single_task={meta['demoted_ambiguous_single_task']}, "
          f"low_evidence={meta['demoted_low_evidence_canonical']})")
    print(f"  -> {OUT}")


if __name__ == "__main__":
    main()
