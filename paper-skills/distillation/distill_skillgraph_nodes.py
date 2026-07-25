"""
distill_skillgraph_nodes.py — SkillGraph(2605.12039)-style skill distillation on mlevolve traces.

Faithful to the paper's distillation V <- M(T+, T-) (Alg.1 line 3): the teacher reads a POOL of
successful (T+) and failed (T-) attempts and emits skill nodes. We approximate the pool with
BATCHES of branches (configurable size) so the input fits context while staying closer to the
paper's pooled distillation than strict per-branch calls.

Each skill node carries the paper's 4 fields {title, principle, condition, category} where
category = "general" | "<task_type>" (task_type folded into category, per paper Appendix B).
evidence_turns (list of "B<branch>.T<turn>" labels) is OUR extension to ground references.

Teacher: DeepSeek (deepseek-v4-flash). Edges/levels/stats/Merge are NOT done
here (see build_edges_levels.py).

Usage:
  python distill_skillgraph_nodes.py --demo            # newest run, batch of 3 richest branches
  python distill_skillgraph_nodes.py --batch 5 --run 20260627_135133
  python distill_skillgraph_nodes.py <branch.md> ...   # explicit, one batch
"""
import os, re, json, sys, pathlib, textwrap, glob, time

REPO = pathlib.Path(__file__).resolve().parents[2]
TRACES = REPO / "paper-skills" / "distillation" / "traces"
OUT = REPO / "paper-skills" / "distillation" / "graph_build"
ENV_FILE = REPO / "mlevolve" / ".env"
PARTIAL_OUT = OUT / "raw_nodes.partial.json"

def load_env():
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
load_env()
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
VALID_SCOPES = {"universal_general", "api_warning", "implementation_note", "task_specific"}
GENERAL_DEMOTE_RE = re.compile(
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
    r"deberta|modernbert|transformer model|transformer models|truncation|pytorch|torch\.",
    flags=re.I,
)
GENERAL_ALLOW_RE = re.compile(
    r"merge conflict|merge marker|stray .*code|generated code|"
    r"define .*before|variable.*before use|function.*before|class.*before|helper.*before|"
    r"script order|definition order|nameerror|"
    r"fit .*train|train fold|train only|avoid leakage|data leakage|"
    r"smoke|one fold|one batch|few rows|minimal repro|sanity check|"
    r"data[- ]flow|pipeline stage|row count|feature dimension|input/output|file.*exist|"
    r"forward output|model interface|tensor.*shape|shape mismatch|shapes? match|"
    r"same device|dtype|scalar loss|head dimension|"
    r"gpu memory|oom|resource budget|memory budget|"
    r"one major component|one change|ablation|incremental",
    flags=re.I,
)

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a skill-graph distiller for an automated ML-solver agent (mlevolve). You read execution
    trajectories — attempts on a Kaggle-style ML competition task — and distill them into procedural
    SKILL NODES. The specific task id and its metric direction (higher- or lower-better) are given in
    the pool below; interpret "improvement" strictly per that direction.

    Operate exactly as the SkillGraph teacher does: you are given a POOL of successful (T+) and a
    POOL of failed (T-) attempts (from several search branches), and emit skills of two classes.
    The `category` field MUST be EXACTLY one of:
      - "general": a domain-independent REASONING / PROCESS / DEBUGGING / VALIDATION habit — HOW you
                   work, NOT WHAT technique. Examples:
                   "verify each sub-goal succeeds before proceeding", "isolate a failure with a minimal
                   repro before fixing", "compare a new metric against the previous best before adopting
                   a change", "define every variable/function before it is called", "fix a typo with
                   global search-and-replace", "fit any scaler/vectorizer on the TRAIN FOLD ONLY to
                   avoid leakage", "read the full traceback and fix the root cause".
      - "<task_type>": use the EXACT task id given in the pool below (e.g. "leaf-classification",
                   "aerial-cactus-identification") for ANY strategy tied to that task. The following are
                   ALWAYS task-specific, NEVER general: optimizer / LR-schedule (AdamW, cosine annealing,
                   linear warmup), training mechanics (AMP / mixed precision, gradient clipping, gradient
                   accumulation), loss functions (label smoothing), model architecture, feature
                   engineering, ensemble methods, early stopping, num_workers, CV fold count.
    Never use a descriptive label like "debugging" or "loss_function". A training technique or model/
    feature choice is NEVER general. When unsure, label task-specific with the pool's task id.

    Rules:
      - **At most 1 general skill per batch.** Mark a skill `general` ONLY if it is a domain-
        independent process habit that appears across multiple branches or is clearly applicable
        across multiple task types. Single-library / single-model / single-API / single-framework
        issues are NEVER general — assign them to the task id with scope=api_warning or
        implementation_note.
      - `scope` (REQUIRED field): one of "universal_general" (true cross-task process SOP),
        "api_warning" (a specific library/API/model gotcha), "implementation_note" (a task-scoped
        implementation detail), or "task_specific" (default for task techniques).
      - principle = HOW (include key params / code intent); condition = WHEN it applies.
      - Do NOT duplicate any title in the existing-skill list below.
      - Prefer many small single-purpose skills.
      - evidence_turns: cite the 1-3 attempt labels "B<branch>.T<turn>" that most directly evidence it.
      - Emit a single ```json block: {"skills":[{title, principle, condition, category, scope, evidence_turns}]}.
""")

USER_TEMPLATE = textwrap.dedent("""\
    ## Existing skill titles (avoid duplicating)
    {EXISTING_TITLES}

    ## Trajectory pool — {NB} search branches from run {RUN_TS}
    Task: **{TASK}**   Metric direction: **{DIRECTION}** (a metric move in this direction = improvement).
    {BRANCH_META}

    Each attempt is labeled "B<branch>.T<turn>". buggy=True means it crashed (Failure line shows the
    exception); buggy=False with a metric means it ran and scored. A metric move in the {DIRECTION}
    direction between two successful attempts = the later strategy improved the objective.

    ### T+ (successful attempts — what worked)
    {SUCCESS_TURNS}

    ### T- (failed attempts — crashes / regressions / dead-ends; Failure line kept)
    {FAILED_TURNS}

    ## Your task
    Distill the pool into skill nodes following the system rules. Use the task id "{TASK}" as the
    task_type for every task-specific skill. A strategy present in T+ whose absence/violation in T-
    caused a crash -> strong task-specific candidate. A process/reasoning pattern is a `general`
    candidate ONLY if it is domain-independent and reusable beyond this task/model/library. If no
    such pattern exists, emit zero `general` skills. A success that IMPROVED the metric (per the
    direction above) -> high-value; preserve its params in `principle`. Emit the ```json object now.
""")


# ---- branch .md parser (format from extract_branches.render_branch) ----
TURN_RE = re.compile(r"## Turn (\d+)\s+\[stage=(\S+)\s+buggy=(True|False)\s+metric=(\S+)\]")
META_RE = re.compile(r"\*\*Turns\*\*:\s*(\d+)\s+\*\*Success\*\*:\s*(\d+)\s+\*\*Buggy\*\*:\s*(\d+)\s+\*\*Best metric.*?\*\*:\s*(\S+)")

def parse_branch(md_path):
    text = pathlib.Path(md_path).read_text(encoding="utf-8")
    run_ts = pathlib.Path(md_path).parent.name
    branch_id = pathlib.Path(md_path).stem.replace("branch_", "")
    m = META_RE.search(text)
    succ, fail = [], []
    for p in re.split(r"(?=## Turn \d+\s)", text):
        tm = TURN_RE.search(p)
        if not tm:
            continue
        tnum, buggy = tm.group(1), tm.group(3) == "True"
        label = f"B{branch_id}.T{tnum}"
        block = f"**[{label}]** (stage={tm.group(2)}, metric={tm.group(4)})\n" + p.strip()
        (fail if buggy else succ).append((label, block))
    task_m = re.search(r"\*\*Task\*\*:\s*(\S+)", text)
    direction = "higher=better" if "higher=better" in text else "lower=better"
    return {
        "run_ts": run_ts, "branch_id": branch_id,
        "task": task_m.group(1) if task_m else "unknown", "direction": direction,
        "n_turns": int(m.group(1)) if m else 0,
        "n_succ": int(m.group(2)) if m else 0,
        "n_bug": int(m.group(3)) if m else 0,
        "best_metric": (m.group(4) if m else "None"),
        "success": succ, "failed": fail,
    }


def render_user(batch, existing_titles):
    meta = "\n".join(f"- B{b['branch_id']}: turns={b['n_turns']} succ={b['n_succ']} bug={b['n_bug']} "
                     f"best_metric={b['best_metric']}" for b in batch)
    succ = "\n\n".join(blk for _, blk in sum((b["success"] for b in batch), [])) or "(none)"
    fail = "\n\n".join(blk for _, blk in sum((b["failed"] for b in batch), [])) or "(none)"
    return USER_TEMPLATE.format(
        EXISTING_TITLES=("\n".join(f"- {t}" for t in existing_titles) if existing_titles else "(none yet)"),
        NB=len(batch), RUN_TS=batch[0]["run_ts"], TASK=batch[0]["task"], DIRECTION=batch[0]["direction"],
        BRANCH_META=meta, SUCCESS_TURNS=succ, FAILED_TURNS=fail)


def extract_json(text):
    for c in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL) + [text.strip()]:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        skills = obj.get("skills", obj) if isinstance(obj, dict) else obj
        if isinstance(skills, list):
            return [s for s in skills if isinstance(s, dict) and s.get("title")]
    return None


def call_teacher(system, user, retries=2, network_retries=4):
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=180, max_retries=2)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for _ in range(retries + 1):
        resp = None
        for attempt in range(network_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=MODEL, messages=msgs, temperature=0.3, max_tokens=8192,
                    response_format={"type": "json_object"})
                break
            except Exception as e:
                if attempt >= network_retries:
                    raise
                wait = min(60, 5 * (attempt + 1))
                print(f"  !! teacher API error: {type(e).__name__}: {e}; retrying in {wait}s", flush=True)
                time.sleep(wait)
        text = resp.choices[0].message.content or ""
        skills = extract_json(text)
        if skills is not None:
            return skills, resp.usage
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": "Previous output was not valid JSON. Re-emit ONLY a valid {\"skills\":[...]} object."}]
    return [], getattr(resp, "usage", None)


def normalize_category(cat, task):
    c = (cat or "").strip().lower().replace("_", "-")
    if c == "general":
        return "general"
    if cat and cat.strip().lower() not in ("task-specific", "task_specific", ""):
        return cat.strip()
    return task  # default task_type = the batch's task id


def normalize_scope(scope, cat):
    s = (scope or "").strip().lower().replace("-", "_")
    if s not in VALID_SCOPES:
        return "universal_general" if cat == "general" else "task_specific"
    if cat == "general":
        return "universal_general" if s == "universal_general" else s
    return "task_specific" if s == "universal_general" else s


def parse_evidence(ev, batch_branch_ids):
    """evidence_turns -> (list of valid 'B.T' labels, set of branch ids)."""
    valid_ids = set(batch_branch_ids)
    labels, branches = [], set()
    for x in (ev or []):
        s = str(x).strip()
        m = re.match(r"B(\d+)\.T(\d+)", s, flags=re.I)
        if not m or m.group(1) not in valid_ids:
            continue
        labels.append(f"B{m.group(1)}.T{m.group(2)}")
        branches.add(m.group(1))
    return labels, sorted(branches, key=int)


def chunk(xs, n):
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def write_nodes(path, nodes, batches, completed_batches, complete):
    path.write_text(json.dumps(
        {"meta": {"teacher": MODEL, "batches": completed_batches, "n_batches_total": len(batches),
                  "n_nodes": len(nodes), "complete": complete,
                  "branches": [pathlib.Path(p).stem for grp in batches for p in grp]},
         "nodes": nodes}, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    raw = sys.argv[1:]
    resume = "--resume" in raw
    raw = [a for a in raw if a != "--resume"]
    batch_size = 5
    if "--batch" in raw:
        batch_size = int(raw[raw.index("--batch") + 1])
        raw = [a for a in raw if a not in ("--batch", str(batch_size))]
    # build batches; each batch is branch .md paths from a SINGLE run (keeps "B<branch>.T<turn>"
    # labels unambiguous). existing_titles accumulates across batches -> cross-run dedup.
    if "--demo" in raw:
        run = "20260627_135133"
        batches = [[str(TRACES / run / f"branch_{b}.md") for b in (1, 4, 2)]]
    elif "--all" in raw:
        batches = []
        for r in sorted({p.parent.name for p in TRACES.glob("*/branch_*.md")}):
            bs = sorted(glob.glob(str(TRACES / r / "branch_*.md")))
            batches += chunk(bs, batch_size)
    elif "--run" in raw:
        run = raw[raw.index("--run") + 1]
        batches = chunk(sorted(glob.glob(str(TRACES / run / "branch_*.md"))), batch_size)
    elif raw:
        batches = chunk(raw, batch_size)
    else:
        print("usage: --demo | --all | --run <ts> | <branch.md>...  [--batch N]"); sys.exit(1)
    if not API_KEY:
        print("ERROR: DEEPSEEK_API_KEY not set"); sys.exit(1)

    OUT.mkdir(parents=True, exist_ok=True)
    existing_titles, nodes, start_batch = [], [], 1
    if resume and PARTIAL_OUT.exists():
        partial = json.load(open(PARTIAL_OUT, encoding="utf-8"))
        nodes = partial.get("nodes", [])
        existing_titles = [n.get("title", "") for n in nodes if n.get("title")]
        start_batch = int(partial.get("meta", {}).get("batches", 0)) + 1
        print(f"=== resume from {PARTIAL_OUT}: completed={start_batch - 1}, nodes={len(nodes)} ===")
    for bi, bpaths in enumerate(batches, 1):
        if bi < start_batch:
            continue
        batch = [parse_branch(p) for p in bpaths]
        bids = [b["branch_id"] for b in batch]
        print(f"\n=== batch {bi}: branches {bids} "
              f"(turns={sum(b['n_turns'] for b in batch)}, "
              f"T+={sum(len(b['success']) for b in batch)}, T-={sum(len(b['failed']) for b in batch)}) ===")
        user = render_user(batch, existing_titles)
        skills, usage = call_teacher(SYSTEM_PROMPT, user)
        btask = batch[0]["task"]
        batch_general_seen = False
        for s in skills:
            t = s["title"].strip()
            if any(t.lower() == e.lower() for e in existing_titles):
                continue
            labels, sbr = parse_evidence(s.get("evidence_turns"), bids)
            cat = normalize_category(s.get("category"), btask)
            scope = normalize_scope(s.get("scope"), cat)
            blob = " ".join([t, s.get("principle") or "", s.get("condition") or ""])
            can_be_general = bool(GENERAL_ALLOW_RE.search(blob)) and not GENERAL_DEMOTE_RE.search(blob)
            demoted_general = False
            promoted_general = False
            if cat != "general" and scope != "api_warning" and can_be_general and not batch_general_seen:
                cat = "general"
                scope = "universal_general"
                promoted_general = True
            if cat == "general" and (
                scope != "universal_general"
                or GENERAL_DEMOTE_RE.search(blob)
                or not can_be_general
            ):
                cat = btask
                scope = "api_warning" if GENERAL_DEMOTE_RE.search(blob) else "implementation_note"
                demoted_general = True
            if cat == "general":
                if batch_general_seen:
                    cat = btask
                    scope = "implementation_note"
                    demoted_general = True
                else:
                    scope = "universal_general"
                    batch_general_seen = True
            elif scope == "universal_general":
                scope = "task_specific"
            existing_titles.append(t)
            nodes.append({
                "id": f"sg_{len(nodes)+1:04d}",
                "title": t,
                "principle": (s.get("principle") or "").strip(),
                "condition": (s.get("condition") or "").strip(),
                "category": cat,
                "scope": scope,
                "evidence_turns": labels,
                "source_branches": [[batch[0]["run_ts"], b] for b in (sbr or [bids[0]])],
            })
            if demoted_general:
                nodes[-1]["distill_scope_action"] = "demoted_from_general_by_guardrail"
            if promoted_general:
                nodes[-1]["distill_scope_action"] = "promoted_to_general_by_guardrail"
            cat = nodes[-1]["category"]
            print(f"  [{cat:24s} {scope:19s}] {t}  ev={labels}")
        uk = f"in={usage.prompt_tokens} out={usage.completion_tokens}" if usage else "?"
        print(f"  -- {len(skills)} skills ({uk})")
        write_nodes(PARTIAL_OUT, nodes, batches, bi, complete=False)

    completed = len(batches)
    write_nodes(OUT / "raw_nodes.json", nodes, batches, completed, complete=True)
    if PARTIAL_OUT.exists():
        PARTIAL_OUT.unlink()
    g = sum(1 for n in nodes if n["category"] == "general")
    print(f"\n=== {len(nodes)} nodes -> {OUT/'raw_nodes.json'}  (general={g}, task-specific={len(nodes)-g}) ===")


if __name__ == "__main__":
    main()
