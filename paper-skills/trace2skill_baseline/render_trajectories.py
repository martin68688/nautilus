#!/usr/bin/env python3
"""Render mlevolve solver traces into Trace2Skill markdown trajectory logs.

This is Stage-1 (trajectory collection) of the faithful Trace2Skill baseline port.
It reads `journal.json` from one or more mlevolve runs and emits one trajectory
log per executed SearchNode, in the exact format the Trace2Skill analysts
(`analysis/run_success_analysis_llm.py`, `analysis/run_error_analysis_llm.py`)
consume unchanged.

Mapping (mlevolve -> Trace2Skill):
  one SearchNode  ==  one trajectory (a single self-contained solution attempt)
  success node (is_buggy=False, metric set)  ->  <id>_SUCCEED.md   (success analyst A+)
  buggy   node (is_buggy=True)               ->  <id>_FAILED.md    (error   analyst A-)

Filename contract (parse_log_filename): the parser takes the LAST '_'-separated
token of the prefix as instance_id, so the unique id must contain NO underscores
(run timestamps like 20260509_185008 contain '_' and would corrupt parsing). We
therefore build the id from hyphen-joined sanitized parts.

Log body contract (strip_log_metadata): file must start with
`# Chat History ...\\n\\n**Timestamp**: ...\\n\\n---\\n\\n` and end with
`\\n---\\n\\n## RESULT\\n...`; the body in between is the trajectory the analyst sees.

IMPORTANT: read journal.json (preserves buggy nodes), NOT records.json (which
drops every buggy node via GlobalMemoryLayer._should_save_node).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# --------------------------------------------------------------------------- #
# Node selection
# --------------------------------------------------------------------------- #

def _metric_rank_key(node: dict):
    """Sort key so that 'best' nodes come first, respecting metric.maximize.

    Returns (rankable_value, has_metric). Nodes without a metric sort last.
    For maximize=True  -> higher value is better -> sort descending.
    For maximize=False -> lower  value is better -> sort ascending  (e.g. log loss).
    """
    m = node.get("metric") or {}
    v = m.get("value")
    if v is None:
        return (0, 0.0)
    maximize = bool(m.get("maximize", False))
    # negate when maximizing so that ascending sort puts best first uniformly
    signed = -float(v) if maximize else float(v)
    return (1, signed)


def _select_success(nodes: list[dict], k: int) -> list[dict]:
    """Top-k success nodes (best metric first), skipping root/no-metric/draft-noise."""
    succ = [
        n for n in nodes
        if not n.get("is_buggy")
        and (n.get("metric") or {}).get("value") is not None
        and n.get("stage") not in (None, "root")
    ]
    succ.sort(key=_metric_rank_key)  # best first
    return succ[:k]


def _dedup_buggy(nodes: list[dict]) -> list[dict]:
    """Keep diverse buggy nodes: dedup by (exc_type, first traceback line)."""
    seen = set()
    out = []
    for n in nodes:
        if not n.get("is_buggy"):
            continue
        exc_type = n.get("exc_type") or "UnknownError"
        term = "".join(n.get("_term_out") or [])
        first = next((ln.strip() for ln in term.splitlines() if ln.strip()
                      and "Traceback" not in ln and "File " not in ln), "")[:80]
        key = (exc_type, first)
        if key in seen:
            continue
        seen.add(key)
        out.append(n)
    return out


def _select_failed(nodes: list[dict], k: int) -> list[dict]:
    bugs = _dedup_buggy([n for n in nodes if n.get("is_buggy")])
    return bugs[:k]


# --------------------------------------------------------------------------- #
# Field rendering helpers
# --------------------------------------------------------------------------- #

def _parse_plan(plan: str | None) -> str:
    """Render a node's plan field. For improve/evolution it is a JSON string with
    keys reason/module/plan; for debug it is free-form root-cause prose. Either way
    return readable text."""
    if not plan:
        return "(no explicit plan recorded)"
    text = plan.strip()
    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except Exception:
            return text
        parts = []
        if obj.get("reason"):
            parts.append(f"**Reasoning:** {obj['reason'].strip()}")
        mods = obj.get("module") or obj.get("modules")
        if mods:
            parts.append(f"**Target components:** {', '.join(mods) if isinstance(mods, list) else mods}")
        pl = obj.get("plan")
        if isinstance(pl, dict) and pl:
            parts.append("**Per-component plan:**")
            for mod, desc in pl.items():
                parts.append(f"  - {mod}: {desc}")
        elif isinstance(pl, str) and pl.strip():
            parts.append(f"**Plan:** {pl.strip()}")
        return "\n".join(parts) if parts else text
    return text


def _truncate(text: str | None, n: int) -> str:
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= n else text[:n].rstrip() + " …[truncated]"


def _tail(text: str, n: int) -> str:
    """Last n chars (for tracebacks, the useful frames are at the end)."""
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= n else "…" + text[-n:].lstrip()


# --------------------------------------------------------------------------- #
# Trajectory rendering
# ----------------------------------------------------------------=========== #

TASK_BLURB = (
    "[Solver context] This is one attempt produced by an automated ML competition "
    "solver (a tree-search agent that writes and executes Python training pipelines). "
    "Task: Spooky Author Identification — predict which of three authors "
    "(EAP=Edgar Allan Poe, HPL=HP Lovecraft, MWS=Mary Shelley) wrote each horror-text "
    "excerpt. Metric: multi-class log loss on validation (LOWER is better). "
    "The agent iterates: each node is one full attempt (plan -> code -> train -> score)."
)


def render_node(node: dict, run_name: str, desc: str) -> str:
    stage = node.get("stage") or "?"
    buggy = bool(node.get("is_buggy"))
    metric = node.get("metric") or {}
    mval = metric.get("value")
    maximize = metric.get("maximize")
    plan_txt = _parse_plan(node.get("plan"))
    code_summary = node.get("code_summary") or ""
    analysis = node.get("analysis") or ""
    created = node.get("created_time") or ""

    # Header (must match strip_log_metadata's header regex)
    out = [
        f"# Chat History - mlevolve run {run_name} (stage={stage})",
        "",
        f"**Timestamp**: {created}",
        "",
        "---",
        "",
        "## User (Task)",
        desc.strip(),
        "",
        TASK_BLURB,
        "",
        "## Assistant (Thought — the approach this attempt planned)",
        plan_txt,
        "",
    ]

    # Action
    out.append("## Assistant (Action — strategy implemented in code)")
    if code_summary:
        out.append(_truncate(code_summary, 1600))
    else:
        out.append("(no code summary recorded; see plan above for the strategy)")
    out.append("")

    # Observation
    if buggy:
        term = "".join(node.get("_term_out") or [])
        exc_type = node.get("exc_type") or "Error"
        exc_msg = (node.get("exc_info") or {}).get("message", "") if isinstance(node.get("exc_info"), dict) else ""
        out.append("## Observation (execution FAILED)")
        if analysis:
            out.append(_truncate(analysis, 1000))
            out.append("")
        out.append(f"**Exception type:** {exc_type}")
        if exc_msg:
            out.append(f"**Exception message:** {exc_msg.strip()}")
        out.append("")
        out.append("**Traceback (tail):**")
        out.append("```")
        out.append(_tail(term, 2200))
        out.append("```")
        out.append("")
    else:
        direction = "higher is better" if maximize else "lower is better"
        out.append("## Observation (execution SUCCEEDED)")
        if analysis:
            out.append(_truncate(analysis, 1000))
            out.append("")
        if mval is not None:
            out.append(f"**Validation metric:** {mval}  ({direction})")
        out.append("")

    # Trailer (must match strip_log_metadata's trailer regex)
    if buggy:
        result = f"FAIL — {node.get('exc_type') or 'exception'} during execution; no valid metric produced."
    else:
        result = f"PASS — validation log loss = {mval} (lower is better)."
    out.append("---")
    out.append("")
    out.append("## RESULT")
    out.append(result)
    out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Filename / id
# --------------------------------------------------------------------------- #

def _sanitize(text: str) -> str:
    """Make a token filename/id-safe: collapse underscores (parser separator) and
    spaces to hyphens."""
    return re.sub(r"[\s_]+", "-", text).strip("-")


def make_filename(node: dict, run_name: str, outcome: str) -> tuple[str, str]:
    """Return (instance_id, filename). instance_id has NO underscores."""
    stage = node.get("stage") or "node"
    step = node.get("step", 0)
    id8 = (node.get("id") or "xxxxxxxx")[:8]
    inst = f"{_sanitize(run_name)}-{stage}-{step}-{id8}"
    fname = f"mlevolve-{inst}_{outcome}.md"
    return inst, fname


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def load_journal(run_dir: Path) -> dict:
    jp = run_dir / "logs" / "journal.json"
    if not jp.is_file():
        raise FileNotFoundError(f"no logs/journal.json under {run_dir}")
    with open(jp, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True,
                    help="mlevolve run dirs (each must contain logs/journal.json)")
    ap.add_argument("--out-dir", required=True, help="output dir for *_SUCCEED/FAILED.md logs")
    ap.add_argument("--desc-file", default=None,
                    help="task description markdown (default: spooky description under mlevolve/data)")
    ap.add_argument("--max-success-per-run", type=int, default=8)
    ap.add_argument("--max-failed-per-run", type=int, default=8)
    ap.add_argument("--min-signal-len", type=int, default=40,
                    help="skip nodes whose plan+analysis+code_summary are all shorter than this")
    args = ap.parse_args()

    # default description
    if args.desc_file:
        desc = Path(args.desc_file).read_text(encoding="utf-8", errors="replace")
    else:
        default_desc = Path(__file__).resolve().parents[2] / \
            "mlevolve/data/spooky-author-identification/prepared/public/description.md"
        desc = default_desc.read_text(encoding="utf-8", errors="replace") if default_desc.is_file() \
            else "(Spooky Author Identification: 3-author horror-text classification, log loss.)"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_succ = total_fail = 0
    for run in args.runs:
        run_dir = Path(run)
        run_name = run_dir.name
        try:
            journal = load_journal(run_dir)
        except FileNotFoundError as e:
            print(f"[skip] {e}")
            continue
        nodes = journal.get("nodes", [])

        def has_signal(n):
            sig = (n.get("plan") or "") + (n.get("analysis") or "") + (n.get("code_summary") or "")
            return len(sig.strip()) >= args.min_signal_len

        succ = [n for n in _select_success(nodes, args.max_success_per_run) if has_signal(n)]
        fail = [n for n in _select_failed(nodes, args.max_failed_per_run) if has_signal(n)]

        pairs = [(n, "SUCCEED") for n in succ] + [(n, "FAILED") for n in fail]
        for node, outcome in pairs:
            inst, fname = make_filename(node, run_name, outcome)
            text = render_node(node, run_name, desc)
            (out_dir / fname).write_text(text, encoding="utf-8")
            if outcome == "SUCCEED":
                total_succ += 1
            else:
                total_fail += 1

        print(f"[{run_name}] nodes={len(nodes)} -> rendered {len(succ)} success + {len(fail)} failed")

    print(f"\nTotal: {total_succ} SUCCEED + {total_fail} FAILED logs -> {out_dir}")


if __name__ == "__main__":
    main()
