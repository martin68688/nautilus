"""Adoption tracker: post-run analysis of memory entry adoption.

Reads each journal node's adoption_log (which memory ids were injected into that node's
prompt), re-fetches each memory entry's original text, checks whether the node's generated
code reflects that entry (keyword match by default; LLM judge interface pluggable via
judge_fn), and writes adoption_report.json + .md to log_dir.

What this measures: "injection → reflection correlation" — of the memory entries each node
saw, which ones show up in the generated code. This is more rigorous than blind grep
(because we know exactly what was injected per-node) but is NOT causal attribution on its
own: the LLM may have known a technique from pretraining. For causal attribution (memory vs
pretrained), run an A/B control (memory on/off) and diff the two reports.
"""
import json
import logging
import re
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger("MLEvolve")


def _fetch_methodology_text(ref_id: str, methodology_kb_path: str) -> str:
    """Re-read original text of a methodology reference by ref_id.

    ref_id is "{category}/{stem}" (dynamic) or "static:{category}/{stem}" (static).
    """
    if not methodology_kb_path:
        return ""
    kb = Path(methodology_kb_path)
    rid = ref_id[len("static:"):] if ref_id.startswith("static:") else ref_id
    cat, _, stem = rid.partition("/")
    for cand in (kb / cat / "references" / f"{stem}.md", kb / cat / f"{stem}_methodology.md"):
        if cand.exists():
            try:
                return cand.read_text(encoding="utf-8")
            except Exception:
                return ""
    return ""


def _fetch_global_memory_text(record_id: str, workspace_dir) -> str:
    """Re-read a global_memory record (description + method) by record_id."""
    rf = Path(workspace_dir) / "global_memory" / "records.json"
    if not rf.exists():
        return ""
    try:
        recs = json.loads(rf.read_text(encoding="utf-8"))
    except Exception:
        return ""
    recs_list = recs if isinstance(recs, list) else recs.get("records", recs.get("memories", []))
    for r in recs_list:
        if isinstance(r, dict) and r.get("record_id") == record_id:
            return f"{r.get('description','')}\n{r.get('method','')}"
    return ""


_STOP = {"the", "this", "that", "when", "should", "must", "never", "always",
         "first", "second", "model", "validation", "training", "methodology",
         "memory", "using", "value", "values", "false", "true", "none"}


def _extract_keywords(text: str) -> list:
    """Heuristic: pull code-ish tokens (quoted names, CamelCase, snake_case) from memory text."""
    kws = set()
    for m in re.findall(r'"([^"]+)"', text):
        if len(m) > 3 and any(c.isalnum() for c in m):
            kws.add(m)
    for m in re.findall(r'\b([A-Z][a-zA-Z]{4,})\b', text):
        kws.add(m)
    for m in re.findall(r'\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b', text):
        if len(m) > 5:
            kws.add(m)
    return [k for k in kws if k.lower() not in _STOP]


def _code_reflects(code: str, memory_text: str, judge_fn: Optional[Callable] = None) -> bool:
    """Does code reflect this memory entry? LLM judge if provided, else keyword match."""
    if judge_fn is not None:
        try:
            return bool(judge_fn(code, memory_text))
        except Exception:
            return False
    kws = _extract_keywords(memory_text)
    if not kws:
        return False
    return any(kw in code for kw in kws)


def run_adoption_analysis(cfg, journal, judge_fn: Optional[Callable] = None) -> dict:
    """Analyze memory adoption across journal nodes. Writes report to cfg.log_dir."""
    nodes = journal.nodes if hasattr(journal, "nodes") else (journal if isinstance(journal, list) else [])
    methodology_kb_path = getattr(cfg, "methodology_kb_path", "") or ""
    workspace_dir = getattr(cfg, "workspace_dir", "")

    by_ref = {}
    for node in nodes:
        log = getattr(node, "adoption_log", None) or []
        code = getattr(node, "code", "") or ""
        for rec in log:
            rid = rec.get("ref_id")
            src = rec.get("source", "?")
            if rid not in by_ref:
                by_ref[rid] = {"ref_id": rid, "source": src, "injected_count": 0,
                               "adopted_count": 0, "node_ids": []}
            by_ref[rid]["injected_count"] += 1
            by_ref[rid]["node_ids"].append(node.id)
            if src == "global_memory":
                mem_text = _fetch_global_memory_text(rid, workspace_dir)
            else:
                mem_text = _fetch_methodology_text(rid, methodology_kb_path)
            if mem_text and _code_reflects(code, mem_text, judge_fn):
                by_ref[rid]["adopted_count"] += 1

    per_memory = list(by_ref.values())
    for m in per_memory:
        m["adoption_rate"] = round(m["adopted_count"] / m["injected_count"], 3) if m["injected_count"] else 0.0
    per_memory.sort(key=lambda x: -x["adoption_rate"])

    total_inj = sum(m["injected_count"] for m in per_memory)
    total_adopt = sum(m["adopted_count"] for m in per_memory)
    by_src = {}
    for m in per_memory:
        d = by_src.setdefault(m["source"], {"injected": 0, "adopted": 0})
        d["injected"] += m["injected_count"]
        d["adopted"] += m["adopted_count"]

    report = {
        "summary": {
            "total_memories": len(per_memory),
            "total_injections": total_inj,
            "total_adopted": total_adopt,
            "overall_adoption_rate": round(total_adopt / total_inj, 3) if total_inj else 0.0,
            "by_source": {s: {"injected": v["injected"], "adopted": v["adopted"],
                              "rate": round(v["adopted"] / v["injected"], 3) if v["injected"] else 0.0}
                          for s, v in by_src.items()},
        },
        "per_memory": per_memory,
    }

    log_dir = Path(getattr(cfg, "log_dir", "."))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "adoption_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_md_report(log_dir / "adoption_report.md", report)
    logger.info(f"[adoption_tracker] report → {log_dir / 'adoption_report.json'} | "
                f"overall={report['summary']['overall_adoption_rate']}")
    return report


def _write_md_report(path: Path, report: dict) -> None:
    s = report["summary"]
    lines = ["# Memory Adoption Report", "",
             f"- total memories: {s['total_memories']}",
             f"- overall adoption: {s['total_adopted']}/{s['total_injections']} = **{s['overall_adoption_rate']:.1%}**",
             "", "## by source"]
    for src, v in s["by_source"].items():
        lines.append(f"- {src}: {v['adopted']}/{v['injected']} = {v['rate']:.1%}")
    lines += ["", "## per memory (sorted by adoption rate)", "",
              "| ref_id | source | injected | adopted | rate |", "|---|---|---|---|---|"]
    for m in report["per_memory"]:
        lines.append(f"| {m['ref_id']} | {m['source']} | {m['injected_count']} | {m['adopted_count']} | {m['adoption_rate']:.1%} |")
    path.write_text("\n".join(lines), encoding="utf-8")
