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


def _build_idf(code_corpus: list) -> dict:
    """Background corpus = all node code in the run. Returns {token: idf}.

    A token present in every code (torch, model, Dataset, AdamW) gets IDF≈0 — no
    discriminative power, filtered out. A token in only a few codes
    (StratifiedShuffleSplit, XGBClassifier, label_smoothing) gets high IDF and is kept.
    This learns what is generic per-run, so no hand-maintained stoplist is needed.
    """
    import math
    N = max(len(code_corpus), 1)
    df: dict = {}
    for code in code_corpus:
        for tok in set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", code or "")):
            df[tok] = df.get(tok, 0) + 1
    return {t: math.log(N / (1 + d)) for t, d in df.items()}


def _code_reflects(code: str, memory_text: str, judge_fn: Optional[Callable] = None,
                   idf: Optional[dict] = None, min_hits: int = 2, idf_floor: float = 1.0) -> bool:
    """Does code reflect this memory entry? LLM judge if provided, else keyword match.

    Keyword match is tightened three ways vs naive `any(kw in code)`:
      (a) drop low-IDF generic tokens via `idf` (torch/model/Dataset → ~0 IDF),
      (b) word-boundary regex instead of substring ('model' won't match 'model_'),
      (c) require >= min_hits distinct high-IDF keywords (a single generic token
          can never count as adoption). min_hits is clamped to len(kws) so a memory
          with only one truly-specific keyword still can match on it.
    """
    if judge_fn is not None:
        try:
            return bool(judge_fn(code, memory_text))
        except Exception:
            return False
    kws = _extract_keywords(memory_text)
    if idf:
        kws = [k for k in kws if idf.get(k, float("inf")) >= idf_floor]
    if not kws:
        return False
    hits = [k for k in kws if re.search(r"\b" + re.escape(k) + r"\b", code or "")]
    return len(hits) >= min(min_hits, len(kws))


def _llm_judge(code: str, memory_text: str, cfg) -> bool:
    """Second-stage semantic confirmation: does code genuinely IMPLEMENT the memory's
    SPECIFIC technique, vs merely sharing generic ML boilerplate (torch/nn.Module/loop)?

    Called only on keyword hits (judge_mode=llm) to cut false positives cheaply. Uses
    DeepSeek (OpenAI-compatible) — cheap/fast and keeps the GLM quota free for the solver.
    Credentials come from os.environ DEEPSEEK_* (loaded from mlevolve/.env at run start);
    `cfg` is accepted only for signature uniformity / fallback. Failure is non-fatal.
    """
    import os
    api_key = os.environ.get("DEEPSEEK_API_KEY") or getattr(cfg, "api_key", "")
    base_url = os.environ.get("DEEPSEEK_BASE_URL") or getattr(cfg, "base_url", "") or None
    model = os.environ.get("DEEPSEEK_MODEL") or "deepseek-chat"
    user = (
        f"Memory entry (a distilled skill describing specific techniques/APIs/patterns):\n"
        f"```\n{memory_text[:2000]}\n```\n\n"
        f"Generated code (first 8000 chars):\n```\n{code[:8000]}\n```\n\n"
        "Does the code actually USE any specific technique/API/pattern the memory entry describes "
        "— meaning there is a real call or application of it (not merely an import, and not just "
        "generic ML boilerplate like torch/nn.Module/a training loop)? An unused import or a "
        "coincidental shared word is NOT adoption; but a partial or variant implementation of any "
        "ONE technique the memory lists DOES count. Output ONLY 'YES' or 'NO'."
    )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model, temperature=0, max_tokens=8,
            messages=[{"role": "system", "content": "Output only YES or NO."},
                      {"role": "user", "content": user}],
        )
        ans = resp.choices[0].message.content or ""
        return ans.strip().upper().startswith("YES")
    except Exception as e:
        logger.warning(f"[adoption_tracker] LLM judge failed: {e}")
        return False


_EMB_MODEL = None
_EMB_CACHE: dict = {}


def _get_emb_model(cfg):
    """Lazy-load the bge embedding model (reuses the project's EmbeddingModel). Cached.

    Only invoked in judge_mode='hybrid'. Falls back to cfg.memory_embedding_* if the
    adoption_tracking block doesn't carry its own device/path.
    """
    global _EMB_MODEL
    if _EMB_MODEL is None:
        from agents.memory.embedding_models import EmbeddingModel
        at = getattr(cfg, "adoption_tracking", None)
        dev = getattr(at, "embedding_device", None) or getattr(cfg, "memory_embedding_device", None) or "cpu"
        name = getattr(at, "embedding_model", None) or getattr(cfg, "memory_embedding_model_path", None) \
               or "BAAI/bge-base-en-v1.5"
        _EMB_MODEL = EmbeddingModel(model_type="local", model_name=name, device=dev)
    return _EMB_MODEL


def _embedding_sim(code: str, memory_text: str, emb_model) -> float:
    """Cosine similarity of code vs memory embeddings. bge caps ~512 tokens, so texts are
    truncated to 2000 chars. Each unique text embedded once (cached). Returns 0.0 if no model."""
    if emb_model is None:
        return 0.0
    import numpy as np

    def _vec(t: str):
        t = t[:2000]
        if t not in _EMB_CACHE:
            _EMB_CACHE[t] = emb_model.encode([t])[0]
        return _EMB_CACHE[t]

    a, b = _vec(code), _vec(memory_text)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def run_adoption_analysis(cfg, journal, judge_fn: Optional[Callable] = None) -> dict:
    """Analyze memory adoption across journal nodes. Writes report to cfg.log_dir."""
    nodes = journal.nodes if hasattr(journal, "nodes") else (journal if isinstance(journal, list) else [])
    methodology_kb_path = getattr(cfg, "methodology_kb_path", "") or ""
    workspace_dir = getattr(cfg, "workspace_dir", "")

    # IDF over the run's own code corpus: generic tokens (torch/model/Dataset) get ~0 IDF.
    idf = _build_idf([getattr(n, "code", "") or "" for n in nodes])
    mode = getattr(getattr(cfg, "adoption_tracking", None), "judge_mode", "keyword")
    use_llm = mode in ("llm", "hybrid")
    use_emb = mode == "hybrid"
    emb_model = _get_emb_model(cfg) if use_emb else None
    logger.info(f"[adoption_tracker] judge_mode={mode}"
                + (": LLM judges every pair (crosses lexical gap)" if mode == "llm"
                   else ": embedding+keyword screen, LLM arbitrates the uncertain middle" if use_emb
                   else ": IDF keyword only"))

    by_ref = {}
    for node in nodes:
        log = getattr(node, "adoption_log", None) or []
        code = getattr(node, "code", "") or ""
        for rec in log:
            rid = rec.get("ref_id")
            src = rec.get("source", "?")
            if rid not in by_ref:
                by_ref[rid] = {"ref_id": rid, "source": src, "injected_count": 0,
                               "keyword_hit_count": 0, "emb_signal_count": 0,
                               "llm_judged_count": 0, "adopted_count": 0, "node_ids": []}
            by_ref[rid]["injected_count"] += 1
            by_ref[rid]["node_ids"].append(node.id)
            mem_text = (_fetch_global_memory_text(rid, workspace_dir) if src == "global_memory"
                        else _fetch_methodology_text(rid, methodology_kb_path))
            if not mem_text:
                continue

            # Cheap signal (all modes): IDF keyword hit at min_hits=1 (a logged evidence signal).
            if _code_reflects(code, mem_text, judge_fn, idf=idf, min_hits=1):
                by_ref[rid]["keyword_hit_count"] += 1
            kw_strict = _code_reflects(code, mem_text, judge_fn, idf=idf, min_hits=2)

            # Adoption decision by mode.
            if mode == "keyword":
                adopted = kw_strict
            elif mode == "llm":
                # LLM judges EVERY pair — no keyword gate. The LLM crosses the lexical gap
                # (XGBoost↔XGBClassifier) that keyword cannot; keyword stays a logged signal only.
                adopted = _llm_judge(code, mem_text, cfg)
                by_ref[rid]["llm_judged_count"] += 1
            else:  # hybrid
                emb_sim = _embedding_sim(code, mem_text, emb_model)
                emb_norm = max(0.0, min(1.0, (emb_sim - 0.35) / 0.4))
                combined = 0.7 * emb_norm + 0.3 * (1.0 if kw_strict else 0.0)
                if combined >= 0.6:
                    adopted = True
                elif combined <= 0.25:
                    adopted = False
                else:
                    adopted = _llm_judge(code, mem_text, cfg)
                    by_ref[rid]["llm_judged_count"] += 1
                by_ref[rid]["emb_signal_count"] += 1

            if adopted:
                by_ref[rid]["adopted_count"] += 1

    per_memory = list(by_ref.values())
    for m in per_memory:
        m["adoption_rate"] = round(m["adopted_count"] / m["injected_count"], 3) if m["injected_count"] else 0.0
    per_memory.sort(key=lambda x: -x["adoption_rate"])

    total_inj = sum(m["injected_count"] for m in per_memory)
    total_kw = sum(m["keyword_hit_count"] for m in per_memory)
    total_emb = sum(m["emb_signal_count"] for m in per_memory)
    total_llm = sum(m["llm_judged_count"] for m in per_memory)
    total_adopt = sum(m["adopted_count"] for m in per_memory)
    by_src = {}
    for m in per_memory:
        d = by_src.setdefault(m["source"], {"injected": 0, "keyword_hit": 0,
                                            "emb_signal": 0, "llm_judged": 0, "adopted": 0})
        d["injected"] += m["injected_count"]
        d["keyword_hit"] += m["keyword_hit_count"]
        d["emb_signal"] += m["emb_signal_count"]
        d["llm_judged"] += m["llm_judged_count"]
        d["adopted"] += m["adopted_count"]

    report = {
        "summary": {
            "total_memories": len(per_memory),
            "total_injections": total_inj,
            "total_keyword_hits": total_kw,
            "total_emb_signals": total_emb,
            "total_llm_judged": total_llm,
            "total_adopted": total_adopt,
            "overall_adoption_rate": round(total_adopt / total_inj, 3) if total_inj else 0.0,
            "keyword_hit_rate": round(total_kw / total_inj, 3) if total_inj else 0.0,
            "judge_mode": mode,
            "by_source": {s: {"injected": v["injected"], "keyword_hit": v["keyword_hit"],
                              "emb_signal": v["emb_signal"], "llm_judged": v["llm_judged"],
                              "adopted": v["adopted"],
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
              "kw_hit = stage-1 IDF keyword hits; adopted = final (after LLM judge if judge_mode=llm)", "",
              "| ref_id | source | injected | kw_hit | adopted | rate |", "|---|---|---|---|---|---|"]
    for m in report["per_memory"]:
        lines.append(f"| {m['ref_id']} | {m['source']} | {m['injected_count']} | {m['keyword_hit_count']} | {m['adopted_count']} | {m['adoption_rate']:.1%} |")
    path.write_text("\n".join(lines), encoding="utf-8")
