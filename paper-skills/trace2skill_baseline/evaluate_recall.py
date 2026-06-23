#!/usr/bin/env python3
"""Evaluate the evolved skill's recall against the 15 ground-truth SOPs.

Gate (from engineering_roadmap §3): recall >= 10/15 against
  paper-skills/experience_kb/small-data-transformer-finetuning/insight.md
which holds 15 SOPs (10 HIGH + 5 MEDIUM) as the `| # | Insight | Evidence | Confidence | File |` rows.

Method: an LLM judge (glm-5.2 by default, temperature=0) sees the full evolved skill
(SKILL.md + references/*.md) and, for each ground-truth SOP, decides whether the
skill contains guidance that semantically covers it, quoting the matching passage.
This is intentionally a per-SOP binary coverage check (evidence-grounded) rather
than a single holistic score.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
from dotenv import load_dotenv


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

# Load GLM creds: mlevolve/.env (gitignored) holds GLM_*; paper-skills/.env for the rest.
load_dotenv(HERE.parent / ".env")
load_dotenv(HERE.parents[1] / "mlevolve" / ".env")
GT_INSIGHT = REPO / "paper-skills/experience_kb/small-data-transformer-finetuning/insight.md"


def parse_gt_sops(path: Path) -> list[dict]:
    """Parse the 15 SOP rows from insight.md. Row shape: | # | Insight | Evidence | Confidence | File |"""
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < 5:
            continue
        if not re.fullmatch(r"#|\d+", cells[0]):
            continue
        if cells[0] == "#":
            continue
        try:
            num = int(cells[0])
        except ValueError:
            continue
        rows.append({
            "num": num,
            "insight": cells[1],
            "evidence": cells[2],
            "confidence": cells[3].upper(),
            "file": cells[4],
        })
    return rows


def load_skill(skill_dir: Path) -> str:
    parts = []
    sm = skill_dir / "SKILL.md"
    if sm.is_file():
        parts.append(f"===== SKILL.md =====\n{sm.read_text(encoding='utf-8', errors='replace')}")
    refs = sorted((skill_dir / "references").glob("*.md")) if (skill_dir / "references").is_dir() else []
    for r in refs:
        parts.append(f"===== references/{r.name} =====\n{r.read_text(encoding='utf-8', errors='replace')}")
    return "\n\n".join(parts) if parts else "(empty skill)"


JUDGE_SYSTEM = (
    "You are a strict but fair evaluator judging whether a distilled ML skill document "
    "covers a specific procedural insight. Decide coverage by SEMANTIC equivalence of the "
    "actionable guidance, not exact wording. Be evidence-grounded: if you say covered, quote "
    "the matching passage from the skill. Respond with STRICT JSON only."
)

JUDGE_USER_TMPL = """Here is the auto-distilled skill document:

<skill>
{skill}
</skill>

Ground-truth SOP #{num} (confidence={conf}):
  Insight:   {insight}
  Evidence:  {evidence}

Question: Does the skill document contain procedural guidance that COVERS this SOP — i.e. an
agent reading the skill would be steered toward the same technique / avoid the same failure?

Respond with EXACTLY this JSON (no prose, no markdown fences):
{{"covered": true|false,
  "match_strength": "full"|"partial"|"none",
  "matched_passage": "<verbatim quote from the skill, or empty>",
  "reasoning": "<one sentence>"}}"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skill-dir", required=True, help="evolved skill dir (SKILL.md + references/)")
    ap.add_argument("--gt", default=str(GT_INSIGHT), help="ground-truth insight.md")
    ap.add_argument("--model", default=os.getenv("GLM_MODEL") or os.getenv("DEEPSEEK_MODEL") or os.getenv("OPENAI_MODEL") or "glm-5.2")
    ap.add_argument("--base-url", default=os.getenv("GLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://open.bigmodel.cn/api/anthropic")
    ap.add_argument("--api-key", default=os.getenv("GLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY"))
    ap.add_argument("--out", default=None, help="write JSON report here")
    ap.add_argument("--max-workers", type=int, default=6)
    args = ap.parse_args()

    if not args.api_key:
        sys.exit("api key required: set GLM_API_KEY / OPENAI_API_KEY / DEEPSEEK_API_KEY")

    gts = parse_gt_sops(Path(args.gt))
    if len(gts) != 15:
        print(f"[warn] parsed {len(gts)} GT SOPs (expected 15) from {args.gt}", file=sys.stderr)
    skill = load_skill(Path(args.skill_dir))
    print(f"GT SOPs: {len(gts)} | skill chars: {len(skill)} | model: {args.model}\n")

    client = anthropic.Anthropic(api_key=args.api_key, base_url=args.base_url, timeout=1200.0)

    def judge(gt):
        msg = JUDGE_USER_TMPL.format(skill=skill, num=gt["num"], conf=gt["confidence"],
                                     insight=gt["insight"], evidence=gt["evidence"])
        # Retry through GLM overload (529) — the SDK's 2 internal retries aren't
        # enough during heavy load. Backoff 10/20/30/40/50s, up to 6 attempts.
        resp = None
        for attempt in range(6):
            try:
                resp = client.messages.create(
                    model=args.model,
                    system=JUDGE_SYSTEM,
                    messages=[{"role": "user", "content": msg}],
                    temperature=0.0,
                    max_tokens=300,
                )
                break
            except Exception as e:
                if attempt == 5:
                    raise
                wait = 10 * (attempt + 1)
                print(f"  [judge #{gt['num']}] {type(e).__name__}; retry {attempt + 1}/5 after {wait}s",
                      file=sys.stderr, flush=True)
                time.sleep(wait)
        raw = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
        try:
            obj = json.loads(raw)
        except Exception:
            obj = {"covered": False, "match_strength": "none", "matched_passage": "",
                   "reasoning": f"PARSE_FAIL: {raw[:200]}"}
        return {**gt, **obj}

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(judge, gt): gt for gt in gts}
        for f in as_completed(futs):
            r = f.result()
            tag = "COVERED " if r.get("covered") else "MISSING "
            print(f"  #{r['num']:>2} [{r['confidence']:>6}] {tag} {r['insight'][:70]}")
            results.append(r)

    results.sort(key=lambda r: r["num"])
    covered = [r for r in results if r.get("covered")]
    full = [r for r in covered if r.get("match_strength") == "full"]
    high = [r for r in results if r["confidence"] == "HIGH"]
    high_cov = [r for r in covered if r["confidence"] == "HIGH"]

    print("\n" + "=" * 60)
    print(f"RECALL: {len(covered)}/{len(results)}  "
          f"(full-strength: {len(full)}/{len(results)})")
    print(f"HIGH recall: {len(high_cov)}/{len(high)}")
    print(f"GATE (>=10/15): {'PASS ✅' if len(covered) >= 10 else 'FAIL ❌'}")
    print("=" * 60)
    missing = [r for r in results if not r.get("covered")]
    if missing:
        print("\nMissing SOPs:")
        for r in missing:
            print(f"  #{r['num']} [{r['confidence']}] {r['insight']}")

    report = {"recall": len(covered), "total": len(results),
              "full_strength": len(full), "high_recall": f"{len(high_cov)}/{len(high)}",
              "gate_pass": len(covered) >= 10, "results": results}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
