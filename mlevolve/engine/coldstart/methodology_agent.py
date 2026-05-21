"""Dynamic methodology search agent.

Reads the task description, scans experience_kb category folders,
uses LLM to find relevant categories, then reads HIGH-confidence
references to build detailed methodology guidance.
"""
import logging
import re
from pathlib import Path
from typing import Any, List

logger = logging.getLogger("MLEvolve")


def _scan_categories(kb_base: Path) -> List[str]:
    """Return list of category path strings (relative to kb_base).

    Supports two layouts:
    - Flat: kb_base/category/  (e.g. experience_kb/small-data-transformer-finetuning/)
    - Nested: kb_base/venue-year/category/  (e.g. paperinsight/naacl-2024/efficient-training/)
    """
    categories = []
    for entry in sorted(kb_base.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        # Flat layout: entry itself contains insight.md
        if (entry / "insight.md").exists():
            categories.append(entry.name)
            continue
        # Nested layout: entry is a venue-year dir containing category subdirs
        for sub in sorted(entry.iterdir()):
            if not sub.is_dir() or sub.name.startswith("."):
                continue
            if (sub / "insight.md").exists():
                categories.append(f"{entry.name}/{sub.name}")
    return categories


def _match_categories_with_llm(task_desc: str, categories: List[str], cfg: Any) -> List[str]:
    """Ask LLM which categories are relevant. Returns up to 5 matches."""
    from openai import OpenAI

    cat_list = "\n".join(f"- {c}" for c in categories)
    user_msg = (
        f"Task description (first 1500 chars):\n{task_desc[:1500]}\n\n"
        f"Available research categories:\n{cat_list}\n\n"
        "Select up to 5 most relevant categories for this task. "
        "Output ONLY the selected category names, one per line, exactly as shown. No explanation."
    )
    try:
        client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url or None)
        resp = client.chat.completions.create(
            model=cfg.model,
            temperature=0,
            max_tokens=256,
            messages=[
                {"role": "system", "content": "You are a research category selector. Output only category names, one per line."},
                {"role": "user", "content": user_msg},
            ],
        )
        response = resp.choices[0].message.content or ""
        matched = []
        for line in response.strip().splitlines():
            line = line.strip().lstrip("- ")
            if line in categories:
                matched.append(line)
        logger.info(f"[MethodologyAgent] LLM matched {len(matched)} categories: {matched}")
        return matched[:5]
    except Exception as e:
        logger.warning(f"[MethodologyAgent] LLM matching failed: {e}")
        return []


def _strip_ref_noise(text: str) -> str:
    """Remove Papers & Evidence and Delta sections from reference content."""
    # Remove ## Papers & Evidence block
    text = re.sub(r"## Papers & Evidence.*?(?=\n## |\Z)", "", text, flags=re.DOTALL)
    # Remove **Delta**: lines
    text = re.sub(r"\*\*Delta\*\*:.*?\n", "", text)
    # Remove frontmatter (--- ... ---)
    text = re.sub(r"^---.*?---\s*", "", text, flags=re.DOTALL)
    return text.strip()


def _read_high_confidence_references(cat_dir: Path) -> str:
    """Read insight.md, find HIGH-confidence rows, read their reference files."""
    insight_file = cat_dir / "insight.md"
    if not insight_file.exists():
        return ""

    insight_text = insight_file.read_text(encoding="utf-8")
    refs_dir = cat_dir / "references"
    ref_contents = []

    in_table = False
    for line in insight_text.splitlines():
        line = line.strip()
        if line.startswith("| # |") or line.startswith("|---|"):
            in_table = True
            continue
        if in_table:
            if not line.startswith("|"):
                break
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if len(cells) < 5:
                continue
            confidence = cells[3].upper()
            if confidence != "HIGH":
                continue
            ref_hint = cells[4].strip()
            # Resolve file path
            ref_name = ref_hint.rsplit("/", 1)[-1]
            ref_path = refs_dir / ref_name
            if not ref_path.exists() and refs_dir.exists():
                # Fuzzy match by slug prefix
                slug = re.sub(r"[^a-z0-9-]", "", cells[1].lower().replace(" ", "-"))[:30]
                candidates = [p for p in refs_dir.glob("*.md") if slug[:15] in p.stem]
                ref_path = candidates[0] if candidates else None

            if ref_path and Path(ref_path).exists():
                try:
                    raw = Path(ref_path).read_text(encoding="utf-8")
                    ref_contents.append(_strip_ref_noise(raw))
                except Exception:
                    continue

    return "\n\n---\n\n".join(ref_contents)


def build_methodology_guidance(task_desc: str, methodology_kb_path: str, cfg: Any) -> str:
    """Scan methodology_kb_path → LLM match → read HIGH-confidence references → return guidance."""
    kb_base = Path(methodology_kb_path)
    if not kb_base.exists():
        logger.info("[MethodologyAgent] methodology_kb_path not found, skipping")
        return ""

    categories = _scan_categories(kb_base)
    if not categories:
        logger.info("[MethodologyAgent] No categories found")
        return ""

    logger.info(f"[MethodologyAgent] Scanning {len(categories)} categories...")
    matched = _match_categories_with_llm(task_desc, categories, cfg)
    if not matched:
        logger.info("[MethodologyAgent] No relevant categories matched")
        return ""

    all_sections = []
    for cat_path in matched:
        cat_dir = kb_base / cat_path
        content = _read_high_confidence_references(cat_dir)
        if content:
            all_sections.append(f"### [{cat_path}]\n\n{content}")
            logger.info(f"[MethodologyAgent] Added references from {cat_path}")

    if not all_sections:
        return ""

    return (
        "\n\n---\n## Methodology Insights from Literature\n"
        "The following detailed techniques from recent papers are relevant to this task:\n\n"
        + "\n\n---\n\n".join(all_sections)
    )
