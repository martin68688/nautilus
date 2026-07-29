"""Dynamic methodology search agent.

Reads the task description, scans experience_kb category folders,
uses LLM to find relevant categories, then reads HIGH-confidence
references to build detailed methodology guidance.
"""
import logging
import hashlib
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
    cat_list = "\n".join(f"- {c}" for c in categories)
    user_msg = (
        f"Task description (first 1500 chars):\n{task_desc[:1500]}\n\n"
        f"Available research categories:\n{cat_list}\n\n"
        "Select up to 5 most relevant categories for this task. "
        "Output ONLY the selected category names, one per line, exactly as shown. No explanation."
    )
    try:
        if (cfg.model or "").lower().startswith("glm"):
            # GLM via the Anthropic-compatible endpoint (Zhipu Coding Plan).
            import anthropic
            client = anthropic.Anthropic(api_key=cfg.api_key, base_url=cfg.base_url or None, timeout=1200.0)
            resp = client.messages.create(
                model=cfg.model,
                temperature=0,
                max_tokens=256,
                system="You are a research category selector. Output only category names, one per line.",
                messages=[{"role": "user", "content": user_msg}],
            )
            response = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text") or ""
        else:
            from openai import OpenAI
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


def _read_high_confidence_candidates(
    cat_dir: Path,
    *,
    category_path: str | None = None,
) -> list[dict[str, str]]:
    """Return one immutable candidate per HIGH-confidence reference."""

    insight_file = cat_dir / "insight.md"
    if not insight_file.exists():
        return []
    insight_text = insight_file.read_text(encoding="utf-8")
    refs_dir = cat_dir / "references"
    output: list[dict[str, str]] = []
    in_table = False
    for line in insight_text.splitlines():
        line = line.strip()
        if line.startswith("| # |") or line.startswith("|---|"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            break
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) < 5 or cells[3].upper() != "HIGH":
            continue
        ref_name = cells[4].strip().rsplit("/", 1)[-1]
        ref_path = refs_dir / ref_name
        if not ref_path.exists() and refs_dir.exists():
            slug = re.sub(
                r"[^a-z0-9-]", "", cells[1].lower().replace(" ", "-")
            )[:30]
            matches = [
                path
                for path in refs_dir.glob("*.md")
                if slug[:15] in path.stem
            ]
            ref_path = matches[0] if matches else None
        if ref_path is None or not Path(ref_path).exists():
            continue
        try:
            text = _strip_ref_noise(Path(ref_path).read_text(encoding="utf-8"))
        except Exception:
            continue
        if not text:
            continue
        category = str(category_path or cat_dir.name)
        ref_id = f"{category}/{Path(ref_path).stem}"
        digest = hashlib.sha256(
            f"{ref_id}\n{text}".encode("utf-8")
        ).hexdigest()
        output.append(
            {
                "candidate_id": f"methodology::{digest[:24]}",
                "claim_id": f"methodology_claim::{digest[:24]}",
                "ref_id": ref_id,
                "category": category,
                "title": cells[1],
                "text": text,
                "content_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
            }
        )
    return output


def _read_high_confidence_references(cat_dir: Path) -> tuple[str, list[str]]:
    """Read insight.md, find HIGH-confidence rows, read their reference files.

    Returns (joined_text, ref_ids). ref_ids are side-channel ids ("{category}/{stem}")
    for adoption tracking only — NEVER injected into the prompt text.
    """
    candidates = _read_high_confidence_candidates(cat_dir)
    return (
        "\n\n---\n\n".join(item["text"] for item in candidates),
        [item["ref_id"] for item in candidates],
    )


def build_methodology_candidates(
    task_desc: str,
    methodology_kb_path: str,
    cfg: Any,
) -> list[dict[str, str]]:
    """Select categories, but return unrendered candidates for Authority."""

    kb_base = Path(methodology_kb_path)
    if not kb_base.exists():
        logger.info("[MethodologyAgent] methodology_kb_path not found, skipping")
        return []
    categories = _scan_categories(kb_base)
    if not categories:
        logger.info("[MethodologyAgent] No categories found")
        return []
    logger.info("[MethodologyAgent] Scanning %s categories...", len(categories))
    matched = _match_categories_with_llm(task_desc, categories, cfg)
    output: list[dict[str, str]] = []
    for category in matched:
        values = _read_high_confidence_candidates(
            kb_base / category,
            category_path=category,
        )
        output.extend(values)
        if values:
            logger.info("[MethodologyAgent] Proposed references from %s", category)
    return output


def build_methodology_guidance(task_desc: str, methodology_kb_path: str, cfg: Any) -> tuple[str, list[str]]:
    """Scan methodology_kb_path → LLM match → read HIGH-confidence references.

    Returns (guidance_text, ref_ids). guidance_text is byte-for-byte identical to before
    (goes into the prompt). ref_ids is a side-channel list for adoption tracking only.
    """
    candidates = build_methodology_candidates(
        task_desc, methodology_kb_path, cfg
    )
    by_category: dict[str, list[dict[str, str]]] = {}
    for candidate in candidates:
        by_category.setdefault(candidate["category"], []).append(candidate)
    all_sections = [
        f"### [{category}]\n\n"
        + "\n\n---\n\n".join(item["text"] for item in values)
        for category, values in by_category.items()
    ]
    all_ref_ids = [item["ref_id"] for item in candidates]

    if not all_sections:
        return "", []

    guidance_text = (
        "\n\n---\n## Methodology Insights from Literature\n"
        "The following detailed techniques from recent papers are relevant to this task:\n\n"
        + "\n\n---\n\n".join(all_sections)
    )
    return guidance_text, all_ref_ids
