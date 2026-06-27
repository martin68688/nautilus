"""
wrap_skill_to_kb.py — 把 Trace2Skill 蒸馏出的 evolved_skill/ 包成 experience_kb 能读的格式。

coldstart 的 _read_high_confidence_references 读 experience_kb/<cat>/insight.md 表里的 HIGH 行,
然后加载对应的 references/<file>。所以这里:
  1. 把 evolved_skill/references/*.md 全拷到 <dst>/references/
  2. 把 evolved_skill/SKILL.md 也拷成 references/skill-overview.md (主内容)
  3. 生成 insight.md 表:每个 reference 文件一行 HIGH,File 列指向它

用法: python wrap_skill_to_kb.py <evolved_skill_dir> <dst_experience_kb_category_dir>
"""
import sys, shutil
from pathlib import Path


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    refs_dst = dst / "references"
    refs_dst.mkdir(exist_ok=True)

    copied = []
    # SKILL.md 主内容先放进去(作为 skill-overview.md, 最先加载)
    skill = src / "SKILL.md"
    if skill.exists():
        shutil.copy(skill, refs_dst / "skill-overview.md")
        copied.append("skill-overview.md")
    # 再拷所有 references
    refs_src = src / "references"
    if refs_src.exists():
        for r in sorted(refs_src.glob("*.md")):
            if r.name == ".gitkeep":
                continue
            shutil.copy(r, refs_dst / r.name)
            copied.append(r.name)

    # 生成 insight.md 表
    rows = []
    for i, name in enumerate(copied, 1):
        title = name.replace(".md", "").replace("-", " ")
        rows.append(f"| {i} | {title} | Distilled from 16 clean runs via Trace2Skill | HIGH | {name} |")

    insight = (
        "# Trace2Skill-distilled skill (auto-generated index)\n\n"
        "Distilled from 16 clean spooky-author-identification runs via vanilla Trace2Skill.\n"
        "Each HIGH row loads its reference file into the agent's coldstart guidance.\n\n"
        "| # | Insight | Evidence | Confidence | File |\n"
        "|---|---------|----------|------------|------|\n"
        + "\n".join(rows) + "\n"
    )
    (dst / "insight.md").write_text(insight, encoding="utf-8")
    # 顶层也留一份 SKILL.md 供人查看
    if skill.exists():
        shutil.copy(skill, dst / "SKILL.md")
    print(f"[wrap] {dst}/insight.md : {len(rows)} HIGH rows | {len(copied)} reference files")


if __name__ == "__main__":
    main()
