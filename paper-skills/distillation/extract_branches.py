"""
extract_branches.py  —  Skill Graph P0: 把 clean run 的搜索树切成 branch (= 一个 draft 分叉 + 子孙),
每个 branch 渲染成一条 trajectory (.md),作为后续逐-branch Trace2Skill 蒸馏的输入。

一个 branch = mlevolve journal.json 里 branch_id 相同的所有节点(按 step 排序)。
渲染格式: header + 每节点一轮(Thought=plan / Action=code_summary / Observation=analysis+metric /
          Failure=_term_out|exc_info) + RESULT(best metric + 成功/buggy 计数)。

用法: python extract_branches.py
输出: paper-skills/distillation/traces/<run_ts>/branch_<id>.md
"""
import json, glob, os, re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "mlevolve" / "runs"
OUT  = REPO / "paper-skills" / "distillation" / "traces"

# 17 verifiably-clean runs (leakage-run-boundary allowlist) — only distill from these
CLEAN = ["20260509_154039","20260509_185008","20260510_025317","20260510_095558","20260510_162636",
         "20260511_014836","20260511_102550","20260513_165253","20260514_023457","20260514_052334",
         "20260515_173948","20260516_104127","20260516_125444","20260517_132158","20260517_151325",
         "20260509_042918",
         "20260627_135133"]  # A100 baseline mlevolve-spooky (elite-peach-mayfly); INDEX_BUG-clean, best 0.1007


def load_nodes(jf: Path):
    J = json.load(open(jf))
    if isinstance(J, dict) and "nodes" in J:
        return J["nodes"]
    if isinstance(J, list):
        return J
    return list(J.values())


def metric_val(n):
    mv = n.get("metric")
    return (mv.get("value") if isinstance(mv, dict) else mv)


def _trunc(s, n=600):
    if s is None:
        return "(none)"
    s = str(s).strip()
    return s if len(s) <= n else s[:n] + " …"


def render_branch(run_ts: str, branch_id, nodes) -> str:
    nodes = sorted(nodes, key=lambda n: n.get("step", 0))
    best = min((metric_val(n) for n in nodes if metric_val(n) is not None and not n.get("is_buggy")),
               default=None)
    n_succ = sum(1 for n in nodes if not n.get("is_buggy") and metric_val(n) is not None)
    n_bug = sum(1 for n in nodes if n.get("is_buggy"))
    out = [f"# Chat History",
           f"",
           f"**Task**: spooky-author-identification   **Run**: {run_ts}   **Branch**: {branch_id}",
           f"**Turns**: {len(nodes)}   **Success**: {n_succ}   **Buggy**: {n_bug}   **Best metric (val log_loss)**: {best}",
           f"",
           f"---",
           f""]
    for i, n in enumerate(nodes, 1):
        mv = metric_val(n)
        out.append(f"## Turn {i}  [stage={n.get('stage')}  buggy={n.get('is_buggy')}  metric={mv}]")
        out.append(f"**Thought (plan)**: {_trunc(n.get('plan'), 500)}")
        out.append(f"**Action (code_summary)**: {_trunc(n.get('code_summary'), 500)}")
        out.append(f"**Observation (analysis)**: {_trunc(n.get('analysis'), 500)}")
        if n.get("is_buggy"):
            term = ""
            to = n.get("_term_out")
            if to:
                term = to[0] if isinstance(to, list) and to else str(to)
            fail = term or str(n.get("exc_info") or "")
            out.append(f"**Failure** (exc_type={n.get('exc_type')}): {_trunc(fail, 400)}")
        out.append("")
    out.append("---")
    out.append("## RESULT")
    out.append(f"Best validation log_loss in this branch: {best}")
    out.append(f"Success nodes: {n_succ} / Buggy nodes: {n_bug}")
    return "\n".join(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    summary = []
    for run in CLEAN:
        jfs = glob.glob(str(RUNS / f"{run}_*" / "logs" / "journal.json"))
        if not jfs:
            print(f"[skip] {run}: no journal.json")
            continue
        nodes = load_nodes(Path(jfs[0]))
        by_branch = {}
        for n in nodes:
            bid = n.get("branch_id")
            if bid is None:
                continue
            by_branch.setdefault(bid, []).append(n)
        run_dir = OUT / run
        run_dir.mkdir(exist_ok=True)
        for bid, bnodes in sorted(by_branch.items()):
            md = render_branch(run, bid, bnodes)
            (run_dir / f"branch_{bid}.md").write_text(md)
            total += 1
        summary.append((run, len(by_branch)))
    print(f"Extracted {total} branch-trajectories → {OUT}")
    for run, nb in summary:
        print(f"  {run}: {nb} branches")
    print(f"\n总计 {total} 个 branch (= {total} 条 trace, 后续逐个蒸馏成 skill)")


if __name__ == "__main__":
    main()
