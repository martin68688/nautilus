"""iMet three-round outer Host micro-F1 curve.

Same method as outputs/leaf_three_rounds_r2_replaced_20260920/plot_curves_replaced.py,
adapted for iMet:
  * metric is host micro-F1 over 3474 multilabel attributes -> HIGHER is better,
    so the staircase is a cumulative MAX (the Leaf figure used a cumulative MIN
    because its metric was log-loss).
  * Host-certified nodes are sparse (1/2/1 across the three rounds), so the
    internal tune nodes are added as a second layer, as explicitly allowed.

Read-only with respect to the iMet run directories; nothing here writes into them.
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, math, glob, os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
IMET = ROOT / 'experiments' / 'autoresearch_imet_plant_20260920_v001' / 'imet'

RUNS = [
    (1, 'R1（无记忆）', IMET / 'runs' / 'imet-r1-20260920-a1'),
    (2, 'R2（仅 R1 记忆）', IMET / 'runs' / 'imet-r2-20260920-b1'),
    (3, 'R3（累计记忆）', IMET / 'remote-recovery' / 'r3-scored-artifacts' / 'imet-r3-20260920-a1'),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_meta(attempt_dir: Path):
    """Candidate artifact metadata. R1/R2 name it metadata.json, R3 names it meta.json."""
    for name in ('metadata.json', 'meta.json'):
        p = attempt_dir / 'worker' / 'artifacts' / name
        if p.exists():
            return json.loads(p.read_text()), name
    return None, None


# ---------------------------------------------------------------- data contract
# One row per registered outer candidate. A node carries a Host score only when
# the harness recomputed it Host-side AND the artifact reload verified; otherwise
# the Host score stays empty and the candidate is drawn as a cross. The internal
# tune score is the candidate's own calibrated tune-split micro-F1, which exists
# for some candidates whose Host scoring stage failed.
rows = []
round_meta = {}
for round_id, label, run_dir in RUNS:
    launch = json.loads((run_dir / 'LAUNCH.json').read_text())
    outcome = json.loads((run_dir / 'RUN_OUTCOME.json').read_text())
    state = json.loads((run_dir / 'STATE.json').read_text())
    round_meta[round_id] = dict(label=label, run_dir=run_dir, launch=launch, outcome=outcome)

    for idx, h in enumerate(state['history']):
        attempt_dir = run_dir / f'attempt-{idx:03d}'
        start_s = float(h['started_elapsed'])
        end_s = float(h['elapsed_seconds'])

        certified = (
            h.get('status') == 'valid'
            and bool(h.get('host_recomputed'))
            and bool(h.get('reload_verified'))
            and isinstance(h.get('search_score'), (int, float))
            and math.isfinite(float(h.get('search_score') or math.nan))
        )

        # Internal tune score + the wall-clock moment it became known (end of the
        # train stage). Stage timings come from execution.json; the leading gap
        # between attempt start and the first stage is the agent's LLM planning.
        meta, meta_name = read_meta(attempt_dir)
        tune = h.get('tune_score')
        if tune is None and meta is not None:
            tune = meta.get('tune_micro_f1')
        tune = float(tune) if isinstance(tune, (int, float)) else None

        tune_min = None
        exec_path = attempt_dir / 'execution.json'
        if tune is not None and exec_path.exists():
            stages = json.loads(exec_path.read_text()).get('stages') or []
            train_s = next((float(s['elapsed_seconds']) for s in stages if s['mode'] == 'train'), None)
            if train_s is not None:
                total_stage = sum(float(s['elapsed_seconds']) for s in stages)
                overhead = max(0.0, (end_s - start_s) - total_stage)
                tune_min = (start_s + overhead + train_s) / 60.0

        rows.append({
            'round': round_id,
            'label': label,
            'attempt': idx,
            'node_id': h['attempt'],
            'start_minutes': start_s / 60.0,
            'elapsed_minutes': end_s / 60.0,
            'certified': certified,
            'host_micro_f1': float(h['search_score']) if certified else None,
            'tune_micro_f1': tune,
            'tune_minutes': tune_min,
            'status': h.get('status'),
            'decision': h.get('decision'),
            'reason': h.get('reason'),
            'meta_file': meta_name,
            'best_so_far': None,
        })

for round_id, *_ in RUNS:
    best = -math.inf
    for row in sorted((x for x in rows if x['round'] == round_id), key=lambda x: x['elapsed_minutes']):
        if row['certified']:
            best = max(best, row['host_micro_f1'])
        row['best_so_far'] = best if math.isfinite(best) else None

fieldnames = ['round', 'label', 'attempt', 'node_id', 'start_minutes', 'elapsed_minutes',
              'certified', 'host_micro_f1', 'tune_micro_f1', 'tune_minutes',
              'best_so_far', 'status', 'decision', 'reason', 'meta_file']
with (OUT / 'source_data_imet.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in sorted(rows, key=lambda x: (x['round'], x['elapsed_minutes'])):
        writer.writerow({k: row.get(k) for k in fieldnames})

summary = []
for round_id, label, run_dir in RUNS:
    rr = [x for x in rows if x['round'] == round_id]
    good = [x for x in rr if x['certified']]
    tuned = [x for x in rr if x['tune_micro_f1'] is not None]
    outcome = round_meta[round_id]['outcome']
    summary.append({
        'round': round_id,
        'label': label,
        'elapsed_minutes': outcome['elapsed_seconds'] / 60.0,
        'registered': len(rr),
        'certified': len(good),
        'tune_nodes': len(tuned),
        'best_host_micro_f1': max(x['host_micro_f1'] for x in good),
        'best_tune_micro_f1': max(x['tune_micro_f1'] for x in tuned),
        'best_host_at_minutes': min(x['elapsed_minutes'] for x in good
                                    if x['host_micro_f1'] == max(y['host_micro_f1'] for y in good)),
    })

# Guardrails: these are the values the run receipts assert. If a source directory
# is ever swapped, the figure fails loudly instead of drawing the wrong numbers.
assert [s['registered'] for s in summary] == [3, 3, 2], [s['registered'] for s in summary]
assert [s['certified'] for s in summary] == [1, 2, 1], [s['certified'] for s in summary]
for s, (_, _, run_dir) in zip(summary, RUNS):
    declared = json.loads((run_dir / 'RUN_OUTCOME.json').read_text())['best_search']
    assert abs(s['best_host_micro_f1'] - declared) < 1e-12, (s['round'], declared)

source_hashes = {}
for round_id, _, run_dir in RUNS:
    for name in ('LAUNCH.json', 'RUN_OUTCOME.json', 'STATE.json'):
        source_hashes[f'R{round_id}/{name}'] = sha256(run_dir / name)

launch0 = round_meta[1]['launch']
(OUT / 'summary_imet.json').write_text(json.dumps({
    'figure': 'iMet three-round outer Host micro-F1',
    'task_id': 'imet-2020-fgvc7',
    'metric': 'host micro-F1 over 3474 multilabel attributes (higher is better)',
    'model': launch0['model'],
    'effort': launch0['effort'],
    'budget_seconds_per_round': launch0['budget_seconds'],
    'data_manifest_sha256': launch0['data_manifest_sha256'],
    'runtime_manifest_sha256': launch0['runtime_manifest_sha256'],
    'rounds': summary,
    'source_hashes': source_hashes,
}, indent=2, ensure_ascii=False) + '\n')

# ---------------------------------------------------------------------- figure
font_path = Path('/System/Library/Fonts/STHeiti Light.ttc')
if font_path.exists():
    font_manager.fontManager.addfont(font_path)
    cjk_name = font_manager.FontProperties(fname=font_path).get_name()
else:
    cjk_name = 'DejaVu Sans'
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': [cjk_name, 'Arial', 'DejaVu Sans'],
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#B7C5D0', 'axes.labelcolor': '#304454',
    'text.color': '#304454', 'xtick.color': '#4F6274', 'ytick.color': '#4F6274',
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.unicode_minus': False,
    'legend.frameon': False,
})
colors = {1: '#377EAD', 2: '#C65E7C', 3: '#15958C'}
markers = {1: 'o', 2: 's', 3: 'D'}

total_nodes = sum(s['registered'] for s in summary)
fig = plt.figure(figsize=(12.8, 8.8), facecolor='white')
fig.text(0.08, 0.958, 'iMet / 三轮外层 Host micro-F1', fontsize=24, color='#23333D')
fig.text(0.08, 0.923, 'imet-2020-fgvc7 · 3474 属性多标签 · 同一固定 search 划分 · micro-F1 越高越好',
         fontsize=12.5, color='#71859A')
fig.text(0.92, 0.955, f'3 轮 / {total_nodes} 个注册候选', ha='right', fontsize=11, color='#71859A')

ax = fig.add_axes([0.09, 0.455, 0.83, 0.40])
strip = fig.add_axes([0.09, 0.305, 0.83, 0.095], sharex=ax)
ax.grid(axis='y', color='#E4ECF2', linewidth=0.8)
ax.set_axisbelow(True)

for round_id, label, _ in RUNS:
    rr = sorted((x for x in rows if x['round'] == round_id), key=lambda x: x['elapsed_minutes'])
    good = [x for x in rr if x['certified']]
    color = colors[round_id]
    end = summary[round_id - 1]['elapsed_minutes']

    # internal tune layer (second layer, drawn first so Host sits on top)
    tuned = sorted((x for x in rr if x['tune_micro_f1'] is not None and x['tune_minutes'] is not None),
                   key=lambda x: x['tune_minutes'])
    if tuned:
        tx = [x['tune_minutes'] for x in tuned]
        ty = [x['tune_micro_f1'] for x in tuned]
        tbest = [max(ty[:i + 1]) for i in range(len(ty))]
        ax.step(tx + [end], tbest + [tbest[-1]], where='post', color=color,
                linestyle=':', linewidth=1.6, alpha=0.75, zorder=1)
        ax.scatter(tx, ty, marker=markers[round_id], s=26, facecolors=color,
                   edgecolors=color, alpha=0.55, linewidths=0, zorder=3)

    # Host-certified layer
    xs = [x['elapsed_minutes'] for x in good]
    ys = [x['host_micro_f1'] for x in good]
    bests = [max(ys[:i + 1]) for i in range(len(ys))]
    ax.step(xs + [end], bests + [bests[-1]], where='post', color=color,
            linestyle='-', linewidth=2.7, zorder=2)
    ax.scatter(xs, ys, marker=markers[round_id], s=48, facecolors='white',
               edgecolors=color, linewidths=1.7, zorder=4)
    ax.plot(end, bests[-1], marker='|', color=color, markersize=12, markeredgewidth=1.8)
    offsets = {1: (-6, -21), 2: (-6, -21), 3: (-6, 11)}
    ax.annotate(f'R{round_id}  {bests[-1]:.5f}', (end, bests[-1]),
                xytext=offsets[round_id], textcoords='offset points', ha='right',
                color=color, fontsize=11, weight='bold')

    level = {1: 3, 2: 2, 3: 1}[round_id]
    strip.hlines(level, 0, end, color=color, alpha=0.22, linewidth=1.2)
    for x in rr:
        if x['certified']:
            strip.scatter(x['elapsed_minutes'], level, marker=markers[round_id], s=32,
                          facecolors='white', edgecolors=color, linewidths=1.1, zorder=4)
        elif x['tune_micro_f1'] is not None:
            strip.scatter(x['elapsed_minutes'], level, marker=markers[round_id], s=22,
                          facecolors=color, edgecolors='none', alpha=0.55, zorder=3)
        else:
            strip.scatter(x['elapsed_minutes'], level, marker='x', s=30,
                          color=color, linewidths=1.1, zorder=3)
    strip.plot(end, level, marker='|', color=color, markersize=11, markeredgewidth=1.5)

ax.set_xlim(0, 123)
ax.set_ylim(0.285, 0.475)
ax.set_ylabel('固定 search micro-F1 ↑')
ax.set_xticks(range(0, 121, 20))
ax.tick_params(axis='x', labelbottom=False, bottom=False)
ax.spines['bottom'].set_visible(False)
ax.set_yticks([0.30, 0.35, 0.40, 0.45])

strip.set_ylim(0.5, 3.5)
strip.set_yticks([3, 2, 1])
strip.set_yticklabels(['R1 · 无记忆', 'R2 · 仅 R1 记忆', 'R3 · 累计记忆'])
strip.set_xticks(range(0, 121, 20))
strip.set_xlabel('距各轮启动的时间 / 分钟（初始化、规划、训练与评分均计入）')
strip.tick_params(axis='y', length=0, labelsize=9)
strip.spines['left'].set_visible(False)
strip.grid(axis='x', color='#EEF3F7', linewidth=0.6)

handles = [Line2D([0], [0], color=colors[i], linestyle='-', linewidth=2.5,
                  marker=markers[i], markerfacecolor='white', markeredgecolor=colors[i],
                  markersize=6, label=f"{summary[i-1]['label']} · {summary[i-1]['certified']} 个 Host 认证")
           for i in (1, 2, 3)]
handles += [
    Line2D([0], [0], color='#71859A', linestyle=':', linewidth=1.6, marker='o',
           markerfacecolor='#71859A', markeredgecolor='none', markersize=5, alpha=0.7,
           label='内部 tune 节点（未经 Host 认证）'),
    Line2D([0], [0], marker='x', color='#71859A', linestyle='None', markersize=6,
           label='候选生成或训练失败'),
]
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.085, 0.905),
           ncol=5, fontsize=9, handlelength=2.1, columnspacing=1.3)

# Summary cards: one column per round, values stacked so nothing collides.
card_xs = [0.09, 0.37, 0.65]
for x, s, color in zip(card_xs, summary, [colors[1], colors[2], colors[3]]):
    fig.text(x, 0.185, s['label'], fontsize=10.5, color='#71859A')
    fig.text(x, 0.118, f"{s['best_host_micro_f1']:.5f}", fontsize=23, color=color)
    fig.text(x, 0.082, f"{s['elapsed_minutes']:.1f} min · {s['certified']}/{s['registered']} Host 认证",
             fontsize=9.5, color='#71859A')
    fig.text(x, 0.055,
             f"最佳出现于 {s['best_host_at_minutes']:.1f} min · 内部 tune 最佳 {s['best_tune_micro_f1']:.5f}",
             fontsize=9.5, color='#71859A')
fig.text(0.09, 0.018,
         '实线阶梯＝Host 认证候选的累计最佳；空心点＝真实 Host 分数；虚点线与实心点＝内部 tune 分数（未经 Host 认证）；'
         '×＝候选生成或训练失败。纵轴未从 0 起。非 Kaggle 官方成绩。',
         fontsize=9.5, color='#71859A')

for ext in ('png', 'svg', 'pdf'):
    kw = dict(dpi=300) if ext == 'png' else {}
    fig.savefig(OUT / f'imet_three_rounds.{ext}', facecolor='white', bbox_inches='tight', **kw)
plt.close(fig)

print(json.dumps(summary, ensure_ascii=False, indent=2))
