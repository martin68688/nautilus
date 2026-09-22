"""iMet three rounds, Leaf-style staircase built from the INTERNAL TUNE nodes.

Same figure method as outputs/leaf_three_rounds_r2_replaced_20260920/plot_curves_replaced.py
(staircase of cumulative best + rug strip + summary cards), but the plotted nodes
are the internal tune evaluations instead of the Host-certified candidates.
Metric is micro-F1, so the staircase is a cumulative MAX.

Node set per candidate = its per-epoch tune values, plus the post-training
threshold-calibration value when that differs from the last epoch:
  * R2 / R3 log per-epoch CALIBRATED `tune_micro_f1`; the last epoch already IS
    the calibrated score, so nothing is added.
  * R1 logs `val_micro_f1_at_0.15`, a fixed-threshold training monitor, and only
    calibrates after training. Its calibration node (0.38823 / 0.30211) is added,
    otherwise R1's staircase would top out at 0.21556 and the round-over-round
    comparison would be an artifact of the logging threshold rather than a result.
R1's fixed-threshold nodes are drawn dimmer so the two quantities stay visible.

Epoch wall-clock is interpolated: train.log is empty, so no per-epoch timestamps
exist. Epochs are spread evenly across each attempt's train stage.

Read-only with respect to the iMet run directories.
"""
from __future__ import annotations
from pathlib import Path
import csv, json, math

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
CALIBRATED_KEY = 'tune_micro_f1'
MONITOR_KEY = 'val_micro_f1_at_0.15'


def read_meta(attempt_dir: Path):
    for name in ('metadata.json', 'meta.json'):
        p = attempt_dir / 'worker' / 'artifacts' / name
        if p.exists():
            return json.loads(p.read_text()), name
    return None, None


rows = []        # every internal tune node
failures = []    # candidates that produced no tune node at all
round_meta = {}

for round_id, label, run_dir in RUNS:
    outcome = json.loads((run_dir / 'RUN_OUTCOME.json').read_text())
    state = json.loads((run_dir / 'STATE.json').read_text())
    round_meta[round_id] = dict(label=label, outcome=outcome)

    for idx, h in enumerate(state['history']):
        attempt_dir = run_dir / f'attempt-{idx:03d}'
        start_s, end_s = float(h['started_elapsed']), float(h['elapsed_seconds'])
        meta, meta_name = read_meta(attempt_dir)

        final_tune = h.get('tune_score')
        if final_tune is None and meta is not None:
            final_tune = meta.get(CALIBRATED_KEY)
        final_tune = float(final_tune) if isinstance(final_tune, (int, float)) else None

        train_s = overhead = None
        exec_path = attempt_dir / 'execution.json'
        if exec_path.exists():
            stages = json.loads(exec_path.read_text()).get('stages') or []
            train_s = next((float(s['elapsed_seconds']) for s in stages if s['mode'] == 'train'), None)
            if train_s is not None:
                overhead = max(0.0, (end_s - start_s) - sum(float(s['elapsed_seconds']) for s in stages))

        traj = (meta or {}).get('trajectory') or []
        if not traj or train_s is None:
            failures.append({'round': round_id, 'node_id': h['attempt'],
                             'minutes': end_s / 60.0, 'reason': h.get('reason') or h.get('status')})
            continue

        key = CALIBRATED_KEY if CALIBRATED_KEY in traj[0] else MONITOR_KEY
        n = len(traj)
        train_start_s = start_s + overhead
        for i, t in enumerate(traj, start=1):
            rows.append({
                'round': round_id, 'node_id': h['attempt'], 'kind': 'epoch',
                'epoch': t.get('epoch', i), 'value_key': key,
                'calibrated': key == CALIBRATED_KEY,
                'minutes': (train_start_s + train_s * i / n) / 60.0,
                'tune_micro_f1': float(t[key]),
            })
        last_epoch_value = float(traj[-1][key])
        if final_tune is not None and abs(final_tune - last_epoch_value) > 1e-9:
            rows.append({
                'round': round_id, 'node_id': h['attempt'], 'kind': 'calibration',
                'epoch': None, 'value_key': CALIBRATED_KEY, 'calibrated': True,
                'minutes': (train_start_s + train_s) / 60.0,
                'tune_micro_f1': float(final_tune),
            })

for round_id, *_ in RUNS:
    best = -math.inf
    for row in sorted((x for x in rows if x['round'] == round_id),
                      key=lambda x: (x['minutes'], x['kind'] == 'calibration')):
        best = max(best, row['tune_micro_f1'])
        row['best_so_far'] = best

with (OUT / 'source_tune_nodes_imet.csv').open('w', newline='') as handle:
    fn = ['round', 'node_id', 'kind', 'epoch', 'minutes', 'tune_micro_f1',
          'best_so_far', 'value_key', 'calibrated']
    w = csv.DictWriter(handle, fieldnames=fn)
    w.writeheader()
    for r in sorted(rows, key=lambda x: (x['round'], x['minutes'])):
        w.writerow({k: r.get(k) for k in fn})

summary = []
for round_id, label, run_dir in RUNS:
    rr = [x for x in rows if x['round'] == round_id]
    ff = [x for x in failures if x['round'] == round_id]
    best = max(x['tune_micro_f1'] for x in rr)
    summary.append({
        'round': round_id, 'label': label,
        'elapsed_minutes': round_meta[round_id]['outcome']['elapsed_seconds'] / 60.0,
        'tune_nodes': len(rr),
        'epoch_nodes': sum(1 for x in rr if x['kind'] == 'epoch'),
        'calibration_nodes': sum(1 for x in rr if x['kind'] == 'calibration'),
        'failed_candidates': len(ff),
        'best_tune_micro_f1': best,
        'best_at_minutes': min(x['minutes'] for x in rr if x['tune_micro_f1'] == best),
    })

assert [s['epoch_nodes'] for s in summary] == [12, 18, 20], [s['epoch_nodes'] for s in summary]
assert [s['calibration_nodes'] for s in summary] == [2, 0, 0]
assert [s['tune_nodes'] for s in summary] == [14, 18, 20]
assert sum(s['tune_nodes'] for s in summary) == 52
assert abs(summary[0]['best_tune_micro_f1'] - 0.3882310785523749) < 1e-12
assert abs(summary[1]['best_tune_micro_f1'] - 0.43860364610280916) < 1e-12
assert abs(summary[2]['best_tune_micro_f1'] - 0.45849145500492566) < 1e-12

(OUT / 'summary_imet_tune_staircase.json').write_text(json.dumps({
    'figure': 'iMet three rounds, staircase over internal tune nodes',
    'task_id': 'imet-2020-fgvc7',
    'metric': 'internal tune micro-F1 (higher is better); NOT Host-recomputed',
    'total_tune_nodes': sum(s['tune_nodes'] for s in summary),
    'epoch_time_is_interpolated': True,
    'r1_monitor_key_note': 'R1 per-epoch values use %s (fixed threshold 0.15); its '
                           'calibration nodes carry the comparable score.' % MONITOR_KEY,
    'rounds': summary,
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
    'font.size': 10, 'axes.labelsize': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#B7C5D0', 'axes.labelcolor': '#304454',
    'text.color': '#304454', 'xtick.color': '#4F6274', 'ytick.color': '#4F6274',
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.unicode_minus': False,
    'legend.frameon': False,
})
colors = {1: '#377EAD', 2: '#C65E7C', 3: '#15958C'}
markers = {1: 'o', 2: 's', 3: 'D'}

fig = plt.figure(figsize=(12.8, 8.4), facecolor='white')
fig.text(0.08, 0.958, 'iMet / 三轮内部 tune 搜索曲线', fontsize=24, color='#23333D')
fig.text(0.08, 0.922, 'imet-2020-fgvc7 · 3474 属性多标签 · 内部 tune 划分 · micro-F1 越高越好',
         fontsize=12.5, color='#71859A')
fig.text(0.92, 0.955, f'3 轮 / {sum(s["tune_nodes"] for s in summary)} 个内部 tune 节点',
         ha='right', fontsize=11, color='#71859A')

ax = fig.add_axes([0.09, 0.445, 0.83, 0.405])
strip = fig.add_axes([0.09, 0.300, 0.83, 0.093], sharex=ax)
ax.grid(axis='y', color='#E4ECF2', linewidth=0.8)
ax.set_axisbelow(True)

for round_id, label, _ in RUNS:
    rr = sorted((x for x in rows if x['round'] == round_id),
                key=lambda x: (x['minutes'], x['kind'] == 'calibration'))
    color = colors[round_id]
    end = summary[round_id - 1]['elapsed_minutes']
    xs = [x['minutes'] for x in rr]
    ys = [x['tune_micro_f1'] for x in rr]
    bests = [max(ys[:i + 1]) for i in range(len(ys))]
    ax.step(xs + [end], bests + [bests[-1]], where='post', color=color, linewidth=2.7, zorder=2)

    # fixed-threshold monitor nodes dimmer than comparable calibrated nodes
    dim = [x for x in rr if not x['calibrated']]
    lit = [x for x in rr if x['calibrated']]
    if dim:
        ax.scatter([x['minutes'] for x in dim], [x['tune_micro_f1'] for x in dim],
                   marker=markers[round_id], s=34, facecolors='white', edgecolors=color,
                   linewidths=1.1, alpha=0.45, zorder=3)
    ax.scatter([x['minutes'] for x in lit], [x['tune_micro_f1'] for x in lit],
               marker=markers[round_id], s=44, facecolors='white', edgecolors=color,
               linewidths=1.6, zorder=4)
    ax.plot(end, bests[-1], marker='|', color=color, markersize=12, markeredgewidth=1.8)
    offsets = {1: (-6, -20), 2: (-6, -20), 3: (-6, 12)}
    ax.annotate(f'R{round_id}  {bests[-1]:.5f}', (end, bests[-1]), xytext=offsets[round_id],
                textcoords='offset points', ha='right', color=color, fontsize=11, weight='bold')

    level = {1: 3, 2: 2, 3: 1}[round_id]
    strip.hlines(level, 0, end, color=color, alpha=0.22, linewidth=1.2)
    for x in rr:
        strip.scatter(x['minutes'], level, marker=markers[round_id], s=24,
                      facecolors='white', edgecolors=color,
                      linewidths=1.0, alpha=1.0 if x['calibrated'] else 0.45, zorder=4)
    for f in (x for x in failures if x['round'] == round_id):
        strip.scatter(f['minutes'], level, marker='x', s=30, color=color, linewidths=1.1, zorder=3)
    strip.plot(end, level, marker='|', color=color, markersize=11, markeredgewidth=1.5)

ax.annotate('R1 的浅色点是固定阈值 0.15 的训练期监控值；该轮阶梯在 89.3 min 跃升，\n'
            '是训练后阈值校准得到 0.38823。R2 / R3 逐 epoch 已是校准值。',
            xy=(0.015, 0.97), xycoords='axes fraction', va='top', ha='left',
            fontsize=8.8, color='#8A9AAB', linespacing=1.5)

ax.set_xlim(0, 123)
ax.set_ylim(0.12, 0.49)
ax.set_ylabel('内部 tune micro-F1 ↑')
ax.set_xticks(range(0, 121, 20))
ax.set_yticks([0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])
ax.tick_params(axis='x', labelbottom=False, bottom=False)
ax.spines['bottom'].set_visible(False)

strip.set_ylim(0.5, 3.5)
strip.set_yticks([3, 2, 1])
strip.set_yticklabels(['R1 · 无记忆', 'R2 · 仅 R1 记忆', 'R3 · 累计记忆'])
strip.set_xticks(range(0, 121, 20))
strip.set_xlabel('距各轮启动的时间 / 分钟（初始化、规划、训练与评分均计入；epoch 时刻按训练阶段等分推算）')
strip.tick_params(axis='y', length=0, labelsize=9)
strip.spines['left'].set_visible(False)
strip.grid(axis='x', color='#EEF3F7', linewidth=0.6)

handles = [Line2D([0], [0], color=colors[i], linewidth=2.5, marker=markers[i],
                  markerfacecolor='white', markeredgecolor=colors[i], markersize=6,
                  label=f"{summary[i-1]['label']} · {summary[i-1]['tune_nodes']} 个 tune 节点")
           for i in (1, 2, 3)]
handles += [
    Line2D([0], [0], marker='o', color='#8A9AAB', linestyle='None', markerfacecolor='white',
           markeredgecolor='#8A9AAB', markersize=5, alpha=0.5,
           label='固定阈值 0.15 监控值（R1）'),
    Line2D([0], [0], marker='x', color='#8A9AAB', linestyle='None', markersize=6,
           label='候选生成或训练失败（无 tune 轨迹）'),
]
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.085, 0.900),
           ncol=5, fontsize=9, handlelength=2.1, columnspacing=1.3)

card_xs = [0.09, 0.37, 0.65]
for x, s, color in zip(card_xs, summary, [colors[1], colors[2], colors[3]]):
    fig.text(x, 0.190, s['label'], fontsize=10.5, color='#71859A')
    fig.text(x, 0.122, f"{s['best_tune_micro_f1']:.5f}", fontsize=23, color=color)
    fig.text(x, 0.088, f"{s['elapsed_minutes']:.1f} min · {s['tune_nodes']} 个 tune 节点 · "
                       f"{s['failed_candidates']} 个候选失败", fontsize=9.5, color='#71859A')
    fig.text(x, 0.062, f"最佳出现于 {s['best_at_minutes']:.1f} min", fontsize=9.5, color='#71859A')
fig.text(0.09, 0.026,
         '阶梯线＝内部 tune 节点的累计最佳；空心点＝内部 tune 分数；×＝候选生成或训练失败（无 tune 轨迹）。',
         fontsize=9, color='#71859A')
fig.text(0.09, 0.008,
         '本图为候选自报的内部 tune 分数，未经 Host 复算。纵轴未从 0 起；epoch 时刻为等分推算，非实测时间戳。非 Kaggle 官方成绩。',
         fontsize=9, color='#71859A')

for ext in ('png', 'svg', 'pdf'):
    kw = dict(dpi=300) if ext == 'png' else {}
    fig.savefig(OUT / f'imet_tune_staircase.{ext}', facecolor='white', bbox_inches='tight', **kw)
plt.close(fig)

print(json.dumps(summary, ensure_ascii=False, indent=2))
