"""iMet three rounds: Host micro-F1 staircase + a separate internal tune-trajectory panel.

Extends plot_imet_three_rounds.py with the per-epoch internal tune nodes
(50 across the three rounds), drawn in their own lighter panel rather than
mixed into the Host curve.

Definition warning, handled explicitly in the figure rather than hidden:
  * R2 / R3 log per-epoch CALIBRATED `tune_micro_f1`; the last epoch equals the
    candidate's final tune score exactly.
  * R1 logs `val_micro_f1_at_0.15`, a fixed-threshold training monitor. Its last
    epoch is 0.21556 while the same candidate's calibrated tune score is 0.38823.
    Plotting the two as one quantity would depress R1 by ~0.17 and manufacture an
    apparent memory gain. R1 is therefore drawn dotted, in its own legend entry,
    with a connector up to its calibrated value.

Epoch wall-clock is interpolated: train.log is empty, so no per-epoch timestamps
exist. Epochs are spread evenly across the train stage of each attempt.

Read-only with respect to the iMet run directories.
"""
from __future__ import annotations
from pathlib import Path
import csv, hashlib, json, math

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


nodes = []        # outer candidates
epochs = []       # inner per-epoch tune nodes
round_meta = {}

for round_id, label, run_dir in RUNS:
    launch = json.loads((run_dir / 'LAUNCH.json').read_text())
    outcome = json.loads((run_dir / 'RUN_OUTCOME.json').read_text())
    state = json.loads((run_dir / 'STATE.json').read_text())
    round_meta[round_id] = dict(label=label, run_dir=run_dir, launch=launch, outcome=outcome)

    for idx, h in enumerate(state['history']):
        attempt_dir = run_dir / f'attempt-{idx:03d}'
        start_s, end_s = float(h['started_elapsed']), float(h['elapsed_seconds'])
        certified = (h.get('status') == 'valid' and bool(h.get('host_recomputed'))
                     and bool(h.get('reload_verified'))
                     and isinstance(h.get('search_score'), (int, float)))

        meta, meta_name = read_meta(attempt_dir)
        tune = h.get('tune_score')
        if tune is None and meta is not None:
            tune = meta.get(CALIBRATED_KEY)
        tune = float(tune) if isinstance(tune, (int, float)) else None

        train_s = overhead = None
        exec_path = attempt_dir / 'execution.json'
        if exec_path.exists():
            stages = json.loads(exec_path.read_text()).get('stages') or []
            train_s = next((float(s['elapsed_seconds']) for s in stages if s['mode'] == 'train'), None)
            if train_s is not None:
                overhead = max(0.0, (end_s - start_s) - sum(float(s['elapsed_seconds']) for s in stages))

        tune_min = None
        if tune is not None and train_s is not None:
            tune_min = (start_s + overhead + train_s) / 60.0

        nodes.append({
            'round': round_id, 'attempt': idx, 'node_id': h['attempt'],
            'elapsed_minutes': end_s / 60.0, 'certified': certified,
            'host_micro_f1': float(h['search_score']) if certified else None,
            'tune_micro_f1': tune, 'tune_minutes': tune_min, 'status': h.get('status'),
        })

        # per-epoch internal tune nodes, evenly spread across the train stage
        traj = (meta or {}).get('trajectory') or []
        if traj and train_s is not None:
            key = CALIBRATED_KEY if CALIBRATED_KEY in traj[0] else MONITOR_KEY
            n = len(traj)
            train_start_s = start_s + overhead
            for i, t in enumerate(traj, start=1):
                epochs.append({
                    'round': round_id, 'node_id': h['attempt'], 'epoch': t.get('epoch', i),
                    'minutes': (train_start_s + train_s * i / n) / 60.0,
                    'value': float(t[key]), 'value_key': key,
                    'calibrated': key == CALIBRATED_KEY,
                    'final_calibrated_tune': tune, 'meta_file': meta_name,
                })

for round_id, *_ in RUNS:
    best = -math.inf
    for row in sorted((x for x in nodes if x['round'] == round_id), key=lambda x: x['elapsed_minutes']):
        if row['certified']:
            best = max(best, row['host_micro_f1'])
        row['best_so_far'] = best if math.isfinite(best) else None

with (OUT / 'source_epochs_imet.csv').open('w', newline='') as handle:
    fn = ['round', 'node_id', 'epoch', 'minutes', 'value', 'value_key', 'calibrated',
          'final_calibrated_tune', 'meta_file']
    w = csv.DictWriter(handle, fieldnames=fn)
    w.writeheader()
    for e in sorted(epochs, key=lambda x: (x['round'], x['node_id'], x['epoch'])):
        w.writerow({k: e[k] for k in fn})

summary = []
for round_id, label, run_dir in RUNS:
    rr = [x for x in nodes if x['round'] == round_id]
    good = [x for x in rr if x['certified']]
    ee = [e for e in epochs if e['round'] == round_id]
    summary.append({
        'round': round_id, 'label': label,
        'elapsed_minutes': round_meta[round_id]['outcome']['elapsed_seconds'] / 60.0,
        'registered': len(rr), 'certified': len(good),
        'epoch_nodes': len(ee),
        'epoch_value_key': sorted({e['value_key'] for e in ee}),
        'best_host_micro_f1': max(x['host_micro_f1'] for x in good),
        'best_tune_micro_f1': max(x['tune_micro_f1'] for x in rr if x['tune_micro_f1'] is not None),
        'best_host_at_minutes': min(x['elapsed_minutes'] for x in good
                                    if x['host_micro_f1'] == max(y['host_micro_f1'] for y in good)),
    })

assert [s['registered'] for s in summary] == [3, 3, 2]
assert [s['certified'] for s in summary] == [1, 2, 1]
assert [s['epoch_nodes'] for s in summary] == [12, 18, 20], [s['epoch_nodes'] for s in summary]
assert sum(s['epoch_nodes'] for s in summary) == 50
assert summary[0]['epoch_value_key'] == [MONITOR_KEY]
assert summary[1]['epoch_value_key'] == [CALIBRATED_KEY]
assert summary[2]['epoch_value_key'] == [CALIBRATED_KEY]
for s, (_, _, run_dir) in zip(summary, RUNS):
    declared = json.loads((run_dir / 'RUN_OUTCOME.json').read_text())['best_search']
    assert abs(s['best_host_micro_f1'] - declared) < 1e-12

(OUT / 'summary_imet_tune.json').write_text(json.dumps({
    'figure': 'iMet three rounds: Host micro-F1 + internal tune trajectories',
    'task_id': 'imet-2020-fgvc7',
    'metric': 'host micro-F1 over 3474 multilabel attributes (higher is better)',
    'internal_tune_nodes_total': sum(s['epoch_nodes'] for s in summary),
    'epoch_time_is_interpolated': True,
    'r1_epoch_key_differs': {
        'r1_key': MONITOR_KEY, 'r2_r3_key': CALIBRATED_KEY,
        'note': 'R1 per-epoch values are a fixed-threshold (0.15) training monitor, '
                'not the calibrated tune metric; not comparable on the same axis.',
    },
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

fig = plt.figure(figsize=(12.8, 11.0), facecolor='white')
fig.text(0.08, 0.972, 'iMet / 三轮外层 Host micro-F1 与内部 tune 轨迹', fontsize=23, color='#23333D')
fig.text(0.08, 0.950, 'imet-2020-fgvc7 · 3474 属性多标签 · 同一固定 search 划分 · micro-F1 越高越好',
         fontsize=12, color='#71859A')
fig.text(0.92, 0.970, f'3 轮 / 8 个外层候选 / {sum(s["epoch_nodes"] for s in summary)} 个内部 tune 节点',
         ha='right', fontsize=11, color='#71859A')

ax = fig.add_axes([0.09, 0.620, 0.83, 0.262])
axB = fig.add_axes([0.09, 0.335, 0.83, 0.225], sharex=ax)
strip = fig.add_axes([0.09, 0.232, 0.83, 0.072], sharex=ax)
for a in (ax, axB):
    a.grid(axis='y', color='#E4ECF2', linewidth=0.8)
    a.set_axisbelow(True)

# ---- Panel A: Host-certified staircase
for round_id, label, _ in RUNS:
    rr = sorted((x for x in nodes if x['round'] == round_id), key=lambda x: x['elapsed_minutes'])
    good = [x for x in rr if x['certified']]
    color = colors[round_id]
    end = summary[round_id - 1]['elapsed_minutes']
    xs = [x['elapsed_minutes'] for x in good]
    ys = [x['host_micro_f1'] for x in good]
    bests = [max(ys[:i + 1]) for i in range(len(ys))]
    ax.step(xs + [end], bests + [bests[-1]], where='post', color=color, linewidth=2.7, zorder=2)
    ax.scatter(xs, ys, marker=markers[round_id], s=48, facecolors='white',
               edgecolors=color, linewidths=1.7, zorder=4)
    ax.plot(end, bests[-1], marker='|', color=color, markersize=12, markeredgewidth=1.8)
    offsets = {1: (-6, -21), 2: (-6, -21), 3: (-6, -21)}
    ax.annotate(f'R{round_id}  {bests[-1]:.5f}', (end, bests[-1]), xytext=offsets[round_id],
                textcoords='offset points', ha='right', color=color, fontsize=11, weight='bold')

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
            strip.scatter(x['elapsed_minutes'], level, marker='x', s=30, color=color,
                          linewidths=1.1, zorder=3)
    for e in (x for x in epochs if x['round'] == round_id):
        strip.scatter(e['minutes'], level, marker='|', s=14, color=color, alpha=0.40, zorder=2)
    strip.plot(end, level, marker='|', color=color, markersize=11, markeredgewidth=1.5)

# ---- Panel B: internal tune trajectories, one polyline per candidate
for round_id, label, _ in RUNS:
    color = colors[round_id]
    by_node = {}
    for e in (x for x in epochs if x['round'] == round_id):
        by_node.setdefault(e['node_id'], []).append(e)
    for node_id, es in by_node.items():
        es.sort(key=lambda x: x['epoch'])
        xs = [e['minutes'] for e in es]
        ys = [e['value'] for e in es]
        calibrated = es[0]['calibrated']
        axB.plot(xs, ys, color=color, alpha=0.75 if calibrated else 0.55,
                 linestyle='-' if calibrated else ':', linewidth=1.9, zorder=2)
        axB.scatter(xs, ys, marker=markers[round_id], s=16, facecolors=color,
                    edgecolors='none', alpha=0.65 if calibrated else 0.45, zorder=3)
        final = es[-1]['final_calibrated_tune']
        if final is not None:
            axB.scatter([xs[-1]], [final], marker=markers[round_id], s=52, facecolors='white',
                        edgecolors=color, linewidths=1.7, zorder=5)
            if not calibrated:
                # make the fixed-threshold -> calibrated jump explicit instead of hiding it
                axB.annotate('', xy=(xs[-1], final), xytext=(xs[-1], ys[-1]),
                             arrowprops=dict(arrowstyle='-|>', color=color, alpha=0.65,
                                             linestyle=(0, (2, 2)), linewidth=1.2,
                                             shrinkA=2, shrinkB=2))

axB.annotate('R1 逐 epoch 记的是固定阈值 0.15 的监控值，\n'
             '训练后阈值校准才得到 0.38823 / 0.30211（虚线箭头）；\n'
             'R2 / R3 逐 epoch 已是校准值，末点即最终 tune 分。',
             xy=(0.015, 0.955), xycoords='axes fraction', va='top', ha='left',
             fontsize=8.8, color='#8A9AAB', linespacing=1.5)

ax.set_xlim(0, 123)
ax.set_ylim(0.372, 0.474)
ax.set_ylabel('固定 search micro-F1 ↑\n（Host 认证）')
ax.set_yticks([0.38, 0.40, 0.42, 0.44, 0.46])
ax.tick_params(axis='x', labelbottom=False, bottom=False)
ax.spines['bottom'].set_visible(False)

axB.set_ylim(0.12, 0.50)
axB.set_ylabel('内部 tune micro-F1\n（未经 Host 认证）')
axB.set_yticks([0.15, 0.25, 0.35, 0.45])
axB.tick_params(axis='x', labelbottom=False, bottom=False)
axB.spines['bottom'].set_visible(False)

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
                  label=f"{summary[i-1]['label']} · {summary[i-1]['certified']} 个 Host 认证 · "
                        f"{summary[i-1]['epoch_nodes']} 个 tune 节点")
           for i in (1, 2, 3)]
handles += [
    Line2D([0], [0], color='#8A9AAB', linewidth=1.9, label='内部 tune 轨迹（校准值，R2/R3）'),
    Line2D([0], [0], color='#8A9AAB', linewidth=1.9, linestyle=':',
           label='内部 tune 轨迹（固定阈值 0.15 监控值，R1）'),
    Line2D([0], [0], marker='x', color='#8A9AAB', linestyle='None', markersize=6,
           label='候选生成或训练失败'),
]
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.085, 0.938),
           ncol=3, fontsize=9, handlelength=2.1, columnspacing=1.4, labelspacing=0.45)

card_xs = [0.09, 0.37, 0.65]
for x, s, color in zip(card_xs, summary, [colors[1], colors[2], colors[3]]):
    fig.text(x, 0.158, s['label'], fontsize=10.5, color='#71859A')
    fig.text(x, 0.104, f"{s['best_host_micro_f1']:.5f}", fontsize=23, color=color)
    fig.text(x, 0.075, f"{s['elapsed_minutes']:.1f} min · {s['certified']}/{s['registered']} Host 认证",
             fontsize=9.5, color='#71859A')
    fig.text(x, 0.053, f"最佳出现于 {s['best_host_at_minutes']:.1f} min · "
                       f"内部 tune 最佳 {s['best_tune_micro_f1']:.5f}",
             fontsize=9.5, color='#71859A')
fig.text(0.09, 0.030,
         '上 panel：实线阶梯＝Host 认证候选的累计最佳，空心点＝真实 Host 分数。'
         '下 panel：每条折线＝一个候选的内部 tune 轨迹，空心点＝该候选最终校准 tune 分。×＝候选生成或训练失败。',
         fontsize=9, color='#71859A')
fig.text(0.09, 0.011,
         '两个 panel 纵轴均未从 0 起，且分属不同指标（Host 复算 vs 候选自报 tune），不可当成同一条曲线。'
         'epoch 时刻为等分推算，非实测时间戳。非 Kaggle 官方成绩。',
         fontsize=9, color='#71859A')

for ext in ('png', 'svg', 'pdf'):
    kw = dict(dpi=300) if ext == 'png' else {}
    fig.savefig(OUT / f'imet_three_rounds_tune.{ext}', facecolor='white', bbox_inches='tight', **kw)
plt.close(fig)

print(json.dumps(summary, ensure_ascii=False, indent=2))
