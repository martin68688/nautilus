"""User-requested display subset: omit R4, preserve original round identities."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import csv
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

OUT = Path(__file__).resolve().parent
rows = list(csv.DictReader((OUT / 'outer_search.csv').open()))
summary = json.loads((OUT / 'summary.json').read_text())['rounds']
terminal_path = OUT.parents[1] / 'experiments/leaf_round4_memory_luna_2h_20260913/all_memory_luna_repeat2_20260914_1747/TERMINAL.json'
terminal = json.loads(terminal_path.read_text())
launch = terminal['controls']['LAUNCH.json']
seal = terminal['core']['FINAL_SNAPSHOT.json']
assert seal['consistent']
assert launch['data_manifest_sha256'] == '7bd8ce4887d28eea1bbe26730227ac4e49160332bd087571d7331110b5580a7f'
verified = {n['id']: n['VERIFIED_RESULT.json'] for n in terminal['nodes'] if 'VERIFIED_RESULT.json' in n}
nodes = [n for n in terminal['core']['journal.json']['nodes'] if n.get('stage') != 'root']
assert len(nodes) == 7
for i, n in enumerate(nodes, 1):
    score = n.get('metric', {}).get('value')
    valid = bool(n.get('is_valid')) and isinstance(score, (int, float)) and math.isfinite(score)
    dt = datetime.fromisoformat(n['finish_time']).replace(tzinfo=timezone.utc)
    elapsed = (dt.timestamp() - launch['started_unix']) / 60
    if valid:
        receipt = verified[n['id']]
        assert receipt['checkpoint_validated'] and receipt['host_recomputed']
        assert abs(receipt['metric'] - score) < 1e-12
    rows.append(dict(round=6, node=i, node_id=n['id'], elapsed_minutes=elapsed,
                     certified=str(valid), search_loss=score if valid else ''))
summary.append(dict(round=6, nodes=7, certified=5, final_search=seal['host_metric'],
                    elapsed_minutes=terminal['controls']['POD_PROCESS_OUTCOME.json']['elapsed_seconds']/60,
                    run_id=terminal['run_id'], stop_reason=terminal['core']['RUN_OUTCOME.json']['reason']))
with (OUT / 'without_r4_outer_search.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(r for r in rows if int(r['round']) != 4)
(OUT / 'without_r4_summary.json').write_text(json.dumps({'rounds': [s for s in summary if s['round'] != 4],
    'terminal_sha256': hashlib.sha256(terminal_path.read_bytes()).hexdigest()}, indent=2))
font = Path('/System/Library/Fonts/STHeiti Light.ttc')
font_manager.fontManager.addfont(font)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': [font_manager.FontProperties(fname=font).get_name(), 'Arial'],
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#BCC8D1', 'axes.linewidth': .7,
    'text.color': '#293E4D', 'axes.labelcolor': '#405565',
    'xtick.color': '#627684', 'ytick.color': '#627684',
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'svg.fonttype': 'none', 'pdf.fonttype': 42,
    'axes.unicode_minus': False, 'legend.frameon': False,
})
colors = ['#397FAC', '#C68A3A', '#8064AD', '#C3637E', '#188F8D', '#253F58']
markers = ['o', 's', 'D', '^', 'P', 'h']
styles = ['-', '-', '--', '-.', '-', '-']
labels = ['R1 首轮', 'R2 第二轮', 'R3 第三轮重跑', 'R4 历史第四轮', 'R5 全记忆轮', 'R6 最新重跑']

# Quantitative hero + subordinate detail; fixed search loss, one seed/split.
# Field mapping: round -> group; elapsed_minutes -> x; search_loss -> y.
# 42 node records -> 23 certified scores. Unscored nodes are not zero losses.
series = []
assert len(rows) == 42
for i, s in enumerate(summary, 1):
    good = [r for r in rows if int(r['round']) == i and r['certified'] == 'True']
    assert len(good) == [2, 2, 10, 1, 3, 5][i - 1]
    x = [float(r['elapsed_minutes']) for r in good]
    y = [float(r['search_loss']) for r in good]
    assert all(math.isfinite(v) and v > 0 for v in y)
    assert x == sorted(x)
    best = [min(y[:j + 1]) for j in range(len(y))]
    assert abs(best[-1] - s['final_search']) < 1e-12
    assert s['elapsed_minutes'] >= x[-1]
    series.append((x, y, best))

fig = plt.figure(figsize=(13.2, 7.0), facecolor='white')
fig.text(.075, .928, 'Leaf  /  Search 成绩对比', fontsize=23)
fig.text(.075, .877, '固定 search log-loss · 越低越好', fontsize=11)
fig.text(.965, .877, 'Luna-low   ·   A10 × 1   ·   同一划分   ·   各轮独立计时',
         ha='right', fontsize=9, color='#718391')
handles = [Line2D([], [], color=c, marker=m, ls=ls, lw=2, ms=5, label=lab)
           for i, (c, m, ls, lab) in enumerate(zip(colors, markers, styles, labels)) if i != 3]
fig.legend(handles=handles, loc='center left', bbox_to_anchor=(.067, .819),
           ncol=5, columnspacing=2.8, handlelength=2.2, fontsize=10)
fig.add_artist(Line2D([.075, .965], [.787, .787], transform=fig.transFigure,
                     color='#DFE6EB', lw=.8))

a = fig.add_axes([.075, .218, .565, .505])
b = fig.add_axes([.735, .402, .23, .321])
fig.text(.075, .749, 'a   五轮对比 · 保留原编号', fontsize=12)
fig.text(.640, .749, '纵轴为对数刻度', fontsize=9, ha='right', color='#718391')
fig.text(.735, .749, 'b   R5 / R6 低损失区域', fontsize=12)
for ax in (a, b):
    ax.grid(axis='y', color='#E7EDF1', lw=.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.7)
for i, ((x, y, best), s, c, m, ls) in enumerate(zip(series, summary, colors, markers, styles)):
    if i == 3:
        continue
    for ax in ([a, b] if i >= 3 else [a]):
        ax.step(x + [s['elapsed_minutes']], best + [best[-1]], where='post',
                color=c, ls=ls, lw=2, zorder=3)
        ax.scatter(x, y, color=c, edgecolor='white', linewidth=.45, marker=m,
                   s=34, alpha=.95, zorder=4)
        ax.plot(s['elapsed_minutes'], best[-1], marker='|', ms=8, color=c, zorder=5)
    if i < 3:
        a.annotate(f'R{i+1}  {best[-1]:.6f}', (s['elapsed_minutes'], best[-1]),
                   xytext=(-3, [10, -17, 23][i]), textcoords='offset points',
                   ha='right', fontsize=9, color=c)
a.set_yscale('log')
a.set(xlim=(0, 120), ylim=(.029, .88), xticks=range(0, 121, 20),
      xlabel='累计墙钟时间 / 分钟', ylabel='Search log-loss ↓')
a.yaxis.set_major_locator(FixedLocator([.04, .06, .1, .2, .4, .8]))
a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
a.yaxis.set_minor_locator(NullLocator())
a.annotate('R5', (summary[4]['elapsed_minutes'], summary[4]['final_search']),
           xytext=(5, 8), textcoords='offset points', fontsize=9, color=colors[4])
a.annotate('R6  0.035167', (summary[5]['elapsed_minutes'], summary[5]['final_search']),
           xytext=(6, -2), textcoords='offset points', fontsize=9, color=colors[5])
b.set(xlim=(0, 90), ylim=(.033, .054), xticks=[0, 15, 30, 45, 60, 75, 90],
      yticks=[.035, .040, .045, .050], xlabel='累计墙钟时间 / 分钟')
b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.3f}'))
b.annotate('R5', (summary[4]['elapsed_minutes'], summary[4]['final_search']),
           xytext=(-1, 8), textcoords='offset points', ha='right', color=colors[4], fontsize=9)
b.annotate('R6', (summary[5]['elapsed_minutes'], summary[5]['final_search']),
           xytext=(3, 8), textcoords='offset points', color=colors[5], fontsize=9)

# A compact aligned score ledger replaces floating endpoint callouts.
fig.text(.735, .296, '轮次', fontsize=9, color='#718391')
fig.text(.865, .296, '最佳 search', ha='right', fontsize=9, color='#718391')
fig.text(.965, .296, '实际结束', ha='right', fontsize=9, color='#718391')
for i, yy in [(4, .261), (5, .228)]:
    s = summary[i]
    fig.text(.735, yy, f'R{i+1}', fontsize=10, color=colors[i])
    fig.text(.865, yy, f"{s['final_search']:.6f}", ha='right', fontsize=10, color=colors[i])
    fig.text(.965, yy, f"{s['elapsed_minutes']:.1f} min", ha='right', fontsize=10)
delta = summary[4]['final_search'] - summary[5]['final_search']
fig.text(.735, .159, f'R6 比 R5 降低 {delta / summary[4]["final_search"] * 100:.2f}%', fontsize=10)
fig.add_artist(Line2D([.075, .965], [.124, .124], transform=fig.transFigure,
                     color='#DFE6EB', lw=.8))
fig.text(.075, .084, '实点：22 个宿主验证成绩   ·   阶梯线：本轮累计最佳   ·   末端短竖线：实际结束   ·   不含 tune 分数',
         fontsize=9, color='#627684')
fig.text(.075, .047, 'R5 / R6 因模型服务不可用提前结束，曲线不延长。已按要求移除 R4 展示；单次固定划分结果，非 Kaggle 官方成绩。',
         fontsize=9, color='#627684')
fig.savefig(OUT / 'leaf_search_without_r4.svg', facecolor='white')
fig.savefig(OUT / 'leaf_search_without_r4.pdf', facecolor='white')
fig.savefig(OUT / 'leaf_search_without_r4.png', dpi=300, facecolor='white')
plt.close(fig)
