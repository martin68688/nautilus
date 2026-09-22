"""Style-only revision: retain every certified score and every real stop time."""
from pathlib import Path
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
colors = ['#397FAC', '#C68A3A', '#8064AD', '#C3637E', '#188F8D']
markers = ['o', 's', 'D', '^', 'P']
styles = ['-', '-', '--', '-.', '-']
labels = ['R1 首轮', 'R2 第二轮', 'R3 第三轮重跑', 'R4 历史第四轮', 'R5 全记忆轮']

# Quantitative hero + subordinate detail; fixed search loss, one seed/split.
# Field mapping: round -> group; elapsed_minutes -> x; search_loss -> y.
# 35 node records -> 18 certified scores. Unscored nodes are not zero losses.
series = []
assert len(rows) == 35
for i, s in enumerate(summary, 1):
    good = [r for r in rows if int(r['round']) == i and r['certified'] == 'True']
    assert len(good) == [2, 2, 10, 1, 3][i - 1]
    x = [float(r['elapsed_minutes']) for r in good]
    y = [float(r['search_loss']) for r in good]
    assert all(math.isfinite(v) and v > 0 for v in y)
    assert x == sorted(x)
    best = [min(y[:j + 1]) for j in range(len(y))]
    assert abs(best[-1] - s['final_search']) < 1e-12
    assert s['elapsed_minutes'] >= x[-1]
    series.append((x, y, best))

fig = plt.figure(figsize=(13.2, 7.0), facecolor='white')
fig.text(.075, .928, 'Leaf  /  五轮 Search 成绩', fontsize=23)
fig.text(.075, .877, '固定 search log-loss · 越低越好', fontsize=11)
fig.text(.965, .877, 'Luna-low   ·   A10 × 1   ·   同一划分   ·   各轮独立计时',
         ha='right', fontsize=9, color='#718391')
handles = [Line2D([], [], color=c, marker=m, ls=ls, lw=2, ms=5, label=lab)
           for c, m, ls, lab in zip(colors, markers, styles, labels)]
fig.legend(handles=handles, loc='center left', bbox_to_anchor=(.067, .819),
           ncol=5, columnspacing=2.8, handlelength=2.4, fontsize=10)
fig.add_artist(Line2D([.075, .965], [.787, .787], transform=fig.transFigure,
                     color='#DFE6EB', lw=.8))

a = fig.add_axes([.075, .218, .565, .505])
b = fig.add_axes([.735, .402, .23, .321])
fig.text(.075, .749, 'a   全部五轮', fontsize=12)
fig.text(.640, .749, '纵轴为对数刻度', fontsize=9, ha='right', color='#718391')
fig.text(.735, .749, 'b   R4 / R5 局部放大', fontsize=12)
for ax in (a, b):
    ax.grid(axis='y', color='#E7EDF1', lw=.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.7)
for i, ((x, y, best), s, c, m, ls) in enumerate(zip(series, summary, colors, markers, styles)):
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
a.set(xlim=(0, 120), ylim=(.035, .88), xticks=range(0, 121, 20),
      xlabel='累计墙钟时间 / 分钟', ylabel='Search log-loss ↓')
a.yaxis.set_major_locator(FixedLocator([.04, .06, .1, .2, .4, .8]))
a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
a.yaxis.set_minor_locator(NullLocator())
a.annotate('R4', (summary[3]['elapsed_minutes'], summary[3]['final_search']),
           xytext=(2, 10), textcoords='offset points', fontsize=9, color=colors[3])
a.annotate('R5', (summary[4]['elapsed_minutes'], summary[4]['final_search']),
           xytext=(5, 8), textcoords='offset points', fontsize=9, color=colors[4])
b.set(xlim=(0, 75), ylim=(.04235, .0458), xticks=[0, 15, 30, 45, 60, 75],
      yticks=[.0425, .0435, .0445, .0455], xlabel='累计墙钟时间 / 分钟')
b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.4f}'))
b.annotate('R4', (summary[3]['elapsed_minutes'], summary[3]['final_search']),
           xytext=(3, 8), textcoords='offset points', color=colors[3], fontsize=9)
b.annotate('R5', (summary[4]['elapsed_minutes'], summary[4]['final_search']),
           xytext=(-1, 8), textcoords='offset points', ha='right', color=colors[4], fontsize=9)

# A compact aligned score ledger replaces floating endpoint callouts.
fig.text(.735, .296, '轮次', fontsize=9, color='#718391')
fig.text(.865, .296, '最佳 search', ha='right', fontsize=9, color='#718391')
fig.text(.965, .296, '实际结束', ha='right', fontsize=9, color='#718391')
for i, yy in [(3, .261), (4, .228)]:
    s = summary[i]
    fig.text(.735, yy, f'R{i+1}', fontsize=10, color=colors[i])
    fig.text(.865, yy, f"{s['final_search']:.6f}", ha='right', fontsize=10, color=colors[i])
    fig.text(.965, yy, f"{s['elapsed_minutes']:.1f} min", ha='right', fontsize=10)
delta = summary[3]['final_search'] - summary[4]['final_search']
fig.text(.735, .174, f'R5 相对下降 {delta / summary[3]["final_search"] * 100:.2f}%', fontsize=10)
fig.text(.735, .146, '差异很小，不代表稳定收益', fontsize=9, color='#718391')
fig.add_artist(Line2D([.075, .965], [.124, .124], transform=fig.transFigure,
                     color='#DFE6EB', lw=.8))
fig.text(.075, .084, '实点：18 个宿主验证成绩   ·   阶梯线：本轮累计最佳   ·   末端短竖线：实际结束   ·   不含 tune 分数',
         fontsize=9, color='#627684')
fig.text(.075, .047, 'R4 / R5 因模型服务不可用提前结束，曲线不延长；R5 是本图编号。单次固定划分结果，非 Kaggle 官方成绩。',
         fontsize=9, color='#627684')
fig.savefig(OUT / 'leaf_five_search_refined.svg', facecolor='white')
fig.savefig(OUT / 'leaf_five_search_refined.pdf', facecolor='white')
fig.savefig(OUT / 'leaf_five_search_refined.png', dpi=300, facecolor='white')
plt.close(fig)
