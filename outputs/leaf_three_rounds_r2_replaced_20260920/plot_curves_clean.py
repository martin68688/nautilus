"""Clean Leaf three-round outer search-loss chart; source data are unchanged."""
from pathlib import Path
import csv, json, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
rows = list(csv.DictReader((OUT / 'source_data_replaced.csv').open()))
summary = json.loads((OUT / 'summary_replaced.json').read_text())['rounds']
for row in rows:
    row['round'] = int(row['round'])
    row['elapsed_minutes'] = float(row['elapsed_minutes'])
    row['certified'] = row['certified'].lower() == 'true'
    row['search_logloss'] = float(row['search_logloss']) if row['search_logloss'] else None

font_path = Path('/System/Library/Fonts/STHeiti Light.ttc')
if font_path.exists():
    font_manager.fontManager.addfont(font_path)
    cjk_name = font_manager.FontProperties(fname=font_path).get_name()
else:
    cjk_name = 'DejaVu Sans'
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': [cjk_name, 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.edgecolor': '#B7C5D0',
    'axes.labelcolor': '#304454',
    'text.color': '#304454',
    'xtick.color': '#4F6274',
    'ytick.color': '#4F6274',
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'axes.unicode_minus': False,
    'legend.frameon': False,
})
colors = {1: '#377EAD', 2: '#C65E7C', 3: '#15958C'}
markers = {1: 'o', 2: 's', 3: 'D'}

fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor='white')
fig.subplots_adjust(left=0.095, right=0.96, bottom=0.14, top=0.84)
fig.text(0.095, 0.925, 'Leaf / 三轮 search loss', fontsize=25, color='#23333D')

for round_id in (1, 2, 3):
    rr = sorted((x for x in rows if x['round'] == round_id and x['certified']),
                key=lambda x: x['elapsed_minutes'])
    xs = [x['elapsed_minutes'] for x in rr]
    ys = [x['search_logloss'] for x in rr]
    best = [min(ys[:i + 1]) for i in range(len(ys))]
    end = summary[round_id - 1]['elapsed_minutes']
    color = colors[round_id]
    ax.step(xs + [end], best + [best[-1]], where='post', color=color, linewidth=2.8)
    ax.scatter(xs, ys, marker=markers[round_id], s=54, facecolors='white',
               edgecolors=color, linewidths=1.7, zorder=3)
    ax.plot(end, best[-1], marker='|', color=color, markersize=13, markeredgewidth=1.8)
    offsets = {1: (-7, 14), 2: (-7, -22), 3: (-7, 12)}
    ax.annotate(f'R{round_id}  {best[-1]:.5f}', (end, best[-1]),
                xytext=offsets[round_id], textcoords='offset points', ha='right',
                color=color, fontsize=12, weight='bold')

ax.set_xlim(0, 120)
ax.set_ylim(0, 0.8)
ax.set_xticks(range(0, 121, 20))
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax.set_xlabel('运行时间 / 分钟')
ax.set_ylabel('Host search log-loss ↓')
ax.grid(axis='y', color='#E4ECF2', linewidth=0.8)
ax.set_axisbelow(True)

handles = [Line2D([0], [0], color=colors[i], linewidth=2.8, marker=markers[i],
                  markerfacecolor='white', markeredgecolor=colors[i], markersize=7,
                  label=f'R{i}') for i in (1, 2, 3)]
ax.legend(handles=handles, loc='upper right', ncol=3, fontsize=11,
          handlelength=2.2, columnspacing=1.8)

fig.savefig(OUT / 'leaf_three_rounds_r2_replaced_clean.png', dpi=300, facecolor='white')
fig.savefig(OUT / 'leaf_three_rounds_r2_replaced_clean.svg', facecolor='white')
fig.savefig(OUT / 'leaf_three_rounds_r2_replaced_clean.pdf', facecolor='white')
plt.close(fig)
print(json.dumps({'rounds': summary}, ensure_ascii=False, indent=2))
