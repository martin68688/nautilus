"""Leaf three-round outer Host search-loss curve with the latest R2 replacing the old R2."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD_DIR = ROOT / 'outputs' / 'leaf_three_rounds'
OLD_SOURCE = OLD_DIR / 'source_data.csv'
OLD_SUMMARY = OLD_DIR / 'summary.json'
R2_SNAPSHOT = OUT / 'current_r2_remote_snapshot.json'
PREFLIGHT = ROOT / 'experiments' / 'leaf_r2_autoresearch_rerun_20260920' / 'PREFLIGHT.json'

# Data contract: preserve old R1/R3 observations; replace every old R2 row
# with the terminal current R2 history. A score is plotted only when Host
# produced a finite certified metric; failed proposals remain as crosses.
old_rows = list(csv.DictReader(OLD_SOURCE.open()))
old_summary = json.loads(OLD_SUMMARY.read_text())
snapshot = json.loads(R2_SNAPSHOT.read_text())
preflight = json.loads(PREFLIGHT.read_text())

rows = []
for row in old_rows:
    if int(row['round']) != 2:
        rr = dict(row)
        rr['round'] = int(rr['round'])
        rr['elapsed_minutes'] = float(rr['elapsed_minutes'])
        rr['certified'] = rr['certified'].lower() == 'true'
        rr['search_logloss'] = float(rr['search_logloss']) if rr['search_logloss'] else None
        rr['status'] = 'certified' if rr['certified'] else 'failed_or_unscored'
        rows.append(rr)

start = snapshot['clock']['started']
latest_best = math.inf
for history in snapshot['state']['history']:
    score = history.get('score')
    valid = isinstance(score, (int, float)) and math.isfinite(score)
    elapsed = (history['finished'] - start) / 60.0
    assert elapsed >= 0
    rows.append({
        'round': 2,
        'label': 'R2（最新重跑）',
        'attempt': int(history['attempt'].split('-')[-1]),
        'node_id': history['attempt'],
        'elapsed_minutes': elapsed,
        'certified': valid,
        'search_logloss': float(score) if valid else None,
        'best_so_far': None,
        'finished_utc': datetime.fromtimestamp(history['finished'], tz=timezone.utc).isoformat(),
        'status': history.get('status', 'failed') if valid else 'failed_or_unscored',
    })

for round_id in (1, 2, 3):
    best = math.inf
    for row in sorted((x for x in rows if x['round'] == round_id), key=lambda x: x['elapsed_minutes']):
        if row['certified']:
            best = min(best, row['search_logloss'])
            row['best_so_far'] = best
        else:
            row['best_so_far'] = best if math.isfinite(best) else None

# Deterministic CSV source for the figure.
fieldnames = ['round', 'label', 'attempt', 'node_id', 'elapsed_minutes', 'certified',
              'search_logloss', 'best_so_far', 'finished_utc', 'status']
with (OUT / 'source_data_replaced.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in sorted(rows, key=lambda x: (x['round'], x['elapsed_minutes'])):
        writer.writerow({k: row.get(k) for k in fieldnames})

old_by_round = {x['round']: x for x in old_summary['rounds']}
summary = []
for round_id in (1, 2, 3):
    rr = [x for x in rows if x['round'] == round_id]
    good = [x for x in rr if x['certified']]
    if round_id == 2:
        duration = snapshot['supervisor']['process']['elapsed'] / 60.0
    else:
        duration = old_by_round[round_id]['elapsed_minutes']
    summary.append({
        'round': round_id,
        'label': 'R1（无记忆）' if round_id == 1 else ('R2（R1记忆，最新重跑）' if round_id == 2 else 'R3（累计记忆）'),
        'elapsed_minutes': duration,
        'registered': len(rr),
        'certified': len(good),
        'best_search_loss': min(x['search_logloss'] for x in good),
    })
assert [x['registered'] for x in summary] == [14, 11, 11]
assert [x['certified'] for x in summary] == [2, 6, 10]
assert abs(summary[1]['best_search_loss'] - 0.06356954282842144) < 1e-12

source_hashes = {
    'old_source_data_csv_sha256': hashlib.sha256(OLD_SOURCE.read_bytes()).hexdigest(),
    'old_summary_json_sha256': hashlib.sha256(OLD_SUMMARY.read_bytes()).hexdigest(),
    'current_r2_remote_snapshot_sha256': hashlib.sha256(R2_SNAPSHOT.read_bytes()).hexdigest(),
    'current_r2_preflight_sha256': hashlib.sha256(PREFLIGHT.read_bytes()).hexdigest(),
    'current_r2_data_manifest_sha256': preflight['input_manifests']['public_manifest_sha256'],
}
(OUT / 'summary_replaced.json').write_text(json.dumps({
    'figure': 'Leaf three-round outer Host search loss',
    'metric': 'fixed search multiclass log-loss (lower is better)',
    'rounds': summary,
    'source_hashes': source_hashes,
    'r2_scope': {'framework': preflight['framework'], 'memory_scope': 'R1-only',
                 'memory_cards': {k: v['cards'] for k, v in preflight['memory'].items()}, 'model': preflight['model'],
                 'endpoint': preflight['endpoint']},
}, indent=2, ensure_ascii=False) + '\n')

# Publication-readable but report-oriented layout; all exports are Python/matplotlib.
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
styles = {1: '--', 2: '-', 3: '-'}

fig = plt.figure(figsize=(12.8, 8.2), facecolor='white')
fig.text(0.08, 0.955, 'Leaf / 三轮外层 search loss', fontsize=24, color='#23333D')
fig.text(0.08, 0.918, '用最新 R2 替换旧 R2 · 同一固定 search 划分 · log-loss 越低越好',
         fontsize=12.5, color='#71859A')
fig.text(0.88, 0.952, '3 轮 / 37 个注册候选', ha='right', fontsize=11, color='#71859A')

ax = fig.add_axes([0.09, 0.40, 0.83, 0.43])
strip = fig.add_axes([0.09, 0.235, 0.83, 0.105], sharex=ax)
ax.grid(axis='y', color='#E4ECF2', linewidth=0.8)
ax.set_axisbelow(True)

for round_id in (1, 2, 3):
    rr = sorted((x for x in rows if x['round'] == round_id), key=lambda x: x['elapsed_minutes'])
    good = [x for x in rr if x['certified']]
    color = colors[round_id]
    xs = [x['elapsed_minutes'] for x in good]
    ys = [x['search_logloss'] for x in good]
    bests = [min(ys[:i + 1]) for i in range(len(ys))]
    end = summary[round_id - 1]['elapsed_minutes']
    ax.step(xs + [end], bests + [bests[-1]], where='post', color=color,
            linestyle=styles[round_id], linewidth=2.7, zorder=2)
    ax.scatter(xs, ys, marker=markers[round_id], s=48, facecolors='white',
               edgecolors=color, linewidths=1.7, zorder=4)
    ax.plot(end, bests[-1], marker='|', color=color, markersize=12, markeredgewidth=1.8)
    offsets = {1: (-6, 12), 2: (-6, -19), 3: (-6, 12)}
    ax.annotate(f'R{round_id}  {bests[-1]:.5f}', (end, bests[-1]),
                xytext=offsets[round_id], textcoords='offset points', ha='right',
                color=color, fontsize=11, weight='bold')
    level = {1: 3, 2: 2, 3: 1}[round_id]
    strip.hlines(level, 0, end, color=color, alpha=0.22, linewidth=1.2)
    for x in rr:
        if x['certified']:
            strip.scatter(x['elapsed_minutes'], level, marker=markers[round_id], s=32,
                          facecolors='white', edgecolors=color, linewidths=1.1, zorder=4)
        else:
            strip.scatter(x['elapsed_minutes'], level, marker='x', s=30,
                          color=color, linewidths=1.1, zorder=3)
    strip.plot(end, level, marker='|', color=color, markersize=11, markeredgewidth=1.5)

ax.set_xlim(0, 120)
ax.set_ylim(0, 0.8)
ax.set_ylabel('固定 search log-loss ↓')
ax.set_xticks(range(0, 121, 20))
ax.tick_params(axis='x', labelbottom=False, bottom=False)
ax.spines['bottom'].set_visible(False)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])

strip.set_ylim(0.5, 3.5)
strip.set_yticks([3, 2, 1])
strip.set_yticklabels(['R1 · 无记忆', 'R2 · 仅 R1 记忆（最新）', 'R3 · 累计记忆'])
strip.set_xticks(range(0, 121, 20))
strip.set_xlabel('距各轮启动的时间 / 分钟（初始化、规划、训练与评分均计入）')
strip.tick_params(axis='y', length=0, labelsize=9)
strip.spines['left'].set_visible(False)
strip.grid(axis='x', color='#EEF3F7', linewidth=0.6)

handles = [Line2D([0], [0], color=colors[i], linestyle=styles[i], linewidth=2.5,
                   marker=markers[i], markerfacecolor='white', markeredgecolor=colors[i],
                   markersize=6, label=f"{summary[i-1]['label']} · {summary[i-1]['certified']} 个认证")
           for i in (1, 2, 3)]
handles += [Line2D([0], [0], marker='x', color='#71859A', linestyle='None', markersize=6,
                   label='未认证／失败候选')]
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.085, 0.895),
           ncol=4, fontsize=9.5, handlelength=2.3, columnspacing=1.6)

# Compact summary cards, deliberately limited to decision-relevant values.
card_y = 0.075
card_xs = [0.09, 0.37, 0.65]
for x, s, color in zip(card_xs, summary, [colors[1], colors[2], colors[3]]):
    fig.text(x, card_y + 0.055, s['label'], fontsize=10.5, color='#71859A')
    fig.text(x, card_y + 0.005, f"{s['best_search_loss']:.5f}", fontsize=23, color=color)
    fig.text(x + 0.15, card_y + 0.012,
             f"{s['elapsed_minutes']:.1f} min · {s['certified']}/{s['registered']} 认证",
             fontsize=9.5, color='#71859A')
fig.text(0.09, 0.028,
         '阶梯线＝认证候选的累计最佳；空心点＝真实 Host 分数；×＝未认证或生成失败。非 Kaggle 官方成绩。',
         fontsize=9.5, color='#71859A')

fig.savefig(OUT / 'leaf_three_rounds_r2_replaced.png', dpi=300, facecolor='white', bbox_inches='tight')
fig.savefig(OUT / 'leaf_three_rounds_r2_replaced.svg', facecolor='white', bbox_inches='tight')
fig.savefig(OUT / 'leaf_three_rounds_r2_replaced.pdf', facecolor='white', bbox_inches='tight')
plt.close(fig)

notes = f'''# Leaf 三轮曲线（最新 R2 替换版）

核心结论：新 R2 的最佳 Host search loss 为 0.06356954；图中 R1、R3 保留原记录，R2 完全替换为 2026-09-20 的新 autoresearch R2。

- 指标：固定 search 集 multiclass log-loss，越低越好。
- R1：14 个注册候选，2 个认证，最佳 0.44389385。
- R2：11 个注册候选，6 个认证，最佳 0.06356954；仅 R1 记忆（18 张 search 卡 + 29 张 inner 卡）。
- R3：11 个注册候选，10 个认证，最佳 0.06677362。
- 仅认证且有限的 Host 分数进入曲线；失败候选保留为 ×，不赋值为 0。
- 认证候选是顺序搜索观测，不是独立重复试验；图中不做显著性或因果记忆结论。

数据来源哈希：
- old source_data.csv: {source_hashes['old_source_data_csv_sha256']}
- current R2 remote snapshot: {source_hashes['current_r2_remote_snapshot_sha256']}
- current R2 data manifest: {source_hashes['current_r2_data_manifest_sha256']}

后端：Python/matplotlib；输出为可编辑 SVG/PDF 和 300 dpi PNG 预览。
'''
(OUT / 'FIGURE_NOTES.md').write_text(notes)
print(json.dumps(summary, ensure_ascii=False, indent=2))
