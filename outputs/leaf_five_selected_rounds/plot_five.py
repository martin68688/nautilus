"""Selected five-run comparison from persisted observations only."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import math
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

OUT = Path(__file__).resolve().parent
BASE = OUT.parents[1]
OLD = OUT.parent / 'leaf_three_rounds_inner'
EXP = BASE / 'experiments/leaf_round4_memory_luna_2h_20260913'
hashes = {}
def load(p):
    raw = p.read_bytes()
    hashes[str(p.relative_to(BASE))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)
def rows(p):
    hashes[str(p.relative_to(BASE))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return list(csv.DictReader(p.open()))
def utc(s):
    d = datetime.fromisoformat(s)
    return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp()
outer = rows(OLD / 'outer_search.csv')
inner = rows(OLD / 'inner_tune.csv')
segments = rows(OLD / 'node_segments.csv')
summary = load(OLD / 'summary.json')['rounds']
for group in (outer, inner, segments):
    for row in group:
        for k in ['round', 'node', 'trial_in_node', 'expanded_slot', 'category', 'trial_count']:
            if k in row: row[k] = int(row[k])
        for k in ['elapsed_minutes', 'search_loss', 'tune_loss', 'best_tune', 'start', 'end']:
            if k in row and row[k] != '': row[k] = float(row[k])
        if 'certified' in row: row['certified'] = row['certified'] == 'True'

historic = load(EXP / 'evidence/TERMINAL_DIAGNOSIS.json')
source = load(EXP / 'all_memory_luna_2h_20260914_0458/SOURCE_EVIDENCE.json')['sources'][0]['runs'][0]
latest = load(EXP / 'all_memory_luna_2h_20260914_0458/TERMINAL.json')
old_launch = load(BASE / 'experiments/leaf_r6_nomemory_luna_3h_20260911/LAUNCH_2H.json')
configs = [
    (historic['control/LAUNCH.json'], historic['control/POD_PROCESS_OUTCOME.json'], source),
    (latest['controls']['LAUNCH.json'], latest['controls']['POD_PROCESS_OUTCOME.json'], latest),
]
for ri, (launch, end, run) in enumerate(configs, 4):
    assert launch['data_manifest_sha256'] == old_launch['data_manifest_sha256']
    assert run['core']['FINAL_SNAPSHOT.json']['consistent']
    duration = end['elapsed_seconds'] / 60
    nodes = [n for n in run['core']['journal.json']['nodes'] if n.get('stage') != 'root' and n.get('finish_time')]
    byid = {n['id']: n for n in run['nodes']}
    slot = 0
    seen = []
    counts = [0, 0, 0]
    for ni, node in enumerate(nodes, 1):
        raw = byid.get(node['id'], {})
        val = node.get('metric', {}).get('value')
        valid = bool(node.get('is_valid')) and isinstance(val, (int, float)) and math.isfinite(val)
        if valid:
            verified = raw['VERIFIED_RESULT.json']
            assert verified['host_recomputed'] and verified['checkpoint_validated']
            assert abs(verified['metric'] - val) < 1e-12
        elapsed = (utc(node['finish_time']) - launch['started_unix']) / 60
        assert 0 <= elapsed <= duration + .1
        outer.append(dict(round=ri, node=ni, node_id=node['id'], elapsed_minutes=elapsed, certified=valid, search_loss=val if valid else ''))
        trials = raw.get('trials', [])
        start = slot + .5
        for ti, t in enumerate(trials, 1):
            assert t['status'] == 'ok'
            loss = t['tune_loss']
            assert math.isfinite(loss) and loss > 0
            slot += 1
            cat = 0 if not seen else 1 if loss < min(seen) - 1e-12 else 2
            counts[cat] += 1
            seen.append(loss)
            inner.append(dict(round=ri, node=ni, node_id=node['id'], trial_in_node=ti, expanded_slot=slot,
                              step=t['trial_id'], tune_loss=loss, best_tune=min(seen), category=cat,
                              parameters=json.dumps(t['parameters'], ensure_ascii=False), elapsed_node_seconds='', elapsed_run_minutes=''))
        if not trials: slot += 1
        segments.append(dict(round=ri, node=ni, node_id=node['id'], start=start, end=slot+.5, trial_count=len(trials), certified=valid))
    good = [o for o in outer if o['round'] == ri and o['certified']]
    summary.append(dict(round=ri, nodes=len(nodes), certified=len(good), inner_trials=len(seen),
                        final_search=min(o['search_loss'] for o in good), final_tune=min(seen), elapsed_minutes=duration,
                        slots=slot, categories=counts, run_id=run['run_id'], stop_reason=run['core']['RUN_OUTCOME.json']['reason']))
assert [s['inner_trials'] for s in summary] == [11, 3, 41, 4, 11]
assert [s['nodes'] for s in summary] == [14, 2, 11, 1, 7]
assert [s['certified'] for s in summary] == [2, 2, 10, 1, 3]
for name, data in [('outer_search.csv', outer), ('inner_tune.csv', inner), ('node_segments.csv', segments)]:
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
(OUT / 'summary.json').write_text(json.dumps({'rounds': summary, 'source_sha256': hashes}, ensure_ascii=False, indent=2))

font = Path('/System/Library/Fonts/STHeiti Light.ttc')
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': [font_manager.FontProperties(fname=font).get_name(), 'Arial', 'DejaVu Sans'],
 'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 12, 'axes.titlepad': 12,
 'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#BBC8D4',
 'axes.labelcolor': '#304454', 'text.color': '#304454', 'xtick.color': '#4F6274', 'ytick.color': '#4F6274',
 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.unicode_minus': False, 'legend.frameon': False})
colors = ['#327EB4', '#CC842C', '#795AAF', '#B95773', '#248E8B']
markers = ['o', 's', 'D', '^', 'P']
styles = ['-', '-', '--', '-.', '-']
labels = ['R1 首轮', 'R2 第二轮', 'R3 第三轮重跑', 'R4 历史第四轮', 'R5 最新全记忆轮']
fig = plt.figure(figsize=(17, 12), facecolor='white')
fig.text(.07, .955, 'Leaf · 五轮成绩曲线对比（展开节点内部调参）', fontsize=24)
fig.text(.07, .919, '新增历史第四轮与最新全记忆轮：最佳 search 0.042718 → 0.042636，仅小幅降低 0.19%', fontsize=13)
fig.text(.07, .886, 'Luna-low · A10 × 1 · 同一固定划分 · 各轮独立计时 · R5 为本图编号，并非第五次启动', fontsize=10.5, color='#71859A')
handles = [Line2D([0], [0], color=col, marker=mk, ls=ls, lw=2,
    label=f"{label}（{s['nodes']} 节点 / {s['certified']} 认证 / {s['inner_trials']} 内部观测）")
    for s, col, mk, ls, label in zip(summary, colors, markers, styles, labels)]
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.065, .866), ncol=3, columnspacing=1.5, fontsize=9.5)
a = fig.add_axes([.07, .545, .57, .24]); b = fig.add_axes([.745, .545, .225, .24])
c = fig.add_axes([.07, .24, .57, .215]); d = fig.add_axes([.775, .24, .195, .215])
strip = fig.add_axes([.07, .113, .57, .095], sharex=c)
for ax in [a, b, c, d]:
    ax.grid(axis='x' if ax == d else 'y', color='#E6EDF3', lw=.7); ax.set_axisbelow(True)
a.set_title('a  实际耗时：host search，点为实测、阶梯线为累计最佳', loc='left')
b.set_title('b  后两轮 search 区域放大', loc='left')
c.set_title('c  全部内部训练／调参展开：tune 分数（对数轴）', loc='left')
d.set_title('d  内部观测结果组成', loc='left')
for ri, (s, col, mk, ls) in enumerate(zip(summary, colors, markers, styles), 1):
    good = [r for r in outer if r['round'] == ri and r['certified']]
    xs = [r['elapsed_minutes'] for r in good]; ys = [r['search_loss'] for r in good]
    best = [min(ys[:i+1]) for i in range(len(ys))]
    for ax in ([a, b] if ri >= 4 else [a]):
        ax.step(xs+[s['elapsed_minutes']], best+[best[-1]], where='post', color=col, ls=ls, lw=2)
        ax.scatter(xs, ys, color=col, marker=mk, s=29, alpha=.85, zorder=4)
        ax.plot(s['elapsed_minutes'], best[-1], marker='|', color=col, ms=9)
    if ri <= 3:
        a.annotate(f"R{ri}  {best[-1]:.6f}", (s['elapsed_minutes'], best[-1]),
                   xytext=(-5, 12 if ri==1 else -18), textcoords='offset points', ha='right', color=col, fontsize=9)
    ir = [r for r in inner if r['round'] == ri]
    x = [r['expanded_slot'] for r in ir]; y = [r['tune_loss'] for r in ir]
    c.step(x+[s['slots']+.5], [r['best_tune'] for r in ir]+[ir[-1]['best_tune']], where='post', color=col, ls=ls, lw=2)
    c.scatter(x, y, color=col, marker=mk, s=24, alpha=.75, zorder=4)
    if ri == 2:
        c.plot([s['slots']+.5, 42.5], [s['final_tune']]*2, color=col, ls='--', lw=1.6)
    for seg in [r for r in segments if r['round'] == ri]:
        level = 6-ri; mid = (seg['start']+seg['end'])/2
        strip.plot([seg['start'], seg['end']], [level]*2, color=col, lw=1, marker='|', ms=7)
        if seg['trial_count']:
            name = f"N{seg['node']} · {seg['trial_count']}" if seg['trial_count'] >= 3 else f"N{seg['node']}"
            strip.text(mid, level+.15, name, ha='center', va='bottom', fontsize=7.2, color=col)
        else: strip.scatter(mid, level, marker='x', s=20, color=col)
    for gi, count in enumerate(s['categories']):
        yy = (2-gi)*1.5 + (3-ri)*.22
        d.barh(yy, count, height=.185, color=col)
        d.text(count+.25, yy, str(count), va='center', color=col, fontsize=9)
a.set(xlim=(0,120), ylim=(0,.8), xticks=range(0,121,20), yticks=[0,.2,.4,.6,.8],
      xlabel='累计墙钟时间 / 分钟（含初始化、规划、训练与评分）', ylabel='固定 search log-loss ↓')
a.text(.025, .20, '后两轮低损失区域见右侧放大图 →', transform=a.transAxes, fontsize=9, color='#71859A')
b.set(xlim=(0,75), ylim=(.0423,.0462), xticks=[0,15,30,45,60,75], xlabel='累计墙钟时间 / 分钟', ylabel='固定 search log-loss ↓')
b.set_yticks([.0425,.0435,.0445,.0455])
b.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:.4f}'))
b.annotate('R4  0.042718\n13.3 分钟提前停止', (summary[3]['elapsed_minutes'], summary[3]['final_search']),
           xytext=(8,.0434), fontsize=9, color=colors[3], arrowprops=dict(arrowstyle='-', color=colors[3]))
b.annotate('R5  0.042636\n68.0 分钟提前停止', (summary[4]['elapsed_minutes'], summary[4]['final_search']),
           xytext=(41,.0434), fontsize=9, color=colors[4], arrowprops=dict(arrowstyle='-', color=colors[4]))
b.text(.03,.94,'差值 0.0000816（相对下降 0.19%）', transform=b.transAxes, va='top', fontsize=8.7)
b.text(0,-.23,'两轮均因模型服务不可用退出，非预算耗尽。', transform=b.transAxes, fontsize=9, color='#71859A')
assert all(math.isfinite(r['tune_loss']) and r['tune_loss'] > 0 for r in inner)
c.set_yscale('log'); c.set(xlim=(.4,43), ylim=(.0007,3.2), ylabel='tune log-loss ↓（对数刻度）')
c.yaxis.set_major_locator(FixedLocator([.001,.01,.1,1])); c.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
c.tick_params(axis='x', labelbottom=False)
annotation = [(23.5,.401943,6,6,'left'), (42.5,.355183,-5,-17,'right'), (42.5,.000934,-5,12,'right'), (4.5,.019554,6,12,'left'), (15.5,.018213,90,6,'left')]
for i,(x,y,dx,dy,ha) in enumerate(annotation):
    suffix = '（延长展示）' if i==1 else ''
    c.annotate(f"R{i+1} 最低 tune {summary[i]['final_tune']:.6f}{suffix}", (x,y), xytext=(dx,dy),
               textcoords='offset points', ha=ha, color=colors[i], fontsize=8.3)
strip.set(ylim=(.7,5.8), yticks=[5,4,3,2,1], yticklabels=['R1','R2','R3','R4','R5'],
          xticks=[1,5,10,15,20,25,30,35,40], xlabel='展开后的试验序号（线段为节点；无内部成绩的节点占一格，标 ×）')
strip.tick_params(axis='y',length=0); strip.spines['left'].set_visible(False); strip.xaxis.label.set_fontsize(10)
d.set(yticks=[3,1.5,0], yticklabels=['首个有效观测','刷新本轮最佳 tune','未刷新本轮最佳 tune'],
      xlabel='内部阶段／试验数', xlim=(0,24), ylim=(-.7,3.7), xticks=[0,5,10,15,20])
d.tick_params(axis='y',length=0,labelsize=9)
d.text(0,-.32,'按内部观测计数，不是外层节点数。\nR1 为训练阶段；R2 为训练；R3–R5 为 Agent 调参。', transform=d.transAxes, fontsize=8.7, color='#71859A')
fig.text(.07,.054,'读图：a、b 为 search；c、d 为 tune，不能混比。70 个内部实测点全部保留；最低 tune 不一定对应最佳 search。',fontsize=10,color='#71859A')
fig.text(.07,.025,'R2 橙色虚线仅为延长展示，其余曲线止于真实结束。单次自适应搜索，微小差距不代表稳定收益；非 Kaggle 官方成绩。',fontsize=10,color='#71859A')
fig.savefig(OUT/'leaf_five_selected_rounds.svg', facecolor='white')
fig.savefig(OUT/'leaf_five_selected_rounds.pdf', facecolor='white')
fig.savefig(OUT/'leaf_five_selected_rounds.png', dpi=300, facecolor='white')
plt.close(fig)
print(json.dumps(summary,ensure_ascii=False,indent=2))
