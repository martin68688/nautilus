from pathlib import Path
import json,csv
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
source=OUT.parent/'pp21_four_rounds_r4_repeat/source_data.csv'
rows=list(csv.DictReader(source.open()));points=[];runs=[]
for k in range(1,5):
 selected=[r for r in rows if int(r['round'])==k]
 p=Path(selected[0]['source']);root=next(x for x in p.parents if (x/'control/LAUNCH.json').exists())
 launch=json.loads((root/'control/LAUNCH.json').read_text());end=json.loads((root/'control/POD_PROCESS_OUTCOME.json').read_text())
 journal=json.loads(next(root.glob('runs/*/logs/journal.json')).read_text())
 wp=root/'control/MODEL_WAIT_CLOCK.events.jsonl'
 events=sorted([json.loads(x) for x in wp.read_text().splitlines()],key=lambda x:x['finished_unix']) if wp.exists() else []
 def paused(t):
  previous=0.
  for e in events:
   now=e['union_paused_seconds'];finish=e['finished_unix']
   if finish<=t:previous=now
   else:
    # Union-clock increments locate the next newly accounted pause interval.
    return previous+max(0.,min(now-previous,t-(finish-(now-previous))))
  return previous
 for row in selected:
  d=json.loads(Path(row['source']).read_text());assert d['checkpoint_validated'] and d['host_recomputed']
  n=next(n for n in journal['nodes'] if n['id']==row['node_id'])
  t=datetime.fromisoformat(n['finish_time']).replace(tzinfo=timezone.utc).timestamp()
  elapsed=(t-launch['started_unix']-paused(t))/60
  assert 0<elapsed<=end['elapsed_seconds']/60+.02
  points.append(dict(round=k,node_id=n['id'],micro_f1=d['micro_f1'],minutes=elapsed,finish_utc=n['finish_time'],clock='wall' if k==1 else 'effective_excluding_model_wait',source=row['source']))
 runs.append(dict(round=k,minutes=end['elapsed_seconds']/60,best=max(float(r['micro_f1']) for r in selected),returncode=end['returncode'],run_id=launch['run_id']))
with (OUT/'source_data.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
(OUT/'run_summary.json').write_text(json.dumps(runs,indent=2))
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial Unicode MS'],'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#CBD5DF','axes.labelcolor':'#52657A','xtick.color':'#708195','ytick.color':'#708195','text.color':'#26394B','svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})
colors=['#5685AB','#AA809C','#D19B45','#259C90'];markers=['o','D','s','P']
labels=['R1 · 无记忆','R2 · R1 记忆','R3 · R1–R2 记忆','R4 重跑 · R1–R3 记忆']
fig=plt.figure(figsize=(15,8.5),facecolor='white')
fig.text(.07,.94,'Plant Pathology',fontsize=26,weight='bold')
fig.text(.07,.89,'四轮记忆迭代 · Host 验证成绩',fontsize=17,color='#607489')
fig.text(.93,.935,'R4 使用重跑记录',ha='right',fontsize=11,color='#259C90',bbox=dict(boxstyle='round,pad=.6',fc='#EAF6F3',ec='none'))
fig.text(.07,.81,'01  成绩随运行推进',fontsize=14,weight='bold')
fig.text(.645,.81,'02  各轮最佳成绩',fontsize=14,weight='bold')
ax=fig.add_axes([.07,.36,.51,.39]);low=fig.add_axes([.07,.235,.51,.075],sharex=ax)
right=fig.add_axes([.73,.30,.17,.43])
for i,run in enumerate(runs):
 pts=sorted([p for p in points if p['round']==i+1],key=lambda p:p['minutes']);x=[p['minutes'] for p in pts];y=[p['micro_f1'] for p in pts];best=np.maximum.accumulate(y)
 ax.step(x+[run['minutes']],list(best)+[best[-1]],where='post',lw=2.4,color=colors[i],label=labels[i],alpha=.95)
 for xx,yy in zip(x,y):
  target=ax if yy>=.83 else low
  target.scatter(xx,yy,s=49,c=colors[i],marker=markers[i],edgecolors='white',linewidth=.8,zorder=4)
 ax.plot(run['minutes'],best[-1],marker='|',ms=11,color=colors[i])
 yy=3-i;right.hlines(yy,.83,run['best'],color=colors[i],lw=4,alpha=.85)
 right.scatter(run['best'],yy,s=80,color=colors[i],edgecolors='white',linewidth=1,zorder=3)
 right.text(run['best']+.0013,yy,f"{run['best']:.5f}",va='center',fontsize=12,color=colors[i],weight='bold')
 fy=.30+.43*(yy+.5)/4
 fig.text(.645,fy+.012,['R1 无记忆','R2 累计记忆','R3 累计记忆','R4 重跑'][i],fontsize=12,va='center')
 fig.text(.645,fy-.023,f"{run['minutes']:.1f} 分钟",fontsize=10,color='#8A99A7',va='center')
ax.set(xlim=(0,124),ylim=(.832,.869),ylabel='Host micro-F1 ↑')
ax.set_yticks([.835,.845,.855,.865]);ax.grid(axis='y',color='#E8EEF3',lw=.9)
ax.tick_params(axis='x',labelbottom=False,bottom=False);ax.spines['bottom'].set_visible(False)
low.set(ylim=(.708,.725),xlabel='本轮预算计时 / 分钟');low.set_yticks([.715]);low.set_xticks([0,30,60,90,120]);low.grid(axis='y',color='#E8EEF3');low.spines['top'].set_visible(False)
low.text(5,.7161,'保留低分候选',fontsize=10,color='#8392A1',va='center')
low.annotate('EfficientNetV2-S  0.71610',xy=(103.0443,.7161004),xytext=(70,.720),fontsize=10,color=colors[3],arrowprops=dict(arrowstyle='-',color=colors[3],lw=.8))
# Explicit axis break preserves the low observation while enlarging the principal score range.
for axis,yedge in [(ax,0),(low,1)]:
 axis.plot([-.008,.008],[yedge-.014,yedge+.014],transform=axis.transAxes,color='#A4B2BF',clip_on=False,lw=1)
ax.legend(loc='center left',ncol=2,fontsize=10,columnspacing=1.5,handlelength=2.4)
right.set(xlim=(.83,.879),ylim=(-.5,3.5));right.set_xticks([.83,.85,.87]);right.set_yticks([]);right.spines['left'].set_visible(False);right.set_xlabel('最佳 Host micro-F1',labelpad=10)
right.grid(axis='x',color='#EEF2F5',zorder=0)
fig.text(.645,.207,'R3 最高；R4 重跑接近 R3',fontsize=12,color='#607489')
fig.text(.07,.137,'+2.22 个百分点',fontsize=20,weight='bold',color='#259C90')
fig.text(.07,.099,'R4 重跑相比无记忆 R1',fontsize=11,color='#708195')
fig.text(.36,.137,'−0.24 个百分点',fontsize=20,weight='bold',color='#607489')
fig.text(.36,.099,'R4 重跑相比 R3',fontsize=11,color='#708195')
fig.text(.645,.139,'13 个真实认证候选',fontsize=15,color='#344E63')
fig.text(.645,.100,'实点 = 候选成绩   阶梯 = 轮内累计最佳',fontsize=10,color='#708195')
fig.text(.07,.047,'纵轴断轴以同时展示主成绩区间与低分候选。R1 为墙钟时间；R2–R4 扣除 LLM 等待。',fontsize=9,color='#8898A7')
fig.text(.07,.022,'仅含 Host 认证分数；R3 预算看门狗收尾，成绩有效。运行配置与计时存在差异，此图不单独证明记忆的因果收益。',fontsize=9,color='#8898A7')
fig.savefig(OUT/'four_rounds_polished.png',dpi=300)
fig.savefig(OUT/'four_rounds_polished.svg')
fig.savefig(OUT/'four_rounds_polished.pdf')
