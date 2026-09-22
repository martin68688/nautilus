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
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial Unicode MS','PingFang SC','Arial'],'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#BBCBD4','axes.labelcolor':'#435B6B','xtick.color':'#708797','ytick.color':'#708797','text.color':'#304958','svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})
colors=['#3C85AF','#BD6781','#C59240','#239586'];markers=['o','D','s','P']
labels=['R1 无记忆','R2 / R1 记忆','R3 / R1–R2 记忆','R4 重跑 / R1–R3 记忆']
fig=plt.figure(figsize=(15,9),facecolor='white');ax=fig.add_axes([.075,.275,.48,.53]);right=fig.add_axes([.765,.25,.135,.555])
fig.text(.065,.943,'Plant Pathology / 前四轮运行成绩对比',fontsize=26)
fig.text(.065,.889,'真实测量点 + 累计最佳阶梯线；只比较 Host 验证分数，不混入 tune 或官方分数',fontsize=12,color='#708797')
fig.text(.075,.838,'a  运行轨迹 · Host search',fontsize=15)
fig.text(.62,.838,'b  各轮最佳 Host F1',fontsize=15)
for i,run in enumerate(runs):
 pts=sorted([p for p in points if p['round']==i+1],key=lambda p:p['minutes']);x=[p['minutes'] for p in pts];y=[p['micro_f1'] for p in pts];best=np.maximum.accumulate(y)
 ax.step(x+[run['minutes']],list(best)+[best[-1]],where='post',lw=2,color=colors[i],label=labels[i])
 ax.scatter(x,y,s=35,c=colors[i],marker=markers[i],zorder=3)
 ax.plot(run['minutes'],best[-1],marker='|',ms=11,color=colors[i])
 yy=3-i;right.hlines(yy,0,run['best'],color=colors[i],lw=2)
 right.scatter(run['best'],yy,s=45,color=colors[i],marker='D' if i==3 else 'o',zorder=3)
 right.text(run['best']+.022,yy,f"{run['best']:.5f}",va='center',fontsize=11,color=colors[i])
 fy=fig.transFigure.inverted().transform(right.transData.transform((0,yy)))[1] if False else 0
 fig.text(.62,.25+.555*(yy+.7)/4.4,labels[i],fontsize=11,va='center')
 fig.text(.98,.25+.555*(yy+.7)/4.4,f"{run['minutes']:.1f} min",fontsize=10,color='#8496A2',va='center',ha='right')
ax.set(xlim=(0,123),ylim=(.70,.89),xlabel='本轮预算计时 / 分钟',ylabel='Host search micro-F1 ↑')
ax.set_xticks([0,30,60,90,120]);ax.set_yticks([.70,.75,.80,.85]);ax.grid(axis='y',color='#E5ECF1',lw=.8)
ax.legend(loc='lower left',fontsize=10)
right.set(xlim=(0,1.14),ylim=(-.7,3.7),xlabel='最佳 Host micro-F1 ↑');right.set_xticks([0,.3,.6,.9]);right.set_yticks([]);right.spines['left'].set_visible(False)
fig.text(.075,.183,'R4 重跑 0.86077   |   R3 最佳 0.86316   |   最初 R1 无记忆 0.83857',fontsize=16,color=colors[3])
fig.text(.075,.116,'实点为 13 个认证候选；阶梯为轮内累计最高分；竖线为实际结束。出分前留空，失败不伪装成 0 分。',fontsize=10,color='#708797')
fig.text(.075,.080,'R1 为墙钟时间；R2–R4 扣除 LLM 等待。点位取候选完成时间，并按等待日志还原预算计时。',fontsize=10,color='#708797')
fig.text(.075,.044,'R3 由预算看门狗收尾（exit 143），认证成绩保留；R4 使用重跑。框架修复与计时口径不同，不能单独归因于记忆。',fontsize=10,color='#708797')
fig.savefig(OUT/'four_rounds.png',dpi=300)
fig.savefig(OUT/'four_rounds.svg')
fig.savefig(OUT/'four_rounds.pdf')
print('Rendered 13 verified measurements across four rounds')
