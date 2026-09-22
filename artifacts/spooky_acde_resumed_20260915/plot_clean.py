from pathlib import Path
import csv,json,tarfile,math,hashlib
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
# Figure contract: show all observed inner trials and independently scored host results.
# Single-seed descriptive trajectories; no causal memory claim or uncertainty estimates.
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial Unicode MS','DejaVu Sans'],'font.size':12,'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#CDD9E2','text.color':'#2D4351','axes.labelcolor':'#2D4351','xtick.color':'#768B9C','ytick.color':'#768B9C','legend.frameon':False})
old=list(csv.DictReader(open(R/'artifacts/spooky_ace_d_overlay_20260914/inner_trials.csv')))
summary={x['round']:x for x in json.load(open(R/'artifacts/spooky_ace_d_overlay_20260914/SUMMARY.json'))}
colors={'A':'#377FAE','C':'#BF657D','D':'#118F85','E':'#7C66B3'};markers={'A':'o','C':'^','D':'D','E':'P'}
labels={'A':'A  无记忆','C':'C  读取 A','D':'D  读取 A+B+C','E':'E  读取 A+B+C+D'}
trials={k:[] for k in colors};hosts={k:[] for k in colors};bounds={};ends={k:v['minutes'] for k,v in summary.items()}
for row in old:
 k=row['round'];trials[k].append({'round':k,'trial_id':row['trial_id'],'tune_loss':float(row['tune_loss']),'segment':'原记录','status':row['status']})
for k in colors:
 p=R/next(x['source'] for x in old if x['round']==k);root=p.parents[7];launch=json.load(open(root/'control/LAUNCH.json'));j=json.load(open(next(root.glob('runs/*/logs/journal.json'))))
 for n in j['nodes']:
  if n['step']==0 or n['metric']['value'] is None:continue
  t=(datetime.fromisoformat(n['finish_time']).replace(tzinfo=timezone.utc).timestamp()-launch['started_unix'])/60
  hosts[k].append((t,n['metric']['value'],n['id']))
archive=R/'experiments/spooky_ac_aws_resume_20260914/terminal-backup-v4/outputs.tar'
with tarfile.open(archive) as tf:
 members={m.name:m for m in tf.getmembers() if m.isfile()}
 def read(name):return json.load(tf.extractfile(members[name]))
 for k in ['A','C']:
  pre=f'outputs/{k}/';launch=read(pre+'control/LAUNCH.json');end=read(pre+'control/POD_PROCESS_OUTCOME.json');bounds[k]=launch['prior_elapsed_seconds']/60;ends[k]=bounds[k]+end['elapsed_seconds']/60
  ids={x['trial_id'] for x in trials[k]};new=[]
  for name in members:
   if name.startswith(pre) and '/inner_tuning/trial-' in name and name.endswith('-outcome.json'):
    t=read(name)
    if t['trial_id'] in ids:continue
    ids.add(t['trial_id']);new.append(t)
  new.sort(key=lambda t:(t['completed_monotonic'] if 'completed_monotonic' in t else (0 if t['node_id']=='2c194b6a94a74db3bacd9af707c669ef' else 1000)+int(t['trial_id'].split(':t')[-1])))
  for t in new:trials[k].append({'round':k,'trial_id':t['trial_id'],'tune_loss':t.get('tune_loss'),'status':t['status'],'segment':'AWS 续跑','elapsed_minutes':bounds[k]+(t['completed_monotonic']-launch['started_monotonic'])/60 if 'completed_monotonic' in t else ''})
  jp=next(n for n in members if n.startswith(pre) and n.endswith('/logs/journal.json'));j=read(jp);existing={x[2] for x in hosts[k]}
  for n in j['nodes']:
   if n['id'] in existing or n['step']==0 or n['metric']['value'] is None:continue
   stamp=datetime.fromisoformat(n['finish_time']).replace(tzinfo=timezone.utc).timestamp();minute=bounds[k]+(stamp-launch['started_unix'])/60
   hosts[k].append((minute,n['metric']['value'],n['id']))
assert [len(trials[k]) for k in colors]==[73,21,15,16]
for k in colors:
 assert all(t['status']=='ok' and math.isfinite(t['tune_loss']) for t in trials[k])
 assert all(0<=t<=ends[k]+.1 for t,_,_ in hosts[k])
with (O/'inner_trials.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['round','trial_id','tune_loss','status','segment','elapsed_minutes']);w.writeheader();w.writerows(sum(trials.values(),[]))
with (O/'host_scores.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['round','elapsed_minutes','host_search_loss','node_id']);w.writerows((k,*x) for k in hosts for x in hosts[k])
fig=plt.figure(figsize=(16,10),facecolor='white');ax=fig.add_axes([.08,.29,.78,.46])
fig.text(.07,.94,'Spooky / 内部调参轨迹 · A / C / D / E',fontsize=25)
fig.text(.07,.894,'已接入 A、C 的 AWS 续跑记录；按各轮实际尝试进度对齐',fontsize=14,color='#768B9C')
fig.text(.94,.94,'125 次有效尝试',ha='right',fontsize=14)
for i,k in enumerate(colors):
 x=.07+i*.226;fig.add_artist(Line2D([x,x+.025],[.83,.83],transform=fig.transFigure,color=colors[k],lw=3));fig.text(x+.032,.824,labels[k],color=colors[k],fontsize=14)
 oldn=summary[k]['trials'];extra=len(trials[k])-oldn
 fig.text(x+.032,.794,f'{oldn} + {extra} 次' if extra else f'{oldn} 次 · 原记录',color='#768B9C',fontsize=11)
for k in colors:
 y=np.array([t['tune_loss'] for t in trials[k]]);x=np.linspace(0,100,len(y));best=np.minimum.accumulate(y);last=int(np.where(y==best[-1])[0][0]);c=colors[k]
 ax.scatter(x,y,color=c,marker=markers[k],alpha=.36,s=23,zorder=3)
 ax.step(x[:last+1],best[:last+1],where='post',color=c,lw=2.5)
 ax.step(x[last:],best[last:],where='post',color=c,lw=2.5)
 ax.scatter(x[last],best[last],s=135,marker='*',c=c,edgecolor='white',lw=.7,zorder=5)
 ax.annotate(f'{k}  {best[-1]:.5f}',(100,best[-1]),xytext=(103,best[-1]+({'A':.005,'C':0,'D':.009,'E':-.004}[k])),color=c,fontsize=12,arrowprops={'arrowstyle':'-','color':c,'lw':.8},annotation_clip=False)

ax.set(xlim=(-1.5,101),ylim=(.25,.485),ylabel='内部 tune log-loss ↓',xlabel='本轮实际调参进度（首次尝试 = 0%，最后一次 = 100%）')
ax.set_xticks([0,25,50,75,100],['0%','25%','50%','75%','100%']);ax.set_yticks([.26,.30,.35,.40,.45]);ax.grid(axis='y',color='#E5ECF2');ax.set_axisbelow(True)
for i,k in enumerate(colors):
 x=.07+i*.226;h=min(y for _,y,_ in hosts[k]);fig.text(x,.163,k+'  宿主 search 最佳',fontsize=11,color='#768B9C');fig.text(x,.12,f'{h:.5f}',fontsize=23,color=colors[k]);fig.text(x+.115,.122,f'{ends[k]:.1f} min',fontsize=11,color='#768B9C')
fig.savefig(O/'inner_clean.png',dpi=300)
fig.savefig(O/'inner_clean.svg')
fig.savefig(O/'inner_clean.pdf')
plt.close(fig)
