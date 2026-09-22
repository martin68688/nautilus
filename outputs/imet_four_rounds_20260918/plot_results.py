from pathlib import Path
import json,csv,hashlib,tarfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
font_manager.fontManager.addfont('/System/Library/Fonts/STHeiti Light.ttc')
plt.rcParams.update({'font.family':['STHeiti','Arial','sans-serif'],'font.size':11,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
old=ROOT/'outputs/imet_rounds_comparison_20260917'
runs=[r for r in json.loads((old/'summary.json').read_text()) if r['id'] in ['A01','A06','A12']]
rows=[]
for r in runs:
 archive=ROOT/r['source_archive']
 assert hashlib.sha256(archive.read_bytes()).hexdigest()==r['archive_sha256']
 with tarfile.open(archive) as t:
  for m in t.getmembers():
   if m.name.startswith('experiment/round1/' if r['id']=='A01' else 'round2/') and m.name.endswith('/VERIFIED_RESULT.json'):
    d=json.load(t.extractfile(m))
    if not (d.get('host_recomputed') and d.get('checkpoint_validated')): continue
    score=d.get('micro_f1',1-d['metric']); minute=(m.mtime-r['started_unix'])/60
    assert 0<=score<=1 and 0<=minute<=r['duration_min']
    rows.append(dict(run_id=r['id'],minute=minute,score=score,node_id=m.name.split('/')[-2],source=str(archive),member=m.name))
 assert len([x for x in rows if x['run_id']==r['id']])==r['search_n']
snapshot=json.loads((OUT/'aws_snapshot.json').read_text()); offset=0
for s in snapshot:
 for p in s['points']:
  assert p['host_recomputed'] and p['checkpoint_validated']
  rows.append(dict(run_id='AWS',minute=(offset+p['mtime']-s['launch']['started_unix'])/60,score=p['score'],node_id=p['node_id'],source=p['path'],member=p['sha256']))
 offset+=s['outcome']['elapsed_seconds']
runs.append(dict(id='AWS',duration_min=offset/60,search_best=max(p['score'] for s in snapshot for p in s['points'])))
labels=['第一轮','第二轮','第三轮','第四轮']
colors=['#397FAC','#C18E43','#238A79','#7955AA']; markers=['o','s','D','^']; styles=['--','-.',':','-']
fig,(ax,b)=plt.subplots(1,2,figsize=(13,5.6),gridspec_kw={'width_ratios':[2.35,1]})
fig.subplots_adjust(left=.075,right=.96,top=.80,bottom=.13,wspace=.38)
fig.text(.075,.93,'iMet｜四轮实验成绩曲线',fontsize=24,weight='bold')
fig.text(.075,.867,'Host 验证 micro-F1（越高越好） · 实点 = 候选成绩 · 阶梯 = 累计最佳',fontsize=12,color='#536574')
for r,label,col,mark,style in zip(runs,labels,colors,markers,styles):
 p=sorted([x for x in rows if x['run_id']==r['id']],key=lambda x:x['minute'])
 x=[a['minute'] for a in p]; y=[a['score'] for a in p]; best=[]
 for v in y:best.append(max(best[-1] if best else 0,v))
 assert abs(max(y)-r['search_best'])<1e-12
 ax.step(x+[r['duration_min']],best+[best[-1]],where='post',color=col,ls=style,lw=2,label=label)
 ax.scatter(x,y,s=45,color=col,marker=mark,zorder=4)
 ax.plot(r['duration_min'],best[-1],marker='|',ms=12,color=col)
ax.set(xlim=(0,122),ylim=(0,.65),xlabel='累计实验时间 / 分钟',ylabel='Host micro-F1',xticks=range(0,121,20))
ax.grid(axis='y',alpha=.2);ax.legend(loc='lower right',frameon=False,fontsize=10)
for i,(r,label,col,mark) in enumerate(zip(runs,labels,colors,markers)):
 v=r['search_best']; b.hlines(i,0,v,color=col,lw=3);b.scatter(v,i,color=col,marker=mark,s=55)
 b.text(v+.018,i,f'{v:.5f}',va='center',color=col,fontsize=12)
 b.text(0,i-.22,label,fontsize=10,color='#384B59')
b.set(xlim=(0,.76),ylim=(3.6,-.7),yticks=[],xticks=[0,.2,.4,.6],xlabel='本轮最佳 Host micro-F1')
b.spines['left'].set_visible(False);b.grid(axis='x',alpha=.15)
fig.savefig(OUT/'imet_four_rounds.png',dpi=300,facecolor='white')
fig.savefig(OUT/'imet_four_rounds.pdf',facecolor='white')
fig.savefig(OUT/'imet_four_rounds.svg',facecolor='white')
with (OUT/'source_data.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'summary.json').write_text(json.dumps(runs,indent=2,ensure_ascii=False))
print([(r['id'],r['search_best'],r['duration_min']) for r in runs])
