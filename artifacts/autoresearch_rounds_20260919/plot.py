from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter,MaxNLocator
O=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial Unicode MS','Arial','DejaVu Sans'],'font.size':14,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#BCD0DC','axes.labelcolor':'#344C59','text.color':'#344C59','xtick.color':'#8096AA','ytick.color':'#8096AA','axes.unicode_minus':False,'legend.frameon':False})
rows=list(csv.DictReader((O/'tune_observations.csv').open()))
colors=['#377FAE','#C86281','#10988F'];markers=['o','^','D'];labels=['R1  无记忆','R2  读取 R1','R3  读取 R1 + R2']
host={'Spooky':[.35028213509959194,.3511051102229948,.34911176412385214],'Leaf':[.44389385340454646,.06820685125413978,.037278732298774535]}
summary={}
for task in ['Spooky','Leaf']:
 fig=plt.figure(figsize=(20,12.5),facecolor='white');ax=fig.add_axes([.08,.29,.76,.45]);total=sum(r['task']==task for r in rows)
 fig.text(.07,.94,f'{task} / 内部调参轨迹 · R1 / R2 / R3',fontsize=29)
 fig.text(.94,.941,f'{total} 个 tune 观测点',ha='right',fontsize=17)
 fig.text(.07,.895,'按各轮实际观测进度对齐；散点为原始记录，阶梯线为累计最低值',fontsize=17,color='#8096AA')
 fig.text(.07,.858,'R1：历史前 2h 截断记录  ·  R2 / R3：累计记忆，各轮预算 2h',fontsize=14,color='#8096AA')
 groups=[]
 for i,rnd in enumerate(['R1','R2','R3']):
  rr=[r for r in rows if r['task']==task and r['round']==rnd];y=np.array([float(r['value']) for r in rr]);x=np.array([float(r['progress_percent']) for r in rr]);assert np.isfinite(y).all() and (y>=0).all();groups.append((x,y))
  pos=.07+i*.305;fig.add_artist(Line2D([pos,pos+.026],[.803,.803],transform=fig.transFigure,color=colors[i],lw=3));fig.text(pos+.033,.797,labels[i],fontsize=18,color=colors[i]);fig.text(pos+.033,.769,f'{len(y)} 个观测点',fontsize=13,color='#8096AA')
  ax.scatter(x,y,s=22 if len(y)>100 else 34,color=colors[i],marker=markers[i],alpha=.28,zorder=3);best=np.minimum.accumulate(y);ax.step(x,best,where='post',lw=2.5,color=colors[i],zorder=4)
  last=int(np.where(y==best[-1])[0][0]);ax.scatter([x[last]],[best[-1]],s=170,marker='*',color=colors[i],edgecolor='white',linewidth=.7,zorder=6)
 ax.set_xlim(-1.5,101);ax.set_ylim((.30,1.0) if task=='Spooky' else (-.065,2.62));ax.grid(axis='y',color='#E8EFF5',lw=1);ax.set_axisbelow(True);ax.set_xticks([0,25,50,75,100]);ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}%'));ax.set_xlabel('本轮观测进度（首次 = 0%，最后一次 = 100%）',labelpad=12,fontsize=15);ax.set_ylabel('内部 tune log-loss ↓',fontsize=15)
 # Spread direct endpoint labels, preserving true values via connectors.
 vals=np.array([min(y) for x,y in groups]);span=ax.get_ylim()[1]-ax.get_ylim()[0];positions=vals.copy();order=np.argsort(vals)
 for a,b in zip(order,order[1:]):positions[b]=max(positions[b],positions[a]+span*.052)
 for i,v in enumerate(vals):ax.annotate(f'R{i+1}  {v:.5f}',xy=(100,v),xytext=(103,positions[i]),color=colors[i],fontsize=14,va='center',annotation_clip=False,arrowprops={'arrowstyle':'-','color':colors[i],'lw':1})
 # Focus panel complements the full-range hero, never substitutes for observations.
 ins=fig.add_axes([.49,.465,.31,.205],facecolor='white');ins.set_title('后期低损失区间 · 放大',fontsize=12,loc='left',color='#8096AA')
 for i,(x,y) in enumerate(groups):ins.scatter(x,y,s=9,color=colors[i],marker=markers[i],alpha=.23);ins.step(x,np.minimum.accumulate(y),where='post',color=colors[i],lw=1.7)
 ins.set_ylim((.332,.365) if task=='Spooky' else (0,.105));ins.set_xlim(-1,101);ins.set_xticks([0,50,100]);ins.set_xticklabels(['0%','50%','100%']);ins.tick_params(labelsize=10);ins.yaxis.set_major_locator(MaxNLocator(4));ins.grid(axis='y',color='#E8EFF5');ins.spines[['top','right']].set_visible(True)
 for i in range(3):
  pos=.07+i*.305;fig.text(pos,.165,f'R{i+1}  宿主 search 最佳',fontsize=14,color='#8096AA');fig.text(pos,.12,f'{host[task][i]:.5f}',fontsize=28,color=colors[i])
  caption=('历史前 120 min' if i==0 else '120 min 预算')
  if task=='Leaf' and i==0:caption='历史实际 93.7 min'
  if task=='Leaf' and i==2:caption='中途受 503 故障影响'
  fig.text(pos+.14,.125,caption,fontsize=12,color='#8096AA')
 note='观测包含检查点、缓存调参或校准；数量不等于独立训练次数。Host 与 tune 分别统计，不混算。'
 fig.text(.07,.064,note,fontsize=12,color='#8096AA')
 fig.text(.07,.037,'R1 为历史 MLEvolve；新轮为 autoresearch。'+('Leaf R1 使用 Luna-low，新轮使用 Terra-low。' if task=='Leaf' else '未做 A/B 对照，不把曲线差异单独归因于记忆。'),fontsize=12,color='#8096AA')
 stem=task.lower()+'_rounds';fig.savefig(O/(stem+'.png'),dpi=300);fig.savefig(O/(stem+'.svg'));fig.savefig(O/(stem+'.pdf'));plt.close(fig)
 summary[task]={'counts':[len(y) for x,y in groups],'best_tune':list(vals),'best_host':host[task]}
from PIL import Image
for name in ['spooky','leaf']:
 im=Image.open(O/(name+'_rounds.png'));im.resize((2000,1250),Image.Resampling.LANCZOS).save(O/(name+'_rounds_preview.png'))
(O/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
