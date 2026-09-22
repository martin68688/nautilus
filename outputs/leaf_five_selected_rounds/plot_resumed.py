"""Search-only quantitative hero and detail: real resumed time, unchanged best.
Structural adaptation of the existing comparison. One seed/fixed split, no CI.
Keep every measured score; failed/unscored nodes never become zero scores.
"""
from pathlib import Path
import csv,json,math
from datetime import datetime,timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator,FuncFormatter,NullLocator
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
rows=list(csv.DictReader((OUT/'without_r4_outer_search.csv').open()))
ss=json.loads((OUT/'without_r4_summary.json').read_text())['rounds']
summ={s['round']:dict(s) for s in ss}
prior={r:summ[r]['elapsed_minutes'] for r in (5,6)}
sources={5:ROOT/'aws-leaf-r5-resume-20260915/PLOT-TERMINAL.json',6:ROOT/'experiments/leaf_round4_memory_luna_2h_20260913/resume_r5_r6_20260914/terra/R6-TERMINAL.json'}
for r,path in sources.items():
 t=json.loads(path.read_text());start=t['controls']['LAUNCH.json']['started_unix']
 summ[r]['elapsed_minutes']=prior[r]+t['controls']['POD_PROCESS_OUTCOME.json']['elapsed_seconds']/60
 old={z['node_id'] for z in rows if int(z['round'])==r}
 for n in t['core']['journal.json']['nodes']:
  if n.get('stage')=='root' or n['id'] in old:continue
  v=n.get('metric',{}).get('value');valid=bool(n.get('is_valid')) and isinstance(v,(int,float)) and math.isfinite(v)
  tm=n.get('finish_time')
  if not tm:continue
  elapsed=prior[r]+(datetime.fromisoformat(tm).replace(tzinfo=timezone.utc).timestamp()-start)/60
  if valid:
   assert r==5
   rec=t['verified'][n['id']];assert rec['host_recomputed'] and rec['checkpoint_validated'] and abs(rec['metric']-v)<1e-12
  rows.append(dict(round=r,node=len(old),node_id=n['id'],elapsed_minutes=elapsed,certified=str(valid),search_loss=v if valid else ''))
with (OUT/'resumed_search.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'resumed_summary.json').write_text(json.dumps(summ,indent=2))
font=Path('/System/Library/Fonts/STHeiti Light.ttc');font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[font_manager.FontProperties(fname=font).get_name(),'Arial'],'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#BCC8D1','axes.linewidth':.7,'text.color':'#293E4D','axes.labelcolor':'#405565','xtick.color':'#627684','ytick.color':'#627684','svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,'legend.frameon':False})
colors={1:'#397FAC',2:'#C68A3A',3:'#8064AD',5:'#188F8D',6:'#253F58'}
markers={1:'o',2:'s',3:'D',5:'P',6:'h'}
fig=plt.figure(figsize=(13.2,7.3),facecolor='white')
fig.text(.075,.93,'Leaf  /  Search 成绩对比 · 续跑后',fontsize=23)
fig.text(.075,.882,'R5 与 R6 的历史最佳均保持不变；曲线已补至实际结束',fontsize=11)
fig.text(.965,.845,'固定 search log-loss ↓  ·  单次固定划分  ·  不含 tune',ha='right',fontsize=9,color='#718391')
fig.legend(handles=[Line2D([],[],color=colors[r],marker=markers[r],lw=2,ls='--' if r==3 else '-',label=f'R{r}') for r in summ],loc='center left',bbox_to_anchor=(.067,.843),ncol=5,columnspacing=2.4)
fig.add_artist(Line2D([.075,.965],[.802,.802],transform=fig.transFigure,color='#DFE6EB'))
a=fig.add_axes([.075,.265,.565,.47]);b=fig.add_axes([.735,.438,.23,.297])
fig.text(.075,.759,'a   全部五轮 · 实点为候选，阶梯线为累计最佳',fontsize=11)
fig.text(.735,.759,'b   R5 / R6 最佳成绩放大',fontsize=11)
for ax in (a,b):ax.grid(axis='y',color='#E7EDF1',lw=.7);ax.set_axisbelow(True)
for r,s in summ.items():
 good=sorted([z for z in rows if int(z['round'])==r and z['certified']=='True'],key=lambda z:float(z['elapsed_minutes']))
 x=[float(z['elapsed_minutes']) for z in good];y=[float(z['search_loss']) for z in good]
 assert all(v>0 and math.isfinite(v) for v in y)
 best=[min(y[:i+1]) for i in range(len(y))];assert abs(best[-1]-s['final_search'])<1e-12
 end=s['elapsed_minutes'];c=colors[r]
 for ax in ([a,b] if r in (5,6) else [a]):
  ax.step(x+[end],best+[best[-1]],where='post',color=c,lw=2,ls='--' if r==3 else '-')
  visible=[(xx,yy) for xx,yy in zip(x,y) if ax is a or .033<=yy<=.054]
  ax.scatter([v[0] for v in visible],[v[1] for v in visible],color=c,marker=markers[r],s=32,edgecolor='white',linewidth=.45,zorder=4)
  ax.plot(end,best[-1],marker='|',ms=8,color=c)
  if r==1:
   # Requested display-only carry-forward; actual end and source data unchanged.
   ax.plot([end,120],[best[-1],best[-1]],color=c,lw=2,ls='-')
  if r in prior:
   ax.plot(prior[r],s['final_search'],marker='o',ms=7,mfc='white',mec=c,zorder=5)
 if r in (1,2,3):a.annotate(f'R{r}  {best[-1]:.6f}',(end,best[-1]),xytext=(-3,{1:10,2:-17,3:24}[r]),textcoords='offset points',ha='right',color=c,fontsize=9)
 else:
  a.annotate(f'R{r}  {best[-1]:.6f}',(end,best[-1]),xytext=(-5,9 if r==5 else -15),textcoords='offset points',ha='right',color=c,fontsize=9)
  b.annotate(f'R{r}',(end,best[-1]),xytext=(-5,9),textcoords='offset points',ha='right',color=c)
a.set_yscale('log');a.set(xlim=(0,123),ylim=(.029,.88),xticks=range(0,121,20),xlabel='累计有效墙钟时间 / 分钟（不计停机间隔）',ylabel='Search log-loss ↓ · 对数刻度')
a.yaxis.set_major_locator(FixedLocator([.04,.06,.1,.2,.4,.8]));a.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'));a.yaxis.set_minor_locator(NullLocator())
b.set(xlim=(0,123),ylim=(.033,.054),xticks=[0,30,60,90,120],yticks=[.035,.040,.045,.050],xlabel='累计墙钟时间 / 分钟')
b.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:.3f}'))
new=[z for z in rows if int(z['round'])==5 and z['certified']=='True' and float(z['elapsed_minutes'])>prior[5]][0]
a.annotate('续跑新候选 0.102981\n未替换历史最佳',(float(new['elapsed_minutes']),float(new['search_loss'])),xytext=(-15,35),textcoords='offset points',ha='center',fontsize=9,color=colors[5],arrowprops={'arrowstyle':'-','color':colors[5]})
fig.text(.735,.338,'轮次',color='#718391',fontsize=9);fig.text(.862,.338,'最佳 search',ha='right',color='#718391',fontsize=9);fig.text(.965,.338,'累计时长',ha='right',color='#718391',fontsize=9)
for r,yy in [(5,.304),(6,.271)]:
 fig.text(.735,yy,f'R{r}',color=colors[r]);fig.text(.862,yy,f"{summ[r]['final_search']:.6f}",ha='right',color=colors[r]);fig.text(.965,yy,f"{summ[r]['elapsed_minutes']:.1f} min",ha='right')
fig.text(.735,.221,'两轮续跑：最佳均未提升',fontsize=10)
fig.add_artist(Line2D([.075,.965],[.185,.185],transform=fig.transFigure,color='#DFE6EB'))
fig.text(.075,.147,'实点：23 个宿主验证成绩；空心圆：续跑开始；竖线：实际结束。R1 实线延长至 120 分钟仅作展示，无新增试验。',fontsize=9,color='#627684')
fig.text(.075,.110,'R5：68.0 分钟后转 Terra-low / AWS A10G（4 核）；R6：80.5 分钟后转 Terra-low / NRP A10。此前为 Luna-low。',fontsize=9,color='#627684')
fig.text(.075,.073,'R5 累计 119.8 分钟，预算到期强制停止、非正常收尾；R6 累计 118.0 分钟，按预算准入策略结束。',fontsize=9,color='#627684')
fig.text(.075,.036,'同一固定 search 划分，非 Kaggle 官方成绩；保留原轮次编号并移除 R4。模型与硬件有变化，不作严格同条件因果比较。',fontsize=9,color='#627684')
fig.savefig(OUT/'leaf_search_resumed_r1_extended.png',dpi=300,facecolor='white')
fig.savefig(OUT/'leaf_search_resumed_r1_extended.svg',facecolor='white')
fig.savefig(OUT/'leaf_search_resumed_r1_extended.pdf',facecolor='white')
plt.close(fig)
