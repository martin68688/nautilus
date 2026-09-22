"""Quantitative comparison: all historical search observations plus the latest repeat.
Python report-sized hero with two magnifications; n=1 fixed split per run, no CI.
Sources preserved; invalid candidates are counts, never zero-score observations.
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
rows=list(csv.DictReader((OUT/'resumed_search.csv').open()))
summary=json.loads((OUT/'resumed_summary.json').read_text())
# Host PASS timestamp and launch timestamp verified in live launcher logs.
start=1789514855.6743057
scored=datetime(2026,9,16,0,20,8,26000,tzinfo=timezone.utc).timestamp()
score=0.4426650482869665
end=(5544.564078320749+1468.4171662000008)/60
rows.append(dict(round='latest',node=3,node_id='8f66b631c3bf4de0b65125c7c36eaeca',
 elapsed_minutes=(scored-start)/60,certified='True',search_loss=score))
summary['latest']={'final_search':score,'elapsed_minutes':end,'candidates':11,
 'certified_candidates':1,'failed_candidates':10,'resume_new_candidates':1,
 'resume_new_certified':0,'resume_start_minutes':5298.728687166935/60,
 'first_score_timestamp_basis':'host PASS; previous rows preserve original timestamps'}
rows=[z for z in rows if str(z['round'])!='2']
for z in rows:
 if z['round']=='latest':z['round']='2'
summary['2']=summary.pop('latest')
with (OUT/'aligned_search.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'aligned_summary.json').write_text(json.dumps(summary,indent=2))
font=Path('/System/Library/Fonts/STHeiti Light.ttc');font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[font_manager.FontProperties(fname=font).get_name(),'Arial'],
 'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#BCC8D1',
 'text.color':'#293E4D','axes.labelcolor':'#405565','xtick.color':'#627684','ytick.color':'#627684',
 'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,'legend.frameon':False})
order=['1','2','3','5','6']
colors=dict(zip(order,['#397FAC','#C68A3A','#8064AD','#188F8D','#253F58','#B6536D']))
markers=dict(zip(order,['o','s','D','P','h','v']))
labels=dict(zip(order,['R1 首轮','R2 最新重跑','R3 第三轮重跑','R5 全记忆轮','R6 历史重跑']))
fig=plt.figure(figsize=(15,8.8),facecolor='white')
fig.text(.065,.945,'Leaf  /  Search 成绩对比 · 完整记录',fontsize=25)
fig.text(.065,.900,'R2 已替换为最新重跑 · 全部实线统一展示至 120 分钟',fontsize=12)
fig.text(.065,.864,'固定 search log-loss ↓  ·  仅宿主认证成绩  ·  各轮独立计时，停机间隔不计入',fontsize=10,color='#718391')
handles=[Line2D([],[],color=colors[r],marker=markers[r],lw=2,ls='-',
 label=f"{labels[r]}  {summary[r]['final_search']:.6f}") for r in order]
fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.06,.836),ncol=3,
 columnspacing=2.6,handlelength=2.2,fontsize=10)
fig.add_artist(Line2D([.065,.97],[.754,.754],transform=fig.transFigure,color='#DFE6EB'))
a=fig.add_axes([.065,.25,.59,.445]);b=fig.add_axes([.755,.505,.215,.19]);c=fig.add_axes([.755,.25,.215,.17])
fig.text(.065,.719,'a   五轮对比 · 实点为候选，阶梯线为累计最佳',fontsize=11)
fig.text(.755,.719,'b   R5 / R6 低损失区域',fontsize=11)
fig.text(.755,.444,'c   R1 / 新 R2 局部放大',fontsize=11)
for ax in (a,b,c):
 ax.grid(axis='y',color='#E7EDF1',lw=.7);ax.set_axisbelow(True)
for r in order:
 s=summary[r];good=sorted([z for z in rows if str(z['round'])==r and z['certified']=='True'],key=lambda z:float(z['elapsed_minutes']))
 x=[float(z['elapsed_minutes']) for z in good];y=[float(z['search_loss']) for z in good]
 assert all(math.isfinite(v) and v>0 for v in y)
 best=[min(y[:i+1]) for i in range(len(y))];assert abs(best[-1]-s['final_search'])<1e-12
 stop=s['elapsed_minutes'];color=colors[r]
 axes=[a]+([b] if r in ('5','6') else [c] if r in ('1','2','latest') else [])
 for ax in axes:
  width=4.5 if ax is a and r=='1' else 1.7 if ax is a and r=='2' else 2
  ax.step(x+[120],best+[best[-1]],where='post',color=color,lw=width,
   ls='-',zorder=3)
  ax.scatter(x,y,color=color,marker=markers[r],s=34,edgecolor='white',linewidth=.5,zorder=5)
  if r in ('5','6','2'):
   t={'5':68.0,'6':80.5,'2':5298.728687166935/60}[r]
   ax.plot(t,best[-1],marker='o',ms=6,mfc='white',mec=color,zorder=7)
 if r in ('3','5','6'):
  a.annotate(f'R{r}  {best[-1]:.6f}',(120,best[-1]),xytext=(-5,14 if r!='6' else -15),textcoords='offset points',ha='right',color=color,fontsize=9)
a.set_yscale('log');a.set(xlim=(0,123),ylim=(.029,.88),xticks=range(0,121,20),
 xlabel='累计运行时间 / 分钟（含规划、训练、恢复和评分）',ylabel='Search log-loss ↓ · 对数刻度')
a.yaxis.set_major_locator(FixedLocator([.04,.06,.1,.2,.4,.8]));a.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'));a.yaxis.set_minor_locator(NullLocator())
a.annotate('新 R2 首个认证成绩\n52.5 分钟 · 0.442665',((scored-start)/60,score),xytext=(33,.56),fontsize=10,color=colors['2'],arrowprops={'arrowstyle':'-','color':colors['2']})
a.text(119,.475,'R1  0.443894',ha='right',color=colors['1'],fontsize=9)
a.text(119,.401,'R2  0.442665',ha='right',color=colors['2'],fontsize=9)
b.set(xlim=(0,123),ylim=(.033,.054),xticks=[0,30,60,90,120],yticks=[.035,.040,.045,.050])
b.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:.3f}'))
for r in ('5','6'):b.text(117,summary[r]['final_search']+.0007,'R'+r,ha='right',fontsize=9,color=colors[r])
c.set(xlim=(50,123),ylim=(.4418,.4452),xticks=[60,80,100,120],yticks=[.442,.443,.444,.445],xlabel='累计运行时间 / 分钟')
c.text(119,.44425,'R1  0.443894',ha='right',color=colors['1'],fontsize=9)
c.text(119,.44215,'R2  0.442665',ha='right',color=colors['2'],fontsize=9)
fig.add_artist(Line2D([.065,.97],[.188,.188],transform=fig.transFigure,color='#DFE6EB'))
notes=[
 '新 R2：11 个候选 / 1 个有效；含断点续跑，最佳 0.442665。原 R2 已从本图移除，R4 仍不展示。',
 '实点：22 个宿主认证成绩；空心圆：续跑衔接。全部实线延长至 120 分钟仅为展示，不代表新增试验或实际跑满。',
 '实际时长：R1 93.7 分钟 · 新 R2 116.9 分钟 · R3 114.2 分钟 · R5 119.8 分钟 · R6 118.0 分钟。',
 '历史各轮最佳方案均为 CPU 训练；本轮为 CPA Luna-low / 4 核 CPU。R5、R6 续跑曾切换 Terra-low，非严格同条件对照。',
 '同一固定 search 划分，非 Kaggle 官方成绩；不含 tune。单次运行，无置信区间，不据此断言记忆或模型的因果收益。']
fig.text(.065,.15,'延长至 120 分钟仅为展示，不代表新增成绩。',fontsize=9,color='#718391')
from matplotlib.transforms import Bbox
crop=Bbox.from_extents(.15,1.12,14.9,8.8)
fig.savefig(OUT/'leaf_search_r2_replaced_aligned.png',dpi=300,facecolor='white',bbox_inches=crop)
fig.savefig(OUT/'leaf_search_r2_replaced_aligned.svg',facecolor='white',bbox_inches=crop)
fig.savefig(OUT/'leaf_search_r2_replaced_aligned.pdf',facecolor='white',bbox_inches=crop)
plt.close(fig)
