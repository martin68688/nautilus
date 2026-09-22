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
with (OUT/'complete_search.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'complete_summary.json').write_text(json.dumps(summary,indent=2))
font=Path('/System/Library/Fonts/STHeiti Light.ttc');font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[font_manager.FontProperties(fname=font).get_name(),'Arial'],
 'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#BCC8D1',
 'text.color':'#293E4D','axes.labelcolor':'#405565','xtick.color':'#627684','ytick.color':'#627684',
 'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,'legend.frameon':False})
order=['1','2','3','5','6','latest']
colors=dict(zip(order,['#397FAC','#C68A3A','#8064AD','#188F8D','#253F58','#B6536D']))
markers=dict(zip(order,['o','s','D','P','h','v']))
labels=dict(zip(order,['R1 首轮','R2 第二轮','R3 第三轮重跑','R5 全记忆轮','R6 历史重跑','本轮 · R1 记忆重跑']))
fig=plt.figure(figsize=(15,8.8),facecolor='white')
fig.text(.065,.945,'Leaf  /  Search 成绩对比 · 完整记录',fontsize=25)
fig.text(.065,.900,'本轮含断点续跑：116.9 分钟，最佳 0.442665；续跑没有新增有效成绩',fontsize=12)
fig.text(.065,.864,'固定 search log-loss ↓  ·  仅宿主认证成绩  ·  各轮独立计时，停机间隔不计入',fontsize=10,color='#718391')
handles=[Line2D([],[],color=colors[r],marker=markers[r],lw=2,ls='--' if r in ('3','latest') else '-',
 label=f"{labels[r]}  {summary[r]['final_search']:.6f}") for r in order]
fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.06,.836),ncol=3,
 columnspacing=2.6,handlelength=2.2,fontsize=10)
fig.add_artist(Line2D([.065,.97],[.754,.754],transform=fig.transFigure,color='#DFE6EB'))
a=fig.add_axes([.065,.25,.59,.445]);b=fig.add_axes([.755,.505,.215,.19]);c=fig.add_axes([.755,.25,.215,.17])
fig.text(.065,.719,'a   全部六条记录 · 实点为候选，阶梯线为累计最佳',fontsize=11)
fig.text(.755,.719,'b   R5 / R6 低损失区域',fontsize=11)
fig.text(.755,.444,'c   本轮与 R1 / R2 局部放大',fontsize=11)
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
  ax.step(x+[stop],best+[best[-1]],where='post',color=color,lw=2,
   ls='--' if r in ('3','latest') else '-',zorder=3)
  ax.scatter(x,y,color=color,marker=markers[r],s=34,edgecolor='white',linewidth=.5,zorder=5)
  ax.plot(stop,best[-1],marker='|',ms=9,color=color,zorder=6)
  if r=='1':ax.plot([stop,120],[best[-1],best[-1]],color=color,lw=2)
  if r in ('5','6','latest'):
   t={'5':68.0,'6':80.5,'latest':5298.728687166935/60}[r]
   ax.plot(t,best[-1],marker='o',ms=6,mfc='white',mec=color,zorder=7)
 if r in ('3','5','6'):
  a.annotate(f'R{r}  {best[-1]:.6f}',(stop,best[-1]),xytext=(-5,14 if r!='6' else -15),textcoords='offset points',ha='right',color=color,fontsize=9)
a.set_yscale('log');a.set(xlim=(0,123),ylim=(.029,.88),xticks=range(0,121,20),
 xlabel='累计运行时间 / 分钟（含规划、训练、恢复和评分）',ylabel='Search log-loss ↓ · 对数刻度')
a.yaxis.set_major_locator(FixedLocator([.04,.06,.1,.2,.4,.8]));a.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'));a.yaxis.set_minor_locator(NullLocator())
a.annotate('本轮首个认证成绩\n52.5 分钟 · 0.442665',((scored-start)/60,score),xytext=(33,.56),fontsize=10,color=colors['latest'],arrowprops={'arrowstyle':'-','color':colors['latest']})
b.set(xlim=(0,123),ylim=(.033,.054),xticks=[0,30,60,90,120],yticks=[.035,.040,.045,.050])
b.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:.3f}'))
for r in ('5','6'):b.text(117,summary[r]['final_search']+.0007,'R'+r,ha='right',fontsize=9,color=colors[r])
c.set(xlim=(50,123),ylim=(.416,.452),xticks=[60,80,100,120],yticks=[.42,.43,.44,.45],xlabel='累计运行时间 / 分钟')
c.text(119,.4467,'R1  0.443894',ha='right',color=colors['1'],fontsize=9)
c.text(119,.4358,'本轮  0.442665',ha='right',color=colors['latest'],fontsize=9)
c.text(119,.4188,'R2  0.422525',ha='right',color=colors['2'],fontsize=9)
fig.add_artist(Line2D([.065,.97],[.188,.188],transform=fig.transFigure,color='#DFE6EB'))
notes=[
 '本轮：11 个候选 / 1 个有效；续跑新增 1 个候选，训练超时、无完整检查点。累计 116.9 分钟，终止状态为 partial。',
 '实点：24 个宿主认证成绩；空心圆：续跑衔接；端点竖线：实际结束。失败候选不画成 0 分；水平线不代表新增试验。',
 'R1 实线按要求展示至 120 分钟，实际结束仍为 93.7 分钟；R4 仍不展示。本轮是新增对照记录，不覆盖原 R2。',
 '历史各轮最佳方案均为 CPU 训练；本轮为 CPA Luna-low / 4 核 CPU。R5、R6 续跑曾切换 Terra-low，非严格同条件对照。',
 '同一固定 search 划分，非 Kaggle 官方成绩；不含 tune。单次运行，无置信区间，不据此断言记忆或模型的因果收益。']
for yy,t in zip([.154,.123,.092,.061,.030],notes):fig.text(.065,yy,t,fontsize=9,color='#627684')
fig.savefig(OUT/'leaf_search_complete_latest.png',dpi=300,facecolor='white')
fig.savefig(OUT/'leaf_search_complete_latest.svg',facecolor='white')
fig.savefig(OUT/'leaf_search_complete_latest.pdf',facecolor='white')
plt.close(fig)
