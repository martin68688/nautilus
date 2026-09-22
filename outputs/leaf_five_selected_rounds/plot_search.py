"""Five selected rounds, host-certified search only; no inferred trial scores."""
from pathlib import Path
import csv
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

OUT = Path(__file__).resolve().parent
rows = list(csv.DictReader((OUT/'outer_search.csv').open()))
summary = json.loads((OUT/'summary.json').read_text())['rounds']
assert len(rows) == 35
font = Path('/System/Library/Fonts/STHeiti Light.ttc')
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':'sans-serif',
 'font.sans-serif':[font_manager.FontProperties(fname=font).get_name(),'Arial','DejaVu Sans'],
 'font.size':11,'axes.labelsize':12,'axes.titlesize':13,'axes.titlepad':14,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#BBC8D4',
 'axes.labelcolor':'#304454','text.color':'#304454','xtick.color':'#4F6274','ytick.color':'#4F6274',
 'svg.fonttype':'none','pdf.fonttype':42,'axes.unicode_minus':False,'legend.frameon':False})
colors=['#327EB4','#CC842C','#795AAF','#B95773','#248E8B']
markers=['o','s','D','^','P']; styles=['-','-','--','-.','-']
labels=['R1 首轮','R2 第二轮','R3 第三轮重跑','R4 历史第四轮','R5 最新全记忆轮']
fig=plt.figure(figsize=(16,7.8),facecolor='white')
fig.text(.075,.945,'Leaf · 五轮 Search 成绩曲线',fontsize=25)
fig.text(.075,.888,'只展示宿主验证的固定 search log-loss（越低越好），不含 tune 分数',fontsize=13)
fig.text(.075,.846,'Luna-low · A10 × 1 · 同一固定划分 · 各轮独立计时',fontsize=11,color='#71859A')
handles=[Line2D([0],[0],color=col,marker=mk,ls=ls,lw=2,
 label=f"{label}  {s['final_search']:.6f}") for s,col,mk,ls,label in zip(summary,colors,markers,styles,labels)]
fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.07,.81),ncol=3,fontsize=10.5,columnspacing=2)
a=fig.add_axes([.075,.25,.555,.43]); b=fig.add_axes([.735,.25,.24,.43])
for ax in [a,b]:
 ax.grid(axis='y',color='#E6EDF3',lw=.8);ax.set_axisbelow(True)
a.set_title('a  全部五轮：实际成绩与本轮累计最佳',loc='left')
b.set_title('b  后两轮低损失区域放大',loc='left')
for ri,(s,col,mk,ls) in enumerate(zip(summary,colors,markers,styles),1):
 good=[r for r in rows if int(r['round'])==ri and r['certified']=='True']
 assert len(good)==[2,2,10,1,3][ri-1]
 xs=[float(r['elapsed_minutes']) for r in good];ys=[float(r['search_loss']) for r in good]
 assert all(math.isfinite(v) and v>0 for v in ys)
 best=[min(ys[:i+1]) for i in range(len(ys))]
 assert abs(best[-1]-s['final_search'])<1e-12
 for ax in ([a,b] if ri>=4 else [a]):
  ax.step(xs+[s['elapsed_minutes']],best+[best[-1]],where='post',color=col,ls=ls,lw=2.2)
  ax.scatter(xs,ys,color=col,marker=mk,s=34,alpha=.8,zorder=4)
  ax.plot(s['elapsed_minutes'],best[-1],marker='|',ms=9,color=col)
 if ri<=3:
  a.annotate(f'R{ri}  {best[-1]:.6f}',(s['elapsed_minutes'],best[-1]),
   xytext=(-5,14 if ri==1 else -20),textcoords='offset points',ha='right',color=col,fontsize=10)
a.set(xlim=(0,120),ylim=(0,.8),xticks=range(0,121,20),yticks=[0,.2,.4,.6,.8],
 xlabel='累计墙钟时间 / 分钟',ylabel='固定 search log-loss ↓')
a.text(.025,.19,'R4 / R5 的微小差距见右侧放大图 →',transform=a.transAxes,fontsize=10,color='#71859A')
b.set(xlim=(0,75),ylim=(.0423,.0462),xticks=[0,15,30,45,60,75],
 yticks=[.0425,.0435,.0445,.0455],xlabel='累计墙钟时间 / 分钟',ylabel='固定 search log-loss ↓')
b.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:.4f}'))
b.text(.035,.945,'差值 0.0000816\n相对下降 0.19%',transform=b.transAxes,va='top',fontsize=10)
b.annotate('R4  0.042718\n13.3 分钟停止',(summary[3]['elapsed_minutes'],summary[3]['final_search']),
 xytext=(8,.04325),fontsize=9.5,color=colors[3],arrowprops=dict(arrowstyle='-',color=colors[3]))
b.annotate('R5  0.042636\n68.0 分钟停止',(summary[4]['elapsed_minutes'],summary[4]['final_search']),
 xytext=(42,.04325),fontsize=9.5,color=colors[4],arrowprops=dict(arrowstyle='-',color=colors[4]))
fig.text(.075,.14,'实点：18 个 host 验证成绩；阶梯线：本轮历史最佳；竖线：实际结束。时间含初始化、规划、审查、训练与评分。',fontsize=10,color='#71859A')
fig.text(.075,.097,'R4、R5 均因模型服务不可用提前结束；曲线不补齐时长。R5 为本图编号，并非第五次启动。',fontsize=10,color='#71859A')
fig.text(.075,.054,'单次自适应搜索，0.19% 的差异不能证明稳定收益；这是内部固定 search 成绩，非 Kaggle 官方成绩。',fontsize=10,color='#71859A')
fig.savefig(OUT/'leaf_five_search.svg',facecolor='white')
fig.savefig(OUT/'leaf_five_search.pdf',facecolor='white')
fig.savefig(OUT/'leaf_five_search.png',dpi=300,facecolor='white')
plt.close(fig)
