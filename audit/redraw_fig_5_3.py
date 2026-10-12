import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
from matplotlib import font_manager
plt.rcParams.update({'font.family':'Liberation Serif','font.size':11})
# exact FINAL_04 CPD values (F05_RQ2_CPD_CRR_five_models.csv)
models=['Logistic\nRegression','Linear SVM','Multinomial\nNB','Random\nForest','XGBoost']
cpd={'Emoji-heavy':[0.016052,0.010134,0.025137,0.005815,-0.014482],
     'Slang-heavy':[0.143987,0.115768,0.004480,0.052529,0.058523],
     'Sarcasm-indicated*':[-0.072680,-0.012803,0.009241,-0.030805,-0.058144]}
cols={'Emoji-heavy':'#2E8582','Slang-heavy':'#D4952B','Sarcasm-indicated*':'#A8475A'}
fig,ax=plt.subplots(figsize=(6.1,3.5),dpi=300)
x=np.arange(len(models)); w=0.26
for i,(k,v) in enumerate(cpd.items()):
    bars=ax.bar(x+(i-1)*w,v,w,color=cols[k],label=k)
    for b,val in zip(bars,v):
        y=val+(0.006 if val>=0 else -0.006)
        ax.text(b.get_x()+b.get_width()/2,y,f'{val:+.3f}',rotation=90,ha='center',va='bottom' if val>=0 else 'top',fontsize=8)
ax.axhline(0,color='black',lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(models,fontsize=10)
ax.set_ylabel('Correctness Parity Difference (CPD)',fontsize=10)
ax.set_ylim(-0.135,0.225)
ax.set_title('CPD versus Reference-Unmarked by Model and Subgroup',fontsize=12,fontweight='bold',pad=26)
ax.legend(ncol=3,frameon=False,loc='lower center',bbox_to_anchor=(0.5,1.0),fontsize=10,handlelength=1.6)
for s in ('top','right'): ax.spines[s].set_visible(False)
fig.text(0.5,0.005,'Positive CPD: subgroup correctness exceeds reference-unmarked.  * N = 14, exploratory only.',ha='center',fontsize=9,style='italic')
fig.tight_layout(rect=(0,0.03,1,1))
fig.savefig('fig_5.3_fixed.png',dpi=300)
from PIL import Image; im=Image.open('fig_5.3_fixed.png'); print(im.size)
