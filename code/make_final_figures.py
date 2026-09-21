import sys; sys.path.insert(0,'.')
import figures as F, numpy as np, pandas as pd, matplotlib.pyplot as plt, os
R='../runs/'
def L(f): return pd.read_csv(R+f).set_index('Model')
iid,d5,d1=L('results_best.csv'),L('results_best_dir0.5.csv'),L('results_best_dir0.1.csv')
s43=[L('results_best_seed43_%s.csv'%k) for k in ('iid','dir0.5','dir0.1')]
# --- tiers
names=['MLP-FL','MLP-INT8','BNN-FL','BNN-INT8IO\n(old)','BNN-INT8IO\nmatched']
acc=[92.08,84.59,97.75,85.08,81.75]; dn=[63.77,18.98,23.52,13.23,8.72]; up=[63.77,63.77,62.27,128.78,62.27]
col=['#1f77b4','#c77aa8','#d95f02','#f0e442','#56b4e9']
fig,ax=plt.subplots(1,2,figsize=(F.COL_DOUBLE,2.6)); x=np.arange(5)
ax[0].bar(x,acc,color=col,edgecolor='black',lw=.4)
for i,a in enumerate(acc): ax[0].text(i,a+1.5,f'{a:.2f}',ha='center',fontsize=7)
ax[0].set_ylim(0,110); ax[0].set_ylabel('Best accuracy (%)'); ax[0].set_title('(a) Detection accuracy',fontsize=8)
ax[1].bar(x,dn,color=col,edgecolor='black',lw=.4,label='Downlink')
ax[1].bar(x,up,bottom=dn,color=col,edgecolor='black',lw=.4,alpha=.45,hatch='//',label='Uplink')
for i in range(5): ax[1].text(i,dn[i]+up[i]+3,f'{dn[i]+up[i]:.1f}',ha='center',fontsize=7)
ax[1].set_ylim(0,165); ax[1].set_ylabel('Traffic per round (KB)'); ax[1].set_title('(b) Round trip (solid = downlink, hatched = uplink)',fontsize=8)
for a in ax:
    a.set_xticks(x); a.set_xticklabels(names,fontsize=6.5,rotation=15,ha='right'); a.grid(axis='y',alpha=.25,ls='--',lw=.5); F._style_axes(a)
fig.tight_layout(); F._save(fig,'fig_precision_tiers')
# --- bipru
fig,ax=plt.subplots(1,2,figsize=(F.COL_SINGLE,2.5))
nm=['BNN-FL','BiPruneFL\nRepro','MLP-FL']; ks=['BNN-MATCHED','BiPruneFL-Repro','MLP']
c=['#d95f02','#009e73','#1f77b4']; x=np.arange(3)
for a,key,t,yl,fmt in ((ax[0],'Accuracy(%)','(a) Accuracy','Best accuracy (%)','{:.2f}'),(ax[1],'PackedPayload(KB)','(b) Downlink','KB per round','{:.1f}')):
    v=[iid.loc[k,key] for k in ks]; a.bar(x,v,color=c,edgecolor='black',lw=.4)
    for i,y in enumerate(v): a.text(i,y*1.01,fmt.format(y),ha='center',va='bottom',fontsize=7)
    a.set_title(t,fontsize=8); a.set_ylabel(yl,fontsize=7); a.set_xticks(x); a.set_xticklabels(nm,fontsize=6.5,rotation=20,ha='right')
    a.set_ylim(0,max(v)*1.2); a.grid(axis='y',alpha=.25,ls='--',lw=.5); F._style_axes(a)
fig.tight_layout(); F._save(fig,'fig_bipru')
# --- noniid
fig,ax=plt.subplots(figsize=(F.COL_SINGLE,2.7)); xs=np.arange(3)
for n in [m for m in F.ORDER if m in iid.index]:
    s=F.STYLE[n]; y=[d.loc[n,'Accuracy(%)'] for d in (iid,d5,d1)]
    ax.plot(xs,y,color=s['c'],marker=s['m'],ls=s['ls'],ms=4,lw=1.2,label=F.LABEL[n]+' (42)')
for n,c2 in (('BNN-MATCHED','#8c2d04'),('BNN-INT8IO','#0072b2')):
    y=[d.loc[n,'Accuracy(%)'] for d in s43]
    ax.plot(xs,y,color=c2,marker='x',ls=':',ms=5,lw=1.3,label=('BNN-FL' if n=='BNN-MATCHED' else 'BNN-INT8IO')+' (43)')
ax.set_xticks(xs); ax.set_xticklabels(['IID',r'$\alpha$=0.5'+'\n(moderate)',r'$\alpha$=0.1'+'\n(severe)'])
ax.set_ylabel('Best accuracy (%)'); ax.legend(fontsize=5.5,ncol=2,loc='lower left',edgecolor='black',framealpha=1)
ax.grid(alpha=.25,ls='--',lw=.5); F._style_axes(ax); fig.tight_layout(); F._save(fig,'fig_noniid')
print(F.FIG_DIR)
