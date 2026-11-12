"""Generate manuscript figures from published numerical results."""
from pathlib import Path
import os,json,csv
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
OUT=ROOT/'outputs/figures';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
NAMES={'glm':'GLM-4.6','deepseek':'DeepSeek flash'};COLORS={'glm':'#256A9A','deepseek':'#B44F26'}
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
def csvout(name,rows):
    with (OUT/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save(fig,name):
    for ext in ('png','pdf','svg'):fig.savefig(OUT/(name+'.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)
def main():
    a=read('published_data/formal_v1/summary.json');m=read('published_data/matched_v1/summary.json');d=read('published_data/ablation_v1/summary.json')
    au=read('published_data/matched_v1/author_sensitivity.json')['results']
    fig,ax=plt.subplots(figsize=(7,4.0));ax.set_xlim(0,10);ax.set_ylim(0,10);ax.axis('off')
    boxes=[(5,9,'PersoNet public test split\n2,530 eligible source groups from 10 books'),
           (5,6.7,'Main study\n600 groups × 2 models × 6 conditions = 7,200 evaluated tasks'),
           (5,4.3,'Matched stability follow-up\n120 groups × 2 models × 2 prompts × 3 versions = 1,440 responses'),
           (5,1.7,f'Quote-deletion follow-up\n240 new anchors → {sum(r["eligible_pairs"] for r in d["paired_results"])} eligible pairs\n{d["terminal_jobs"]-240} edited-text responses')]
    for x,y,t in boxes:ax.text(x,y,t,ha='center',va='center',fontsize=10,bbox={'boxstyle':'round,pad=0.65','fc':'#F2F5F7','ec':'#667884'})
    for y1,y2 in [(8.2,7.5),(5.85,5.1),(3.4,2.65)]:ax.annotate('',xy=(5,y2),xytext=(5,y1),arrowprops={'arrowstyle':'->','color':'#444444','lw':1.5})
    save(fig,'Figure_1_Study_design')
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.5),gridspec_kw={'width_ratios':[1,1.3]})
    ax=axes[0]
    for p in NAMES:
        rs=[r for r in a['main_summary'] if r['provider']==p]
        ax.plot(list('ABCDEF'),[100*r['book_macro_accuracy'] for r in rs],'-o',color=COLORS[p],label=NAMES[p],ms=5)
    ax.set_ylim(0,60);ax.set_ylabel('Reference agreement (%)');ax.set_xlabel('Prompt condition');ax.axhline(20,ls=':',color='#888888',lw=1);ax.text(0,21,'Uniform guess',fontsize=8,color='#666666');ax.set_title('A   Main study',loc='left',fontweight='bold');ax.legend(frameon=False,fontsize=9,loc='lower right')
    ax=axes[1];rs=a['joint_evidence_outcomes'];bottom=np.zeros(4)
    keys=['correct_traceable','incorrect_traceable','correct_not_traceable','incorrect_not_traceable']
    labs=['Agrees, traceable','Disagrees, traceable','Agrees, not traceable','Disagrees or invalid, not traceable']
    cs=['#256A9A','#CE784C','#92B5CD','#D9D9D9']
    for k,l,c in zip(keys,labs,cs):
        v=np.array([100*r[k]/r['n'] for r in rs]);ax.bar(range(4),v,bottom=bottom,color=c,label=l,width=.65);bottom+=v
    ax.set_xticks(range(4),['GLM\nC','GLM\nF','DeepSeek\nC','DeepSeek\nF']);ax.set_ylim(0,100);ax.set_ylabel('All responses (%)');ax.set_title('B   Traceability and agreement',loc='left',fontweight='bold')
    handles,legend_labels=ax.get_legend_handles_labels();fig.legend(handles,legend_labels,frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(.5,0),ncol=2)
    fig.tight_layout();save(fig,'Figure_2_Agreement_traceability');csvout('Figure_2A_source',a['main_summary']);csvout('Figure_2B_source',rs)
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.5),gridspec_kw={'width_ratios':[1,1.25]})
    primary=[r for r in m['paired_results'] if r['primary']];ax=axes[0];x=np.arange(2);w=.31
    ax.bar(x-w/2,[100*r['repeat_flip_rate_book_macro'] for r in primary],w,label='Same-order repeat',color='#9BA9B2')
    ax.bar(x+w/2,[100*r['rotation_flip_rate_book_macro'] for r in primary],w,label='Option rotation',color='#256A9A')
    ax.set_xticks(x,[NAMES[r['provider']] for r in primary]);ax.set_ylim(0,40);ax.set_ylabel('Trait changes from anchor (%)');ax.set_title('A   Matched F responses',loc='left',fontweight='bold');ax.legend(frameon=False,fontsize=8,loc='upper left')
    ax=axes[1];forest=[]
    for j,r in enumerate(primary):
        ar=next(v for v in au if v['provider']==r['provider'])
        for k,(est,ci,label) in enumerate([(r['difference'],[r['ci95_low'],r['ci95_high']],'Book'),(ar['equal_author_difference'],ar['t_ci95'],'Author')]):
            y=3-2*j-k;ax.errorbar(est*100,y,xerr=[[100*(est-ci[0])],[100*(ci[1]-est)]],fmt='o' if k==0 else 's',color=COLORS[r['provider']],capsize=3)
            forest.append({'provider':r['provider'],'aggregation':label,'difference':est,'ci_low':ci[0],'ci_high':ci[1]})
    ax.set_yticks([3,2,1,0],['GLM · books','GLM · authors','DeepSeek · books','DeepSeek · authors']);ax.axvline(0,color='#888888',ls='--',lw=1);ax.set_xlabel('Rotation minus repeat (percentage points)');ax.set_ylim(-.6,3.6);ax.set_title('B   Aggregation sensitivity',loc='left',fontweight='bold')
    fig.tight_layout();save(fig,'Figure_3_Matched_stability');csvout('Figure_3_source',forest)
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.7),gridspec_kw={'width_ratios':[1,1.25]});ax=axes[0];dr=d['paired_results']
    ax.bar(x-w/2,[100*r['control_flip_rate_book_macro'] for r in dr],w,label='Control deletion',color='#9BA9B2')
    ax.bar(x+w/2,[100*r['cited_flip_rate_book_macro'] for r in dr],w,label='Cited-text deletion',color='#B44F26')
    ax.set_xticks(x,[NAMES[r['provider']] for r in dr]);ax.set_ylim(0,max(50,max(r['cited_flip_rate_book_macro']*100 for r in dr)+12));ax.set_ylabel('Trait changes from fresh anchor (%)');ax.set_title('A   Matched deletion',loc='left',fontweight='bold');ax.legend(frameon=False,fontsize=8)
    for i,r in enumerate(dr):ax.text(i,-.24,f"{r['valid_triples']} valid / {r['eligible_pairs']} eligible\nfrom 120 anchors",transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
    ax=axes[1]
    for i,r in enumerate(dr):
        est=r['difference'];ci=r['ci95']
        if ci:
            ax.errorbar(est*100,1-i,xerr=[[100*(est-ci[0])],[100*(ci[1]-est)]],fmt='o',color=COLORS[r['provider']],capsize=4)
        else:ax.plot(est*100,1-i,'o',mfc='white',mec=COLORS[r['provider']],ms=7)
        note=f"{r['books']} books"
        ax.text(.08,.91-i*.45,note,transform=ax.transAxes,fontsize=8)
    ax.set_yticks([1,0],[NAMES[r['provider']] for r in dr]);ax.set_ylim(-.5,1.5);ax.axvline(0,color='#888888',ls='--',lw=1);ax.set_xlabel('Cited minus control (percentage points)');ax.set_title('B   Paired book-level difference',loc='left',fontweight='bold');fig.tight_layout();save(fig,'Figure_4_Quote_deletion')
    csvout('Figure_4_source',[{k:v for k,v in r.items() if k!='exclusions'} for r in dr])
    fig,axes=plt.subplots(1,2,figsize=(8.0,3.7),sharey=True)
    for ax,p in zip(axes,NAMES):
        rr=[r for r in m['selection_rules'] if r['provider']==p];labels=['Traceable','Stable','Both']
        for r,l in zip(rr,labels):
            xx=100*r['coverage'];yy=100*r['retained_reference_accuracy'];rand=100*r['same_book_random_retention_expected_accuracy']
            ax.plot([xx,xx],[rand,yy],color=COLORS[p],lw=1.5);ax.scatter(xx,yy,color=COLORS[p],s=40);ax.scatter(xx,rand,marker='x',color='#555555',s=35)
            offset=(-10,12) if p=='glm' and l=='Both' else ((10,-1) if p=='glm' and l=='Stable' else (4,4))
            ax.annotate(l,(xx,yy),xytext=offset,textcoords='offset points',fontsize=9,ha='right' if p=='glm' and l=='Both' else 'left')
        ax.set_xlim(40,108);ax.set_ylim(35,68);ax.set_title(NAMES[p]);ax.set_xlabel('Retained responses (%)')
    axes[0].set_ylabel('Agreement among retained responses (%)');fig.text(.5,.01,'Circles: observed selection    Crosses: same-book random-retention expectation',ha='center',fontsize=8);fig.tight_layout(rect=(0,.04,1,1));save(fig,'Figure_A1_Selective_review');csvout('Figure_A1_source',m['selection_rules'])
    print(json.dumps({'figures':5,'formats':['png 300 dpi','pdf','svg'],'directory':str(OUT)}))
if __name__=='__main__':main()
