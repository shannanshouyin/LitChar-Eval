"""Frozen conditional paired deletion analysis; original labels are descriptive only."""
from pathlib import Path
from collections import defaultdict, Counter
import csv
import json
import statistics as st
from .core import score_response
from .statistics import book_level_summary, holm_adjust

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def mean(xs):return st.mean(xs) if xs else None
def write_csv(p,rows):
    if rows:
        with p.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def paired_metrics(a,c,r):
    valid=all(x['valid'] for x in (a,c,r))
    cf=int(not(a['valid'] and c['valid'] and a['trait']==c['trait']))
    rf=int(not(a['valid'] and r['valid'] and a['trait']==r['trait']))
    return {'all_three_valid':valid,'cited_flip':cf if valid else None,'control_flip':rf if valid else None,
            'difference':cf-rf if valid else None,'operational_difference':cf-rf}

def main():
    out=ROOT/'results/ablation_v1';gold={g['sample_id']:g for g in read(ROOT/'data/processed/matched120_gold.json')}
    scores=[];events=[];cost=0
    for phase in ('base','edit'):
        assert read(out/phase/'last_execution.json')['all_jobs_terminal']
        variants={v['variant_id']:v for v in read(ROOT/f'data/processed/ablation_{phase}_variants.json')}
        logs=[json.loads(l) for l in (out/phase/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
        events+=logs
        ends=[r for r in logs if r['event']=='end'];cost+=sum(r['estimated_charge_cny'] for r in ends)
        terminal=[r for r in ends if r['status'] in ('response','refusal','transport_exhausted')]
        assert len({r['job_id'] for r in terminal})==len(terminal)
        for r in terminal:
            item=variants[r['variant_id']]['item'];g=gold[r['sample_id']];resp=r.get('response') or {}
            first=(resp.get('choices') or [{}])[0];msg=first.get('message') or {}
            s=score_response(msg.get('content'),item,g['gt'],'F',first.get('finish_reason'),msg.get('refusal') or ('refused' if r['status']=='refusal' else None))
            scores.append({**{k:r[k] for k in ('job_id','sample_id','provider','kind','utc')},'book':g['book'],'author':g['author'],
                           'response_model':resp.get('model'),'api_status':r['status'],**{k:v for k,v in s.items() if k!='quotes'}})
    index={(s['sample_id'],s['provider'],s['kind']):s for s in scores}
    eligibility=read(ROOT/'data/processed/ablation_eligibility.json');pairs=[]
    for e in eligibility:
        if e['eligible']:
            a,c,r=[index[(e['sample_id'],e['provider'],k)] for k in ('anchor','cited_deletion','control_deletion')]
            pairs.append({**{k:e[k] for k in ('sample_id','provider','book','author','removed_words')},**paired_metrics(a,c,r),
                          'anchor_correct':a['correct'],'cited_correct':c['correct'],'control_correct':r['correct']})
    summary=[];bookrows=[];sensitivity=[]
    for p in ('glm','deepseek'):
        es=[e for e in eligibility if e['provider']==p];ps=[r for r in pairs if r['provider']==p];vs=[r for r in ps if r['all_three_valid']]
        effects={};bounds=[];operations=[]
        for b in sorted({e['book'] for e in es}):
            allb=[r for r in ps if r['book']==b];vb=[r for r in vs if r['book']==b]
            n=len(allb);missing=n-len(vb);total=sum(r['difference'] for r in vb)
            if vb:effects[b]=mean([r['difference'] for r in vb])
            lo=(total-missing)/n if n else None;hi=(total+missing)/n if n else None
            if n:bounds.append((lo,hi));operations.append(mean([r['operational_difference'] for r in allb]))
            bookrows.append({'provider':p,'book':b,'anchors':12,'eligible_pairs':n,'valid_triples':len(vb),
                             'cited_flip_rate':mean([r['cited_flip'] for r in vb]),'control_flip_rate':mean([r['control_flip'] for r in vb]),
                             'difference':effects.get(b),'eligible_lower_bound':lo,'eligible_upper_bound':hi})
        inference=book_level_summary(list(effects.values())) if len(effects)==10 else None
        result={'provider':p,'anchors':120,'eligible_pairs':len(ps),'valid_triples':len(vs),'books':len(effects),
                'exclusions':dict(Counter(e['exclusion_reason'] for e in es if not e['eligible'])),
                'removed_words_median':st.median([r['removed_words'] for r in ps]) if ps else None,
                'cited_flip_rate_micro':mean([r['cited_flip'] for r in vs]),'control_flip_rate_micro':mean([r['control_flip'] for r in vs]),
                'cited_flip_rate_book_macro':mean([r['cited_flip_rate'] for r in bookrows if r['provider']==p and r['valid_triples']]),
                'control_flip_rate_book_macro':mean([r['control_flip_rate'] for r in bookrows if r['provider']==p and r['valid_triples']]),
                'difference':mean(list(effects.values())),'ci95':inference['t_ci95'] if inference else None,
                'sign_flip_p':inference['sign_flip_p'] if inference else None,'holm_p':None,
                'eligible_lower_bound':mean([x[0] for x in bounds]),'eligible_upper_bound':mean([x[1] for x in bounds]),
                'operational_difference':mean(operations),'inference_available':bool(inference)}
        summary.append(result)
        authors=defaultdict(set)
        for g in gold.values():authors[g['author']].add(g['book'])
        ae={a:mean([effects[b] for b in bs if b in effects]) for a,bs in authors.items() if any(b in effects for b in bs)}
        if len(ae)>=2:
            sensitivity.append({'provider':p,'kind':'equal_author','omitted':'',**book_level_summary(list(ae.values()))})
        for b in effects:sensitivity.append({'provider':p,'kind':'leave_one_book','omitted':b,'macro_difference':mean([v for k,v in effects.items() if k!=b])})
        for a,bs in authors.items():sensitivity.append({'provider':p,'kind':'leave_one_author','omitted':a,'macro_difference':mean([v for k,v in effects.items() if k not in bs])})
    if all(r['inference_available'] for r in summary):
        for r,pv in zip(summary,holm_adjust([r['sign_flip_p'] for r in summary])):r['holm_p']=pv
    authorrows=[r for r in sensitivity if r['kind']=='equal_author']
    if len(authorrows)==2:
        for r,pv in zip(authorrows,holm_adjust([r['sign_flip_p'] for r in authorrows])):r['holm_p']=pv
    value={'complete':True,'terminal_jobs':len(scores),'attempts':sum(e['event']=='start' for e in events),
           'valid_responses':sum(r['valid'] for r in scores),'estimated_cost_cny':cost,
           'utc_range':[min(e['utc'] for e in events),max(e['utc'] for e in events)],'paired_results':summary,
           'sensitivity':sensitivity,'note':'Conditional on traceable uniquely located anchor quotes and matched control availability. Changed-text reference agreement is descriptive, not task accuracy.'}
    write_csv(out/'responses.csv',scores);write_csv(out/'paired_groups.csv',pairs);write_csv(out/'book_effects.csv',bookrows)
    (out/'summary.json').write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['QUOTE DELETION FOLLOW-UP','Post-result experiment; paired cited-token versus same-passage matched-token deletion.',json.dumps(value,ensure_ascii=False,indent=2)]
    (out/'REPORT.txt').write_text('\n\n'.join(lines),encoding='utf-8');print(json.dumps(value,ensure_ascii=False))
if __name__=='__main__':main()
