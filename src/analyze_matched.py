"""Frozen paired stability analysis for the post-result matched follow-up."""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import csv
import json
import statistics as st
from .core import score_response,quote_norm,locate_quote,passages_for
from .statistics import book_level_summary,holm_adjust

ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def mean(v):return st.mean(v) if v else None
def write_csv(path,rows):
    if rows:
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def triple_metrics(anchor,repeat,reorder):
    valid=all(r['valid'] for r in (anchor,repeat,reorder))
    repeat_bad=not(anchor['valid'] and repeat['valid'] and anchor['trait']==repeat['trait'])
    reorder_bad=not(anchor['valid'] and reorder['valid'] and anchor['trait']==reorder['trait'])
    return {'all_three_valid':valid,'repeat_flip':int(anchor['trait']!=repeat['trait']) if valid else None,
            'rotation_flip':int(anchor['trait']!=reorder['trait']) if valid else None,
            'difference':int(anchor['trait']!=reorder['trait'])-int(anchor['trait']!=repeat['trait']) if valid else None,
            'stable_trait':valid and anchor['trait']==repeat['trait']==reorder['trait'],
            'operational_repeat_failure':int(repeat_bad),'operational_rotation_failure':int(reorder_bad),
            'operational_difference':int(reorder_bad)-int(repeat_bad)}

def infer(books,scale=1):
    effects={b:mean(v) for b,v in books.items() if v}
    if len(effects)!=10:
        return {'books':len(effects),'difference':mean(list(effects.values())),
                'ci95_low':None,'ci95_high':None,'sign_flip_p':None,'inference_available':False}
    s=book_level_summary([x/scale for x in effects.values()])
    return {'books':10,'difference':s['macro_difference']*scale,'ci95_low':s['t_ci95'][0]*scale,
            'ci95_high':s['t_ci95'][1]*scale,'sign_flip_p':s['sign_flip_p'],'inference_available':True}

def main():
    out=ROOT/'results/matched_v1'
    variants={v['variant_id']:v for v in read(ROOT/'data/processed/matched_variants.json')}
    gold={g['sample_id']:g for g in read(ROOT/'data/processed/matched120_gold.json')}
    logs=[json.loads(l) for l in (out/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
    ends=[r for r in logs if r['event']=='end'];terminal=[r for r in ends if r['status'] in ('response','refusal','transport_exhausted')]
    expected=sum(len(v['conditions'])*2 for v in variants.values())
    assert len({r['job_id'] for r in terminal})==len(terminal)
    if len(terminal)!=expected:print(json.dumps({'terminal_jobs':len(terminal),'expected':expected,'analysis':'deferred_until_complete'}));return
    scores=[];quote_sets={};usage=[]
    for r in terminal:
        v=variants[r['variant_id']];item=v['item'];g=gold[r['sample_id']]
        response=r.get('response') or {};first=(response.get('choices') or [{}])[0];msg=first.get('message') or {}
        s=score_response(msg.get('content'),item,g['gt'],r['condition'],first.get('finish_reason'),msg.get('refusal') or ('refused' if r['status']=='refusal' else None))
        if r['status']=='transport_exhausted':s['status']='transport_exhausted'
        meta={k:r[k] for k in ('job_id','variant_id','sample_id','provider','condition','kind','replicate')}
        scores.append({**meta,'book':g['book'],'author':g['author'],'api_status':r['status'],'response_model':response.get('model'),
                       'utc':r['utc'],'elapsed_seconds':r['elapsed_seconds'],'reasoning_content_chars':len(msg.get('reasoning_content') or ''),
                       **{k:x for k,x in s.items() if k!='quotes'}})
        quote_sets[r['job_id']]=set()
        if s['valid']:
            obj=json.loads(msg['content']);passages={p['passage_id']:p['text'] for p in passages_for(item,r['condition'])}
            quote_sets[r['job_id']]={(e['passage_id'],quote_norm(e['quote'])) for e in obj['evidence'] if e['passage_id'] in passages and 0<len(e['quote'].split())<=25 and locate_quote(e['quote'],passages[e['passage_id']])['matched']}
        u=response.get('usage') or {}
        usage.append({**meta,'prompt_tokens':u.get('prompt_tokens',0) or 0,'completion_tokens':u.get('completion_tokens',0) or 0,
                      'cached_tokens':u.get('prompt_cache_hit_tokens',(u.get('prompt_tokens_details') or {}).get('cached_tokens',0)) or 0,
                      'reasoning_tokens':(u.get('completion_tokens_details') or {}).get('reasoning_tokens'),'usage_present':bool(u)})
    index={(r['sample_id'],r['provider'],r['condition'],r['kind']):r for r in scores}
    triples=[]
    for p in ('glm','deepseek'):
        for c in 'EF':
            for sid,g in gold.items():
                a,b,r=[index[(sid,p,c,k)] for k in ('anchor','repeat','reorder')]
                m=triple_metrics(a,b,r)
                triples.append({'sample_id':sid,'book':g['book'],'author':g['author'],'provider':p,'condition':c,
                                **m,'anchor_valid':a['valid'],'anchor_correct':a['correct'],'anchor_traceable':a['traceable'],
                                'stable_reference_disagree':m['stable_trait'] and not a['correct'],
                                'all_three_traceable':all(x['traceable'] for x in (a,b,r)),
                                'rotation_shared_quote':bool(quote_sets[a['job_id']]&quote_sets[r['job_id']]),
                                'repeat_shared_quote':bool(quote_sets[a['job_id']]&quote_sets[b['job_id']])})
    results=[];bookrows=[];sensitivity=[];selection=[]
    for p in ('glm','deepseek'):
        for c in 'EF':
            rr=[r for r in triples if r['provider']==p and r['condition']==c]
            valid=[r for r in rr if r['all_three_valid']];bybook=defaultdict(list)
            for r in valid:bybook[r['book']].append(r['difference'])
            stat=infer(bybook)
            bounds=[];operations=[]
            for book in sorted({r['book'] for r in rr}):
                allb=[r for r in rr if r['book']==book];vb=[r for r in allb if r['all_three_valid']]
                n=len(allb);missing=n-len(vb);total=sum(r['difference'] for r in vb)
                lower=(total-missing)/n;upper=(total+missing)/n;bounds.append((lower,upper))
                operational=mean([r['operational_difference'] for r in allb]);operations.append(operational)
                bookrows.append({'provider':p,'condition':c,'book':book,'n':n,'valid_triples':len(vb),
                                 'repeat_flip_rate':mean([r['repeat_flip'] for r in vb]),'rotation_flip_rate':mean([r['rotation_flip'] for r in vb]),
                                 'difference':mean([r['difference'] for r in vb]),'all_group_lower_bound':lower,'all_group_upper_bound':upper,
                                 'operational_difference':operational})
            results.append({'provider':p,'condition':c,'primary':c=='F','groups':len(rr),'valid_triples':len(valid),
                            'repeat_flip_rate_micro':mean([r['repeat_flip'] for r in valid]),'rotation_flip_rate_micro':mean([r['rotation_flip'] for r in valid]),
                            'repeat_flip_rate_book_macro':mean([mean([r['repeat_flip'] for r in valid if r['book']==b]) for b in bybook]),
                            'rotation_flip_rate_book_macro':mean([mean([r['rotation_flip'] for r in valid if r['book']==b]) for b in bybook]),
                            **stat,'holm_p':None,'all_group_lower_bound':mean([b[0] for b in bounds]),'all_group_upper_bound':mean([b[1] for b in bounds]),
                            'operational_difference':mean(operations),'stable_triples':sum(r['stable_trait'] for r in rr),
                            'stable_reference_disagree':sum(r['stable_reference_disagree'] for r in rr),
                            'stable_reference_disagree_all_traceable':sum(r['stable_reference_disagree'] and r['all_three_traceable'] for r in rr),
                            'rotation_flips_sharing_quote':sum(r['rotation_flip']==1 and r['rotation_shared_quote'] for r in valid)})
            if c=='F':
                effects={b:mean(v) for b,v in bybook.items()};authors=defaultdict(set)
                for g in gold.values():authors[g['author']].add(g['book'])
                for b in effects:sensitivity.append({'provider':p,'analysis':'leave_one_book','omitted':b,'difference':mean([v for k,v in effects.items() if k!=b])})
                for a,bs in authors.items():sensitivity.append({'provider':p,'analysis':'leave_one_author','omitted':a,'difference':mean([v for k,v in effects.items() if k not in bs])})
                sensitivity.append({'provider':p,'analysis':'equal_author_descriptive','omitted':'','difference':mean([mean([effects[b] for b in bs if b in effects]) for bs in authors.values() if any(b in effects for b in bs)])})
                for rule in ('anchor_traceable','all_three_valid_same_trait','anchor_traceable_and_all_three_valid_same_trait'):
                    kept=[r for r in rr if (r['anchor_traceable'] if rule=='anchor_traceable' else r['stable_trait'] if rule=='all_three_valid_same_trait' else r['anchor_traceable'] and r['stable_trait'])]
                    random_correct=sum(sum(k['book']==b for k in kept)*mean([int(x['anchor_correct']) for x in rr if x['book']==b]) for b in {x['book'] for x in rr})
                    correct=sum(r['anchor_correct'] for r in kept);errors=sum(not r['anchor_correct'] for r in rr)
                    selection.append({'provider':p,'rule':rule,'n':len(rr),'retained':len(kept),'coverage':len(kept)/len(rr),
                                      'retained_reference_accuracy':correct/len(kept) if kept else None,
                                      'same_book_random_retention_expected_accuracy':random_correct/len(kept) if kept else None,
                                      'retained_reference_errors':len(kept)-correct,'all_anchor_reference_errors':errors,
                                      'error_retention_fraction':(len(kept)-correct)/errors if errors else None,
                                      'accuracy_minus_random_expectation':(correct-random_correct)/len(kept) if kept else None})
    primary=[r for r in results if r['primary']]
    if all(r['inference_available'] for r in primary):
        for r,pv in zip(primary,holm_adjust([r['sign_flip_p'] for r in primary])):r['holm_p']=pv
    interaction=[];tindex={(r['sample_id'],r['provider'],r['condition']):r for r in triples}
    for p in ('glm','deepseek'):
        diffs=defaultdict(list)
        for sid,g in gold.items():
            e,f=tindex[(sid,p,'E')],tindex[(sid,p,'F')]
            if e['all_three_valid'] and f['all_three_valid']:diffs[g['book']].append(f['difference']-e['difference'])
        interaction.append({'provider':p,'all_six_valid_n':sum(len(v) for v in diffs.values()),**infer(diffs,scale=2),'interpretation':'secondary exploratory F-minus-E excess flip contrast'})
    summary=[]
    for p in ('glm','deepseek'):
        for c in 'EF':
            for k in ('anchor','repeat','reorder'):
                rr=[r for r in scores if r['provider']==p and r['condition']==c and r['kind']==k]
                summary.append({'provider':p,'condition':c,'kind':k,'n':len(rr),'valid':sum(r['valid'] for r in rr),
                                'correct':sum(r['correct'] for r in rr),'traceable':sum(r['traceable'] for r in rr),
                                'length_compliant':sum(r['length_compliant'] for r in rr)})
    report={'generated_utc':datetime.now(timezone.utc).isoformat(),'complete':True,'planned_jobs':expected,'terminal_jobs':len(terminal),'attempts':len(ends),
            'api_status_counts':dict(Counter(r['status'] for r in ends)),'scoring_status_counts':dict(Counter(r['status'] for r in scores)),
            'first_call_utc':min(r['utc'] for r in logs),'last_call_utc':max(r['utc'] for r in logs),
            'estimated_cost_cny':sum(r['estimated_charge_cny'] for r in ends),
            'models':{p:sorted({r['response_model'] for r in scores if r['provider']==p and r['response_model']}) for p in ('glm','deepseek')},
            'reasoning_content_nonempty':sum(r['reasoning_content_chars']>0 for r in scores),
            'usage_missing':sum(not u['usage_present'] for u in usage),'summary':summary,'paired_results':results,'secondary_interaction':interaction,'selection_rules':selection,
            'interpretation':'Post-result follow-up on existing groups. Primary family: two F contrasts, equal-book, common valid triples. Sign flips require independent symmetric book differences; same authors and fixed corpus limit inference. Invalid cases reported in coverage and full-group bounds. Shared quotation does not prove semantic contradiction or model reasoning faithfulness.'}
    for name,rr in [('scores',scores),('usage',usage),('triples',triples),('condition_summary',summary),('paired_results',results),('by_book',bookrows),('sensitivity',sensitivity),('selection_rules',selection),('secondary_interaction',interaction)]:write_csv(out/(name+'.csv'),rr)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['同题重复与换序后续实验报告','',f'完成{len(terminal)}/{expected}项；120题、10部作品、2模型；E分析、F证据分析。',
           '三种回答均在本批次重新采集：anchor原顺序、repeat同输入重复、reorder仅选项换序。',
           '主要指标为相对同一anchor的换序翻转率减同序重复翻转率，使用三次均有效的共同题集，再按作品同权平均。','']
    for r in results:
        lines.append(f"{r['provider']} {r['condition']}: 有效三元组{r['valid_triples']}/{r['groups']}；作品平均超额翻转差={r['difference']}，95% CI=[{r['ci95_low']}, {r['ci95_high']}]，p={r['sign_flip_p']}，Holm p={r['holm_p']}。")
    lines+=['','F的两模型比较组成主要检验族；E、F−E及筛选规则为次要/探索性分析。',
            '本批为已有题目的事后扩展，不是新语料独立验证。引文可定位不代表语义充分；不显著不代表等效。',
            f"费用估算{report['estimated_cost_cny']:.2f}元；GLM单价未核实，非实际账单。"]
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'complete':True,'terminal_jobs':len(terminal),'primary':primary,'scoring_status_counts':report['scoring_status_counts'],'estimated_cost_cny':report['estimated_cost_cny']},ensure_ascii=False))

if __name__=='__main__':main()
