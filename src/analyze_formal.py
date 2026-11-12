"""Prespecified formal analysis. Inferential tables require complete collection."""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import json
import csv
import statistics as stats
from .core import score_response
from .statistics import book_level_summary,holm_adjust

ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def csv_write(path,rows):
    if not rows:return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def mean(values):return stats.mean(values) if values else None
def aggregate(rows):
    books=defaultdict(list)
    for r in rows:books[r['book']].append(int(r['correct']))
    return {'n':len(rows),'valid':sum(r['valid'] for r in rows),'correct':sum(r['correct'] for r in rows),
            'micro_accuracy':mean([int(r['correct']) for r in rows]),
            'book_macro_accuracy':mean([mean(v) for v in books.values()]),
            'traceable':sum(r['traceable'] for r in rows),'valid_quotes':sum(r['valid_quotes'] for r in rows),
            'matched_quotes':sum(r['matched_quotes'] for r in rows),'nonempty_quotes':sum(r['nonempty_quotes'] for r in rows),
            'length_compliant':sum(r['length_compliant'] for r in rows),
            'key_order_compliant':sum(r['key_order_compliant'] for r in rows)}

def main():
    out=ROOT/'results/formal_v1';out.mkdir(parents=True,exist_ok=True)
    variants={v['variant_id']:v for v in read(ROOT/'data/processed/formal_variants.json')}
    gold={g['sample_id']:g for g in read(ROOT/'data/processed/formal600_gold.json')}
    logs=[json.loads(l) for l in (out/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
    ends=[r for r in logs if r['event']=='end'];terminal=[r for r in ends if r['status'] in ('response','refusal','transport_exhausted')]
    assert len({r['job_id'] for r in terminal})==len(terminal)
    expected=sum(len(v['conditions'])*2 for v in variants.values())
    if len(terminal)!=expected:
        print(json.dumps({'terminal_jobs':len(terminal),'expected':expected,'analysis':'deferred_until_complete'}));return
    scores=[]; quotes=[];usage=[]
    for r in terminal:
        variant=variants[r['variant_id']];item=variant['item'];g=gold[r['sample_id']]
        response=r.get('response') or {};choices=response.get('choices') or [];first=choices[0] if choices else {};msg=first.get('message') or {}
        score=score_response(msg.get('content'),item,g['gt'],r['condition'],first.get('finish_reason'),
                             msg.get('refusal') or ('refused' if r['status']=='refusal' else None))
        if r['status']=='transport_exhausted':score['status']='transport_exhausted'
        meta={k:r[k] for k in ('job_id','variant_id','sample_id','kind','replicate','provider','condition')}
        row={**meta,'book':g['book'],'author':g['author'],'character':g['character'],'api_status':r['status'],
             'response_model':response.get('model'),'response_utc':r['utc'],'latency_seconds':r['elapsed_seconds'],
             'reasoning_content_chars':len(msg.get('reasoning_content') or ''),
             **{k:v for k,v in score.items() if k!='quotes'}}
        scores.append(row)
        for index,q in enumerate(score['quotes']):quotes.append({**meta,'quote_index':index,**q})
        u=response.get('usage') or {}
        usage.append({**meta,'prompt_tokens':u.get('prompt_tokens',0) or 0,'completion_tokens':u.get('completion_tokens',0) or 0,
                      'cached_tokens':u.get('prompt_cache_hit_tokens',(u.get('prompt_tokens_details') or {}).get('cached_tokens',0)) or 0,
                      'reasoning_tokens':(u.get('completion_tokens_details') or {}).get('reasoning_tokens'),
                      'usage_available':bool(u)})
    main_rows=[r for r in scores if r['kind']=='main'];lookup={(r['sample_id'],r['provider'],r['condition']):r for r in main_rows}
    summary=[];bybook=[];joint=[]
    for p in ('glm','deepseek'):
        for c in 'ABCDEF':
            rows=[r for r in main_rows if r['provider']==p and r['condition']==c]
            summary.append({'provider':p,'condition':c,**aggregate(rows)})
            for b in sorted({r['book'] for r in rows}):bybook.append({'provider':p,'condition':c,'book':b,**aggregate([r for r in rows if r['book']==b])})
            if c in 'CF':
                joint.append({'provider':p,'condition':c,'n':len(rows),
                              'correct_traceable':sum(r['correct'] and r['traceable'] for r in rows),
                              'correct_not_traceable':sum(r['correct'] and not r['traceable'] for r in rows),
                              'incorrect_traceable':sum(not r['correct'] and r['traceable'] for r in rows),
                              'incorrect_not_traceable':sum(not r['correct'] and not r['traceable'] for r in rows),
                              'invalid_subset':sum(not r['valid'] for r in rows)})
    comparisons=[];book_effects=[];sensitivity=[]
    contrasts=[('C-B',{'C':1,'B':-1},True),('F-E',{'F':1,'E':-1},True),
               ('interaction',{'F':1,'E':-1,'C':-1,'B':1},False),
               ('D-A',{'D':1,'A':-1},False),('E-B',{'E':1,'B':-1},False),('F-C',{'F':1,'C':-1},False)]
    for p in ('glm','deepseek'):
        for name,weights,primary in contrasts:
            diffs=defaultdict(list);characters=defaultdict(list);filtered={s:defaultdict(list) for s in ('valid_pairs','length_compliant_pairs','response_only')}
            for sid,g in gold.items():
                paired=[lookup[(sid,p,c)] for c in weights]
                delta=sum(weights[c]*int(lookup[(sid,p,c)]['correct']) for c in weights)
                diffs[g['book']].append(delta);characters[(g['book'],g['character'])].append(delta)
                for label,keep in [('valid_pairs',all(r['valid'] for r in paired)),
                                   ('length_compliant_pairs',all(r['valid'] and r['length_compliant'] for r in paired)),
                                   ('response_only',all(r['api_status']!='transport_exhausted' for r in paired))]:
                    if keep:filtered[label][g['book']].append(delta)
            effects={b:mean(v) for b,v in diffs.items()}
            # Interaction ranges [-2,2]; scale only for the bounded generic summary utility.
            scale=2 if name=='interaction' else 1
            stat=book_level_summary([v/scale for v in effects.values()]);stat['macro_difference']*=scale;stat['t_ci95']=[v*scale for v in stat['t_ci95']]
            comparisons.append({'provider':p,'contrast':name,'primary':primary,'paired_n':len(gold),
                                'book_macro_difference':stat['macro_difference'],'ci95_low':stat['t_ci95'][0],
                                'ci95_high':stat['t_ci95'][1],'sign_flip_p':stat['sign_flip_p'],'holm_p':None})
            for b,v in effects.items():book_effects.append({'provider':p,'contrast':name,'book':b,'n':len(diffs[b]),'difference':v})
            if primary:
                for b in effects:sensitivity.append({'provider':p,'contrast':name,'analysis':'leave_one_book','omitted':b,'books':len(effects)-1,'paired_n':sum(len(v) for k,v in diffs.items() if k!=b),'difference':mean([v for k,v in effects.items() if k!=b])})
                author_books=defaultdict(set)
                for g in gold.values():author_books[g['author']].add(g['book'])
                for a,bs in author_books.items():sensitivity.append({'provider':p,'contrast':name,'analysis':'leave_one_author','omitted':a,'books':len(effects)-len(bs),'paired_n':sum(len(v) for b,v in diffs.items() if b not in bs),'difference':mean([v for b,v in effects.items() if b not in bs])})
                author_equal=mean([mean([effects[b] for b in bs]) for bs in author_books.values()])
                sensitivity.append({'provider':p,'contrast':name,'analysis':'equal_author_descriptive','omitted':'','books':len(effects),'paired_n':len(gold),'difference':author_equal})
                character_balanced=mean([mean([mean(v) for (book,_),v in characters.items() if book==b]) for b in effects])
                sensitivity.append({'provider':p,'contrast':name,'analysis':'equal_character_within_book','omitted':'','books':len(effects),'paired_n':len(gold),'difference':character_balanced})
                for label,books in filtered.items():sensitivity.append({'provider':p,'contrast':name,'analysis':label,'omitted':'','books':len(books),'paired_n':sum(len(v) for v in books.values()),'difference':mean([mean(v) for v in books.values()])})
    primary=[r for r in comparisons if r['primary']]
    for row,pval in zip(primary,holm_adjust([r['sign_flip_p'] for r in primary])):row['holm_p']=pval
    robustness=[];robust_pairs=[];repeat_summary=[]
    for kind in ('reorder','repeat','irrelevant_history'):
        for p in ('glm','deepseek'):
            for c in ('EF' if kind=='irrelevant_history' else 'ABCDEF'):
                rows=[r for r in scores if r['kind']==kind and r['provider']==p and r['condition']==c]
                paired=[]
                for r in rows:
                    base=lookup[(r['sample_id'],p,c)]
                    pair={'sample_id':r['sample_id'],'book':r['book'],'provider':p,'condition':c,'kind':kind,'replicate':r['replicate'],
                          'both_valid':r['valid'] and base['valid'],'same_trait_valid':r['valid'] and base['valid'] and r['trait']==base['trait'],
                          'base_correct':base['correct'],'variant_correct':r['correct'],'difference':int(r['correct'])-int(base['correct'])}
                    paired.append(pair);robust_pairs.append(pair)
                bdiff=defaultdict(list)
                for r in paired:bdiff[r['book']].append(r['difference'])
                robustness.append({'kind':kind,'provider':p,'condition':c,**aggregate(rows),
                                   'paired_valid_n':sum(r['both_valid'] for r in paired),'same_trait_valid_n':sum(r['same_trait_valid'] for r in paired),
                                   'paired_book_macro_change':mean([mean(v) for v in bdiff.values()])})
                if kind=='repeat':
                    triples=defaultdict(list)
                    for r in rows:triples[r['sample_id']].append(r)
                    allvalid=consistent=0
                    for sid,rr in triples.items():
                        rr=rr+[lookup[(sid,p,c)]];assert len(rr)==3
                        valid=all(r['valid'] for r in rr);allvalid+=valid;consistent+=valid and len({r['trait'] for r in rr})==1
                    repeat_summary.append({'provider':p,'condition':c,'groups':len(triples),'all_three_valid':allvalid,'all_three_same_trait_valid':consistent})
    estimated_cost=sum(r['estimated_charge_cny'] for r in ends)
    report={'generated_utc':datetime.now(timezone.utc).isoformat(),'complete':True,'planned_jobs':expected,'terminal_jobs':len(terminal),
            'attempts':len(ends),'api_status_counts':dict(Counter(r['status'] for r in ends)),
            'main_scoring_status':dict(Counter(r['status'] for r in main_rows)),
            'collection_first_utc':min(r['utc'] for r in logs),'collection_last_utc':max(r['utc'] for r in logs),
            'models':{p:sorted({r['response_model'] for r in scores if r['provider']==p and r['response_model']}) for p in ('glm','deepseek')},
            'reasoning_content_nonempty':sum(r['reasoning_content_chars']>0 for r in scores),
            'estimated_cost_cny':estimated_cost,'cost_note':'No-cache peak DeepSeek and unverified GLM 4/16 assumptions; includes reservations for missing usage/errors. Not actual bill.',
            'main_summary':summary,'comparisons':comparisons,'joint_evidence_outcomes':joint,'robustness':robustness,'repeat_triples':repeat_summary,
            'interpretation':'Primary Holm family has four comparisons. Secondary and robustness analyses are exploratory/descriptive. Exact quote matching measures traceability, not semantic support. Book t intervals and sign flips assume book independence; shared authors and fixed classic corpus limit generalization. No equivalence inference from nonsignificance.'}
    for name,rows in [('scores',scores),('quotes',quotes),('usage',usage),('main_summary',summary),('by_book',bybook),
                      ('comparisons',comparisons),('book_effects',book_effects),('sensitivity',sensitivity),
                      ('joint_evidence',joint),('robustness',robustness),('robustness_pairs',robust_pairs),('repeat_triples',repeat_summary)]:csv_write(out/(name+'.csv'),rows)
    (out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['第二篇正式实验结果','',f'完成 {len(terminal)}/{expected} 项；600来源组、10部作品、2个模型。',
           '条件：A局部直接、B局部分析、C局部证据分析；D历史直接、E历史分析、F历史证据分析。',
           '严格正确率：格式无效、拒答、截断及重试耗尽均计0；不修复或重抽模型答案。','',
           '模型 条件 样本 有效 正确 微平均 作品同权平均']
    for r in summary:lines.append(f"{r['provider']} {r['condition']} {r['n']} {r['valid']} {r['correct']} {r['micro_accuracy']:.4f} {r['book_macro_accuracy']:.4f}")
    lines+=['','主要比较（效应为准确率差，正值表示证据提示较高；四项Holm校正）：']
    for r in primary:lines.append(f"{r['provider']} {r['contrast']}: {r['book_macro_difference']:+.4f}, 95% CI [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}], p={r['sign_flip_p']:.4f}, Holm p={r['holm_p']:.4f}")
    lines+=['','完整分作品结果、留一书/作者敏感性、证据联合结果及稳健性对照见同目录CSV与summary.json。',
            '引文可逐字定位不代表其足以支持人物特质判断；不显著不代表等效。作品为固定经典小说语料，共享作者可能使书级独立性假设偏强。',
            f'费用估算 {estimated_cost:.2f} 元：GLM单价未核实，非账户实际账单。']
    (out/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'complete':True,'terminal_jobs':len(terminal),'main_scoring_status':report['main_scoring_status'],'primary':primary,'estimated_cost_cny':estimated_cost},ensure_ascii=False))

if __name__=='__main__':main()
