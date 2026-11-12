"""Freeze anchors, then derive deterministic quote and matched-token deletions."""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import re
from .core import digest,quote_norm,score_response,passages_for
from .prepare_formal import save_frozen

ROOT=Path(__file__).resolve().parents[1]
SEED=20261023
FIELDS={'H1':'history','L1':'former','T1':'target','L2':'post'}
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def deletion_pair(item,evidence,sid,provider):
    if not isinstance(evidence,list) or not 1<=len(evidence)<=2:return None,'quote_count'
    removed={};texts={p['passage_id']:p['text'] for p in passages_for(item,'F')}
    quote_meta=[]
    for e in evidence:
        pid=e['passage_id'];quote=e['quote']
        if pid not in texts or not 0<len(quote.split())<=25:return None,'invalid_quote'
        hay=quote_norm(texts[pid]);needle=quote_norm(quote)
        hits=[m.start() for m in re.finditer('(?='+re.escape(needle)+')',hay)] if needle else []
        if len(hits)!=1:return None,'absent_or_ambiguous_quote'
        start=hits[0];end=start+len(needle);tokens=list(re.finditer(r'\S+',hay))
        indices={i for i,t in enumerate(tokens) if t.start()<end and t.end()>start}
        if not indices:return None,'empty_span'
        removed.setdefault(pid,set()).update(indices)
        quote_meta.append({'passage_id':pid,'normalized_start':start,'normalized_end':end,'token_indices':sorted(indices)})
    controls={};spans=[]
    for pid,indices in removed.items():
        n=len(texts[pid].split());ordered=sorted(indices);runs=[]
        for i in ordered:
            if not runs or i!=runs[-1][-1]+1:runs.append([i])
            else:runs[-1].append(i)
        chosen=set()
        for j,run in enumerate(runs):
            length=len(run)
            candidates=[k for k in range(n-length+1) if not(set(range(k,k+length))&(indices|chosen))]
            if not candidates:return None,'no_same_passage_matched_span'
            k=min(candidates,key=lambda k:digest([SEED,'delete_control',sid,provider,pid,j,k]))
            chosen.update(range(k,k+length));spans.append({'passage_id':pid,'cited_start_token':run[0],
                                                       'control_start_token':k,'token_count':length})
        controls[pid]=chosen
    cited=dict(item);control=dict(item)
    for pid,indices in removed.items():
        words=texts[pid].split();field=FIELDS[pid]
        cited[field]=' '.join(w for i,w in enumerate(words) if i not in indices)
        control[field]=' '.join(w for i,w in enumerate(words) if i not in controls[pid])
        assert len(cited[field].split())==len(control[field].split())
    return {'cited_item':cited,'control_item':control,'spans':spans,'quote_spans':quote_meta,
            'removed_words':sum(len(v) for v in removed.values())},None

def prepare_base():
    items=read(ROOT/'data/processed/matched120_inputs.json');gold=read(ROOT/'data/processed/matched120_gold.json')
    assert len(items)==len(gold)==120
    variants=[{'variant_id':i['sample_id']+'__deletion_anchor','sample_id':i['sample_id'],'kind':'anchor','replicate':0,'conditions':['F'],'item':i} for i in items]
    save_frozen(ROOT/'data/processed/ablation_base_variants.json',variants)
    old={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('dev_v1','formal_v1','matched_v1') for p in sorted((ROOT/'results'/folder).rglob('*')) if p.is_file()}
    save_frozen(ROOT/'data/processed/ablation_prior_results_sha256.json',old)
    print(json.dumps({'phase':'base','jobs':240}))

def prepare_edits():
    out=ROOT/'results/ablation_v1/base';status=read(out/'last_execution.json');assert status['all_jobs_terminal']
    items={v['sample_id']:v['item'] for v in read(ROOT/'data/processed/ablation_base_variants.json')}
    gold={g['sample_id']:g for g in read(ROOT/'data/processed/matched120_gold.json')}
    events=[json.loads(l) for l in (out/'attempts.jsonl').read_text(encoding='utf-8').splitlines()]
    terminal={r['job_id']:r for r in events if r['event']=='end' and r['status'] in ('response','refusal','transport_exhausted')};assert len(terminal)==240
    records=[];variants=[]
    for r in sorted(terminal.values(),key=lambda r:r['job_id']):
        sid=r['sample_id'];p=r['provider'];item=items[sid]
        response=r.get('response') or {};first=(response.get('choices') or [{}])[0];msg=first.get('message') or {}
        score=score_response(msg.get('content'),item,gold[sid]['gt'],'F',first.get('finish_reason'),msg.get('refusal') or ('refused' if r['status']=='refusal' else None))
        reason='invalid_anchor';pair=None
        if score['valid']:pair,reason=deletion_pair(item,json.loads(msg['content'])['evidence'],sid,p)
        record={'sample_id':sid,'provider':p,'book':gold[sid]['book'],'author':gold[sid]['author'],
                'anchor_job_id':r['job_id'],'eligible':pair is not None,'exclusion_reason':reason}
        if pair:
            record.update({k:pair[k] for k in ('spans','quote_spans','removed_words')})
            for kind,field in [('cited_deletion','cited_item'),('control_deletion','control_item')]:
                variants.append({'variant_id':sid+'__'+p+'__'+kind,'sample_id':sid,'kind':kind,'replicate':0,
                                 'conditions':['F'],'providers':[p],'item':pair[field]})
        records.append(record)
    save_frozen(ROOT/'data/processed/ablation_edit_variants.json',variants)
    save_frozen(ROOT/'data/processed/ablation_eligibility.json',records)
    save_frozen(ROOT/'data/processed/ablation_derivation.json',{'anchor_log_sha256':hashlib.sha256((out/'attempts.jsonl').read_bytes()).hexdigest(),
                'variants_sha256':digest(variants),'eligibility_sha256':digest(records),'eligible_pairs':sum(r['eligible'] for r in records),'edit_jobs':len(variants),
                'exclusions':dict(Counter(r['exclusion_reason'] for r in records if not r['eligible']))})
    print(json.dumps({'phase':'edit','eligible_pairs':sum(r['eligible'] for r in records),'jobs':len(variants)}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('base','edit'),required=True);args=parser.parse_args()
    prepare_base() if args.phase=='base' else prepare_edits()
