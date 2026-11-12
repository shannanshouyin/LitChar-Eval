"""Outcome-blind selection from the prior 600 groups for a matched follow-up."""
from pathlib import Path
from collections import Counter
import json
import hashlib
from .core import digest
from .prepare_formal import save_frozen

ROOT=Path(__file__).resolve().parents[1]
SEED=20261023

def prepare():
    processed=ROOT/'data/processed'
    source=json.loads((processed/'formal600_inputs.json').read_text(encoding='utf-8'))
    gold=json.loads((processed/'formal600_gold.json').read_text(encoding='utf-8'))
    refs={r['sample_id']:r for r in gold};byid={r['sample_id']:r for r in source}
    parent=json.loads((ROOT/'results/formal_v1/frozen_manifest.json').read_text(encoding='utf-8'))['frozen']
    assert digest(gold)==parent['gold_sha256']
    preparation=json.loads((processed/'formal_preparation.json').read_text(encoding='utf-8'))
    assert digest(source)==preparation['input_sha256']
    selected=[];variants=[]
    for book in sorted({g['book'] for g in gold}):
        # Only source IDs and book membership affect selection; outcomes never read.
        ids=sorted((g['sample_id'] for g in gold if g['book']==book),key=lambda s:digest([SEED,'matched_selection',s]))[:12]
        assert len(ids)==12
        rotate_order=sorted(ids,key=lambda s:digest([SEED,'matched_rotation',s]))
        for sid in ids:
            item=byid[sid];selected.append(item)
            shift=1+rotate_order.index(sid)%4
            rotated={**item,'options':item['options'][shift:]+item['options'][:shift]}
            for kind,rep,content in [('anchor',0,item),('repeat',1,item),('reorder',0,rotated)]:
                variants.append({'variant_id':sid+'__matched_'+kind,'sample_id':sid,'kind':kind,'replicate':rep,
                                 'conditions':['E','F'],'rotation':shift if kind=='reorder' else 0,'item':content})
    selected_gold=[refs[i['sample_id']] for i in selected]
    assert len(selected)==120 and len({i['sample_id'] for i in selected})==120
    parent_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for folder in ('dev_v1','formal_v1') for p in sorted((ROOT/'results'/folder).rglob('*')) if p.is_file()}
    save_frozen(processed/'matched_prior_results_sha256.json',parent_hashes)
    for name,value in [('matched120_inputs.json',selected),('matched120_gold.json',selected_gold),('matched_variants.json',variants),
                       ('matched_preparation.json',{'seed':SEED,'parent_inputs_sha256':digest(source),'parent_gold_sha256':digest(gold),
                        'inputs_sha256':digest(selected),'gold_sha256':digest(selected_gold),'variants_sha256':digest(variants),
                        'groups':120,'book_counts':dict(Counter(g['book'] for g in selected_gold)),
                        'planned_jobs':1440,'rotation_counts':dict(Counter(v['rotation'] for v in variants if v['kind']=='reorder')),
                        'selection':'12 per book from prior 600; stable hash; no model outcomes read; post-result follow-up, not independent new corpus'})]:
        save_frozen(processed/name,value)
    print(json.dumps({'groups':120,'books':10,'planned_jobs':1440}))

if __name__=='__main__':prepare()
