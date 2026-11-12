"""Freeze test selection and robustness variants before any formal API calls."""
from pathlib import Path
from collections import defaultdict, Counter
import json
import random
import hashlib
from .core import norm, clean_text, trim_boundary_overlap, digest, build_messages
from .statistics import capped_allocation

ROOT=Path(__file__).resolve().parents[1]
SEED=20261022
EXPECTED='ecb37dc9d3f566836096a369ace2cc8182ee37ac971702822ade68b09509890d'

def save_frozen(path, value):
    text=json.dumps(value, ensure_ascii=False, indent=2)+'\n'
    if path.exists() and path.read_text(encoding='utf-8')!=text:
        raise RuntimeError(f'Frozen file differs: {path.name}')
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text,encoding='utf-8')

def prepare():
    raw=(ROOT/'data/raw/full_test_data.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==EXPECTED
    groups=defaultdict(list); excluded=Counter()
    for index,r in enumerate(json.loads(raw)):
        if len(r['options'])!=5 or len({norm(x).casefold() for x in r['options']})!=5:
            excluded['duplicate_options']+=1; continue
        if not norm(r['history']): excluded['empty_history']+=1; continue
        assert r['options'][ord(r['answer'][1])-97]==r['gt']
        history,n=trim_boundary_overlap(clean_text(r['history']),clean_text(r['snippet_former_context']))
        if not history: excluded['empty_history_after_boundary_dedup']+=1; continue
        gid=digest([norm(r[k]) for k in ('book_name','character','snippet_underlined')])
        groups[(r['book_name'],gid)].append((index,r,history,n))
    bybook=defaultdict(list)
    for book,gid in groups: bybook[book].append(gid)
    allocation=capped_allocation({b:len(gs) for b,gs in bybook.items()},600)
    items=[]; gold=[]
    for book in sorted(bybook):
        gids=sorted(bybook[book],key=lambda g:digest([SEED,'formal_group',g]))[:allocation[book]]
        for gid in gids:
            index,r,history,n=min(groups[(book,gid)],key=lambda x:digest([SEED,'formal_row',x[0]]))
            options=list(r['options']);random.Random(int(digest([SEED,'formal_options',gid]),16)).shuffle(options)
            sid='test_'+gid[:20]
            items.append({'sample_id':sid,'character':r['character'],'options':options,
                          'history':' '.join(history.split()[-1200:]),'former':clean_text(r['snippet_former_context']),
                          'target':clean_text(r['snippet_underlined']),'post':clean_text(r['snippet_post_context'])})
            author=book.split(' by ')[-1].replace('Fyodor Dostoevsky','Fyodor Dostoyevsky')
            gold.append({'sample_id':sid,'source_row':index,'group_id':gid,'book':book,'author':author,
                         'character':r['character'],'gt':r['gt'],'original_answer':r['answer'],
                         'history_boundary_removed_words':n})
    goldmap={g['sample_id']:g for g in gold}; itemmap={i['sample_id']:i for i in items}
    subsets={}
    for label,count in [('reorder',6),('repeat',3),('irrelevant_history',6)]:
        subsets[label]=[]
        for book in sorted(bybook):
            candidates=[i['sample_id'] for i in items if goldmap[i['sample_id']]['book']==book]
            subsets[label]+=sorted(candidates,key=lambda s:digest([SEED,label,s]))[:count]
    variants=[{'variant_id':i['sample_id']+'__main','sample_id':i['sample_id'],'kind':'main',
               'replicate':0,'conditions':list('ABCDEF'),'item':i} for i in items]
    for sid in subsets['reorder']:
        original=itemmap[sid]; item=dict(original); options=list(original['options'])
        # A nonzero cyclic rotation changes every option position, preserves all traits.
        shift=1+int(digest([SEED,'rotation',sid]),16)%4
        item['options']=options[shift:]+options[:shift]
        variants.append({'variant_id':sid+'__reorder','sample_id':sid,'kind':'reorder',
                         'replicate':0,'conditions':list('ABCDEF'),'rotation':shift,'item':item})
    for sid in subsets['repeat']:
        for rep in (1,2):
            variants.append({'variant_id':sid+f'__repeat{rep}','sample_id':sid,'kind':'repeat',
                             'replicate':rep,'conditions':list('ABCDEF'),'item':itemmap[sid]})
    for sid in subsets['irrelevant_history']:
        original=itemmap[sid]; book=goldmap[sid]['book']
        candidates=[i for i in items if goldmap[i['sample_id']]['book']!=book]
        donor=min(candidates,key=lambda i:(abs(len(i['history'].split())-len(original['history'].split())),
                                          digest([SEED,'donor',sid,i['sample_id']])))
        item=dict(original); item['history']=donor['history']
        variants.append({'variant_id':sid+'__irrelevant_history','sample_id':sid,'kind':'irrelevant_history',
                         'replicate':0,'conditions':list('EF'),'donor_sample_id':donor['sample_id'],
                         'history_word_difference':len(item['history'].split())-len(original['history'].split()),'item':item})
    out=ROOT/'data/processed'
    for name,value in [('formal600_inputs.json',items),('formal600_gold.json',gold),('formal_variants.json',variants),
                       ('formal_preparation.json',{'source_sha256':EXPECTED,'seed':SEED,'excluded':dict(excluded),
                        'eligible_groups':len(groups),'allocation':allocation,'robustness_subsets':subsets,
                        'input_sha256':digest(items),'gold_sha256':digest(gold),'variants_sha256':digest(variants),
                        'jobs':2*sum(len(v['conditions']) for v in variants)})]:
        save_frozen(out/name,value)
    assert len(items)==600 and len(goldmap)==600 and sum(len(v['conditions']) for v in variants)*2==8880
    print(json.dumps({'sample_size':len(items),'allocation':allocation,'jobs':8880},ensure_ascii=False))

if __name__=='__main__':prepare()
