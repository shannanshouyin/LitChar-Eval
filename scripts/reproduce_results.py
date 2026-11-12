"""Recompute article contrasts from published book-level results."""
from pathlib import Path
from collections import defaultdict
import csv,json,sys,shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.statistics import book_level_summary,holm_adjust

def rows(path):
    with (ROOT/'published_data'/path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def contrast(stage,provider,name,values):
    result={'study':stage,'provider':provider,'contrast':name,'books':len(values),
            'difference':sum(values)/len(values),'ci_low':None,'ci_high':None,
            'raw_p':None,'holm_p':None}
    if len(values)==10:
        summary=book_level_summary(values)
        result.update(ci_low=summary['t_ci95'][0],ci_high=summary['t_ci95'][1],
                      raw_p=summary['sign_flip_p'])
    return result

def adjust(family):
    if all(r['raw_p'] is not None for r in family):
        for row,p in zip(family,holm_adjust([r['raw_p'] for r in family])):
            row['holm_p']=p

def main():
    output=ROOT/'outputs';output.mkdir(exist_ok=True)
    main_groups=defaultdict(list)
    for row in rows('formal_v1/book_effects.csv'):
        if row['contrast'] in ('C-B','F-E'):
            main_groups[(row['provider'],row['contrast'])].append(float(row['difference']))
    main_results=[contrast('main',p,c,v) for (p,c),v in sorted(main_groups.items())]
    adjust(main_results)
    matched_groups=defaultdict(list)
    for row in rows('matched_v1/by_book.csv'):
        if row['condition']=='F' and row['difference']!='':
            matched_groups[row['provider']].append(float(row['difference']))
    matched_results=[contrast('matched',p,'rotation-minus-repeat',v) for p,v in sorted(matched_groups.items())]
    adjust(matched_results)
    deletion_groups=defaultdict(list)
    for row in rows('ablation_v1/book_effects.csv'):
        if row['difference']!='':
            deletion_groups[row['provider']].append(float(row['difference']))
    deletion_results=[contrast('deletion',p,'cited-minus-control',v) for p,v in sorted(deletion_groups.items())]
    adjust(deletion_results)
    results=main_results+matched_results+deletion_results
    with (output/'comparisons.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    authors=json.loads((ROOT/'config/book_authors.json').read_text(encoding='utf-8'))
    groups=defaultdict(lambda:defaultdict(list))
    for row in rows('matched_v1/by_book.csv'):
        if row['condition']=='F' and row['difference']!='':
            groups[row['provider']][authors[row['book']]].append(float(row['difference']))
    author_results=[]
    for provider,author_groups in sorted(groups.items()):
        effects={a:sum(v)/len(v) for a,v in author_groups.items()}
        summary=book_level_summary(list(effects.values()))
        author_results.append({'provider':provider,'effects':effects,**summary})
    for row,p in zip(author_results,holm_adjust([r['sign_flip_p'] for r in author_results])):
        row['exploratory_holm_p']=p
    (output/'author_sensitivity.json').write_text(json.dumps(author_results,indent=2),encoding='utf-8')
    tables=output/'tables';tables.mkdir(exist_ok=True)
    for path in (ROOT/'published_data/manuscript_tables').glob('*.csv'):
        shutil.copy2(path,tables/path.name)
    print('Wrote outputs/comparisons.csv, author_sensitivity.json and tables/.')
if __name__=='__main__':main()
