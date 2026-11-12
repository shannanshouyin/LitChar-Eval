"""Prespecified book-level paired summaries; no inference from pilot results."""
import itertools
import math
import numpy as np
from scipy.stats import t

def holm_adjust(pvalues):
    p=np.asarray(pvalues,dtype=float)
    if p.ndim!=1 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Expected finite p-values in [0,1]')
    order=np.argsort(p);adjusted=np.empty(len(p));running=0.
    for rank,index in enumerate(order):
        running=max(running,(len(p)-rank)*p[index]);adjusted[index]=min(1.,running)
    return adjusted.tolist()

def book_level_summary(book_differences):
    d=np.asarray(book_differences,dtype=float)
    if len(d)<2 or len(d)>16 or not np.isfinite(d).all():
        raise ValueError('Use 2..16 finite book differences')
    if np.any(np.abs(d)>1):raise ValueError('Accuracy difference outside [-1,1]')
    estimate=float(d.mean());se=float(d.std(ddof=1)/math.sqrt(len(d)))
    margin=float(t.ppf(.975,len(d)-1)*se)
    # Exact enumeration under independent symmetric book-level differences.
    # Not a design-based randomization test; books are a fixed convenience corpus.
    null=[abs(float(np.mean(d*np.array(signs)))) for signs in itertools.product((-1,1),repeat=len(d))]
    p=sum(v>=abs(estimate)-1e-12 for v in null)/len(null)
    return {'books':len(d),'macro_difference':estimate,'t_ci95':[estimate-margin,estimate+margin],
            'sign_flip_p':p,'assumptions':'independent symmetric book-level differences; few fixed books; not population-representative'}

def capped_allocation(capacities, total):
    if total>sum(capacities.values()) or total<0:raise ValueError('Insufficient capacity')
    allocation={b:0 for b in sorted(capacities)}
    for _ in range(total):
        b=min((b for b in allocation if allocation[b]<capacities[b]),key=lambda b:(allocation[b],b))
        allocation[b]+=1
    return allocation

def planning_half_width(capacities, n, discordance, within_book_correlation):
    allocation=capped_allocation(capacities,n)
    if any(v==0 for v in allocation.values()):raise ValueError('Each book needs a sample')
    b=len(allocation)
    # Delta=0 planning approximation Var(paired difference)=discordance.
    # No finite-population correction, no observed test outcomes.
    variance=sum(discordance/k*(1+(k-1)*within_book_correlation) for k in allocation.values())/(b*b)
    return float(t.ppf(.975,b-1)*math.sqrt(variance))
