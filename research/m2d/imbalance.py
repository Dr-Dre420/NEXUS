"""M2D RESEARCH — class-imbalance study (section 10). Hypothesis-driven, NOT a sweep.

H1: the extreme PV/new-stress base rate stops LightGBM growing past a stump, so
    propagation features never get a chance to be split on.
H2: if H1 holds, rebalancing should (a) grow more trees AND (b) let a propagation
    feature acquire non-trivial split gain. If trees grow but the B->B+P gap stays
    at zero, the limitation is informational, not an imbalance artifact.
Run on DEVELOPMENT seeds only; the holdout is never used for tuning.
"""
import sys,os
sys.path.append('.'); sys.path.append('research/m2d')
import numpy as np, pandas as pd
from experiment import load, feature_sets, fit_eval
from features_research import CANDIDATE_SETS

SEEDS=[42,101,202,303,404,505,606,707,808,909]
REGIMES={
 'default':        (None,None),
 'balanced_weight':(None,'balanced'),
 'low_min_child':  ({'min_child_samples':3},None),
 'balanced+lowmin':({'min_child_samples':3},'balanced'),
}
rows=[]
for target in ['propagation_vulnerability','rt_new_stress']:
    for seed in SEEDS:
        df=load(seed); A,B=feature_sets(df)
        P5=B+[c for c in CANDIDATE_SETS['P5_process'] if c in df.columns]
        for rname,(params,cw) in REGIMES.items():
            for spec,feats in [('B',B),('B+P5_process',P5)]:
                r=fit_eval(df,feats,target,params=params,class_weight=cw)
                if r is None: continue
                r.pop('preds',None)
                rows.append(dict(experiment_id=f'imbalance|{target}|{rname}|{spec}|seed{seed}',
                                 target=target,regime=rname,model_spec=spec,seed=seed,
                                 pr_auc=r['pr_auc'],roc_auc=r['roc_auc'],
                                 n_trees=r['n_trees'],pos_test=r['pos_test'],
                                 n_unique_pred=r['n_unique_pred']))
    print(f'  {target} done',flush=True)
d=pd.DataFrame(rows); d.to_csv('research/m2d/imbalance_results.csv',index=False)
for target in d.target.unique():
    t=d[d.target==target]
    g=t.groupby(['regime','model_spec']).agg(n=('seed','nunique'),pr=('pr_auc','mean'),
                                             roc=('roc_auc','mean'),trees=('n_trees','mean'),
                                             uniq=('n_unique_pred','mean')).round(4)
    print(f"\n=== CLASS IMBALANCE | {target} ===")
    print(g.to_string())
    piv=t.pivot_table(index=['regime','seed'],columns='model_spec',values='pr_auc')
    if 'B' in piv and 'B+P5_process' in piv:
        dd=(piv['B+P5_process']-piv['B']).groupby('regime').agg(['mean','median',
             lambda x:int((x>1e-9).sum()),'count'])
        dd.columns=['mean_d_PR','median_d_PR','wins','n']
        print("  B+P5_process minus B, by regime:"); print(dd.round(4).to_string())
