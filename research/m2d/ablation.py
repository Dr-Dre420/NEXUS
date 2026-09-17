"""M2D RESEARCH — ablation (section 8): is any lift NEW information or a re-expression of B?"""
import sys,os
sys.path.append('.'); sys.path.append('research/m2d')
import numpy as np, pandas as pd
from experiment import load, feature_sets, fit_eval, NETWORK_COLS
from features_research import RESEARCH_FEATURES, CANDIDATE_SETS

PEER_CTX=['peer_buffer_mean_t','peer_shortfall_mean_4w','peer_dpd_mean_4w',
          'peer_debt_burden','peer_predicted_stress_mean','expected_peer_debt_burden',
          'borrower_liability_share','group_size']
TARGET='rt_new_stress'
SEEDS=[42,101,202,303,404,505,606,707,808,909]

rows=[]
for seed in SEEDS:
    df=load(seed); A,B=feature_sets(df)
    Bnopeer=[c for c in B if c not in PEER_CTX]
    R=[c for c in RESEARCH_FEATURES if c in df.columns]
    specs={
      'ab1_B':                          B,
      'ab2_B_plus_propagation':         B+R,
      'ab3_B_peer_context_only':        Bnopeer+PEER_CTX,
      'ab4_propagation_without_peerctx':Bnopeer+R,
      'ab5_individual_plus_propagation':A+R,
      'ab6_propagation_alone':          R,
      'ab7_B_minus_peerctx':            Bnopeer,
    }
    for name,f in specs.items():
        r=fit_eval(df,f,TARGET)
        if r is None: continue
        r.pop('preds',None)
        rows.append(dict(experiment_id=f'ablation|{TARGET}|{name}|seed{seed}',
                         ablation=name,seed=seed,target=TARGET,n_features=len(f),
                         pr_auc=r['pr_auc'],roc_auc=r['roc_auc'],pos_test=r['pos_test']))
    print(f'  seed {seed} done',flush=True)

d=pd.DataFrame(rows); d.to_csv('research/m2d/ablation_results.csv',index=False)
g=d.groupby('ablation').agg(n=('seed','nunique'),pr=('pr_auc','mean'),pr_sd=('pr_auc','std'),
                            roc=('roc_auc','mean'),roc_sd=('roc_auc','std'),nf=('n_features','mean'))
base=g.loc['ab1_B']
g['d_pr']=g.pr-base.pr; g['d_roc']=g.roc-base.roc
print("\n=== ABLATION (target rt_new_stress, 10 independent worlds) ===")
print(g.round(4).to_string())
