"""M2D RESEARCH — feature redundancy / conditional-information audit (section 9)."""
import sys,os,pickle
sys.path.append('.'); sys.path.append('research/m2d')
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import spearmanr
from experiment import load, feature_sets
from features_research import RESEARCH_FEATURES

rows=[]
for seed in [42,303,606,909]:
    df=load(seed); A,B=feature_sets(df)
    X=df[B].fillna(0).values
    rng=np.random.default_rng(0); sub=rng.choice(len(df),9000,replace=False)
    for f in ['borrower_propagation_exposure']+RESEARCH_FEATURES:
        if f not in df.columns: continue
        y=df[f].fillna(0).values
        if np.std(y)==0: rows.append(dict(seed=seed,feature=f,r2_from_B=np.nan,note='constant')); continue
        rf=RandomForestRegressor(n_estimators=40,max_depth=10,n_jobs=-1,random_state=0).fit(X[sub],y[sub])
        r2=rf.score(X,y)
        best=max(((c,abs(spearmanr(df[f],df[c]).statistic)) for c in B if df[c].std()>0), key=lambda z:z[1])
        rows.append(dict(seed=seed,feature=f,r2_from_B=round(float(r2),4),
                         top_B_corr_feature=best[0],top_B_spearman=round(float(best[1]),4)))
d=pd.DataFrame(rows)
piv=d.pivot_table(index='feature',values='r2_from_B',aggfunc='mean').sort_values('r2_from_B',ascending=False)
print("=== HOW MUCH OF EACH PROPAGATION FEATURE IS RECOVERABLE FROM MODEL B's 51 FEATURES? ===")
print("   (RandomForest R^2; 1.0 = fully redundant, 0.0 = entirely new information)")
print(piv.round(4).to_string())
print()
print("=== strongest single-B-feature rank correlation ===")
print(d.groupby('feature')[['top_B_spearman']].mean().round(4).join(
      d.groupby('feature')['top_B_corr_feature'].agg(lambda x:x.mode()[0])).to_string())
d.to_csv('research/m2d/redundancy_results.csv',index=False)
