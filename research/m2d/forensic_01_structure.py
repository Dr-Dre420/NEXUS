"""M2D Forensic Phase 1 — structural analysis of the frozen M2C representation.
Read-only with respect to all frozen artifacts."""
import sys, os, json, pickle
sys.path.append(os.path.abspath('.'))
import numpy as np, pandas as pd

d = pickle.load(open('data/frozen_m2c/world_state.pkl','rb'))
df = d['next_df_prop']
print(f"seed={d['seed']}  as_of={d['as_of_week']}  rows={len(df)}")

B_EXCL = ['borrower_id','group_id','week','split','current_stress','next_period_stress',
          'propagation_vulnerability','scenario_family','group_covered_amount','borrower_propagation_exposure']
Bf = [c for c in df.columns if c not in B_EXCL and c not in ('model_b_score','model_c_score','model_c_score_raw')]
print(f"\nB feature count (excl. score cols): {len(Bf)}")

# --- Q3/Q4/Q5: is exposure a deterministic function of two B features? ---
print("\n=== Q4/Q5: EXPOSURE vs ITS OWN CONSTITUENTS (both are Model B features) ===")
recon = df['peer_predicted_stress_mean'] * df['borrower_liability_share']
err = np.max(np.abs(recon - df['borrower_propagation_exposure']))
print(f"  exposure == peer_predicted_stress_mean * borrower_liability_share ?  max_abs_err={err:.3e}")
print(f"  'peer_predicted_stress_mean' in Model B feature set: {'peer_predicted_stress_mean' in Bf}")
print(f"  'borrower_liability_share'    in Model B feature set: {'borrower_liability_share' in Bf}")
print("  => exposure carries NO information absent from B; it is a product of two B columns.")

# --- Q4: correlation of exposure with every B feature ---
print("\n=== Q4: |Pearson| and |Spearman| of exposure vs top B features ===")
from scipy.stats import spearmanr
e = df['borrower_propagation_exposure'].values
rows=[]
for c in Bf:
    v = df[c].values.astype(float)
    if np.nanstd(v) == 0: continue
    p = abs(np.corrcoef(e, v)[0,1])
    s = abs(spearmanr(e, v).statistic)
    rows.append((c, p, s))
rows.sort(key=lambda r: -r[1])
for c,p,s in rows[:10]:
    print(f"  {c:32} |pearson|={p:.4f}  |spearman|={s:.4f}")

# --- R^2 of exposure regressed on B features (redundancy) ---
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
X = df[Bf].fillna(0).values
lin = LinearRegression().fit(X, e)
print(f"\n  Linear R^2 of exposure on all {len(Bf)} B features: {lin.score(X,e):.6f}")
sub = np.random.default_rng(0).choice(len(df), 8000, replace=False)
rf = RandomForestRegressor(n_estimators=40, max_depth=10, n_jobs=-1, random_state=0).fit(X[sub], e[sub])
print(f"  RandomForest R^2 (8k subsample, out-of-fit on full): {rf.score(X,e):.6f}")

# --- Q6/Q7: compression / dynamic range ---
print("\n=== Q6/Q7: COMPRESSION OF THE EXPOSURE SCALAR ===")
for c in ['borrower_propagation_exposure','peer_predicted_stress_mean','borrower_liability_share']:
    v=df[c]
    print(f"  {c:32} min={v.min():.3e} p50={v.median():.3e} p99={v.quantile(.99):.3e} max={v.max():.3e} nuniq={v.nunique()}")
q=df['borrower_propagation_exposure']
print(f"  fraction of exposure mass in top 1% of rows: {q.nlargest(int(len(q)*.01)).sum()/q.sum():.3f}")
print(f"  fraction of rows with exposure < 1e-3: {(q<1e-3).mean():.3f}")
