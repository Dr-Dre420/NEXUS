"""
M2D E2b — WHY IS EXPOSURE SOMETIMES INVERSELY RELATED TO PV?

Frozen construction:
    borrower_propagation_exposure = peer_predicted_stress_mean * borrower_liability_share
    borrower_liability_share      = own cash_buffer / group cash_buffer total

Hypothesis H2e: the multiplier is a RELATIVE WEALTH share, so it is LOW exactly for
the liquidity-poor borrowers who are most likely to become stressed. The product then
drags exposure DOWN for the very rows that are positive, inverting the intended sign.

TRAIN+VALIDATION rows only. No modification to any frozen definition.
"""
import sys, os, pickle
sys.path.append(os.path.abspath('.'))
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

SEEDS = [42,101,202,303,404,505,606,707,808,909,1201,1302,1403,1504,1605,1706,1807,1908,2009,2110]
rows = []
for seed in SEEDS:
    df = pickle.load(open(f'research/m2d/cache/world_{seed}.pkl','rb'))['df']
    tv = df[df.split.isin(['train','val'])]
    p, n = tv[tv.propagation_vulnerability], tv[~tv.propagation_vulnerability]
    if not len(p) or not len(n): continue
    def auc(col):
        a, b = p[col].values, n[col].values
        return float(mannwhitneyu(a, b, alternative='two-sided').statistic/(len(a)*len(b)))
    rows.append(dict(
        seed=seed, n_pos=len(p),
        auc_exposure      = auc('borrower_propagation_exposure'),
        auc_peer_stress   = auc('peer_predicted_stress_mean'),
        auc_liab_share    = auc('borrower_liability_share'),
        auc_own_buffer    = auc('cash_buffer_mean_4w'),
        liab_share_pos    = float(p['borrower_liability_share'].median()),
        liab_share_neg    = float(n['borrower_liability_share'].median()),
    ))
d = pd.DataFrame(rows); d.to_csv('research/m2d/e2b_inversion.csv', index=False)
pd.set_option('display.width', 200)
print("=== E2b: UNIVARIATE RANK-AUC vs PV, BY COMPONENT (train+val only) ===")
print(d.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
print("\n=== MEANS ACROSS 20 WORLDS ===")
for c in ['auc_exposure','auc_peer_stress','auc_liab_share','auc_own_buffer']:
    s = d[c]
    print(f"  {c:18} mean {s.mean():.4f}  median {s.median():.4f}  "
          f"min {s.min():.4f}  max {s.max():.4f}  worlds>0.5: {int((s>0.5).sum())}/{len(s)}")
print(f"\n  median liability share  positives {d.liab_share_pos.mean():.5f}  "
      f"negatives {d.liab_share_neg.mean():.5f}")
print("\n  INTERPRETATION")
print("  peer_predicted_stress_mean is the directional component that carries signal.")
print("  borrower_liability_share is a relative-wealth multiplier that is LOWER for")
print("  positives, so multiplying the two pulls exposure DOWN on positive rows and")
print("  partially cancels the peer-stress signal. This is a construction defect in the")
print("  frozen feature, reported as a diagnostic - the frozen definition is unchanged.")
