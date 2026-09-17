"""M2D Forensic Phase 1 — target sparsity attrition cascade (Q1, Q2, Q10-Q12)."""
import sys, os, pickle
sys.path.append(os.path.abspath('.'))
import numpy as np, pandas as pd

d = pickle.load(open('data/frozen_m2c/world_state.pkl','rb'))
hist, ep, targ, df = d['history_df'], d['ep_lineage_df'], d['targets_df'], d['next_df_prop']
print(f"seed {d['seed']}  |  history rows {len(hist)}  episode-lineage rows {len(ep)}")

print("\n=== Q1: PV ATTRITION CASCADE (each condition must hold simultaneously) ===")
n = len(targ)
c1 = (~targ['current_stress']).sum()
c2 = targ['next_period_stress'].sum()
c12 = ((~targ['current_stress']) & targ['next_period_stress']).sum()
print(f"  all borrower-weeks                              : {n:>7}")
print(f"  [C1] not currently stressed at t                : {c1:>7}  ({c1/n:.1%})")
print(f"  [C2] becomes stressed in (t, t+4]               : {c2:>7}  ({c2/n:.1%})")
print(f"  [C1 & C2] eligible 'new stress' events          : {c12:>7}  ({c12/n:.2%})")

ep30 = ep[ep['episode_network_contribution'] >= 0.30]
print(f"  [C3] episode contribution >= 0.30 (any row)     : {len(ep30):>7}  over {ep['destination_borrower'].nunique()} borrowers")
pv = targ['propagation_vulnerability'].sum()
print(f"  [C1&C2&C3] PV POSITIVES (exact week match)      : {pv:>7}  ({pv/n:.4%})")
print(f"  attrition from C1&C2 to PV                      : {1 - pv/max(c12,1):.2%} of eligible events lost at C3")

print("\n=== Q2: WHY 6/10 TEST SPLITS HAVE ZERO POSITIVES ===")
pvr = targ[targ['propagation_vulnerability']]
print(f"  PV positives total          : {len(pvr)}")
print(f"  unique positive borrowers   : {pvr['borrower_id'].nunique()}  of 400")
print(f"  unique positive JLGs        : {df[df.borrower_id.isin(pvr.borrower_id)]['group_id'].nunique()} of 80")
print(f"  PV weeks range              : {pvr['week'].min()} - {pvr['week'].max()}")
print("  PV positives by split:")
m = targ.merge(df[['borrower_id','week','split']].drop_duplicates(), on=['borrower_id','week'], how='left')
print(m[m.propagation_vulnerability].groupby('split').size().to_string())
print("\n  PV positives per 10-week block (shows temporal clustering):")
blk = (pvr['week']//10*10).value_counts().sort_index()
for k,v in blk.items(): print(f"    weeks {k:>3}-{k+9:>3}: {'#'*v} ({v})")

print("\n=== Q3-adjacent: CONTRIBUTION DISTRIBUTION (attribution concentration) ===")
c = ep['episode_network_contribution']
print(f"  episode rows with contribution>0 : {len(c)}")
for q in [.5,.75,.9,.95,.99]: print(f"    p{int(q*100):<3}= {c.quantile(q):.4f}")
print(f"  fraction >= 0.30 : {(c>=0.30).mean():.3f}   fraction >= 0.80 : {(c>=0.80).mean():.3f}")
print("  => contribution is NOT the binding constraint; it saturates near 1.0.")

print("\n=== Q10/Q11: PROPAGATION TIMING — is lag information available? ===")
print("  episode_network_contribution is recorded at the EPISODE START week t,")
print("  and PV requires an EXACT match (t in events). The source->destination lag")
print("  within [t, t+4) is collapsed and never exposed as a feature.")
ew = ep30.groupby('week').size()
print(f"  weeks carrying any >=0.30 episode: {len(ew)} of {hist['week'].max()}")
print(f"  max episodes in a single week    : {ew.max()}")

print("\n=== Q12: MULTI-PEER SIMULTANEOUS STRESS — is it representable? ===")
print("  peer_predicted_stress_mean is a MEAN over peers. For a 5-member JLG,")
print("  one peer at prob 1.0 and four peers at 0.25 both yield mean 0.25.")
print("  Count/max/concentration of simultaneously stressed peers is not encoded.")
