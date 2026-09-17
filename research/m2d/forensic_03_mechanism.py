"""M2D Forensic — Q8/Q9: does the exposure feature match the actual transmission mechanism?"""
import sys,os,pickle
sys.path.append('.')
import numpy as np, pandas as pd
d=pickle.load(open('data/frozen_m2c/world_state.pkl','rb'))
df=d['next_df_prop']; bs=pickle.load(open('data/frozen_m2c/baseline_state.pkl','rb'))

print("=== Q8: IS 'borrower_liability_share' THE ACTUAL TRANSMISSION WEIGHT? ===")
lw = np.array([bs.households[bs.borrower_to_household[b]].liability_weight for b in sorted(bs.borrower_to_household)])
print(f"  generator household liability_weight: unique={np.unique(lw)[:5]}  n_unique={len(np.unique(lw))}")
print("  mechanics.apply_group_liability_coverage weights contributions by hh.liability_weight")
print(f"  feature 'borrower_liability_share' is defined in features.py as:")
print(f"      cash_buffer / (group_buffer_sum + 1e-6)     <- a RELATIVE WEALTH share, not the legal weight")
print("  => the feature does NOT correspond to the generator's transmission weight.")

print("\n=== Q9: IS DESTINATION VULNERABILITY IN THE EXPOSURE? ===")
print("  exposure = peer_predicted_stress_mean * borrower_liability_share")
print("  borrower_liability_share is HIGH when the borrower is RICH relative to peers.")
cur=df[df.week==d['as_of_week']]
c=np.corrcoef(cur['borrower_liability_share'], cur['cash_buffer_mean_4w'])[0,1]
print(f"  corr(liability_share, own cash_buffer_mean_4w) = {c:+.4f}")
print("  => the multiplier RISES with the borrower's own liquidity, so a borrower with")
print("     MORE cushion receives a LARGER 'exposure'. Absolute destination fragility")
print("     (buffer relative to own obligation) is absent.")
adq = cur['cash_buffer_mean_4w']/(cur['amount_due_mean_4w']+1e-6)
print(f"  corr(exposure, buffer/amount_due adequacy)     = {np.corrcoef(cur['borrower_propagation_exposure'],adq)[0,1]:+.4f}")

print("\n=== Q13: HOW MANY TIMES IS THE NETWORK COLLAPSED BEFORE THE MODEL SEES IT? ===")
print("  1) per-peer stress probs -> MEAN over peers      (loses count/max/variance)")
print("  2) mean * scalar share   -> ONE float            (loses which peer, how many, how long)")
print("  => two lossy collapses, before any learning.")
