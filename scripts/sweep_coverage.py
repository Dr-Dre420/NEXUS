import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import SyntheticWorldGenerator
from src.targets import construct_targets, assign_temporal_splits

def run_sweep():
    slack_regimes = ["conservative", "base", "high"]
    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    
    for regime in slack_regimes:
        print("============================================================")
        print(f"REGIME: {regime.upper()} SLACK (50% MAX MEMBER COVERAGE)")
        print("============================================================")
        
        all_h = []
        all_l = []
        all_t = []
        
        for s in seeds:
            gen = SyntheticWorldGenerator(n_borrowers=400, seed=s, coverage_fraction=0.50, slack_regime=regime)
            h, l = gen.simulate(156)
            t = construct_targets(h, l)
            t = assign_temporal_splits(t)
            
            h['seed'] = s
            l['seed'] = s
            t['seed'] = s
            
            all_h.append(h)
            all_l.append(l)
            all_t.append(t)
            
        df_h = pd.concat(all_h, ignore_index=True)
        df_l = pd.concat(all_l, ignore_index=True)
        df_t = pd.concat(all_t, ignore_index=True)
        
        print("\n--- liquid cash-buffer distribution ---")
        print(df_h['cash_buffer'].describe())
        
        print("\n--- non-liquid reserve distribution ---")
        print(df_h['non_liquid_reserve'].describe())
        
        print("\n--- amount_due distribution ---")
        print(df_h[df_h['amount_due'] > 0]['amount_due'].describe())
        
        print("\n--- days_past_due distribution ---")
        print(df_h[df_h['days_past_due'] > 0]['days_past_due'].describe())
        
        print("\n--- defaulted loan count ---")
        default_count = df_h[df_h['is_defaulted'] == True].groupby(['seed', 'borrower_id']).ngroups
        print(f"Total unique loans defaulted: {default_count}")
        
        # Repayment lifecycle check
        print("\n--- DPD / LOAN-LIFECYCLE DIAGNOSTICS ---")
        active_loans = df_h[(df_h['is_defaulted'] == False) & (df_h['is_closed'] == False)]
        active_with_dpd = active_loans[active_loans['days_past_due'] > 0]
        frac_active_dpd = len(active_with_dpd) / len(active_loans) if len(active_loans) > 0 else 0
        print(f"Fraction of active loan-weeks with positive DPD: {frac_active_dpd:.4f}")
        print("Active loan DPD distribution:")
        print(active_with_dpd['days_past_due'].describe())
        
        total_unique_loans = df_h.groupby(['seed', 'borrower_id']).ngroups
        print(f"Default rate: {default_count / total_unique_loans:.4f}")
        
        cured_loans = 0
        # A crude way to detect cures is loans that had DPD > 0 then went back to 0 without defaulting
        for (s, b), group in df_h.groupby(['seed', 'borrower_id']):
            was_delinquent = False
            for _, row in group.iterrows():
                if row['days_past_due'] > 0:
                    was_delinquent = True
                elif was_delinquent and row['days_past_due'] == 0 and not row['is_defaulted']:
                    cured_loans += 1
                    break
        print(f"Cured loan count (unique): {cured_loans}")
        print(f"Cure rate (of ever-delinquent): {cured_loans / max(1, (default_count + cured_loans)):.4f} (approx)")
        
        print("\n--- group coverage distribution ---")
        print(df_h[df_h['group_covered_amount'] > 0]['group_covered_amount'].describe())
        
        print("\n--- propagation contribution-share distribution ---")
        if not df_l.empty:
            print(df_l['contribution_share'].describe())
            print("\nFractions:")
            print(f"  Exactly 0.0: {(df_l['contribution_share'] == 0.0).mean():.4f}")
            print(f"  (0.0, 0.30): {((df_l['contribution_share'] > 0.0) & (df_l['contribution_share'] < 0.30)).mean():.4f}")
            print(f"  [0.30, 0.50): {((df_l['contribution_share'] >= 0.30) & (df_l['contribution_share'] < 0.50)).mean():.4f}")
            print(f"  [0.50, 0.80): {((df_l['contribution_share'] >= 0.50) & (df_l['contribution_share'] < 0.80)).mean():.4f}")
            print(f"  >= 0.80: {(df_l['contribution_share'] >= 0.80).mean():.4f}")
        else:
            print("No lineage events.")
            
        print("\n--- propagation-vulnerability prevalence ---")
        pv = df_t['propagation_vulnerability'].sum()
        eligible = df_t['next_period_stress'].sum()
        print(f"PV: {pv} / {eligible} (Rate: {pv/max(1, eligible):.4f})")
        print("By split:")
        print(df_t.groupby('split')['propagation_vulnerability'].sum())
        
        print("\n--- scenario prevalence ---")
        print(df_h[df_h['shortfall'] > 0].groupby('scenario_family')['borrower_id'].count())
        
        print("\n--- healthy + stressed-peer cases ---")
        # Define healthy as cash_buffer > 500, amount_due == 0
        df_h['is_healthy'] = (df_h['cash_buffer'] > 500) & (df_h['amount_due'] == 0)
        df_h['is_stressed'] = (df_h['cash_buffer'] == 0) | (df_h['amount_due'] > 0)
        
        group_stress = df_h.groupby(['seed', 'week', 'group_id'])['is_stressed'].sum().reset_index()
        group_stress.rename(columns={'is_stressed': 'group_stressed_count'}, inplace=True)
        
        merged = df_h.merge(group_stress, on=['seed', 'week', 'group_id'])
        cases = merged[(merged['is_healthy']) & (merged['group_stressed_count'] > 0)]
        print(f"Number of (borrower-week) cases where healthy despite stressed peers: {len(cases)}")
        
        highly_stressed_cases = merged[(merged['is_healthy']) & (merged['group_stressed_count'] >= 2)]
        print(f"Number of (borrower-week) cases where healthy despite HIGHLY stressed peers: {len(highly_stressed_cases)}")
        
        print("\n\n")

if __name__ == "__main__":
    run_sweep()
