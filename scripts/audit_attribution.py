import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import SyntheticWorldGenerator
from src.targets import construct_targets

def main():
    print("Generating synthetic world data (Seed 42)...")
    generator = SyntheticWorldGenerator(seed=42, n_borrowers=400, slack_regime='conservative')
    # Using 156 weeks
    history_df, lineage_df, ep_lineage_df = generator.simulate(n_weeks=156)
    
    print("\n==============================================")
    print("MARGINAL VS EPISODE AUDIT")
    print("==============================================")
    
    marg = lineage_df['contribution_share'] if not lineage_df.empty else pd.Series(dtype=float)
    ep = ep_lineage_df['episode_network_contribution'] if not ep_lineage_df.empty else pd.Series(dtype=float)
    
    print(f"{'Metric':<25} | {'Marginal (Weekly)':<20} | {'Episode (4-week)':<20}")
    print("-" * 70)
    print(f"{'Count':<25} | {len(marg):<20} | {len(ep):<20}")
    if len(marg) > 0 and len(ep) > 0:
        print(f"{'Min':<25} | {marg.min():.4f}{'':<14} | {ep.min():.4f}")
        print(f"{'Mean':<25} | {marg.mean():.4f}{'':<14} | {ep.mean():.4f}")
        print(f"{'p10':<25} | {marg.quantile(0.10):.4f}{'':<14} | {ep.quantile(0.10):.4f}")
        print(f"{'p25':<25} | {marg.quantile(0.25):.4f}{'':<14} | {ep.quantile(0.25):.4f}")
        print(f"{'Median':<25} | {marg.median():.4f}{'':<14} | {ep.median():.4f}")
        print(f"{'p75':<25} | {marg.quantile(0.75):.4f}{'':<14} | {ep.quantile(0.75):.4f}")
        print(f"{'p90':<25} | {marg.quantile(0.90):.4f}{'':<14} | {ep.quantile(0.90):.4f}")
        print(f"{'Max':<25} | {marg.max():.4f}{'':<14} | {ep.max():.4f}")
        
        print("\nDistribution:")
        print(f"{'Fraction exactly 0':<25} | {(marg == 0).mean():.4f}{'':<14} | {(ep == 0).mean():.4f}")
        print(f"{'Fraction 0 < x < 0.30':<25} | {((marg > 0) & (marg < 0.30)).mean():.4f}{'':<14} | {((ep > 0) & (ep < 0.30)).mean():.4f}")
        print(f"{'Fraction 0.30 <= x < 0.50':<25} | {((marg >= 0.30) & (marg < 0.50)).mean():.4f}{'':<14} | {((ep >= 0.30) & (ep < 0.50)).mean():.4f}")
        print(f"{'Fraction 0.50 <= x < 0.80':<25} | {((marg >= 0.50) & (marg < 0.80)).mean():.4f}{'':<14} | {((ep >= 0.50) & (ep < 0.80)).mean():.4f}")
        print(f"{'Fraction >= 0.80':<25} | {(marg >= 0.80).mean():.4f}{'':<14} | {(ep >= 0.80).mean():.4f}")
    
    print("\n==============================================")
    print("TARGET RECONCILIATION")
    print("==============================================")
    
    # 1. Total marginal lineage events
    total_marg = len(lineage_df)
    
    # 2. Total episode events
    total_ep = len(ep_lineage_df)
    
    # 3. Eligible currently non-stressed borrowers
    targets_df = construct_targets(history_df, ep_lineage_df)
    eligible_df = targets_df[targets_df['current_stress'] == False]
    eligible_count = len(eligible_df)
    
    # 4. PV positives with episode contribution >= 0.30
    pv_positives_df = targets_df[targets_df['propagation_vulnerability'] == True]
    pv_count = len(pv_positives_df)
    
    # 5. Unique PV borrowers
    unique_pv_borrowers = pv_positives_df['borrower_id'].nunique()
    
    # 6. Unique affected JLGs
    # Need to map borrower_id to group_id
    b_to_g = history_df[['borrower_id', 'group_id']].drop_duplicates().set_index('borrower_id')['group_id']
    unique_jlgs = pv_positives_df['borrower_id'].map(b_to_g).nunique()
    
    print(f"{'Metric':<40} | {'Count':<10}")
    print("-" * 55)
    print(f"{'Total marginal lineage events':<40} | {total_marg:<10}")
    print(f"{'Total network-affected episodes':<40} | {total_ep:<10}")
    print(f"{'Eligible currently non-stressed':<40} | {eligible_count:<10}")
    print(f"{'PV positives (Episode >= 0.30)':<40} | {pv_count:<10}")
    print(f"{'Unique PV borrowers':<40} | {unique_pv_borrowers:<10}")
    print(f"{'Unique affected JLGs':<40} | {unique_jlgs:<10}")
    
    print("\n==============================================")
    print("REPRESENTATIVE CASES (Seed 42)")
    print("==============================================")
    
    if not ep_lineage_df.empty:
        # Get one strong, one weak, one non-propagated
        strong_cases = ep_lineage_df[ep_lineage_df['episode_network_contribution'] >= 0.50]
        weak_cases = ep_lineage_df[(ep_lineage_df['episode_network_contribution'] > 0) & (ep_lineage_df['episode_network_contribution'] < 0.30)]
        
        # Non-propagated (affected but not formally stressed)
        pv_idx = list(zip(pv_positives_df['borrower_id'], pv_positives_df['week']))
        ep_lineage_df['is_pv'] = ep_lineage_df.apply(lambda row: (row['destination_borrower'], row['week']) in pv_idx, axis=1)
        non_propagated = ep_lineage_df[(ep_lineage_df['is_pv'] == False) & (ep_lineage_df['episode_network_contribution'] > 0)]
        
        samples = []
        if not strong_cases.empty:
            samples.append(("STRONG CONTRIBUTION", strong_cases.iloc[0]))
        if not weak_cases.empty:
            samples.append(("WEAK CONTRIBUTION", weak_cases.iloc[0]))
        if not non_propagated.empty:
            samples.append(("NON-PROPAGATED (AFFECTED BUT NO PV)", non_propagated.iloc[0]))
            
        for label, row in samples:
            print(f"\n--- {label} ---")
            print(f"Week: {row['week']}")
            print(f"Destination Borrower: {row['destination_borrower']}")
            print(f"Full-World Burden: {row['fw_burden']:.2f}")
            print(f"Counterfactual Burden: {row['cf_burden']:.2f}")
            print(f"Incremental Network Burden: {row['network_incremental_burden']:.2f}")
            print(f"Bounded Contribution Share: {row['episode_network_contribution']:.4f}")
            print(f"Final PV Label: {'Positive' if row.get('is_pv', False) else 'Negative'}")
    
if __name__ == '__main__':
    main()
