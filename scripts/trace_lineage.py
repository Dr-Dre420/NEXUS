import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import SyntheticWorldGenerator
from src.targets import construct_targets

def trace():
    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    timing_metrics = []
    
    for seed in seeds:
        gen = SyntheticWorldGenerator(n_borrowers=400, seed=seed, coverage_fraction=0.50, slack_regime="conservative")
        h, l = gen.simulate(156)
        
        if len(l) > 0:
            t = construct_targets(h, l)
            
            for idx, row in l.iterrows():
                week = row['week']
                dest_b = row['destination_borrower']
                share = row['contribution_share']
                
                targets_dest = t[t['borrower_id'] == dest_b]
                
                # Trace source borrower
                group_info = h[(h['borrower_id'] == dest_b) & (h['week'] == week)]
                if len(group_info) == 0: continue
                group_id = group_info.iloc[0]['group_id']
                
                group_members = h[(h['group_id'] == group_id) & (h['week'] <= week)]
                source_candidates = group_members[(group_members['shortfall'] > 0) & (group_members['borrower_id'] != dest_b)]
                if len(source_candidates) == 0: continue
                
                first_source_shortfall = source_candidates['week'].min()
                source_b = source_candidates[source_candidates['week'] == first_source_shortfall].iloc[0]['borrower_id']
                
                stress_events = targets_dest[(targets_dest['week'] >= first_source_shortfall) & (targets_dest['current_stress'] == True)]
                if len(stress_events) > 0:
                    dest_stress_week = stress_events['week'].min()
                else:
                    dest_stress_week = -1
                    
                timing_metrics.append({
                    "seed": seed,
                    "source": source_b,
                    "dest": dest_b,
                    "source_shortfall_week": first_source_shortfall,
                    "dest_stress_week": dest_stress_week,
                    "lag": dest_stress_week - first_source_shortfall if dest_stress_week != -1 else -1,
                    "share": share
                })
                
    if timing_metrics:
        df_metrics = pd.DataFrame(timing_metrics)
        print("\n--- TIMING AUDIT ---")
        print(df_metrics['lag'].describe())
        valid_lags = df_metrics[df_metrics['lag'] >= 0]
        print(f"\nOccurring within 4 weeks: {len(valid_lags[valid_lags['lag'] <= 4])} / {len(df_metrics)}")
        print(f"Never stressed (lag = -1): {len(df_metrics[df_metrics['lag'] == -1])} / {len(df_metrics)}")
        
if __name__ == "__main__":
    trace()
