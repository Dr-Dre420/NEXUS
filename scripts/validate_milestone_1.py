import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import SyntheticWorldGenerator
from src.targets import construct_targets, assign_temporal_splits, load_config

def main():
    config = load_config()
    threshold = config['propagation']['meaningful_contribution_share']
    
    print(f"Starting Milestone 1 Validation...")
    print(f"Propagation threshold: {threshold}")
    
    results = []
    total_unique_borrowers = set()
    total_unique_jlgs = set()
    
    seeds = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010]
    
    for seed in seeds:
        print(f"Generating world for seed {seed}...")
        gen = SyntheticWorldGenerator(seed=seed, n_borrowers=400)
        history_df, lineage_df = gen.simulate(n_weeks=156)
        
        targets_df = construct_targets(history_df, lineage_df)
        targets_df = assign_temporal_splits(targets_df)
        
        # Merge with history to get group_id for reporting
        history_subset = history_df[['week', 'borrower_id', 'group_id']]
        targets_df = targets_df.merge(history_subset, on=['week', 'borrower_id'], how='left')
        
        # Count propagation events
        prop_events = targets_df[targets_df['propagation_vulnerability'] == True]
        
        if seed == 1001:
            print("Weeks with propagation vulnerability:", sorted(prop_events['week'].unique()))
            
        train_events = prop_events[prop_events['split'] == 'train'].shape[0]
        val_events = prop_events[prop_events['split'] == 'val'].shape[0]
        test_events = prop_events[prop_events['split'] == 'test'].shape[0]
        
        results.append({
            'World Seed': seed,
            'Train Events': train_events,
            'Validation Events': val_events,
            'Test Events': test_events
        })
        
        unique_borrowers = prop_events['borrower_id'].unique()
        unique_groups = prop_events['group_id'].unique()
        
        total_unique_borrowers.update(unique_borrowers)
        total_unique_jlgs.update(unique_groups)

    results_df = pd.DataFrame(results)
    
    print("\nEvent-Count Validation Output:")
    print("-" * 60)
    print(results_df.to_string(index=False))
    print("-" * 60)
    
    total_test_events = results_df['Test Events'].sum()
    print(f"\nTotal Propagation-attributed episodes (across all seeds): {results_df[['Train Events', 'Validation Events', 'Test Events']].sum().sum()}")
    print(f"Unique affected borrowers (across all seeds): {len(total_unique_borrowers)}")
    print(f"Unique affected JLGs (across all seeds): {len(total_unique_jlgs)}")
    print(f"Held-out (Test) propagation event count (across all seeds): {total_test_events}")
    
    if total_test_events < 50:
        print("\nWARNING: UNDERPOWERED EVALUATION")
        print("The held-out propagation events are too sparse for meaningful downstream evaluation.")
        print("Do not proceed to model development. Adjust the synthetic-world mechanics/scenario frequency first.")
    else:
        print("\nSUCCESS: Evaluation power appears sufficient for model development.")

if __name__ == "__main__":
    main()
