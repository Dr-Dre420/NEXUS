import pandas as pd
from src.targets import construct_targets, compute_stress_state, assign_temporal_splits, load_config

def test_current_stress_leakage_concept():
    # Construct a dummy history
    history = pd.DataFrame({
        'week': [1, 2],
        'borrower_id': ['B1', 'B1'],
        'days_past_due': [0, 15],
        'cash_buffer': [100.0, 0.0]
    })
    
    config = load_config()
    stress_df = compute_stress_state(history, config)
    
    # Assert that current stress doesn't look ahead
    assert not stress_df[stress_df['week'] == 1]['is_stressed'].iloc[0]
    assert stress_df[stress_df['week'] == 2]['is_stressed'].iloc[0]
    
def test_purge_gaps():
    targets = pd.DataFrame({
        'week': list(range(1, 157))
    })
    targets_with_splits = assign_temporal_splits(targets)
    
    # Assert there are purge weeks
    assert 'purge' in targets_with_splits['split'].values
    
    config = load_config()
    purge_gap = config['temporal']['purge_gap_weeks']
    
    # Count purge weeks - there should be 2 purge gaps of size `purge_gap`
    purge_count = targets_with_splits[targets_with_splits['split'] == 'purge'].shape[0]
    assert purge_count >= purge_gap * 2
