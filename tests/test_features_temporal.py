import pytest
import pandas as pd
import numpy as np

from src.features import generate_features, build_feature_matrices

def test_features_temporal_isolation():
    # 1. Mock history dataframe
    history_data = []
    for w in range(1, 10):
        history_data.append({
            'week': w,
            'borrower_id': 'B1',
            'group_id': 'G1',
            'cash_buffer': w * 100.0,
            'weekly_income': 2000.0,
            'weekly_expenses': 1000.0,
            'amount_due': w * 50.0,
            'days_past_due': 0,
            'shortfall': 0.0,
            'principal_remaining': 50000.0,
            'group_covered_amount': 0.0,
            'scenario_family': 'none'
        })
    history_df = pd.DataFrame(history_data)
    
    # Generate features
    features_df = generate_features(history_df)
    
    # 2. Mock target dataframe
    targets_data = []
    for w in range(5, 10):
        targets_data.append({
            'week': w,
            'borrower_id': 'B1',
            'split': 'train',
            'current_stress': False,
            'next_period_stress': False,
            'propagation_vulnerability': False
        })
    targets_df = pd.DataFrame(targets_data)
    
    # Build matrices
    curr_df, next_df = build_feature_matrices(targets_df, features_df)
    
    # --- Tests ---
    
    # Assert network leakage columns are excluded
    assert 'group_covered_amount' not in curr_df.columns
    assert 'scenario_family' not in curr_df.columns
    
    # For Next-Period Stress at week 5, the cash_buffer_mean_4w should be the average of weeks 2,3,4,5
    # cash buffers: week 2=200, 3=300, 4=400, 5=500. Mean = 350.
    next_week_5 = next_df[(next_df['week'] == 5) & (next_df['borrower_id'] == 'B1')].iloc[0]
    assert next_week_5['cash_buffer_mean_4w'] == 350.0
    
    # For Current Stress at week 5, the features must ONLY use data strictly before 5 (up to 4).
    # cash buffers: week 1=100, 2=200, 3=300, 4=400. Mean = 250.
    curr_week_5 = curr_df[(curr_df['week'] == 5) & (curr_df['borrower_id'] == 'B1')].iloc[0]
    assert curr_week_5['cash_buffer_mean_4w'] == 250.0
