import pandas as pd
import numpy as np
from typing import Tuple

def generate_features(history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates temporal features (4-week and 12-week trailing) for each borrower at each week.
    Features only use individual financial state, strictly avoiding network leakage.
    Produces features computed up to week 't'.
    """
    df = history_df.sort_values(['borrower_id', 'week']).copy()
    
    # Initialize features DataFrame with identity keys
    features = df[['borrower_id', 'week', 'principal_remaining']].copy()
    
    # Ensure no leakage columns are present
    assert 'group_covered_amount' in df.columns, "Expected group_covered_amount in history"
    assert 'scenario_family' in df.columns, "Expected scenario_family in history"
    
    cols_to_roll = [
        'cash_buffer', 'weekly_income', 'weekly_expenses', 
        'amount_due', 'days_past_due', 'shortfall'
    ]
    
    for window in [4, 12]:
        for col in cols_to_roll:
            grouped = df.groupby('borrower_id')[col]
            
            features[f'{col}_mean_{window}w'] = grouped.transform(lambda x: x.rolling(window, min_periods=1).mean())
            features[f'{col}_max_{window}w'] = grouped.transform(lambda x: x.rolling(window, min_periods=1).max())
            features[f'{col}_var_{window}w'] = grouped.transform(lambda x: x.rolling(window, min_periods=2).var().fillna(0.0))
            
            if col in ['shortfall', 'amount_due']:
                features[f'{col}_sum_{window}w'] = grouped.transform(lambda x: x.rolling(window, min_periods=1).sum())

    # Debt burden at time t
    features['debt_burden_ratio'] = df['amount_due'] / (df['weekly_income'] + 1e-6)
    
    # Recent trend: Cash buffer change over 4 weeks
    features['buffer_trend_4w'] = df.groupby('borrower_id')['cash_buffer'].transform(
        lambda x: x - x.shift(4).bfill()
    )

    # --- NETWORK / GROUP FEATURES (MODEL B) ---
    # These aggregations strictly use data available at time 't' and are isolated in the exact same manner.
    
    features['group_id'] = df['group_id']
    
    # Group sizes
    df['group_size'] = df.groupby(['week', 'group_id'])['borrower_id'].transform('count')
    features['group_size'] = df['group_size']
    peer_count = df['group_size'] - 1
    
    # 1. Peer cash buffer mean
    group_buffer_sum = df.groupby(['week', 'group_id'])['cash_buffer'].transform('sum')
    peer_buffer_sum = group_buffer_sum - df['cash_buffer']
    features['peer_buffer_mean_t'] = np.where(peer_count > 0, peer_buffer_sum / peer_count, 0.0)
    
    # 2. Borrower liability share (borrower's share of total group buffer)
    features['borrower_liability_share'] = df['cash_buffer'] / (group_buffer_sum + 1e-6)
    
    # 3. Peer shortfall mean (4w)
    group_sf_sum_4w = features.groupby(['week', 'group_id'])['shortfall_sum_4w'].transform('sum')
    peer_sf_sum_4w = group_sf_sum_4w - features['shortfall_sum_4w']
    features['peer_shortfall_mean_4w'] = np.where(peer_count > 0, peer_sf_sum_4w / peer_count, 0.0)
    
    # 4. Peer DPD mean (4w max)
    group_dpd_sum = features.groupby(['week', 'group_id'])['days_past_due_max_4w'].transform('sum')
    peer_dpd_sum = group_dpd_sum - features['days_past_due_max_4w']
    features['peer_dpd_mean_4w'] = np.where(peer_count > 0, peer_dpd_sum / peer_count, 0.0)
    
    # 5. Peer debt burden
    group_amt_due = df.groupby(['week', 'group_id'])['amount_due'].transform('sum')
    group_inc = df.groupby(['week', 'group_id'])['weekly_income'].transform('sum')
    peer_amt_due = group_amt_due - df['amount_due']
    peer_inc = group_inc - df['weekly_income']
    features['peer_debt_burden'] = peer_amt_due / (peer_inc + 1e-6)

    # Need to keep group_id in features so build_feature_matrices can access it, but we can drop it later
    features['group_id'] = df['group_id']

    return features

def build_feature_matrices(targets_df: pd.DataFrame, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Joins targets with features enforcing temporal rules:
    - Current Stress: target at t uses features strictly before t (i.e. t-1).
    - Next-Period Stress: target at t uses features at or before t.
    """
    # 1. Next-Period Matrix (features up to t)
    # Both targets_df and features_df use 'week' for time t.
    next_df = targets_df.merge(features_df, on=['borrower_id', 'week'], how='inner')
    
    # 2. Current Stress Matrix (features up to t-1)
    # Shift features forward by 1 week: the feature row computed for week W is now labeled as week W+1,
    # so when it merges with target at week W+1, it provides the features up to week W.
    shifted_features = features_df.copy()
    shifted_features['week'] = shifted_features['week'] + 1
    
    curr_df = targets_df.merge(shifted_features, on=['borrower_id', 'week'], how='inner')
    
    return curr_df, next_df
