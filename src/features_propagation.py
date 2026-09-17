import pandas as pd
import numpy as np

def generate_propagation_features(df: pd.DataFrame, oof_estimates: pd.DataFrame) -> pd.DataFrame:
    """
    Computes propagation-aware exposure features (Model C) by aggregating historical neighbor stress estimates.
    This operates on a feature matrix (e.g. Next-Period Matrix) merged with OOF estimates.
    
    Features:
    - peer_predicted_stress_mean
    - borrower_propagation_exposure
    - expected_peer_debt_burden
    """
    # Merge OOF Current-Stress predictions
    out_df = df.merge(
        oof_estimates[['borrower_id', 'week', 'oof_current_stress_prob']], 
        on=['borrower_id', 'week'], 
        how='left'
    ).copy()
    
    # Fill nan with 0 for the first few weeks where history was too short to train
    out_df['oof_current_stress_prob'] = out_df['oof_current_stress_prob'].fillna(0.0)
    
    # 1. peer_predicted_stress_mean (leave-one-out)
    group_pred_sum = out_df.groupby(['week', 'group_id'])['oof_current_stress_prob'].transform('sum')
    peer_pred_sum = group_pred_sum - out_df['oof_current_stress_prob']
    peer_count = out_df['group_size'] - 1
    
    out_df['peer_predicted_stress_mean'] = np.where(peer_count > 0, peer_pred_sum / peer_count, 0.0)
    
    # 2. borrower_propagation_exposure
    # Combines group predicted stress and borrower's liability share (frozen group-first architecture)
    out_df['borrower_propagation_exposure'] = out_df['peer_predicted_stress_mean'] * out_df['borrower_liability_share']
    
    # 3. expected_peer_debt_burden
    # Debt burden of peers, weighted by their probability of stress.
    out_df['expected_amount_due'] = out_df['amount_due_mean_4w'] * out_df['oof_current_stress_prob']
    out_df['expected_income'] = out_df['weekly_income_mean_4w'] * (1 - out_df['oof_current_stress_prob'])
    
    group_exp_due_sum = out_df.groupby(['week', 'group_id'])['expected_amount_due'].transform('sum')
    peer_exp_due_sum = group_exp_due_sum - out_df['expected_amount_due']
    
    group_exp_inc_sum = out_df.groupby(['week', 'group_id'])['expected_income'].transform('sum')
    peer_exp_inc_sum = group_exp_inc_sum - out_df['expected_income']
    
    out_df['expected_peer_debt_burden'] = peer_exp_due_sum / (peer_exp_inc_sum + 1e-6)
    
    # Drop intermediate columns
    drop_cols = ['expected_amount_due', 'expected_income', 'oof_current_stress_prob']
    out_df = out_df.drop(columns=drop_cols)
    
    return out_df
