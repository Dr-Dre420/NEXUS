import pytest
import pandas as pd
import numpy as np

from src.models.model_a_current import ModelACurrent
from src.features_propagation import generate_propagation_features

def test_historical_oof_predictions():
    # Create fake curr_df with weeks 1 to 20
    data = []
    for w in range(1, 21):
        for b in range(1, 11):
            data.append({
                'week': w,
                'borrower_id': f'B{b}',
                'group_id': 'G1' if b <= 5 else 'G2',
                'current_stress': True if b == 1 and w > 10 else False,
                'cash_buffer': 100.0,
                'weekly_income': 2000.0,
                'amount_due': 500.0,
                'group_size': 5,
                'borrower_liability_share': 0.2
            })
    curr_df = pd.DataFrame(data)
    
    model = ModelACurrent()
    oof_preds = model.generate_historical_predictions(curr_df)
    
    # Assert weeks <= 12 have exactly 0 probability (because we wait for min_history=12)
    early_preds = oof_preds[oof_preds['week'] <= 12]
    assert all(early_preds['oof_current_stress_prob'] == 0.0)
    
    # Check that predictions for week 15 exist
    later_preds = oof_preds[oof_preds['week'] == 15]
    assert len(later_preds) == 10
    
    # Ensure provenance tag exists
    assert 'neighbor_provenance' in oof_preds.columns

def test_propagation_features_and_healthy_borrower_protection():
    # Setup mock features and oof estimates
    # Test a healthy borrower B1 with a highly stressed peer B2
    features_data = [
        {'borrower_id': 'B1', 'week': 15, 'group_id': 'G1', 'group_size': 2, 'borrower_liability_share': 0.5, 'amount_due_mean_4w': 100, 'weekly_income_mean_4w': 500},
        {'borrower_id': 'B2', 'week': 15, 'group_id': 'G1', 'group_size': 2, 'borrower_liability_share': 0.5, 'amount_due_mean_4w': 100, 'weekly_income_mean_4w': 500}
    ]
    features_df = pd.DataFrame(features_data)
    
    oof_data = [
        {'borrower_id': 'B1', 'week': 15, 'oof_current_stress_prob': 0.05}, # Healthy borrower
        {'borrower_id': 'B2', 'week': 15, 'oof_current_stress_prob': 0.95}  # Highly stressed neighbor
    ]
    oof_df = pd.DataFrame(oof_data)
    
    out_df = generate_propagation_features(features_df, oof_df)
    
    # B1's peer is B2 (stress prob 0.95). B1 peer_predicted_stress_mean should be 0.95.
    b1_row = out_df[out_df['borrower_id'] == 'B1'].iloc[0]
    assert np.isclose(b1_row['peer_predicted_stress_mean'], 0.95)
    
    # But B1's exposure is modulated by liability share (0.5), protecting the healthy borrower from raw saturation
    assert np.isclose(b1_row['borrower_propagation_exposure'], 0.95 * 0.5) 
    
    # B2's peer is B1 (stress prob 0.05).
    b2_row = out_df[out_df['borrower_id'] == 'B2'].iloc[0]
    assert np.isclose(b2_row['peer_predicted_stress_mean'], 0.05)
    assert np.isclose(b2_row['borrower_propagation_exposure'], 0.05 * 0.5)
