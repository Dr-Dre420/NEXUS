import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import SyntheticWorldGenerator
from src.targets import construct_targets, assign_temporal_splits
from src.features import generate_features, build_feature_matrices
from src.features_propagation import generate_propagation_features
from src.models.model_a_current import ModelACurrent
from src.models.model_b import ModelB
from src.models.model_c import ModelC

def audit_prevalence():
    print("============================================================")
    print("1. PROPAGATION-TARGET PREVALENCE AUDIT")
    print("============================================================")
    
    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    all_targets = []
    all_lineage = []
    
    for s in seeds:
        gen = SyntheticWorldGenerator(n_borrowers=400, seed=s)
        h_df, l_df = gen.simulate(n_weeks=156)
        targets = construct_targets(h_df, l_df)
        targets = assign_temporal_splits(targets)
        
        targets['seed'] = s
        all_targets.append(targets)
        
        l_df['seed'] = s
        all_lineage.append(l_df)
        
    df = pd.concat(all_targets, ignore_index=True)
    l_df = pd.concat(all_lineage, ignore_index=True)
    
    # We only care about rows where next_period_stress is True
    eligible = df[df['next_period_stress'] == True]
    
    print(f"Total borrowers eligible for PV (Next-Period Stress == True): {len(eligible)}")
    pv_count = df['propagation_vulnerability'].sum()
    print(f"Propagation-Vulnerable Positives (Threshold >= 0.30): {pv_count}")
    print(f"Positive Rate among eligible: {pv_count / len(eligible):.4f}")
    
    print("\nBy Split (Count of PV Positives):")
    split_counts = df.groupby('split')['propagation_vulnerability'].sum()
    print(split_counts)
    
    print("\nContribution Share Distribution (All Lineage Events):")
    print(l_df['contribution_share'].describe())
    
    print("\nContribution Share Distribution (Meaningful >= 0.30):")
    meaningful = l_df[l_df['contribution_share'] >= 0.30]
    print(meaningful['contribution_share'].describe())
    
def audit_features():
    print("\n============================================================")
    print("3. MODEL B VS MODEL C FEATURE REDUNDANCY AUDIT")
    print("============================================================")
    
    gen = SyntheticWorldGenerator(n_borrowers=400, seed=42)
    h_df, l_df = gen.simulate(n_weeks=156)
    targets = construct_targets(h_df, l_df)
    targets = assign_temporal_splits(targets)
    features_df = generate_features(h_df)
    curr_df, next_df = build_feature_matrices(targets, features_df)
    
    model_a_current = ModelACurrent()
    oof_estimates = model_a_current.generate_historical_predictions(curr_df)
    
    next_df_prop = generate_propagation_features(next_df, oof_estimates)
    
    prop_features = ['peer_predicted_stress_mean', 'borrower_propagation_exposure', 'expected_peer_debt_burden']
    
    print("Model C Propagation Feature Distributions (Train Set):")
    train_df = next_df_prop[next_df_prop['split'] == 'train']
    print(train_df[prop_features].describe())
    
    print("\nNon-Null / Non-Zero Rate:")
    for f in prop_features:
        non_zero = (train_df[f] > 0).mean()
        print(f"{f}: {non_zero:.2%} > 0")
        
    print("\nCorrelation with existing Model B static network features:")
    b_features = ['peer_buffer_mean_t', 'borrower_liability_share', 'peer_debt_burden']
    corr_matrix = train_df[prop_features + b_features].corr()
    print(corr_matrix.loc[prop_features, b_features])
    
    return next_df_prop

def audit_healthy(df):
    print("\n============================================================")
    print("5. HEALTHY-BORROWER PROTECTION AUDIT")
    print("============================================================")
    
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test'].copy()
    
    mb = ModelB()
    mc = ModelC()
    
    y_train = train_df['propagation_vulnerability'].astype(int)
    mb.fit(train_df, y_train, test_df, test_df['propagation_vulnerability'].astype(int))
    mc.fit(train_df, y_train, test_df, test_df['propagation_vulnerability'].astype(int))
    
    test_df['mb_prob'] = mb.predict_proba(test_df)
    test_df['mc_prob'] = mc.predict_proba(test_df)
    
    print("Healthy Borrower + Highly Stressed Peers (Diagnostic Case C)")
    # High cash buffer, high peer predicted stress, but low liability share -> weak exposure
    # Or high liability share -> strong exposure
    
    cases = test_df[(test_df['cash_buffer_mean_4w'] > 100) & (test_df['peer_predicted_stress_mean'] > 0.5)]
    if not cases.empty:
        sample = cases.head(3)[['borrower_id', 'week', 'cash_buffer_mean_4w', 'peer_predicted_stress_mean', 'borrower_propagation_exposure', 'mb_prob', 'mc_prob']]
        print(sample.to_string(index=False))
    else:
        print("No cases found in test.")
        
    print("\nStressed + Isolated (Diagnostic Case E)")
    cases2 = test_df[(test_df['shortfall_sum_4w'] > 50) & (test_df['borrower_propagation_exposure'] == 0)]
    if not cases2.empty:
        sample2 = cases2.head(3)[['borrower_id', 'week', 'shortfall_sum_4w', 'borrower_propagation_exposure', 'mb_prob', 'mc_prob']]
        print(sample2.to_string(index=False))
    else:
        print("No cases found in test.")

if __name__ == "__main__":
    audit_prevalence()
    df = audit_features()
    audit_healthy(df)
