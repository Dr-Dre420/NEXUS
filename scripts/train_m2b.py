import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import SyntheticWorldGenerator
from src.targets import construct_targets, assign_temporal_splits
from src.features import generate_features, build_feature_matrices
from src.models.m0_baseline import BaselineM0
from src.models.model_a import ModelA
from src.models.model_b import ModelB

def evaluate_model(name: str, model, X: pd.DataFrame, y: pd.Series, split_name: str) -> dict:
    if len(y) == 0:
        print(f"{name: <25} | Empty dataset.")
        return {}
        
    preds_proba = model.predict_proba(X)
    y_int = y.astype(int)
    
    if len(y_int.unique()) > 1:
        roc_auc = roc_auc_score(y_int, preds_proba)
        pr_auc = average_precision_score(y_int, preds_proba)
    else:
        roc_auc = float('nan')
        pr_auc = float('nan')
        
    print(f"{name: <25} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    return {'roc_auc': roc_auc, 'pr_auc': pr_auc}
    
def main():
    print("Generating synthetic world data...")
    generator = SyntheticWorldGenerator(n_borrowers=400, seed=42)
    history_df, lineage_df = generator.simulate(n_weeks=156)
    
    print("Constructing targets...")
    targets_df = construct_targets(history_df, lineage_df)
    targets_df = assign_temporal_splits(targets_df)
    
    print("Generating temporal individual and network features...")
    features_df = generate_features(history_df)
    
    print("Building temporally isolated feature matrices...")
    curr_df, next_df = build_feature_matrices(targets_df, features_df)
    
    target_col = 'next_period_stress'
    
    print(f"\n==============================================")
    print(f"Evaluating Next-Period Stress Task (M2B)")
    print(f"==============================================")
    
    train_df = next_df[next_df['split'] == 'train']
    val_df = next_df[next_df['split'] == 'val']
    test_df = next_df[next_df['split'] == 'test']
    
    X_train, y_train = train_df, train_df[target_col]
    X_val, y_val = val_df, val_df[target_col]
    X_test, y_test = test_df, test_df[target_col]
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)} (Target=1: {y_val.sum()})")
    print(f"Test samples: {len(X_test)} (Target=1: {y_test.sum()})")
    
    # 1. M0
    m0 = BaselineM0(dpd_threshold=7.0)
    
    # 2. Model A (Individual only)
    print("\nTraining Model A (Individual Baseline)...")
    model_a = ModelA(drop_shortfall=False)
    model_a.fit(X_train, y_train.astype(int), X_val, y_val.astype(int))
    
    # 3. Model A-no-shortfall (Diagnostic)
    print("Training Model A-no-shortfall (Diagnostic)...")
    model_a_no_sf = ModelA(drop_shortfall=True)
    model_a_no_sf.fit(X_train, y_train.astype(int), X_val, y_val.astype(int))
    
    # 4. Model B (Individual + Network)
    print("Training Model B (Individual + Network Context)...")
    model_b = ModelB()
    model_b.fit(X_train, y_train.astype(int), X_val, y_val.astype(int))
    
    print("\n--- Validation Results ---")
    evaluate_model("M0 Baseline", m0, X_val, y_val, "Validation")
    evaluate_model("Model A", model_a, X_val, y_val, "Validation")
    evaluate_model("Model A-no-shortfall", model_a_no_sf, X_val, y_val, "Validation")
    evaluate_model("Model B", model_b, X_val, y_val, "Validation")
    
    print("\n--- Held-Out Test Results ---")
    evaluate_model("M0 Baseline", m0, X_test, y_test, "Test")
    evaluate_model("Model A", model_a, X_test, y_test, "Test")
    evaluate_model("Model A-no-shortfall", model_a_no_sf, X_test, y_test, "Test")
    evaluate_model("Model B", model_b, X_test, y_test, "Test")
    
    print("\n--- Feature Counts ---")
    print(f"Model A features: {len(model_a.features)}")
    print(f"Model A-no-shortfall features: {len(model_a_no_sf.features)}")
    print(f"Model B features: {len(model_b.features)}")
    
    # Model B Feature Importance
    importances = model_b.model.feature_importance(importance_type='gain')
    feature_names = model_b.model.feature_name()
    imp_df = pd.DataFrame({'feature': feature_names, 'gain': importances}).sort_values(by='gain', ascending=False)
    print("\n--- Model B Feature Importances (Top 15) ---")
    print(imp_df.head(15).to_string(index=False))

if __name__ == "__main__":
    main()
