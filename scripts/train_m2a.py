import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import SyntheticWorldGenerator
from src.targets import construct_targets, assign_temporal_splits
from src.features import generate_features, build_feature_matrices
from src.models.m0_baseline import BaselineM0
from src.models.model_a import ModelA

def evaluate_model(name: str, model, X: pd.DataFrame, y: pd.Series, split_name: str):
    if len(y) == 0:
        print(f"--- {name} on {split_name} ---")
        print("Empty dataset.")
        return
        
    preds_proba = model.predict_proba(X)
    y_int = y.astype(int)
    
    # In case there's only 1 class in the split, AUC calculation fails
    if len(y_int.unique()) > 1:
        roc_auc = roc_auc_score(y_int, preds_proba)
        pr_auc = average_precision_score(y_int, preds_proba)
    else:
        roc_auc = float('nan')
        pr_auc = float('nan')
        
    print(f"--- {name} on {split_name} ---")
    print(f"Samples: {len(y)}, Target=1: {y_int.sum()} ({y_int.mean():.2%})")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    
def main():
    print("Generating synthetic world data...")
    generator = SyntheticWorldGenerator(n_borrowers=400, seed=42)
    history_df, lineage_df = generator.simulate(n_weeks=156)
    
    print("Constructing targets (Current Stress, Next-Period Stress, Propagation Vulnerability)...")
    targets_df = construct_targets(history_df, lineage_df)
    targets_df = assign_temporal_splits(targets_df)
    
    print("Generating temporal features (4w and 12w trailing)...")
    features_df = generate_features(history_df)
    
    print("Building temporally isolated feature matrices...")
    curr_df, next_df = build_feature_matrices(targets_df, features_df)
    
    # We will evaluate Next-Period Stress as the primary predictive task for M2A
    target_col = 'next_period_stress'
    
    print(f"\n==============================================")
    print(f"Evaluating Next-Period Stress Task")
    print(f"==============================================")
    
    # Split datasets
    train_df = next_df[next_df['split'] == 'train']
    val_df = next_df[next_df['split'] == 'val']
    test_df = next_df[next_df['split'] == 'test']
    
    X_train, y_train = train_df, train_df[target_col]
    X_val, y_val = val_df, val_df[target_col]
    X_test, y_test = test_df, test_df[target_col]
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # M0 Baseline
    print("\nTraining M0 Baseline (Deterministic Delinquency Rule)...")
    # Setting threshold to 7 days past due maximum over last 4 weeks
    m0 = BaselineM0(dpd_threshold=7.0)
    
    # Model A
    print("\nTraining Model A (LightGBM on individual tabular features)...")
    model_a = ModelA()
    model_a.fit(X_train, y_train.astype(int), X_val, y_val.astype(int))
    
    print("\n--- Validation Results ---")
    evaluate_model("M0 Baseline", m0, X_val, y_val, "Validation")
    evaluate_model("Model A", model_a, X_val, y_val, "Validation")
    
    print("\n--- Held-Out Test Results ---")
    evaluate_model("M0 Baseline", m0, X_test, y_test, "Test")
    evaluate_model("Model A", model_a, X_test, y_test, "Test")
    
if __name__ == "__main__":
    main()
