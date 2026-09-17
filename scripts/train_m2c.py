import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import SyntheticWorldGenerator
from src.targets import construct_targets, assign_temporal_splits
from src.features import generate_features, build_feature_matrices
from src.features_propagation import generate_propagation_features
from src.models.m0_baseline import BaselineM0
from src.models.model_a import ModelA
from src.models.model_a_current import ModelACurrent
from src.models.model_b import ModelB
from src.models.model_c import ModelC

def evaluate_model(name: str, model, X: pd.DataFrame, y: pd.Series, split_name: str, threshold: float = 0.5) -> dict:
    if len(y) == 0:
        return {}
        
    preds_proba = model.predict_proba(X)
    y_int = y.astype(int)
    preds_bin = (preds_proba >= threshold).astype(int)
    
    if len(y_int.unique()) > 1:
        roc_auc = roc_auc_score(y_int, preds_proba)
        pr_auc = average_precision_score(y_int, preds_proba)
        prec = precision_score(y_int, preds_bin, zero_division=0)
        rec = recall_score(y_int, preds_bin, zero_division=0)
    else:
        roc_auc = float('nan')
        pr_auc = float('nan')
        prec = float('nan')
        rec = float('nan')
        
    print(f"{name: <25} | PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")
    return {'roc_auc': roc_auc, 'pr_auc': pr_auc, 'prec': prec, 'rec': rec}
    
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
    
    print("Generating historical OOF neighbor estimates using Model A Current-Stress...")
    model_a_current = ModelACurrent()
    oof_estimates = model_a_current.generate_historical_predictions(curr_df)
    
    print("Generating propagation-aware exposure features (Model C)...")
    next_df_prop = generate_propagation_features(next_df, oof_estimates)
    
    train_df = next_df_prop[next_df_prop['split'] == 'train']
    val_df = next_df_prop[next_df_prop['split'] == 'val']
    test_df = next_df_prop[next_df_prop['split'] == 'test']
    
    print(f"\n==============================================")
    print(f"TASK A: Generic Next-Period Stress")
    print(f"==============================================")
    
    y_train_np = train_df['next_period_stress']
    y_val_np = val_df['next_period_stress']
    y_test_np = test_df['next_period_stress']
    
    # 1. M0
    m0 = BaselineM0(dpd_threshold=7.0)
    
    # 2. Model A
    model_a = ModelA(drop_shortfall=False)
    model_a.fit(train_df, y_train_np.astype(int), val_df, y_val_np.astype(int))
    
    # 3. Model B
    model_b = ModelB()
    model_b.fit(train_df, y_train_np.astype(int), val_df, y_val_np.astype(int))
    
    print("\n--- Held-Out Test Results ---")
    evaluate_model("M0 Baseline", m0, test_df, y_test_np, "Test")
    evaluate_model("Model A", model_a, test_df, y_test_np, "Test")
    evaluate_model("Model B", model_b, test_df, y_test_np, "Test")
    
    print(f"\n==============================================")
    print(f"TASK B: Propagation Vulnerability")
    print(f"==============================================")
    
    y_train_pv = train_df['propagation_vulnerability']
    y_val_pv = val_df['propagation_vulnerability']
    y_test_pv = test_df['propagation_vulnerability']
    
    print(f"Train samples: {len(train_df)} (Target=1: {y_train_pv.sum()})")
    print(f"Validation samples: {len(val_df)} (Target=1: {y_val_pv.sum()})")
    print(f"Test samples: {len(test_df)} (Target=1: {y_test_pv.sum()})")
    
    # Train Model B on PV
    print("\nTraining Model B on Propagation Vulnerability...")
    model_b_pv = ModelB()
    model_b_pv.fit(train_df, y_train_pv.astype(int), val_df, y_val_pv.astype(int))
    
    # Train Model C on PV
    print("Training Model C (Propagation-Aware) on PV...")
    model_c = ModelC()
    model_c.fit(train_df, y_train_pv.astype(int), val_df, y_val_pv.astype(int))
    
    print("\n--- Held-Out Test Results ---")
    # For precision/recall we use a reasonable probability threshold
    # Since calibration scales things, using 0.1 for B might not match C.
    evaluate_model("Model B", model_b_pv, test_df, y_test_pv, "Test", threshold=0.05)
    evaluate_model("Model C", model_c, test_df, y_test_pv, "Test", threshold=0.05)
    
    # Model C Feature Importance
    importances = model_c.model.feature_importance(importance_type='gain')
    feature_names = model_c.model.feature_name()
    imp_df = pd.DataFrame({'feature': feature_names, 'gain': importances}).sort_values(by='gain', ascending=False)
    print("\n--- Model C Feature Importances (Top 15) ---")
    print(imp_df.head(15).to_string(index=False))

if __name__ == "__main__":
    main()
