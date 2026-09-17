import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import SyntheticWorldGenerator
from src.targets import construct_targets, assign_temporal_splits
from src.features import generate_features, build_feature_matrices
from src.models.model_a import ModelA

def main():
    generator = SyntheticWorldGenerator(n_borrowers=400, seed=42)
    history_df, lineage_df = generator.simulate(n_weeks=156)
    
    targets_df = construct_targets(history_df, lineage_df)
    targets_df = assign_temporal_splits(targets_df)
    
    features_df = generate_features(history_df)
    curr_df, next_df = build_feature_matrices(targets_df, features_df)
    
    target_col = 'next_period_stress'
    
    train_df = next_df[next_df['split'] == 'train']
    val_df = next_df[next_df['split'] == 'val']
    
    X_train, y_train = train_df, train_df[target_col]
    X_val, y_val = val_df, val_df[target_col]
    
    model_a = ModelA()
    model_a.fit(X_train, y_train.astype(int), X_val, y_val.astype(int))
    
    importances = model_a.model.feature_importance(importance_type='gain')
    feature_names = model_a.model.feature_name()
    
    imp_df = pd.DataFrame({'feature': feature_names, 'gain': importances})
    imp_df = imp_df.sort_values(by='gain', ascending=False)
    
    print("--- Feature Importances (Gain) ---")
    print(imp_df.to_string(index=False))

if __name__ == "__main__":
    main()
