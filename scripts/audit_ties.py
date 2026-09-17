import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import SyntheticWorldGenerator
from src.targets import construct_targets, assign_temporal_splits
from src.features import generate_features, build_feature_matrices
from src.features_propagation import generate_propagation_features
from src.models.model_c import ModelC
from src.models.model_a_current import ModelACurrent

def main():
    seed = 909
    generator = SyntheticWorldGenerator(n_borrowers=400, seed=seed, slack_regime='conservative')
    history_df, lineage_df, ep_lineage_df = generator.simulate(n_weeks=156)
    targets_df = assign_temporal_splits(construct_targets(history_df, ep_lineage_df))
    features_df = generate_features(history_df)
    curr_df, next_df = build_feature_matrices(targets_df, features_df)
    
    oof_estimates = ModelACurrent().generate_historical_predictions(curr_df)
    next_df_prop = generate_propagation_features(next_df, oof_estimates)
    
    train_df = next_df_prop[next_df_prop['split'] == 'train']
    val_df = next_df_prop[next_df_prop['split'] == 'val']
    test_df = next_df_prop[next_df_prop['split'] == 'test']
    
    y_tr_pv, y_val_pv, y_te_pv = train_df['propagation_vulnerability'], val_df['propagation_vulnerability'], test_df['propagation_vulnerability']
    
    mc = ModelC()
    mc.fit(train_df, y_tr_pv, val_df, y_val_pv)
    
    raw_preds = mc.model.predict(test_df[mc.features])
    cal_preds = mc.predict_proba(test_df)
    
    print(f"Raw unique: {len(np.unique(raw_preds))}")
    print(f"Cal unique: {len(np.unique(cal_preds))}")
    
    # Are there rank inversions?
    df = pd.DataFrame({'raw': raw_preds, 'cal': cal_preds, 'y': y_te_pv})
    df = df.sort_values('raw')
    df['cal_diff'] = df['cal'].diff()
    inversions = df[df['cal_diff'] < 0]
    print(f"Inversions: {len(inversions)}")

    df = df.sort_values('cal')
    print("Top 10 cal preds (with corresponding raw and y):")
    print(df.tail(10))

if __name__ == '__main__':
    main()
