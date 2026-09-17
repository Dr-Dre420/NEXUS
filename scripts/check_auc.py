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
    
    raw_auc = roc_auc_score(y_te_pv, raw_preds)
    cal_auc = roc_auc_score(y_te_pv, cal_preds)
    
    spearman = spearmanr(raw_preds, cal_preds).statistic
    
    print(f"Raw AUC: {raw_auc}")
    print(f"Cal AUC: {cal_auc}")
    print(f"Spearman rank correlation: {spearman}")
    print(f"Classes: {mc.calibrator.classes_}")
    print(f"Coef: {mc.calibrator.coef_[0][0]}")

if __name__ == '__main__':
    main()
