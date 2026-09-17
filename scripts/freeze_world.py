import os
import sys
import pickle
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import SyntheticWorldGenerator
from src.targets import construct_targets, assign_temporal_splits
from src.features import generate_features, build_feature_matrices
from src.features_propagation import generate_propagation_features
from src.models.model_c import ModelC
from src.models.model_b import ModelB
from src.models.model_a_current import ModelACurrent

def freeze_world(seed: int = 909):
    print(f"Freezing world seed {seed}...")
    
    # 1. Generate World
    generator = SyntheticWorldGenerator(n_borrowers=400, seed=seed, slack_regime='conservative')
    history_df, lineage_df, ep_lineage_df = generator.simulate(n_weeks=156)
    
    # 2. Targets & Features
    targets_df = assign_temporal_splits(construct_targets(history_df, ep_lineage_df))
    features_df = generate_features(history_df)
    curr_df, next_df = build_feature_matrices(targets_df, features_df)
    
    # 3. Model A OOF Estimates & Propagation Features
    ma = ModelACurrent()
    oof_estimates = ma.generate_historical_predictions(curr_df)
    next_df_prop = generate_propagation_features(next_df, oof_estimates)
    
    # 4. Train Models
    train_df = next_df_prop[next_df_prop['split'] == 'train']
    val_df = next_df_prop[next_df_prop['split'] == 'val']
    test_df = next_df_prop[next_df_prop['split'] == 'test']
    
    y_tr_pv = train_df['propagation_vulnerability']
    y_val_pv = val_df['propagation_vulnerability']
    
    mb = ModelB()
    mb.fit(train_df, y_tr_pv, val_df, y_val_pv)
    
    mc = ModelC()
    mc.fit(train_df, y_tr_pv, val_df, y_val_pv)
    
    # 5. Predict on Full Timeline
    next_df_prop['model_b_score'] = mb.predict_proba(next_df_prop)
    next_df_prop['model_c_score_raw'] = mc.model.predict(next_df_prop[mc.features])
    next_df_prop['model_c_score'] = mc.predict_proba(next_df_prop)
    
    # 6. Save Artifacts
    save_dir = os.path.join('data', 'frozen_m2c')
    os.makedirs(save_dir, exist_ok=True)
    
    frozen_data = {
        'seed': seed,
        'history_df': history_df,
        'lineage_df': lineage_df,
        'ep_lineage_df': ep_lineage_df,
        'targets_df': targets_df,
        'next_df_prop': next_df_prop,
        'as_of_week': test_df['week'].min(), # First week of test set is our "current" demo time
    }
    
    with open(os.path.join(save_dir, 'world_state.pkl'), 'wb') as f:
        pickle.dump(frozen_data, f)
        
    with open(os.path.join(save_dir, 'model_b.pkl'), 'wb') as f:
        pickle.dump(mb, f)
        
    with open(os.path.join(save_dir, 'model_c.pkl'), 'wb') as f:
        pickle.dump(mc, f)

    print(f"World state saved to {save_dir}/world_state.pkl")
    print(f"Demo / as-of week established at: {frozen_data['as_of_week']}")

if __name__ == '__main__':
    freeze_world(909)
