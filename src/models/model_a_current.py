import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, Any, List

class ModelACurrent:
    """
    Model A Current-Stress used exclusively for generating historical/OOF neighbor estimates.
    Target: current_stress
    Features: strictly < t (using curr_df)
    """
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 5,
            'verbose': -1,
            'random_state': 42
        }
        self.features = None
        
    def _filter_features(self, X: pd.DataFrame) -> List[str]:
        # Exclude everything network related, future outcomes, and metadata
        exclude_cols = [
            'borrower_id', 'group_id', 'week', 'split', 
            'current_stress', 'next_period_stress', 'propagation_vulnerability',
            'scenario_family', 'group_covered_amount',
            'group_size', 'peer_buffer_mean_t', 'borrower_liability_share',
            'peer_shortfall_mean_4w', 'peer_dpd_mean_4w', 'peer_debt_burden'
        ]
        return [c for c in X.columns if c not in exclude_cols]
        
    def generate_historical_predictions(self, curr_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates historical expanding-window predictions for Current Stress.
        For predictions at week t, the model is trained ONLY on weeks < t.
        Returns a DataFrame with ['borrower_id', 'week', 'oof_current_stress_prob']
        """
        df = curr_df.copy().sort_values('week')
        self.features = self._filter_features(df)
        
        df['oof_current_stress_prob'] = 0.0
        
        weeks = sorted(df['week'].unique())
        
        # Step size of 4 weeks (1 month) for retraining
        step = 4
        
        # We need at least 12 weeks of history to train a stable model
        min_history = 12
        
        for i in range(len(weeks)):
            w_start = weeks[i]
            
            # Only retrain/predict on step boundaries, or if it's the very first prediction block
            if w_start > min_history and (w_start % step == 0):
                w_end = w_start + step
                
                train_mask = df['week'] < w_start
                predict_mask = (df['week'] >= w_start) & (df['week'] < w_end)
                
                if train_mask.sum() > 100:
                    X_tr = df.loc[train_mask, self.features]
                    y_tr = df.loc[train_mask, 'current_stress'].astype(int)
                    
                    train_data = lgb.Dataset(X_tr, label=y_tr)
                    
                    # Train model without early stopping since we don't use future validation data
                    model = lgb.train(self.params, train_data, num_boost_round=50)
                    
                    if predict_mask.sum() > 0:
                        df.loc[predict_mask, 'oof_current_stress_prob'] = model.predict(df.loc[predict_mask, self.features])
                        
        # Ensure provenance is attached
        df['neighbor_provenance'] = 'Model A Current-Stress Historical (t-cutoff)'
        
        return df[['borrower_id', 'week', 'oof_current_stress_prob', 'neighbor_provenance']]
