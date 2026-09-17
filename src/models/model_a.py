import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, Any, List

class ModelA:
    """
    Model A: First learned individual-financial model.
    Uses LightGBM trained strictly on individual/household financial features.
    Excludes all group/network features.
    """
    def __init__(self, params: Dict[str, Any] = None, drop_shortfall: bool = False):
        self.params = params or {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 5,
            'verbose': -1,
            'random_state': 42
        }
        self.drop_shortfall = drop_shortfall
        self.model = None
        self.features = None
        
    def _filter_features(self, X: pd.DataFrame) -> List[str]:
        # Strictly exclude metadata, future outcomes, and ALL NETWORK features
        exclude_cols = [
            'borrower_id', 'group_id', 'week', 'split', 
            'current_stress', 'next_period_stress', 'propagation_vulnerability',
            'scenario_family', 'group_covered_amount',
            'group_size', 'peer_buffer_mean_t', 'borrower_liability_share',
            'peer_shortfall_mean_4w', 'peer_dpd_mean_4w', 'peer_debt_burden'
        ]
        
        # If 'no_shortfall' diagnostic is requested, we drop all shortfall/dpd/amount_due features
        if self.drop_shortfall:
            shortfall_cols = [c for c in X.columns if 'shortfall' in c or 'days_past_due' in c or 'amount_due' in c or 'debt_burden' in c]
            exclude_cols.extend(shortfall_cols)
            
        return [c for c in X.columns if c not in exclude_cols]
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: pd.DataFrame = None, y_val: pd.Series = None):
            
        self.features = self._filter_features(X_train)
        
        train_data = lgb.Dataset(X_train[self.features], label=y_train)
        
        valid_sets = [train_data]
        valid_names = ['train']
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val[self.features], label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('val')
            
        # Basic LightGBM training with early stopping on validation split
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=200,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)] if len(valid_sets) > 1 else []
        )
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been fitted.")
        return self.model.predict(X[self.features])
        
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return self.predict_proba(X) >= threshold
