import pandas as pd
import numpy as np
import lightgbm as lgb
from typing import Dict, Any, List

class ModelB:
    """
    Model B: Individual + Network Context.
    Uses LightGBM trained on individual financial features PLUS aggregated network/peer context.
    Excludes hidden lineage, scenario metadata, and future outcomes.
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
        self.model = None
        self.features = None
        
    def _filter_features(self, X: pd.DataFrame) -> List[str]:
        # Strictly exclude metadata, future outcomes, and hidden graph mechanisms.
        # ALLOWS the computed Model B network features (peer_buffer_mean_t, etc.)
        exclude_cols = [
            'borrower_id', 'group_id', 'week', 'split', 
            'current_stress', 'next_period_stress', 'propagation_vulnerability',
            'scenario_family', 'group_covered_amount'
        ]
        
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
