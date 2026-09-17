import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from typing import Dict, Any, List

class ModelC:
    """
    Model C: Individual + Network + Propagation-Aware Exposure.
    Uses LightGBM targeting Propagation Vulnerability.
    Output is calibrated using Platt scaling on the validation set.
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
        self.calibrator = None
        self.features = None
        
    def _filter_features(self, X: pd.DataFrame) -> List[str]:
        exclude_cols = [
            'borrower_id', 'group_id', 'week', 'split', 
            'current_stress', 'next_period_stress', 'propagation_vulnerability',
            'scenario_family', 'group_covered_amount'
        ]
        return [c for c in X.columns if c not in exclude_cols]
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: pd.DataFrame, y_val: pd.Series):
            
        self.features = self._filter_features(X_train)
        
        train_data = lgb.Dataset(X_train[self.features], label=y_train)
        val_data = lgb.Dataset(X_val[self.features], label=y_val, reference=train_data)
        
        # 1. Train LightGBM with early stopping on validation
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
        )
        
        # 2. Platt Scaling (Logistic Calibration) on validation set
        val_preds_raw = self.model.predict(X_val[self.features])
        
        if len(np.unique(y_val)) > 1:
            self.calibrator = LogisticRegression()
            # Reshape for sklearn
            self.calibrator.fit(val_preds_raw.reshape(-1, 1), y_val)
        else:
            self.calibrator = None  # Skip calibration if only one class
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been fitted.")
            
        raw_preds = self.model.predict(X[self.features])
        
        if self.calibrator is not None:
            # Return probability of class 1
            calibrated_preds = self.calibrator.predict_proba(raw_preds.reshape(-1, 1))[:, 1]
            return calibrated_preds
        else:
            return raw_preds
        
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return self.predict_proba(X) >= threshold
