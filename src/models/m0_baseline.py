import pandas as pd
import numpy as np

class BaselineM0:
    """
    A genuinely simple operational baseline (M0) based on recent repayment/delinquency behavior.
    Uses a deterministic rule: if the maximum days past due in the recent 4-week window 
    meets or exceeds a frozen threshold, predict stress.
    """
    def __init__(self, dpd_threshold: float = 7.0):
        # The threshold is frozen before held-out evaluation (e.g. 7 days past due)
        self.dpd_threshold = dpd_threshold
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        # M0 is deterministic and does not require fitting
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns binary probabilities (0.0 or 1.0) based on the deterministic rule.
        """
        if 'days_past_due_max_4w' not in X.columns:
            raise ValueError("Required feature 'days_past_due_max_4w' missing from input matrix.")
            
        preds = (X['days_past_due_max_4w'] >= self.dpd_threshold).astype(float).values
        return preds
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_proba(X) >= 0.5
