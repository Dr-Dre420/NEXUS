import os
import pickle
import pandas as pd
from typing import Dict, Any

class DataStore:
    def __init__(self):
        self.world_state = None
        self.model_b = None
        self.model_c = None
        self._loaded = False
        
    def load(self, data_dir: str = 'data/frozen_m2c'):
        if self._loaded:
            return
            
        world_path = os.path.join(data_dir, 'world_state.pkl')
        mb_path = os.path.join(data_dir, 'model_b.pkl')
        mc_path = os.path.join(data_dir, 'model_c.pkl')
        
        if not os.path.exists(world_path):
            raise FileNotFoundError(f"Frozen world state not found at {world_path}. Run scripts/freeze_world.py first.")
            
        with open(world_path, 'rb') as f:
            self.world_state = pickle.load(f)
            
        with open(mb_path, 'rb') as f:
            self.model_b = pickle.load(f)
            
        with open(mc_path, 'rb') as f:
            self.model_c = pickle.load(f)
            
        baseline_path = os.path.join(data_dir, 'baseline_state.pkl')
        with open(baseline_path, 'rb') as f:
            self.baseline_state = pickle.load(f)
            
        evaluation_path = os.path.join(data_dir, 'evaluation.json')
        import json
        with open(evaluation_path, 'r') as f:
            self.evaluation_metrics = json.load(f)
            
        self._loaded = True
        
    def get_as_of_week(self) -> int:
        return self.world_state['as_of_week']
        
    def get_borrower_state(self, borrower_id: int, week: int) -> pd.Series:
        df = self.world_state['next_df_prop']
        matches = df[(df['borrower_id'] == borrower_id) & (df['week'] == week)]
        if len(matches) == 0:
            return None
        return matches.iloc[0]

    def get_baseline_state_copy(self) -> Dict[str, Any]:
        import copy
        return copy.deepcopy(self.baseline_state)
        
    def get_evaluation_metrics(self) -> Dict[str, Any]:
        return self.evaluation_metrics

store = DataStore()
