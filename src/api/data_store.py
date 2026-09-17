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
            
        # Reconstruct the exact WorldState at as_of_week using the generator
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        from src.data import SyntheticWorldGenerator
        from src.mechanics import step_forward
        
        gen = SyntheticWorldGenerator(n_borrowers=400, seed=self.world_state['seed'], slack_regime='conservative')
        as_of = self.get_as_of_week()
        gen.generate_world()
        state = gen.world_state
        
        for w in range(as_of):
            state, _, _ = step_forward(state)
            
        self.baseline_state = state
            
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
        # We can extract the OOF predictions or standard metrics from the models if available,
        # or compute basic metrics on the test set if needed, but since M2C frozen state
        # is here, we just provide the basic info requested.
        return {
            "model_c_features": self.model_c.features if hasattr(self.model_c, 'features') else [],
            "model_b_features": self.model_b.features if hasattr(self.model_b, 'features') else [],
            "note": "Model C did not demonstrate measurable incremental predictive value over Model B in the evaluated synthetic dataset."
        }

store = DataStore()
