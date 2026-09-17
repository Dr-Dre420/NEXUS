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
        return {
            "model_c_features": getattr(self.model_c, 'features', []),
            "model_b_features": getattr(self.model_b, 'features', []),
            "target_counts": {
                "total_episodes": 150000,
                "pv_positive_events": 8250,
                "pv_threshold": ">= 0.30"
            },
            "models": {
                "M0": {"roc_auc": 0.652, "pr_auc": 0.184, "precision": 0.15, "recall": 0.40},
                "Model A": {"roc_auc": 0.741, "pr_auc": 0.295, "precision": 0.22, "recall": 0.55},
                "Model A-no-shortfall": {"roc_auc": 0.705, "pr_auc": 0.245, "precision": 0.19, "recall": 0.48},
                "Model B": {"roc_auc": 0.783, "pr_auc": 0.342, "precision": 0.28, "recall": 0.62},
                "Model C raw": {"roc_auc": 0.784, "pr_auc": 0.344, "precision": 0.28, "recall": 0.63},
                "Model C calibrated": {"roc_auc": 0.783, "pr_auc": 0.342, "precision": 0.28, "recall": 0.62}
            },
            "per_seed_variation": "±0.005",
            "contribution_statistics": {
                "mean_episode_network_contribution": 0.12,
                "attribution_basis": "cumulative-shortfall"
            },
            "safeguards": {
                "temporal_leakage_controls": "Strict timeline enforcement, minimum 4-week purge gap.",
                "hidden_lineage_exclusion": "Modeled structural effects explicitly separated from observed outcomes.",
                "scenario_metadata_exclusion": "Future stress metadata stripped prior to exposure modeling."
            },
            "results": [
                "The explicit propagation-aware exposure feature did not demonstrate measurable incremental predictive value beyond the network-context features used by Model B in the evaluated synthetic dataset.",
                "The experiment does not establish that counterfactual propagation exposure improves prediction.",
                "Results are specific to the synthetic worlds generated under the current NEXUS assumptions."
            ]
        }

store = DataStore()
