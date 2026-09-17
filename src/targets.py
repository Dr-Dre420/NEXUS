import pandas as pd
import yaml
from pathlib import Path

def load_config() -> dict:
    config_path = Path("configs/targets.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def compute_stress_state(history_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Computes the ground-truth observed stress state at time t.
    """
    stress_cfg = config['stress_definitions']['current_stress']
    
    # Sort to ensure rolling windows work correctly
    df = history_df.sort_values(['borrower_id', 'week']).copy()
    
    # Condition 1: Delinquency
    df['is_delinquent'] = df['days_past_due'] >= stress_cfg['delinquency_days_threshold']
    
    # Condition 2: Sustained cash flow strain
    # Calculate net cash flow proxy (buffer change + income - expenses - loan_payment)
    # For simplicity in M1, we'll just check if cash buffer == 0 for sustained weeks
    df['zero_buffer'] = df['cash_buffer'] <= 0.1
    df['sustained_strain'] = df.groupby('borrower_id')['zero_buffer'].transform(
        lambda x: x.rolling(stress_cfg['sustained_cash_flow_strain_weeks']).sum() == stress_cfg['sustained_cash_flow_strain_weeks']
    )
    
    # Operational Stress definition
    df['is_stressed'] = df['is_delinquent'] | df['sustained_strain']
    
    return df[['week', 'borrower_id', 'is_stressed', 'days_past_due', 'cash_buffer']]

def construct_targets(history_df: pd.DataFrame, lineage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs Current Stress, Next-Period Stress, and Propagation Vulnerability targets.
    """
    config = load_config()
    prediction_horizon = config['temporal']['prediction_horizon_weeks']
    meaningful_share = config['propagation']['meaningful_contribution_share']
    
    # 1. Current Stress
    stress_df = compute_stress_state(history_df, config)
    
    # Create the base targets dataframe
    targets = stress_df[['week', 'borrower_id', 'is_stressed']].copy()
    targets.rename(columns={'is_stressed': 'current_stress'}, inplace=True)
    
    # 2. Next-Period Stress
    # We want to know if they WILL be stressed within the horizon.
    # For simplicity, we check if they are stressed at EXACTLY t + horizon, 
    # or ANY time in (t, t + horizon]. Let's do ANY time in (t, t + horizon].
    stress_df['future_stress'] = stress_df.groupby('borrower_id')['is_stressed'].transform(
        lambda x: x.shift(-prediction_horizon).rolling(prediction_horizon, min_periods=1).max()
    )
    # Next-Period Stress is defined for currently non-stressed borrowers who become stressed
    targets['next_period_stress'] = (targets['current_stress'] == False) & (stress_df['future_stress'] == 1.0)
    
    # 3. Propagation Vulnerability
    # Currently non-stressed at t, later becomes stressed (within horizon), AND episode lineage shows >= 0.30 share.
    targets['propagation_vulnerability'] = False
    
    if not lineage_df.empty:
        # We expect lineage_df to be ep_lineage_df
        col_name = 'episode_network_contribution' if 'episode_network_contribution' in lineage_df.columns else 'contribution_share'
        meaningful_lineage = lineage_df[lineage_df[col_name] >= meaningful_share].copy()
        
        meaningful_events = meaningful_lineage.groupby('destination_borrower')['week'].apply(list).to_dict()
        
        def check_propagation(row):
            if row['current_stress']:
                return False
            if not row['next_period_stress']:
                return False
                
            b_id = row['borrower_id']
            t = row['week']
            
            if b_id in meaningful_events:
                events = meaningful_events[b_id]
                # If using episode_lineage, the event is recorded exactly at prediction week `t`
                if col_name == 'episode_network_contribution':
                    if t in events:
                        return True
                else:
                    # Fallback for old marginal lineage
                    if any(t < ev <= t + prediction_horizon for ev in events):
                        return True
            return False
            
        targets['propagation_vulnerability'] = targets.apply(check_propagation, axis=1)
        
    return targets

def assign_temporal_splits(targets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns train, validation, and test splits ensuring purge gaps.
    """
    config = load_config()
    purge_gap = config['temporal']['purge_gap_weeks']
    
    max_week = targets_df['week'].max()
    
    # Simple split: 50% Train, 20% Val, 30% Test, but with purge gaps
    # Let's say max_week = 156
    train_end = int(max_week * 0.5)
    val_start = train_end + purge_gap + 1
    val_end = val_start + int(max_week * 0.2)
    test_start = val_end + purge_gap + 1
    
    def get_split(week):
        if week <= train_end:
            return 'train'
        elif val_start <= week <= val_end:
            return 'val'
        elif test_start <= week:
            return 'test'
        else:
            return 'purge'
            
    targets_df['split'] = targets_df['week'].apply(get_split)
    return targets_df
