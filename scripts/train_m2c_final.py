import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import SyntheticWorldGenerator
from src.targets import construct_targets, assign_temporal_splits
from src.features import generate_features, build_feature_matrices
from src.features_propagation import generate_propagation_features
from src.models.m0_baseline import BaselineM0
from src.models.model_a import ModelA
from src.models.model_a_current import ModelACurrent
from src.models.model_b import ModelB
from src.models.model_c import ModelC

def eval_metrics(model, X, y, threshold=0.5):
    if len(y) == 0:
        return {'pr_auc': np.nan, 'roc_auc': np.nan, 'prec': np.nan, 'rec': np.nan}
    if not hasattr(model, 'predict_proba'):
        preds_proba = model.predict(X).astype(float)
        preds_bin = model.predict(X, threshold=threshold).astype(int)
    else:
        preds_proba = model.predict_proba(X)
        preds_bin = (preds_proba >= threshold).astype(int)
        
    y_int = y.astype(int)
    if len(y_int.unique()) > 1:
        roc = roc_auc_score(y_int, preds_proba)
        pr = average_precision_score(y_int, preds_proba)
        prec = precision_score(y_int, preds_bin, zero_division=0)
        rec = recall_score(y_int, preds_bin, zero_division=0)
    else:
        roc, pr, prec, rec = np.nan, np.nan, np.nan, np.nan
    return {'pr_auc': pr, 'roc_auc': roc, 'prec': prec, 'rec': rec, 'pos': y_int.sum()}

def main():
    seeds = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909]
    n_weeks = 156
    
    # Aggregation stores
    all_ep_lineage = []
    all_marg_lineage = []
    reconciliation = []
    
    results = {
        'np_stress': {'m0': [], 'a': [], 'b': [], 'a_no_sf': []},
        'pv': {'b': [], 'c': []}
    }
    
    feature_importances = {'b': [], 'c': []}
    healthy_audit_cases = []
    
    for seed in seeds:
        print(f"==================================================")
        print(f"PROCESSING WORLD SEED: {seed}")
        print(f"==================================================")
        
        generator = SyntheticWorldGenerator(n_borrowers=400, seed=seed, slack_regime='conservative')
        history_df, lineage_df, ep_lineage_df = generator.simulate(n_weeks=n_weeks)
        
        # Add provenance
        history_df['world_seed'] = seed
        if not lineage_df.empty: lineage_df['world_seed'] = seed
        if not ep_lineage_df.empty: ep_lineage_df['world_seed'] = seed
        
        all_marg_lineage.append(lineage_df)
        all_ep_lineage.append(ep_lineage_df)
        
        targets_df = construct_targets(history_df, ep_lineage_df)
        targets_df = assign_temporal_splits(targets_df)
        
        features_df = generate_features(history_df)
        curr_df, next_df = build_feature_matrices(targets_df, features_df)
        
        # Neighbor estimates (historical OOF)
        model_a_current = ModelACurrent()
        oof_estimates = model_a_current.generate_historical_predictions(curr_df)
        
        next_df_prop = generate_propagation_features(next_df, oof_estimates)
        
        # Add seed to feature matrices for provenance tracking if desired, but we split independently
        train_df = next_df_prop[next_df_prop['split'] == 'train']
        val_df = next_df_prop[next_df_prop['split'] == 'val']
        test_df = next_df_prop[next_df_prop['split'] == 'test']
        
        # Reconciliation stats
        recon = {
            'seed': seed,
            'marg_events': len(lineage_df),
            'ep_events': len(ep_lineage_df),
            'eligible': len(targets_df[targets_df['current_stress'] == False]),
            'pv_pos': targets_df['propagation_vulnerability'].sum(),
            'unique_b': targets_df[targets_df['propagation_vulnerability']]['borrower_id'].nunique()
        }
        reconciliation.append(recon)
        
        # --- TASK A: NEXT-PERIOD STRESS ---
        y_tr_np = train_df['next_period_stress']
        y_val_np = val_df['next_period_stress']
        y_te_np = test_df['next_period_stress']
        
        m0 = BaselineM0(dpd_threshold=7.0)
        ma = ModelA(drop_shortfall=False)
        ma.fit(train_df, y_tr_np, val_df, y_val_np)
        
        mb = ModelB()
        mb.fit(train_df, y_tr_np, val_df, y_val_np)
        
        ma_no_sf = ModelA(drop_shortfall=True)
        ma_no_sf.fit(train_df, y_tr_np, val_df, y_val_np)
        
        results['np_stress']['m0'].append(eval_metrics(m0, test_df, y_te_np))
        results['np_stress']['a'].append(eval_metrics(ma, test_df, y_te_np))
        results['np_stress']['b'].append(eval_metrics(mb, test_df, y_te_np))
        results['np_stress']['a_no_sf'].append(eval_metrics(ma_no_sf, test_df, y_te_np))
        
        # --- TASK B: PROPAGATION VULNERABILITY ---
        y_tr_pv = train_df['propagation_vulnerability']
        y_val_pv = val_df['propagation_vulnerability']
        y_te_pv = test_df['propagation_vulnerability']
        
        mb_pv = ModelB()
        mb_pv.fit(train_df, y_tr_pv, val_df, y_val_pv)
        
        mc = ModelC()
        mc.fit(train_df, y_tr_pv, val_df, y_val_pv)
        
        results['pv']['b'].append(eval_metrics(mb_pv, test_df, y_te_pv, threshold=0.05))
        results['pv']['c'].append(eval_metrics(mc, test_df, y_te_pv, threshold=0.05))
        
        # Feature Importances for C
        if hasattr(mc.model, 'feature_importance'):
            imp_c = dict(zip(mc.model.feature_name(), mc.model.feature_importance(importance_type='gain')))
            feature_importances['c'].append(imp_c)
        if hasattr(mb_pv.model, 'feature_importance'):
            imp_b = dict(zip(mb_pv.model.feature_name(), mb_pv.model.feature_importance(importance_type='gain')))
            feature_importances['b'].append(imp_b)
            
        # --- HEALTHY BORROWER AUDIT (Collect on last seed) ---
        if seed == 909:
            # We want to find examples of:
            # healthy + weak exposure, healthy + strong exposure, healthy + highly stressed peer
            h_df = test_df[test_df['current_stress'] == False].copy()
            h_df['mc_score'] = mc.predict_proba(h_df)
            h_df['network_exposure'] = h_df['borrower_propagation_exposure']
            
            # 1. Healthy + weak exposure
            weak = h_df[h_df['network_exposure'] < 0.1].sort_values('mc_score').head(1)
            # 2. Healthy + strong exposure (but not necessarily PV)
            strong = h_df[h_df['network_exposure'] >= 0.5].sort_values('mc_score', ascending=False).head(1)
            # 3. Genuine PV positive
            gen_pv = h_df[h_df['propagation_vulnerability'] == True].sort_values('mc_score', ascending=False).head(1)
            
            healthy_audit_cases.extend([
                ("Healthy + Weak Exposure", weak),
                ("Healthy + Strong Exposure", strong),
                ("Genuine PV", gen_pv)
            ])
            
    print("\n\n==================================================")
    print("FINAL REPORT GENERATION")
    print("==================================================")
    
    # 1. TARGET RECONCILIATION
    print("\nB. TARGET RECONCILIATION")
    df_recon = pd.DataFrame(reconciliation)
    print(df_recon.to_string(index=False))
    print("\nAggregate:")
    print(df_recon.sum().to_string())
    
    # 2. NEXT PERIOD STRESS
    print("\nC. NEXT-PERIOD STRESS")
    for mod_name in ['m0', 'a', 'b']:
        df_res = pd.DataFrame(results['np_stress'][mod_name])
        print(f"\nModel: {mod_name.upper()}")
        print(f"Mean PR-AUC:  {df_res['pr_auc'].mean():.4f} +/- {df_res['pr_auc'].std():.4f}")
        print(f"Mean ROC-AUC: {df_res['roc_auc'].mean():.4f} +/- {df_res['roc_auc'].std():.4f}")
        print(f"Mean Prec:    {df_res['prec'].mean():.4f}")
        print(f"Mean Recall:  {df_res['rec'].mean():.4f}")
        
    # 3. PROPAGATION VULNERABILITY
    print("\nD. PROPAGATION VULNERABILITY")
    for mod_name in ['b', 'c']:
        df_res = pd.DataFrame(results['pv'][mod_name])
        print(f"\nModel: {mod_name.upper()}")
        print(f"Mean PR-AUC:  {df_res['pr_auc'].mean():.4f} +/- {df_res['pr_auc'].std():.4f}")
        print(f"Mean ROC-AUC: {df_res['roc_auc'].mean():.4f} +/- {df_res['roc_auc'].std():.4f}")
        print(f"Mean Prec:    {df_res['prec'].mean():.4f}")
        print(f"Mean Recall:  {df_res['rec'].mean():.4f}")
        
    # 4. MODEL A-NO-SHORTFALL
    print("\nE. MODEL A-NO-SHORTFALL")
    df_res_sf = pd.DataFrame(results['np_stress']['a_no_sf'])
    df_res_a = pd.DataFrame(results['np_stress']['a'])
    print(f"A PR-AUC:       {df_res_a['pr_auc'].mean():.4f}")
    print(f"A-no-sf PR-AUC: {df_res_sf['pr_auc'].mean():.4f}")
    
    # 5. MODEL C FEATURE AUDIT
    print("\nF. MODEL B/C FEATURE AUDIT (Averaged across 10 seeds)")
    agg_imp_c = pd.DataFrame(feature_importances['c']).mean().sort_values(ascending=False)
    print("\nTop 10 Model C Features (Gain):")
    print(agg_imp_c.head(10).to_string())
    
    # 6. CONTRIBUTION-SHARE AUDIT
    print("\nG. CONTRIBUTION-SHARE AUDIT")
    global_ep = pd.concat(all_ep_lineage, ignore_index=True) if all_ep_lineage else pd.DataFrame()
    global_marg = pd.concat(all_marg_lineage, ignore_index=True) if all_marg_lineage else pd.DataFrame()
    
    if not global_ep.empty:
        ep = global_ep['episode_network_contribution']
        print(f"Count: {len(ep)}")
        print(f"Min: {ep.min():.4f}")
        print(f"p10: {ep.quantile(0.1):.4f}")
        print(f"p25: {ep.quantile(0.25):.4f}")
        print(f"Median: {ep.median():.4f}")
        print(f"Mean: {ep.mean():.4f}")
        print(f"p75: {ep.quantile(0.75):.4f}")
        print(f"p90: {ep.quantile(0.9):.4f}")
        print(f"Max: {ep.max():.4f}")
        print(f"Fraction exactly 0: {(ep == 0).mean():.4f}")
        print(f"Fraction 0 < x < 0.30: {((ep > 0) & (ep < 0.3)).mean():.4f}")
        print(f"Fraction 0.30 <= x < 0.50: {((ep >= 0.3) & (ep < 0.5)).mean():.4f}")
        print(f"Fraction 0.50 <= x < 0.80: {((ep >= 0.5) & (ep < 0.8)).mean():.4f}")
        print(f"Fraction >= 0.80: {(ep >= 0.8).mean():.4f}")
        
    print("\nI. HEALTHY-BORROWER AUDIT")
    for label, case_df in healthy_audit_cases:
        print(f"\n--- {label} ---")
        if case_df.empty:
            print("No case found.")
        else:
            row = case_df.iloc[0]
            print(f"Borrower: {row['borrower_id']}")
            print(f"Network Exposure: {row['network_exposure']:.4f}")
            print(f"Calibrated Operational Score: {row['mc_score']:.4f}")
            print(f"Is Stressed? {row['current_stress']}")
            print(f"Recommendation: {'Intervene' if row['mc_score'] >= 0.05 else 'Monitor/Ignore'}")

if __name__ == '__main__':
    main()
