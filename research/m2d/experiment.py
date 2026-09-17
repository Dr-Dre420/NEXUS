"""
M2D RESEARCH — incrementality experiment harness.

Compares A / B / B+candidate on IDENTICAL seeds, rows, targets, temporal splits,
purge gap and calibration procedure. Worlds are never pooled for splitting.
"""
import sys, os, pickle, json, itertools
sys.path.append(os.path.abspath('.')); sys.path.append(os.path.abspath('research/m2d'))
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_score, recall_score, brier_score_loss)
from features_research import RESEARCH_FEATURES, CANDIDATE_SETS

CACHE = 'research/m2d/cache'

META = ['borrower_id','group_id','week','split','current_stress','next_period_stress',
        'propagation_vulnerability','scenario_family','group_covered_amount',
        'rt_pv_window','rt_net_stress_any','rt_new_stress',
        'model_b_score','model_c_score','model_c_score_raw']

# Model A = individual financial only (no peer/network context at all)
NETWORK_COLS = ['group_size','peer_buffer_mean_t','borrower_liability_share',
                'peer_shortfall_mean_4w','peer_dpd_mean_4w','peer_debt_burden',
                'peer_predicted_stress_mean','expected_peer_debt_burden',
                'borrower_propagation_exposure']

LGB_PARAMS = dict(objective='binary', metric='auc', learning_rate=0.05,
                  num_leaves=31, max_depth=5, verbose=-1, random_state=42)


def load(seed):
    return pickle.load(open(os.path.join(CACHE, f'world_{seed}.pkl'), 'rb'))['df']


def feature_sets(df):
    base = [c for c in df.columns if c not in META and not c.startswith('r_')]
    A = [c for c in base if c not in NETWORK_COLS]
    B = [c for c in base if c != 'borrower_propagation_exposure']
    return A, B


def fit_eval(df, feats, target, params=None, class_weight=None, calibrate=True):
    """Train on train split, calibrate on val split, evaluate on test split."""
    tr, va, te = df[df.split=='train'], df[df.split=='val'], df[df.split=='test']
    ytr, yva, yte = (tr[target].astype(int), va[target].astype(int), te[target].astype(int))
    if ytr.sum() == 0 or yte.sum() == 0 or yte.nunique() < 2:
        return None
    p = dict(LGB_PARAMS); 
    if params: p.update(params)
    if class_weight == 'balanced':
        p['scale_pos_weight'] = float((len(ytr)-ytr.sum())/max(ytr.sum(),1))
    dtr = lgb.Dataset(tr[feats], label=ytr)
    dva = lgb.Dataset(va[feats], label=yva, reference=dtr)
    cbs = [lgb.early_stopping(20, verbose=False)] if yva.sum() > 0 else []
    m = lgb.train(p, dtr, num_boost_round=200,
                  valid_sets=[dtr, dva], valid_names=['train','val'], callbacks=cbs)
    raw_te = m.predict(te[feats])
    # Platt calibration fitted on VALIDATION ONLY; test labels never used.
    cal_te = raw_te
    if calibrate and yva.nunique() > 1:
        cal = LogisticRegression().fit(m.predict(va[feats]).reshape(-1,1), yva)
        cal_te = cal.predict_proba(raw_te.reshape(-1,1))[:,1]
    thr = 0.05
    return dict(
        n_train=len(tr), n_val=len(va), n_test=len(te),
        pos_train=int(ytr.sum()), pos_val=int(yva.sum()), pos_test=int(yte.sum()),
        train_weeks=f"{tr.week.min()}-{tr.week.max()}",
        val_weeks=f"{va.week.min()}-{va.week.max()}",
        test_weeks=f"{te.week.min()}-{te.week.max()}",
        n_features=len(feats), n_trees=m.num_trees(),
        pr_auc=float(average_precision_score(yte, raw_te)),
        roc_auc=float(roc_auc_score(yte, raw_te)),
        pr_auc_cal=float(average_precision_score(yte, cal_te)),
        roc_auc_cal=float(roc_auc_score(yte, cal_te)),
        precision=float(precision_score(yte,(raw_te>=thr).astype(int),zero_division=0)),
        recall=float(recall_score(yte,(raw_te>=thr).astype(int),zero_division=0)),
        brier=float(brier_score_loss(yte, cal_te)),
        n_unique_pred=int(len(np.unique(raw_te))),
        preds=raw_te,
    )
