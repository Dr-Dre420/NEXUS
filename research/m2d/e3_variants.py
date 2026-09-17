"""
M2D E3 - VARIANTS x MODEL CLASS x SEEDS   (research questions 3, 6, 7, 8)

Pre-registered in e3_preregistration.md BEFORE running. Prediction: H3-null.

Frozen M2C is the reference. Target, splits, purge gap, horizon and leakage
protections are untouched. Calibration is fitted on validation only; test labels
are never used for any modelling or selection choice.
"""
import sys, os, json, pickle, warnings
sys.path.append(os.path.abspath('.'))
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
warnings.filterwarnings('ignore')

SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909,
         1201, 1302, 1403, 1504, 1605, 1706, 1807, 1908, 2009, 2110]
TARGET = 'propagation_vulnerability'
EXPO = 'borrower_propagation_exposure'

META = ['borrower_id', 'group_id', 'week', 'split', 'current_stress', 'next_period_stress',
        TARGET, 'scenario_family', 'group_covered_amount', 'rt_pv_window',
        'rt_net_stress_any', 'rt_new_stress', 'model_b_score', 'model_c_score',
        'model_c_score_raw']

NETWORK = ['group_size', 'peer_buffer_mean_t', 'borrower_liability_share',
           'peer_shortfall_mean_4w', 'peer_dpd_mean_4w', 'peer_debt_burden',
           'peer_predicted_stress_mean', 'expected_peer_debt_burden', EXPO]

LGB_P = dict(objective='binary', metric='auc', learning_rate=0.05, num_leaves=31,
             max_depth=5, verbose=-1, random_state=42)


def add_variant_features(df):
    """Every feature is a function of information available at or before week t."""
    d = df.sort_values(['borrower_id', 'week']).copy()
    g = d.groupby('borrower_id')[EXPO]

    # C1 - persistence / trend / volatility / elevated duration (backward windows only)
    d['v_expo_roll4'] = g.transform(lambda s: s.rolling(4, min_periods=1).mean())
    d['v_expo_trend4'] = d[EXPO] - g.shift(4).fillna(0.0)
    d['v_expo_vol4'] = g.transform(lambda s: s.rolling(4, min_periods=1).std()).fillna(0.0)
    hi = d.groupby('week')[EXPO].transform(lambda s: s.quantile(0.90))
    d['v_expo_elevated'] = (d[EXPO] >= hi).astype(float)
    d['v_expo_elev_dur'] = d.groupby('borrower_id')['v_expo_elevated'].transform(
        lambda s: s.rolling(8, min_periods=1).sum())

    # C2 - monotone transforms (no-ops for an axis-aligned tree, meaningful for a linear model)
    d['v_expo_log1p'] = np.log1p(d[EXPO])
    cap = d.groupby('week')[EXPO].transform(lambda s: s.quantile(0.99))
    d['v_expo_winsor'] = np.minimum(d[EXPO], cap)
    d['v_expo_pctrank'] = d.groupby('week')[EXPO].rank(pct=True)

    # C3 - corrected propagation term: source stress x ABSOLUTE destination fragility.
    # Replaces the relative-wealth multiplier that E2b showed is anti-predictive.
    oblig = d['amount_due_mean_4w'] + d['weekly_expenses_mean_4w']
    d['v_dest_fragility'] = oblig / (d['cash_buffer_mean_4w'] + oblig + 1e-9)
    d['v_expo_corrected'] = d['peer_predicted_stress_mean'] * d['v_dest_fragility']
    return d


def feature_sets(d):
    base = [c for c in d.columns if c not in META and not c.startswith('v_')]
    A = [c for c in base if c not in NETWORK]
    B = [c for c in base if c != EXPO]
    C = base
    return {
        'A': A,
        'B': B,
        'C': C,
        'C1': C + ['v_expo_roll4', 'v_expo_trend4', 'v_expo_vol4', 'v_expo_elev_dur'],
        'C2': C + ['v_expo_log1p', 'v_expo_winsor', 'v_expo_pctrank'],
        'C3': B + ['v_expo_corrected', 'v_dest_fragility'],
    }


def m0_scores(d):
    """Deterministic recent-delinquency baseline. No learning, no fitting."""
    return (d['days_past_due_max_4w'].values.astype(float)
            + 1e-6 * d['shortfall_mean_4w'].values.astype(float))


def fit_predict(kind, tr, va, te, feats):
    ytr = tr[TARGET].astype(int)
    yva = va[TARGET].astype(int)
    if ytr.sum() == 0:
        return None, None
    if kind == 'lgbm':
        dtr = lgb.Dataset(tr[feats], label=ytr)
        vs, vn, cbs = [dtr], ['train'], []
        if yva.sum() > 0:
            vs.append(lgb.Dataset(va[feats], label=yva, reference=dtr))
            vn.append('val')
            cbs = [lgb.early_stopping(20, verbose=False)]
        m = lgb.train(LGB_P, dtr, num_boost_round=200, valid_sets=vs,
                      valid_names=vn, callbacks=cbs)
        return m.predict(te[feats]), m.predict(va[feats])

    def clean(x):
        return x[feats].replace([np.inf, -np.inf], 0).fillna(0)
    sc = StandardScaler().fit(clean(tr))
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(clean(tr)), ytr)
    return (lr.predict_proba(sc.transform(clean(te)))[:, 1],
            lr.predict_proba(sc.transform(clean(va)))[:, 1])


def main():
    rows = []
    for seed in SEEDS:
        world = pickle.load(open(f'research/m2d/cache/world_{seed}.pkl', 'rb'))['df']
        d = add_variant_features(world)
        sets = feature_sets(d)
        tr, va, te = d[d.split == 'train'], d[d.split == 'val'], d[d.split == 'test']
        yte = te[TARGET].astype(int)
        scoreable = bool(yte.sum() > 0 and yte.nunique() > 1)

        def log(spec, kind, raw, val_raw):
            r = dict(experiment_id=f'E3|{TARGET}|{spec}|{kind}|seed{seed}', seed=seed,
                     spec=spec, model=kind, target=TARGET,
                     train_weeks=f'{tr.week.min()}-{tr.week.max()}',
                     val_weeks=f'{va.week.min()}-{va.week.max()}',
                     test_weeks=f'{te.week.min()}-{te.week.max()}',
                     n_features=len(sets[spec]) if spec in sets else 0,
                     pos_train=int(tr[TARGET].sum()), pos_val=int(va[TARGET].sum()),
                     pos_test=int(yte.sum()), scoreable=scoreable)
            if raw is None:
                r['status'] = 'no_train_positives'
            elif not scoreable:
                r['status'] = 'unscoreable_no_test_positives'
            else:
                r['pr_auc_raw'] = float(average_precision_score(yte, raw))
                r['roc_auc_raw'] = float(roc_auc_score(yte, raw))
                r['n_unique_raw'] = int(len(np.unique(raw)))
                if val_raw is not None and va[TARGET].astype(int).nunique() > 1:
                    cal = LogisticRegression().fit(
                        np.asarray(val_raw).reshape(-1, 1), va[TARGET].astype(int))
                    c = cal.predict_proba(np.asarray(raw).reshape(-1, 1))[:, 1]
                    r['pr_auc_cal'] = float(average_precision_score(yte, c))
                    r['roc_auc_cal'] = float(roc_auc_score(yte, c))
                    r['n_unique_cal'] = int(len(np.unique(c)))
                r['status'] = 'ok'
            rows.append(r)

        log('M0', 'deterministic', m0_scores(te), m0_scores(va))
        for spec, feats in sets.items():
            for kind in ('lgbm', 'logreg'):
                raw, vr = fit_predict(kind, tr, va, te, feats)
                log(spec, kind, raw, vr)
        print(f'  seed {seed} done (test positives {int(yte.sum())})', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv('research/m2d/e3_results.csv', index=False)
    df.to_json('research/m2d/e3_results.json', orient='records', indent=2)
    print(f'\nwrote {len(df)} experiment rows to research/m2d/e3_results.csv')


if __name__ == '__main__':
    main()
