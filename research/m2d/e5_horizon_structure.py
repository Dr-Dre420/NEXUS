"""
M2D E5 - TEMPORAL HORIZON AND NETWORK STRUCTURE  (research questions 4 and 5)

Q4 hypothesis: propagation information may be predictive at a different lag or
horizon than the frozen 4-week PV window.

Q5 hypothesis: propagation effects may exist only under particular network
structures (liability share, group exposure concentration, peer stress load), and
be averaged away when all borrowers are pooled.

Discipline
  - The frozen PV target is NEVER replaced. Alternative horizons are separate,
    clearly labelled RESEARCH targets built from observed stress only.
  - Association is measured with univariate rank-AUC on TRAIN+VALIDATION rows.
    The test split is never inspected here.
  - The quantity probed is `peer_predicted_stress_mean`, which E2b identified as
    the directional component of propagation, plus the frozen exposure for contrast.
"""
import sys, os, pickle, warnings
sys.path.append(os.path.abspath('.'))
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
warnings.filterwarnings('ignore')

SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909,
         1201, 1302, 1403, 1504, 1605, 1706, 1807, 1908, 2009, 2110]
EXPO = 'borrower_propagation_exposure'
PEER = 'peer_predicted_stress_mean'


def rank_auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(mannwhitneyu(pos, neg, alternative='two-sided').statistic / (len(pos) * len(neg)))


def research_horizon_targets(d):
    """RESEARCH targets over observed stress at several horizons.

    These do NOT replace `propagation_vulnerability`; they exist only to ask
    whether propagation features associate with stress at some other lag.
    Each is built from `current_stress` shifted BACKWARD in time relative to the
    prediction week, i.e. the label is strictly in the future of the features.
    """
    d = d.sort_values(['borrower_id', 'week']).copy()
    g = d.groupby('borrower_id')['current_stress']
    out = {}
    for h in (1, 2, 4, 8, 12):
        # stressed at any point in (t, t+h], and not already stressed at t
        fut = g.transform(lambda s: s.shift(-1).rolling(h, min_periods=1).max())
        out[f'rh_stress_{h}w'] = (~d['current_stress'].astype(bool)) & (fut == 1)
    # persistence: stressed for at least 4 consecutive weeks inside (t, t+12]
    run = g.transform(lambda s: s.shift(-1).rolling(4, min_periods=4).min())
    fut12 = run.rolling(12, min_periods=1).max()
    out['rh_persistent_4of12'] = (~d['current_stress'].astype(bool)) & (fut12 == 1)
    for k, v in out.items():
        d[k] = v.fillna(False).astype(bool)
    return d


def main():
    hor_rows, str_rows = [], []
    for seed in SEEDS:
        d = pickle.load(open(f'research/m2d/cache/world_{seed}.pkl', 'rb'))['df']
        d = research_horizon_targets(d)
        tv = d[d.split.isin(['train', 'val'])]

        # ---------------- Q4: horizon sweep ----------------
        targets = ['propagation_vulnerability'] + [c for c in d.columns if c.startswith('rh_')]
        for t in targets:
            p, n = tv[tv[t]], tv[~tv[t]]
            hor_rows.append(dict(
                seed=seed, target=t, n_pos=int(len(p)), rate=float(tv[t].mean()),
                auc_peer_stress=rank_auc(p[PEER].values, n[PEER].values),
                auc_exposure=rank_auc(p[EXPO].values, n[EXPO].values)))

        # ---------------- Q5: structure strata ----------------
        # Strata are defined from time-t observable structure only.
        grp_expo = tv.groupby(['week', 'group_id'])[EXPO].transform('sum')
        strata = {
            'liab_share_low':   tv['borrower_liability_share'] <= tv['borrower_liability_share'].quantile(0.33),
            'liab_share_high':  tv['borrower_liability_share'] >= tv['borrower_liability_share'].quantile(0.67),
            'group_expo_low':   grp_expo <= grp_expo.quantile(0.33),
            'group_expo_high':  grp_expo >= grp_expo.quantile(0.67),
            'peer_stress_low':  tv[PEER] <= tv[PEER].quantile(0.67),
            'peer_stress_high': tv[PEER] > tv[PEER].quantile(0.67),
        }
        for name, mask in strata.items():
            s = tv[mask]
            p, n = s[s.propagation_vulnerability], s[~s.propagation_vulnerability]
            str_rows.append(dict(
                seed=seed, stratum=name, n_rows=int(len(s)), n_pos=int(len(p)),
                pv_rate=float(s.propagation_vulnerability.mean()) if len(s) else np.nan,
                auc_peer_stress=rank_auc(p[PEER].values, n[PEER].values),
                auc_exposure=rank_auc(p[EXPO].values, n[EXPO].values)))
        print(f'  seed {seed} done', flush=True)

    H = pd.DataFrame(hor_rows); S = pd.DataFrame(str_rows)
    H.to_csv('research/m2d/e5_horizon.csv', index=False)
    S.to_csv('research/m2d/e5_structure.csv', index=False)
    pd.set_option('display.width', 200)

    print('\n=== Q4: DOES PROPAGATION ASSOCIATE WITH STRESS AT A DIFFERENT HORIZON? ===')
    print('    univariate rank-AUC, train+validation only, 20 worlds')
    g = H.groupby('target').agg(
        mean_pos=('n_pos', 'mean'), rate=('rate', 'mean'),
        peer_auc_mean=('auc_peer_stress', 'mean'), peer_auc_sd=('auc_peer_stress', 'std'),
        peer_gt50=('auc_peer_stress', lambda s: int((s > .5).sum())),
        expo_auc_mean=('auc_exposure', 'mean'),
        expo_gt50=('auc_exposure', lambda s: int((s > .5).sum())))
    order = ['propagation_vulnerability', 'rh_stress_1w', 'rh_stress_2w', 'rh_stress_4w',
             'rh_stress_8w', 'rh_stress_12w', 'rh_persistent_4of12']
    print(g.reindex([o for o in order if o in g.index]).round(4).to_string())

    print('\n=== Q5: DOES THE PROPAGATION SIGNAL VARY BY NETWORK STRUCTURE? ===')
    print('    PV target, univariate rank-AUC, train+validation only')
    g2 = S.groupby('stratum').agg(
        rows=('n_rows', 'mean'), pos=('n_pos', 'mean'), pv_rate=('pv_rate', 'mean'),
        peer_auc=('auc_peer_stress', 'mean'), peer_sd=('auc_peer_stress', 'std'),
        peer_gt50=('auc_peer_stress', lambda s: int((s > .5).sum())),
        expo_auc=('auc_exposure', 'mean'),
        expo_gt50=('auc_exposure', lambda s: int((s > .5).sum())))
    print(g2.round(4).to_string())
    print('\n  (peer_gt50 / expo_gt50 = number of the 20 worlds with rank-AUC above chance)')


if __name__ == '__main__':
    main()
