"""
M2D E6 - IS THE Q5 STRUCTURE FINDING CIRCULAR?

E5 reported that inside the high-group-exposure stratum, peer_predicted_stress_mean
separates PV positives from negatives with rank-AUC 0.89 in 20/20 worlds - the most
stable association found anywhere in M2D.

Before reporting that as a finding I have to rule out two artefacts:

  A1 SELF-CONDITIONING. `group_exposure` and `peer_stress_high` are both functions of
     peer_predicted_stress_mean, so measuring that same variable's AUC inside a
     stratum defined by it can be partly mechanical.
  A2 BASE-RATE ARTEFACT. PV is rarer in the low strata, and rank-AUC is unstable at
     tiny positive counts (E1b), so a "weak" low stratum may just be underpowered.

Test: repeat the stratification using an INDEPENDENT conditioning variable that is
not built from peer stress - group size and the group's aggregate DPD - and compare.
If the association survives conditioning on a variable unrelated to the measured one,
it is not purely self-conditioning.

Train+validation rows only.
"""
import sys, os, pickle, warnings
sys.path.append(os.path.abspath('.'))
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
warnings.filterwarnings('ignore')

SEEDS = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909,
         1201, 1302, 1403, 1504, 1605, 1706, 1807, 1908, 2009, 2110]
PEER = 'peer_predicted_stress_mean'
EXPO = 'borrower_propagation_exposure'


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(mannwhitneyu(pos, neg, alternative='two-sided').statistic / (len(pos) * len(neg)))


def main():
    rows = []
    for seed in SEEDS:
        d = pickle.load(open(f'research/m2d/cache/world_{seed}.pkl', 'rb'))['df']
        tv = d[d.split.isin(['train', 'val'])].copy()

        # conditioning variables that do NOT use peer predicted stress
        tv['_grp_dpd'] = tv.groupby(['week', 'group_id'])['peer_dpd_mean_4w'].transform('sum')
        tv['_grp_shortfall'] = tv.groupby(['week', 'group_id'])['peer_shortfall_mean_4w'].transform('sum')

        # Most groups carry zero DPD/shortfall, so a quantile cut degenerates.
        # Split on strictly-positive vs zero instead, which actually partitions.
        dpd_hi = tv['_grp_dpd'] > 0
        sf_hi = tv['_grp_shortfall'] > 0
        strata = {
            # self-conditioned (the E5 stratum, repeated for comparison)
            'SELF_peer_stress_high': tv[PEER] > tv[PEER].quantile(0.67),
            # independent conditioning: built from realised delinquency/shortfall,
            # never from peer predicted stress
            'IND_grp_dpd_pos': dpd_hi,
            'IND_grp_dpd_zero': ~dpd_hi,
            'IND_grp_shortfall_pos': sf_hi,
            'IND_grp_shortfall_zero': ~sf_hi,
            'ALL_rows': pd.Series(True, index=tv.index),
        }
        for name, mask in strata.items():
            s = tv[mask]
            p, n = s[s.propagation_vulnerability], s[~s.propagation_vulnerability]
            rows.append(dict(seed=seed, stratum=name, n_rows=len(s), n_pos=len(p),
                             auc_peer=auc(p[PEER].values, n[PEER].values),
                             auc_expo=auc(p[EXPO].values, n[EXPO].values)))
        print(f'  seed {seed} done', flush=True)

    D = pd.DataFrame(rows)
    D.to_csv('research/m2d/e6_circularity.csv', index=False)
    pd.set_option('display.width', 200)

    g = D.groupby('stratum').agg(
        rows=('n_rows', 'mean'), pos=('n_pos', 'mean'),
        peer_auc=('auc_peer', 'mean'), peer_sd=('auc_peer', 'std'),
        peer_gt50=('auc_peer', lambda s: int((s > .5).sum())),
        expo_auc=('auc_expo', 'mean'),
        expo_gt50=('auc_expo', lambda s: int((s > .5).sum())))
    print('\n=== E6: SELF-CONDITIONED vs INDEPENDENTLY CONDITIONED STRATA ===')
    print(g.round(4).to_string())

    print('\n=== READING ===')
    a = g.loc['ALL_rows', 'peer_auc']
    self_s = g.loc['SELF_peer_stress_high', 'peer_auc']
    ind_hi = g.loc['IND_grp_dpd_pos', 'peer_auc']
    ind_lo = g.loc['IND_grp_dpd_zero', 'peer_auc']
    print(f'  unconditioned                       : {a:.4f}')
    print(f'  self-conditioned (peer stress high) : {self_s:.4f}  <- inflated by construction')
    print(f'  independent  (group DPD > 0)         : {ind_hi:.4f}')
    print(f'  independent  (group DPD == 0)        : {ind_lo:.4f}')
    if ind_hi > a + 0.02:
        print('  The association SURVIVES independent conditioning, so it is not purely')
        print('  self-conditioning, though the self-conditioned number overstates it.')
    else:
        print('  The association does NOT strengthen under independent conditioning, so the')
        print('  E5 stratum result is substantially an artefact of self-conditioning.')


if __name__ == '__main__':
    main()
