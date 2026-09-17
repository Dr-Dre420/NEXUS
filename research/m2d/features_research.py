"""
M2D RESEARCH propagation representations.

RESEARCH ONLY. Does not modify src/features_propagation.py or any frozen artifact.

Every feature is computable from information available at or before prediction time t:
  - peer stress uses the SAME out-of-fold Model-A current-stress estimates as M2C
  - liability weight is a static contractual parameter fixed at origination (t=0)
  - rolling windows look backwards only
No hidden lineage, no future outcomes, no scenario labels.
"""
import numpy as np
import pandas as pd

DECAY = 0.6          # weekly persistence decay for P3
STRESS_HI = 0.5      # "severely stressed peer" threshold for P2


def _loo(df, col, gcol='group_id', wcol='week'):
    """Leave-one-out peer sum for `col` within (week, group)."""
    tot = df.groupby([wcol, gcol])[col].transform('sum')
    return tot - df[col]


def build_research_propagation(df, oof, liability_weight_map):
    """
    df   : next-period feature matrix (must carry borrower_id, group_id, week, group_size,
           cash_buffer_mean_4w, amount_due_mean_4w, weekly_expenses_mean_4w)
    oof  : OOF current-stress probabilities (borrower_id, week, oof_current_stress_prob)
    liability_weight_map : {borrower_id -> contractual joint-liability weight}
    Returns df + research propagation columns.
    """
    out = df.merge(oof[['borrower_id', 'week', 'oof_current_stress_prob']],
                   on=['borrower_id', 'week'], how='left').copy()
    out['oof_current_stress_prob'] = out['oof_current_stress_prob'].fillna(0.0)
    out = out.sort_values(['borrower_id', 'week']).reset_index(drop=True)

    s = out['oof_current_stress_prob']
    peer_n = (out['group_size'] - 1).clip(lower=1)

    # contractual transmission weight (static, known at origination)
    out['r_liability_weight'] = out['borrower_id'].map(liability_weight_map).astype(float)

    # ---------- P1: liability-weighted peer stress (corrected channel) ----------
    # M2C multiplies peer-mean stress by cash_buffer/group_buffer (a WEALTH share).
    # The generator transmits through normalized contractual liability_weight instead.
    out['_sw'] = s * out['r_liability_weight']
    peer_sw = _loo(out, '_sw')
    peer_w = _loo(out, 'r_liability_weight')
    out['r_liab_weighted_peer_stress'] = peer_sw / (peer_w + 1e-9)

    # ---------- P2: multi-peer concentration (mean cannot represent this) ----------
    # max peer stress: one severe peer != several mild peers
    gmax = out.groupby(['week', 'group_id'])['oof_current_stress_prob'].transform('max')
    is_max = (s >= gmax - 1e-12)
    gmax2 = out.groupby(['week', 'group_id'])['oof_current_stress_prob'].transform(
        lambda x: x.nlargest(2).min() if len(x) > 1 else x.max())
    out['r_peer_stress_max'] = np.where(is_max, gmax2, gmax)
    # count of simultaneously severely-stressed peers
    out['_hi'] = (s > STRESS_HI).astype(float)
    out['r_peer_stress_count_hi'] = _loo(out, '_hi')
    # concentration: how unevenly peer stress is distributed
    out['_s2'] = s ** 2
    peer_s = _loo(out, 'oof_current_stress_prob')
    peer_s2 = _loo(out, '_s2')
    out['r_peer_stress_hhi'] = peer_s2 / (peer_s ** 2 + 1e-9)

    # ---------- P3: time-decayed peer-stress persistence ----------
    out['_peer_mean'] = peer_s / peer_n
    g = out.groupby('borrower_id')['_peer_mean']
    out['r_peer_stress_decayed'] = sum(
        (DECAY ** k) * g.shift(k).fillna(0.0) for k in range(4))
    # velocity: is peer stress rising or falling?
    out['r_peer_stress_velocity'] = out['_peer_mean'] - g.shift(4).fillna(0.0)

    # ---------- P4: destination fragility (absolute, not relative) ----------
    oblig = out['amount_due_mean_4w'] + out['weekly_expenses_mean_4w']
    out['r_buffer_adequacy'] = out['cash_buffer_mean_4w'] / (oblig + 1e-6)
    out['r_dest_vulnerability'] = 1.0 / (1.0 + out['r_buffer_adequacy'])

    # ---------- P5: composite process feature ----------
    # source stress x transmission channel x persistence x destination vulnerability
    out['r_propagation_process'] = (
        out['r_liab_weighted_peer_stress']
        * out['r_peer_stress_decayed']
        * out['r_dest_vulnerability']
    )

    out = out.drop(columns=['_sw', '_hi', '_s2', '_peer_mean', 'oof_current_stress_prob'])
    return out


RESEARCH_FEATURES = [
    'r_liability_weight',
    'r_liab_weighted_peer_stress',
    'r_peer_stress_max',
    'r_peer_stress_count_hi',
    'r_peer_stress_hhi',
    'r_peer_stress_decayed',
    'r_peer_stress_velocity',
    'r_buffer_adequacy',
    'r_dest_vulnerability',
    'r_propagation_process',
]

CANDIDATE_SETS = {
    'P0_m2c_exposure':   ['borrower_propagation_exposure'],
    'P1_liab_channel':   ['r_liab_weighted_peer_stress'],
    'P2_concentration':  ['r_peer_stress_max', 'r_peer_stress_count_hi', 'r_peer_stress_hhi'],
    'P3_persistence':    ['r_peer_stress_decayed', 'r_peer_stress_velocity'],
    'P4_dest_fragility': ['r_buffer_adequacy', 'r_dest_vulnerability'],
    'P5_process':        ['r_propagation_process'],
    'P6_full':           RESEARCH_FEATURES,
}
