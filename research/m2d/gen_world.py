"""
M2D RESEARCH — per-seed world generation + feature/target caching.

Writes research caches ONLY to research/m2d/cache/.
Never touches data/frozen_m2c/.
"""
import sys, os, pickle, argparse
sys.path.append(os.path.abspath('.'))
import numpy as np, pandas as pd

from src.data import SyntheticWorldGenerator
from src.targets import construct_targets, assign_temporal_splits, load_config
from src.features import generate_features, build_feature_matrices
from src.features_propagation import generate_propagation_features
from src.models.model_a_current import ModelACurrent
sys.path.append(os.path.abspath('research/m2d'))
from features_research import build_research_propagation

CACHE = 'research/m2d/cache'
N_WEEKS = 156


def build_research_targets(targets_df, ep_lineage, horizon=4):
    """
    RESEARCH TARGETS — clearly separate from the frozen M2C `propagation_vulnerability`.
    The frozen column is carried through untouched for reference.
    """
    t = targets_df.copy()
    if ep_lineage.empty:
        for c in ['rt_pv_window', 'rt_net_stress_any', 'rt_new_stress']:
            t[c] = False
        return t

    ep = ep_lineage
    # RT_A: same >=0.30 contribution rule, but the episode may start anywhere in
    # (t, t+horizon] rather than exactly at t. Relaxes only the timing match.
    ev30 = ep[ep['episode_network_contribution'] >= 0.30].groupby(
        'destination_borrower')['week'].apply(set).to_dict()
    # RT_B: any strictly positive network contribution at episode start t.
    evany = ep[ep['episode_network_contribution'] > 0].groupby(
        'destination_borrower')['week'].apply(set).to_dict()

    elig = (~t['current_stress']) & t['next_period_stress']

    def hit(row, table, window):
        if not (not row['current_stress'] and row['next_period_stress']):
            return False
        ws = table.get(row['borrower_id'])
        if not ws:
            return False
        tt = row['week']
        if window:
            return any(tt <= w <= tt + horizon for w in ws)
        return tt in ws

    t['rt_pv_window'] = t.apply(lambda r: hit(r, ev30, True), axis=1)
    t['rt_net_stress_any'] = t.apply(lambda r: hit(r, evany, False), axis=1)
    # RT_C: plain new-stress event (no network attribution at all) — used as a
    # diagnostic denominator, NOT as a propagation target.
    t['rt_new_stress'] = elig
    return t


def run(seed):
    os.makedirs(CACHE, exist_ok=True)
    out = os.path.join(CACHE, f'world_{seed}.pkl')
    if os.path.exists(out):
        print(f'  seed {seed}: cached, skip'); return

    gen = SyntheticWorldGenerator(n_borrowers=400, seed=seed, slack_regime='conservative')
    hist, lin, ep = gen.simulate(n_weeks=N_WEEKS)

    liab = {b: gen.world_state.households[h].liability_weight
            for b, h in gen.world_state.borrower_to_household.items()}

    targ = assign_temporal_splits(build_research_targets(construct_targets(hist, ep), ep))
    feats = generate_features(hist)
    cur, nxt = build_feature_matrices(targ, feats)

    oof = ModelACurrent().generate_historical_predictions(cur)
    m2c = generate_propagation_features(nxt, oof)          # frozen M2C exposure
    full = build_research_propagation(m2c, oof, liab)      # + research features

    pickle.dump({'seed': seed, 'df': full, 'ep': ep, 'targets': targ,
                 'liability_weight': liab}, open(out, 'wb'))
    n = {c: int(full[c].sum()) for c in
         ['propagation_vulnerability', 'rt_pv_window', 'rt_net_stress_any', 'rt_new_stress']
         if c in full.columns}
    print(f'  seed {seed}: rows={len(full)} positives={n}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('seeds', nargs='+', type=int)
    a = ap.parse_args()
    for s in a.seeds:
        run(s)
