"""
M2D round 2 — extract BOTH lineage tables per world.

The cached worlds carry episode lineage only. Research question 2 asks about
marginal_weekly_network_contribution as well, which is DIAGNOSTIC ONLY and is
never substituted for episode attribution.

Writes to research/m2d/cache_lineage/. Frozen M2C artifacts are untouched.
"""
import sys, os, pickle, argparse
sys.path.append(os.path.abspath('.'))
from src.data import SyntheticWorldGenerator

OUT = 'research/m2d/cache_lineage'
N_WEEKS = 156


def run(seed):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f'lineage_{seed}.pkl')
    if os.path.exists(p):
        print(f'  seed {seed}: cached', flush=True); return
    gen = SyntheticWorldGenerator(n_borrowers=400, seed=seed, slack_regime='conservative')
    hist, marginal, episode = gen.simulate(n_weeks=N_WEEKS)
    pickle.dump({'seed': seed, 'marginal': marginal, 'episode': episode,
                 'history': hist[['week', 'borrower_id', 'group_id', 'shortfall',
                                  'group_covered_amount', 'cash_buffer', 'amount_due',
                                  'days_past_due', 'is_defaulted']]},
                open(p, 'wb'))
    print(f'  seed {seed}: marginal={len(marginal)} episode={len(episode)}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('seeds', nargs='+', type=int)
    for s in ap.parse_args().seeds:
        run(s)
