"""
M2D E1 — SIGNAL PREVALENCE CENSUS  (research question 1)

Hypothesis: sparse and concentrated positives, not feature design, explain most of
the cross-seed instability of the frozen M2C comparison.

Changed component: nothing. This is pure measurement on the frozen PV target.
The target, splits, purge gap and horizon are untouched.
"""
import sys, os, json, pickle
sys.path.append(os.path.abspath('.'))
import numpy as np, pandas as pd

CACHE = 'research/m2d/cache'
SEEDS = [42,101,202,303,404,505,606,707,808,909,1201,1302,1403,1504,1605,1706,1807,1908,2009,2110]
TARGET = 'propagation_vulnerability'   # frozen M2C target, never redefined

rows, per_group, per_period = [], [], []

for seed in SEEDS:
    d = pickle.load(open(os.path.join(CACHE, f'world_{seed}.pkl'), 'rb'))
    df, ep = d['df'], d['ep']
    pos = df[df[TARGET]]

    by_split = df.groupby('split')[TARGET].agg(['sum', 'size'])
    g = df.groupby('group_id')[TARGET].sum()
    gp = g[g > 0]

    # temporal concentration of positives (Gini over 10-week blocks)
    blocks = (pos['week'] // 10 * 10).value_counts()
    def gini(x):
        x = np.sort(np.asarray(x, float))
        if x.sum() == 0: return 0.0
        n = len(x); idx = np.arange(1, n + 1)
        return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)

    rows.append(dict(
        seed=seed,
        n_rows=len(df),
        pv_positives=int(df[TARGET].sum()),
        pv_rate=float(df[TARGET].mean()),
        train_pos=int(by_split.loc['train', 'sum']) if 'train' in by_split.index else 0,
        val_pos=int(by_split.loc['val', 'sum']) if 'val' in by_split.index else 0,
        test_pos=int(by_split.loc['test', 'sum']) if 'test' in by_split.index else 0,
        test_scoreable=bool(('test' in by_split.index) and by_split.loc['test', 'sum'] > 0),
        unique_pos_borrowers=int(pos['borrower_id'].nunique()),
        unique_pos_groups=int(len(gp)),
        total_groups=int(df['group_id'].nunique()),
        max_pos_in_one_group=int(g.max()),
        share_pos_in_top_group=float(g.max() / max(df[TARGET].sum(), 1)),
        episode_rows=int(len(ep)),
        episode_rows_ge_030=int((ep['episode_network_contribution'] >= 0.30).sum()) if len(ep) else 0,
        new_stress_events=int(df['rt_new_stress'].sum()),
        pv_share_of_new_stress=float(df[TARGET].sum() / max(df['rt_new_stress'].sum(), 1)),
        temporal_gini_10w=gini(blocks.values) if len(blocks) else 0.0,
        weeks_with_any_pos=int(pos['week'].nunique()),
    ))
    for gid, c in gp.items():
        per_group.append(dict(seed=seed, group_id=gid, positives=int(c)))
    for b, c in blocks.sort_index().items():
        per_period.append(dict(seed=seed, week_block=int(b), positives=int(c)))

d = pd.DataFrame(rows)
d.to_csv('research/m2d/e1_prevalence_per_seed.csv', index=False)
pd.DataFrame(per_group).to_csv('research/m2d/e1_prevalence_per_group.csv', index=False)
pd.DataFrame(per_period).to_csv('research/m2d/e1_prevalence_per_period.csv', index=False)

pd.set_option('display.width', 200)
print("=== E1: PV PREVALENCE PER WORLD (frozen target, 20 independent seeds) ===")
print(d[['seed','pv_positives','pv_rate','train_pos','val_pos','test_pos','test_scoreable',
         'unique_pos_borrowers','unique_pos_groups','new_stress_events','pv_share_of_new_stress']]
      .to_string(index=False, float_format=lambda v: f'{v:.5f}'))

print("\n=== AGGREGATE ===")
print(f"  worlds                         : {len(d)}")
print(f"  PV rate            mean/median : {d.pv_rate.mean():.5f} / {d.pv_rate.median():.5f}")
print(f"  PV positives       min..max    : {d.pv_positives.min()} .. {d.pv_positives.max()}")
print(f"  TEST positives     min..max    : {d.test_pos.min()} .. {d.test_pos.max()}")
print(f"  worlds with TEST positives     : {int(d.test_scoreable.sum())}/{len(d)}")
print(f"  worlds with 0 TEST positives   : {int((~d.test_scoreable).sum())}/{len(d)}")
print(f"  unique positive borrowers      : mean {d.unique_pos_borrowers.mean():.1f} of 400")
print(f"  unique positive JLGs           : mean {d.unique_pos_groups.mean():.1f} of {int(d.total_groups.mean())}")
print(f"  PV as share of new-stress      : mean {d.pv_share_of_new_stress.mean():.3f}")
print(f"  temporal Gini (10w blocks)     : mean {d.temporal_gini_10w.mean():.3f}")
print(f"  share of positives in 1 group  : mean {d.share_pos_in_top_group.mean():.3f}")

# power: probability a 30-week test window contains >=k positives
print("\n=== POWER IMPLICATION ===")
tot = d.pv_positives.sum(); te = d.test_pos.sum()
print(f"  total PV positives across 20 worlds : {tot}")
print(f"  of which land in a TEST split       : {te} ({te/tot:.1%})")
print(f"  median test positives per world     : {d.test_pos.median():.1f}")
print("  A PR-AUC comparison on <10 positives has extremely wide sampling error;")
print("  this is a power limitation of the evaluation population, not a model defect.")
