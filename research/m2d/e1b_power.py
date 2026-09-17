"""
M2D E1b — HOW MUCH INSTABILITY DOES SPARSITY ALONE EXPLAIN?  (research question 1)

Hypothesis: at the observed positive counts, test PR-AUC carries so much sampling
error that two models of EQUAL true quality routinely appear to differ by more than
the frozen M2C study's reported lift. If so, sparsity alone explains the instability.

Null construction: both models see the same latent signal of identical strength and
differ only by independent idiosyncratic noise of equal variance. Neither is better.
Any observed delta is therefore pure sampling error.

Nothing about the target, splits, horizon or models under study is modified; this
uses only the empirically observed positive counts.
"""
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score

rng = np.random.default_rng(20260917)
N_TEST = 12000
COUNTS = [3, 4, 7, 8, 10, 20, 50, 100, 200]
REPS = 3000
SIGNAL, IDIO = 0.8, 0.5      # equal-quality models, equal idiosyncratic noise

print("=== E1b-1: APPARENT PR-AUC DELTA BETWEEN TWO EQUALLY GOOD MODELS ===")
print(f"{'positives':>10}{'median|d|':>11}{'p90|d|':>9}{'p99|d|':>9}{'max|d|':>9}{'P(|d|>0.0587)':>15}")
rows = []
for k in COUNTS:
    y = np.zeros(N_TEST, int); y[:k] = 1
    d = np.empty(REPS)
    for i in range(REPS):
        rng.shuffle(y)
        lat = rng.normal(0, 1, N_TEST) + SIGNAL * y
        a = lat + rng.normal(0, IDIO, N_TEST)
        b = lat + rng.normal(0, IDIO, N_TEST)
        d[i] = average_precision_score(y, a) - average_precision_score(y, b)
    ad = np.abs(d)
    p_exceed = float((ad > 0.0587).mean())      # frozen M2C headline lift
    rows.append(dict(positives=k, median=np.median(ad), p90=np.quantile(ad, .90),
                     p99=np.quantile(ad, .99), mx=ad.max(), p_exceed_m2c_lift=p_exceed))
    print(f"{k:>10}{np.median(ad):>11.4f}{np.quantile(ad,.90):>9.4f}"
          f"{np.quantile(ad,.99):>9.4f}{ad.max():>9.4f}{p_exceed:>15.3f}")
pd.DataFrame(rows).to_csv('research/m2d/e1b_power_null.csv', index=False)

print("\n=== E1b-2: SPREAD OF PR-AUC FOR ONE FIXED-QUALITY MODEL ===")
print(f"{'positives':>10}{'mean PR':>10}{'sd':>9}{'sd/mean':>9}{'p05':>9}{'p95':>9}")
rows2 = []
for k in COUNTS:
    y = np.zeros(N_TEST, int); y[:k] = 1
    v = np.empty(1500)
    for i in range(1500):
        rng.shuffle(y)
        v[i] = average_precision_score(y, rng.normal(0, 1, N_TEST) + SIGNAL * y)
    rows2.append(dict(positives=k, mean=v.mean(), sd=v.std(), cv=v.std()/max(v.mean(),1e-9),
                      p05=np.quantile(v, .05), p95=np.quantile(v, .95)))
    print(f"{k:>10}{v.mean():>10.4f}{v.std():>9.4f}{v.std()/v.mean():>9.2f}"
          f"{np.quantile(v,.05):>9.4f}{np.quantile(v,.95):>9.4f}")
pd.DataFrame(rows2).to_csv('research/m2d/e1b_power_spread.csv', index=False)

print("\n=== INTERPRETATION AT THE OBSERVED DESIGN POINT ===")
obs = pd.read_csv('research/m2d/e1_prevalence_per_seed.csv')
sc = obs[obs.test_scoreable]
print(f"  scoreable worlds: {len(sc)}/20, test positives per scoreable world: "
      f"{sorted(sc.test_pos.tolist())}")
r8 = next(r for r in rows if r['positives'] == 8)
print(f"  at 8 test positives, two EQUALLY GOOD models differ by >0.0587 PR-AUC "
      f"{r8['p_exceed_m2c_lift']:.1%} of the time by chance alone.")
