"""
M2D E4 - CONFIRMATORY TEST OF THE TWO SURVIVING VARIANTS

E3 produced two variants (C2, C3 under logistic regression) that passed the
pre-registered rule on the frozen PV target. Three reasons to distrust that before
reporting it as a finding:

  1. MULTIPLE COMPARISONS. E3 ran 10 variant-vs-B tests. At alpha=0.05 roughly one
     false positive is expected by construction.
  2. SANITY FAILURE. Model A (individual features only, a strict SUBSET of B) also
     "beat" B in both model classes. A cannot genuinely carry more information than
     a superset of itself, so the comparison is noise-dominated at n=9 worlds and
     42 test positives.
  3. POWER. E1b measured a PR-AUC coefficient of variation of 2.3-4.3 at these counts.

This experiment applies the correction and then runs the decisive confirmatory test:
if C2/C3 carry real incremental information they must show it on a target where
power actually exists.

`rt_new_stress` is a clearly labelled RESEARCH target (a newly stressed borrower,
with no network attribution). It does NOT replace the frozen PV target, which is
untouched and reported alongside. Splits, purge gap and horizon are unchanged.
"""
import sys, os, pickle, warnings
sys.path.append(os.path.abspath('.'))
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath('research/m2d'))
from e3_variants import add_variant_features, feature_sets, LGB_P, SEEDS

TARGETS = ['propagation_vulnerability', 'rt_new_stress']


def fit_predict(kind, tr, va, te, feats, target):
    ytr = tr[target].astype(int)
    yva = va[target].astype(int)
    if ytr.sum() == 0:
        return None
    if kind == 'lgbm':
        dtr = lgb.Dataset(tr[feats], label=ytr)
        vs, vn, cbs = [dtr], ['train'], []
        if yva.sum() > 0:
            vs.append(lgb.Dataset(va[feats], label=yva, reference=dtr))
            vn.append('val')
            cbs = [lgb.early_stopping(20, verbose=False)]
        m = lgb.train(LGB_P, dtr, num_boost_round=200, valid_sets=vs,
                      valid_names=vn, callbacks=cbs)
        return m.predict(te[feats])
    def clean(x):
        return x[feats].replace([np.inf, -np.inf], 0).fillna(0)
    sc = StandardScaler().fit(clean(tr))
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(clean(tr)), ytr)
    return lr.predict_proba(sc.transform(clean(te)))[:, 1]


def main():
    rows = []
    for seed in SEEDS:
        d = add_variant_features(
            pickle.load(open(f'research/m2d/cache/world_{seed}.pkl', 'rb'))['df'])
        sets = feature_sets(d)
        tr, va, te = d[d.split == 'train'], d[d.split == 'val'], d[d.split == 'test']
        for target in TARGETS:
            yte = te[target].astype(int)
            if yte.sum() == 0 or yte.nunique() < 2:
                rows.append(dict(seed=seed, target=target, status='unscoreable',
                                 pos_test=int(yte.sum())))
                continue
            for spec in ['A', 'B', 'C', 'C2', 'C3']:
                for kind in ('lgbm', 'logreg'):
                    p = fit_predict(kind, tr, va, te, sets[spec], target)
                    if p is None:
                        continue
                    rows.append(dict(
                        experiment_id=f'E4|{target}|{spec}|{kind}|seed{seed}',
                        seed=seed, target=target, spec=spec, model=kind, status='ok',
                        pos_test=int(yte.sum()),
                        pr_auc=float(average_precision_score(yte, p)),
                        roc_auc=float(roc_auc_score(yte, p))))
        print(f'  seed {seed} done', flush=True)

    df = pd.DataFrame(rows)
    df.to_csv('research/m2d/e4_results.csv', index=False)
    df.to_json('research/m2d/e4_results.json', orient='records', indent=2)

    ok = df[df.status == 'ok']
    print('\n' + '=' * 78)
    print('E4 CONFIRMATORY ANALYSIS')
    print('=' * 78)

    for target in TARGETS:
        t = ok[ok.target == target]
        if t.empty:
            continue
        npos = int(t.groupby('seed').pos_test.first().sum())
        print(f'\n--- TARGET: {target} '
              f'(worlds {t.seed.nunique()}/20, test positives {npos}) ---')
        print(f"{'model':>8}{'spec':>6}{'n':>4}{'meanD':>9}{'medD':>9}"
              f"{'win':>5}{'loss':>6}{'p_raw':>9}{'p_bonf':>9}{'verdict':>16}")
        n_tests = 8  # 4 specs x 2 model classes compared against B
        for kind in ['lgbm', 'logreg']:
            piv = t[t.model == kind].pivot_table(index='seed', columns='spec', values='pr_auc')
            if 'B' not in piv:
                continue
            for spec in ['A', 'C', 'C2', 'C3']:
                if spec not in piv:
                    continue
                dd = (piv[spec] - piv['B']).dropna()
                if not len(dd):
                    continue
                try:
                    p = wilcoxon(dd).pvalue if dd.abs().sum() > 1e-12 else 1.0
                except Exception:
                    p = float('nan')
                pb = min(1.0, p * n_tests)
                w = int((dd > 1e-9).sum()); l = int((dd < -1e-9).sum())
                verdict = 'survives' if (pb < 0.05 and dd.mean() > 0) else 'null/unstable'
                if spec == 'A':
                    verdict = 'SANITY' + ('-FAIL' if dd.mean() > 0 else '-ok')
                print(f"{kind:>8}{spec:>6}{len(dd):>4}{dd.mean():>9.4f}{dd.median():>9.4f}"
                      f"{w:>5}{l:>6}{p:>9.4f}{pb:>9.4f}{verdict:>16}")

    print('\n=== NOTE ON MODEL A ===')
    print('  A is a strict SUBSET of B. If A appears to beat B, the comparison is')
    print('  noise-dominated and no variant result on that target can be trusted.')


if __name__ == '__main__':
    main()
