import numpy as np, pandas as pd
from scipy.stats import wilcoxon
pd.set_option('display.width', 220)

d = pd.read_csv('research/m2d/e3_results.csv')
ok = d[d.status == 'ok']
print("=== SCOREABILITY (frozen PV target) ===")
print(f"  worlds total: {d.seed.nunique()}   scoreable: {ok.seed.nunique()}   "
      f"unscoreable: {d.seed.nunique()-ok.seed.nunique()}")
print(f"  test positives across scoreable worlds: "
      f"{int(ok.groupby('seed').pos_test.first().sum())}")
print(f"  per-world test positives: {sorted(ok.groupby('seed').pos_test.first().tolist())}")

print("\n=== P1 CHECK: are monotone transforms a no-op for the tree? (C vs C2, lgbm) ===")
t = ok[ok.model=='lgbm'].pivot_table(index='seed', columns='spec', values='pr_auc_raw')
if 'C' in t and 'C2' in t:
    diff = (t['C2']-t['C']).abs()
    print(f"  max |C2 - C| PR-AUC across worlds: {diff.max():.3e}")
    print("  " + ("CONFIRMED - transforms are no-ops for the tree, as predicted."
                  if diff.max() < 1e-9 else
                  "NOT a no-op: the tree used the transformed columns as extra split candidates."))

print("\n=== AGGREGATE BY SPEC AND MODEL CLASS (raw scores, scoreable worlds only) ===")
g = ok.groupby(['model','spec']).agg(
    n=('seed','nunique'), pr_mean=('pr_auc_raw','mean'), pr_med=('pr_auc_raw','median'),
    pr_sd=('pr_auc_raw','std'), roc_mean=('roc_auc_raw','mean')).round(4)
print(g.to_string())

print("\n=== PRE-REGISTERED TEST: EACH VARIANT vs MODEL B, PER MODEL CLASS ===")
print(f"{'model':>12}{'spec':>6}{'n':>4}{'meanD':>9}{'medD':>9}{'win':>5}{'loss':>6}{'tie':>5}{'wilcox_p':>10}{'LOO worst':>11}")
res=[]
for kind in ['lgbm','logreg']:
    sub = ok[ok.model==kind].pivot_table(index='seed', columns='spec', values='pr_auc_raw')
    if 'B' not in sub: continue
    for spec in ['A','C','C1','C2','C3']:
        if spec not in sub: continue
        dd = (sub[spec]-sub['B']).dropna()
        if not len(dd): continue
        try: p = wilcoxon(dd).pvalue if dd.abs().sum()>1e-12 else 1.0
        except Exception: p = float('nan')
        loo = min(dd.drop(k).mean() for k in dd.index) if len(dd)>1 else dd.mean()
        w=int((dd>1e-9).sum()); l=int((dd<-1e-9).sum()); tie=int((dd.abs()<=1e-9).sum())
        print(f"{kind:>12}{spec:>6}{len(dd):>4}{dd.mean():>9.4f}{dd.median():>9.4f}"
              f"{w:>5}{l:>6}{tie:>5}{p:>10.4f}{loo:>11.4f}")
        res.append(dict(model=kind,spec=spec,n=len(dd),mean_delta=dd.mean(),
                        median_delta=dd.median(),wins=w,losses=l,ties=tie,
                        wilcoxon_p=p,loo_worst_mean=loo))
pd.DataFrame(res).to_csv('research/m2d/e3_vs_baseline.csv', index=False)

print("\n=== DECISION RULE (all four must pass) ===")
for r in res:
    if r['spec']=='A': continue
    c1 = r['mean_delta']>0
    c2 = r['wins'] >= 0.7*r['n']
    c3 = (r['wilcoxon_p']<0.05)
    c4 = r['loo_worst_mean']>0
    verdict = 'PURSUE' if all([c1,c2,c3,c4]) else 'null / unstable'
    print(f"  {r['model']:>7} {r['spec']:<3} mean>0:{str(c1):<5} wins>=70%:{str(c2):<5} "
          f"p<0.05:{str(c3):<5} LOO-robust:{str(c4):<5} -> {verdict}")

print("\n=== Q7 CALIBRATION: does calibration change RANKING? ===")
cal = ok.dropna(subset=['pr_auc_cal'])
if len(cal):
    cal = cal.assign(d_pr=cal.pr_auc_cal-cal.pr_auc_raw, d_roc=cal.roc_auc_cal-cal.roc_auc_raw,
                     tie_collapse=cal.n_unique_raw-cal.n_unique_cal)
    print(f"  rows compared              : {len(cal)}")
    print(f"  mean d PR-AUC (cal - raw)  : {cal.d_pr.mean():+.5f}")
    print(f"  mean d ROC-AUC (cal - raw) : {cal.d_roc.mean():+.5f}")
    print(f"  rows where ranking changed : {int((cal.d_roc.abs()>1e-9).sum())}/{len(cal)}")
    print(f"  mean distinct-value collapse (raw -> cal): {cal.tie_collapse.mean():.1f}")
    print("  Platt is monotone, so any ROC change comes from distinct raw scores")
    print("  collapsing into float ties, which destroys ranking resolution.")
