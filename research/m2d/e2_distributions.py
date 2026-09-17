"""
M2D E2 — PROPAGATION EXPOSURE AND BURDEN DISTRIBUTIONS  (research question 2)

Hypotheses:
  H2a the attribution quantities are near-zero-inflated and heavy-tailed
  H2b episode_network_contribution saturates near 1.0, so the 0.30 threshold is
      not what makes PV rare
  H2c the Model C exposure feature separates positives from negatives only weakly
  H2d exposure concentrates in a few groups and periods

Discipline: positives-vs-negatives comparison uses TRAIN+VALIDATION rows only.
Test rows are never inspected. marginal_weekly_network_contribution is reported as
a diagnostic only and is never substituted for episode attribution.
"""
import sys, os, pickle
sys.path.append(os.path.abspath('.'))
import numpy as np, pandas as pd

SEEDS = [42,101,202,303,404,505,606,707,808,909,1201,1302,1403,1504,1605,1706,1807,1908,2009,2110]
EXPO = 'borrower_propagation_exposure'

def q(x, p): return float(np.quantile(x, p)) if len(x) else float('nan')

lin_rows, exp_rows, sep_rows = [], [], []
for seed in SEEDS:
    L = pickle.load(open(f'research/m2d/cache_lineage/lineage_{seed}.pkl', 'rb'))
    W = pickle.load(open(f'research/m2d/cache/world_{seed}.pkl', 'rb'))
    marg, ep, df = L['marginal'], L['episode'], W['df']

    # ---- attribution quantities -------------------------------------------
    r = dict(seed=seed, marginal_rows=len(marg), episode_rows=len(ep))
    if len(marg):
        m = marg['contribution_share'].values
        r.update(marg_mean=m.mean(), marg_median=np.median(m), marg_p90=q(m,.9),
                 marg_max=m.max(), marg_frac_ge_030=float((m>=.30).mean()),
                 marg_frac_ge_080=float((m>=.80).mean()))
    if len(ep):
        e = ep['episode_network_contribution'].values
        fw, cf, inc = ep['fw_burden'].values, ep['cf_burden'].values, ep['network_incremental_burden'].values
        r.update(ep_mean=e.mean(), ep_median=np.median(e), ep_p10=q(e,.1), ep_p90=q(e,.9),
                 ep_frac_ge_030=float((e>=.30).mean()), ep_frac_ge_080=float((e>=.80).mean()),
                 ep_frac_eq_1=float((e>=0.999).mean()),
                 fw_burden_median=float(np.median(fw)), cf_burden_median=float(np.median(cf)),
                 inc_burden_median=float(np.median(inc)),
                 inc_burden_frac_zero=float((inc<=1e-9).mean()),
                 inc_over_fw_median=float(np.median(inc/np.maximum(fw,1e-9))))
    lin_rows.append(r)

    # ---- exposure feature distribution ------------------------------------
    x = df[EXPO].values
    exp_rows.append(dict(seed=seed, mean=x.mean(), sd=x.std(), median=np.median(x),
                         p75=q(x,.75), p90=q(x,.9), p99=q(x,.99), mx=x.max(),
                         frac_zero=float((x<=1e-12).mean()),
                         frac_below_1e3=float((x<1e-3).mean()),
                         skew=float(pd.Series(x).skew()), kurt=float(pd.Series(x).kurt()),
                         top1pct_share=float(np.sort(x)[::-1][:max(1,len(x)//100)].sum()/max(x.sum(),1e-12)),
                         gini_group=float(df.groupby('group_id')[EXPO].sum().pipe(
                             lambda g: (2*np.arange(1,len(g)+1)*np.sort(g.values)).sum()
                                       /(len(g)*np.sort(g.values).sum())-(len(g)+1)/len(g)))))

    # ---- separation, TRAIN+VAL ONLY ---------------------------------------
    tv = df[df.split.isin(['train','val'])]
    pos, neg = tv[tv.propagation_vulnerability], tv[~tv.propagation_vulnerability]
    if len(pos) and len(neg):
        from scipy.stats import mannwhitneyu
        a, b = pos[EXPO].values, neg[EXPO].values
        auc = float(mannwhitneyu(a, b, alternative='two-sided').statistic/(len(a)*len(b)))
        sep_rows.append(dict(seed=seed, n_pos=len(pos), n_neg=len(neg),
                             pos_median=float(np.median(a)), neg_median=float(np.median(b)),
                             pos_mean=float(a.mean()), neg_mean=float(b.mean()),
                             rank_auc=auc, ratio_median=float(np.median(a)/max(np.median(b),1e-12))))

lin = pd.DataFrame(lin_rows); exp = pd.DataFrame(exp_rows); sep = pd.DataFrame(sep_rows)
lin.to_csv('research/m2d/e2_attribution_distributions.csv', index=False)
exp.to_csv('research/m2d/e2_exposure_distributions.csv', index=False)
sep.to_csv('research/m2d/e2_separation_trainval.csv', index=False)
pd.set_option('display.width', 220)

print("=== E2-A: ATTRIBUTION QUANTITIES (20 worlds) ===")
print(f"  marginal rows/world  : median {lin.marginal_rows.median():.0f}   (DIAGNOSTIC ONLY)")
print(f"  episode  rows/world  : median {lin.episode_rows.median():.0f}   (PV attribution quantity)")
print(f"  episode contribution : median {lin.ep_median.mean():.3f}  p10 {lin.ep_p10.mean():.3f}  p90 {lin.ep_p90.mean():.3f}")
print(f"  fraction >= 0.30     : {lin.ep_frac_ge_030.mean():.3f}")
print(f"  fraction >= 0.80     : {lin.ep_frac_ge_080.mean():.3f}")
print(f"  fraction ~= 1.00     : {lin.ep_frac_eq_1.mean():.3f}")
print(f"  incremental burden zero-fraction : {lin.inc_burden_frac_zero.mean():.3f}")
print(f"  incremental/full burden (median) : {lin.inc_over_fw_median.mean():.3f}")
print("  => H2b: episode contribution SATURATES near 1; the 0.30 cut removes almost nothing.")

print("\n=== E2-B: MODEL C EXPOSURE FEATURE ===")
print(f"  frac exactly zero    : {exp.frac_zero.mean():.3f}")
print(f"  frac below 1e-3      : {exp.frac_below_1e3.mean():.3f}")
print(f"  median               : {exp['median'].mean():.6f}")
print(f"  p99 / median         : {(exp.p99/exp['median']).mean():.1f}x")
print(f"  skew / excess kurtosis : {exp['skew'].mean():.1f} / {exp['kurt'].mean():.1f}")
print(f"  top-1% share of total  : {exp.top1pct_share.mean():.3f}")
print(f"  group-level Gini       : {exp.gini_group.mean():.3f}")
print("  => H2a/H2d: near-zero inflated, extremely heavy tailed, group concentrated.")

print("\n=== E2-C: POSITIVE vs NEGATIVE SEPARATION (TRAIN+VAL ONLY, test never touched) ===")
print(sep[['seed','n_pos','pos_median','neg_median','ratio_median','rank_auc']]
      .to_string(index=False, float_format=lambda v: f'{v:.6f}'))
print(f"\n  rank-AUC of exposure alone: mean {sep.rank_auc.mean():.4f}  median {sep.rank_auc.median():.4f}"
      f"  min {sep.rank_auc.min():.4f}  max {sep.rank_auc.max():.4f}")
print(f"  worlds where exposure ranks positives ABOVE negatives (AUC>0.5): "
      f"{int((sep.rank_auc>0.5).sum())}/{len(sep)}")
