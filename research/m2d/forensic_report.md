# M2D FORENSIC REPORT — why the frozen M2C propagation representation is weak

Baseline: commit `b5d3dd4` (m2c-integrity-fix). Nothing in `data/frozen_m2c/` or `src/` was modified.
Note: `src/graph.py`, `src/propagation.py`, `src/calibration.py`, `src/train.py`, `src/evaluate.py`
named in the brief do not exist in this repository.

## Finding 1 (decisive) — the exposure feature contains no information Model B lacks

```
borrower_propagation_exposure = peer_predicted_stress_mean x borrower_liability_share
```
Verified on the frozen seed-909 matrix: `max_abs_err = 0.000e+00`.
**Both factors are already Model B features.** The feature is an algebraic product of two
columns B already holds, so it cannot add information — only a shortcut through an
axis-aligned split, which a 1-split stump never exercises.

RandomForest reconstruction of the feature from B's 51 features: **R² = 0.9987**.

## Finding 2 — every alternative propagation representation is also near-redundant

R² recovering each feature from B's 51 features (mean over seeds 42/303/606/909):

| feature | R² from B | closest B feature (|Spearman|) |
|---|---|---|
| r_peer_stress_hhi | 0.9998 | peer_predicted_stress_mean (0.989) |
| r_peer_stress_count_hi | 0.9998 | expected_peer_debt_burden (0.992) |
| r_peer_stress_max | 0.9998 | peer_predicted_stress_mean (0.999) |
| r_buffer_adequacy | 0.9996 | cash_buffer_mean_12w (0.972) |
| borrower_propagation_exposure | 0.9987 | peer_predicted_stress_mean (0.804) |
| r_peer_stress_decayed | 0.9986 | peer_predicted_stress_mean (0.999) |
| r_liab_weighted_peer_stress | 0.9884 | peer_predicted_stress_mean (0.999) |
| r_propagation_process | 0.9738 | peer_predicted_stress_mean (0.914) |
| r_peer_stress_velocity | 0.8008 | peer_dpd_mean_4w (0.422) |
| r_liability_weight | 0.7452 | principal_remaining (0.067) |

**Root cause: the JLG has only 5 members, so each borrower has 4 peers.** With 4 peers the
mean, max, count and HHI of peer stress are near-collinear — there is not enough group
cardinality for concentration statistics to diverge from the mean. This is a structural
property of the synthetic world, not of the feature design.

## Finding 3 — the exposure uses the wrong transmission channel

`mechanics.apply_group_liability_coverage` weights member contributions by the household's
static `liability_weight` (`random.uniform(0.1, 1.0)`, 400 distinct values).
The feature `borrower_liability_share` is defined in `features.py` as
`cash_buffer / group_buffer_sum` — a **relative wealth share**, correlated +0.85 with the
borrower's own cash buffer. It is not the generator's transmission weight.

Consequence: the exposure multiplier *rises* with the borrower's own liquidity, so a
better-cushioned borrower is assigned larger "exposure". The contractual weight that
actually governs transmission is never exposed as a feature at all.

## Finding 4 — destination fragility, timing, persistence and multiplicity are all absent

- Destination resilience enters only through the wealth share above (relative, not absolute).
  Correlation of exposure with true buffer adequacy (buffer / obligation) is only +0.16.
- `peer_predicted_stress_mean` is contemporaneous at t: no lag, decay, velocity or duration.
- The episode contribution is recorded at the episode start week and PV requires an exact
  week match, so source→destination lag inside `[t, t+4)` is collapsed and never featurised.
- A mean over peers cannot distinguish one peer at 1.0 from four peers at 0.25.
- The network is collapsed **twice** before the learner sees anything: per-peer probabilities
  → mean, then mean → one scalar.

## Finding 5 — PV rarity is caused by new-stress rarity, not by the attribution rule

Attrition cascade, seed 909 (62,400 borrower-weeks):

| stage | count | share |
|---|---|---|
| not currently stressed at t | 60,408 | 96.8% |
| **becomes stressed in (t, t+4]** | **116** | **0.19%** |
| episode contribution ≥ 0.30 (exact week) | 31 | 0.0497% |

Only **10 unique borrowers in 9 of 80 JLGs** ever go PV-positive across 156 weeks.
The contribution distribution saturates (96.7% of non-zero episodes are ≥ 0.30), so the
0.30 threshold is *not* the binding constraint.

Relaxing the attribution rule barely helps — measured across all 20 research worlds:
PV 31 → window-relaxed 36 → any-contribution 32, while plain new-stress is 116.
**You cannot fix the evidence problem by loosening the network-attribution rule.**

## Finding 6 — why 6/10 test splits hold zero positives

Splits (max_week 156, purge 8): train 1–78, purge 79–86, val 87–118, purge 119–126,
test 127–156. Valid, with purge gaps on both sides.
With ~4–38 positives scattered over 156 weeks and concentrated in <10 borrowers, a 30-week
test window catches 0–8 by chance. That is sampling, not a bug.

## Diagnosis

The M2C propagation experiment is **structurally incapable** of demonstrating incremental
value, for two independent reasons:
1. Model C's only extra feature is a deterministic function of two Model B columns.
2. The target is too rare for the test split to discriminate anything.
