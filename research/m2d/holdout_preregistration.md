# M2D RESEARCH — HOLDOUT PRE-REGISTRATION

Written BEFORE any holdout world was scored. Fixes the candidate, target, metric and
decision rule so the holdout cannot be used for selection.

## Selection stage (RESEARCH VALIDATION)
Seeds: 42, 101, 202, 303, 404, 505, 606, 707, 808, 909 (the 10 M2C worlds).
Target used for selection: `rt_new_stress` (RESEARCH TARGET).
Reason: the frozen M2C `propagation_vulnerability` target was scoreable on only
4/10 seeds with 22 test positives total and is underpowered for model comparison.

## Selection outcome (dev means vs Model B)
| candidate | mean d_PR | mean d_ROC | PR wins/10 | ROC wins/10 |
|---|---|---|---|---|
| P5_process | +0.0049 | +0.0539 | 3 | 3 |
| P1_liab_channel | +0.0047 | +0.0236 | 4 | 5 |
| P6_full | -0.0005 | +0.0474 | 6 | 7 |
| P0_m2c_exposure (frozen C) | -0.0035 | +0.0169 | 4 | 3 |

## Pre-registered candidates
- PRIMARY: `B+P5_process` — highest mean ROC delta on validation; matches the
  source x channel x persistence x destination form; single interpretable scalar.
- SECONDARY: `B+P6_full` — most ROC wins; upper bound for this feature family.

## Pre-registered holdout
Fresh seeds, never used in formulation or selection:
1201, 1302, 1403, 1504, 1605, 1706, 1807, 1908, 2009, 2110
Each world generated and split independently; no pooling.

## Pre-registered decision rule
The candidate is declared PROMISING only if ALL hold on the holdout:
1. mean PR-AUC delta vs B > 0, AND
2. wins on >= 7 of 10 independent worlds, AND
3. paired Wilcoxon p < 0.05 on the per-seed PR-AUC delta, AND
4. >= 100 test positives in total across the holdout.
Otherwise: NOT promising. The holdout is scored ONCE and not re-inspected for tuning.
