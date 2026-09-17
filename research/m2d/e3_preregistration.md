# M2D E3 — PRE-REGISTRATION (written before any E3 experiment was run)

## What E2b established (train+validation only)

The frozen construction is `exposure = peer_predicted_stress_mean x borrower_liability_share`.
Measured univariate rank-AUC against the frozen PV target, across 20 independent worlds:

| component | mean rank-AUC | worlds > 0.5 |
|---|---|---|
| `peer_predicted_stress_mean` | 0.7154 | 18/20 |
| `borrower_liability_share` | 0.0901 | 0/20 |
| product (= frozen exposure) | 0.4482 | 8/20 |

Median liability share is 0.0047 on positives and 0.2064 on negatives (a ~44x gap), because
the share is `own_buffer / group_buffer` and positives are liquidity-poor. `cash_buffer_mean_4w`
is itself inverse (AUC 0.107), so the multiplier is largely a proxy for the borrower's own
buffer and encodes vulnerability with the wrong sign for a multiplicative term.

## The tempting but wrong inference

"Repair the multiplier and Model C will beat Model B."

## Pre-registered prediction (what I actually expect)

**H3-null: no corrected variant will show stable incremental value over Model B.**

Reason: `peer_predicted_stress_mean` is ALREADY a Model B feature. Removing the harmful
multiplier does not add information to B - it recovers a column B already holds. Any variant
built from these components should therefore land within sampling noise of B.

## Falsifiable sub-predictions

- **P1** Monotone transforms of exposure (log, rank, winsorize) are mathematically no-ops for
  an axis-aligned tree. Tree results for C2 must equal C exactly. If they differ by more than
  float noise, my harness has a bug, not a finding.
- **P2** The same transforms CAN matter for logistic regression, so C2-vs-C may differ there.
- **P3** A corrected interaction (C3) will not beat B by a margin that is stable across seeds,
  even though its univariate AUC should exceed the frozen exposure's.
- **P4** Given E1b, per-seed deltas on the frozen PV target will be dominated by sampling error
  (PR-AUC coefficient of variation 2.3-4.3 at the observed positive counts).

## Decision rule (fixed in advance)

A variant is called "worth pursuing" only if ALL hold:
1. mean PR-AUC delta vs B > 0 on the frozen PV target, AND
2. positive delta on >= 70% of scoreable worlds, AND
3. paired Wilcoxon p < 0.05 across worlds, AND
4. the result survives leave-one-world-out (drop the single most influential world and the
   sign and significance are unchanged).

Anything less is reported as null or unstable. No seed is dropped for being unfavourable.

## Variants (all time-valid, no hidden lineage, no future information)

- **A**  individual financial only
- **B**  individual + network context (frozen baseline to beat)
- **C**  B + frozen `borrower_propagation_exposure` (frozen Model C)
- **C1** C + exposure persistence: 4-week rolling mean, trend, volatility, elevated-duration count
- **C2** C + monotone transforms of exposure: log1p, winsorized at p99, within-week percentile rank
- **C3** B + corrected propagation term: `peer_predicted_stress_mean x absolute destination fragility`
         where fragility = obligation / (buffer + obligation), replacing the relative-wealth multiplier
- **M0** deterministic recent-delinquency baseline

## Model classes
LightGBM (frozen M2C configuration) and L2 logistic regression on standardized features.

## Evaluation
Frozen PV target. Frozen chronological splits, purge gap and horizon. Calibration fitted on
validation only; test labels never used for any choice. All 20 worlds processed independently;
unscoreable worlds are reported, never hidden.
