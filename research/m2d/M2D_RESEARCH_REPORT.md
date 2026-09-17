# M2D RESEARCH REPORT — propagation signal stability

**Branch:** `research-m2d` · **Starting commit:** `67fbb9a` · **Frozen reference:** `m2c-integrity-fix` @ `b5d3dd4`

Frozen M2C artifacts, target definitions, splits, purge gap, horizon and leakage
protections were not modified. A test asserts the canonical evaluation artifact is
byte-identical (newline-normalised) to the committed blob.

---

## Research questions and what was done

| # | Question | Experiment | Outcome |
|---|---|---|---|
| 1 | Signal prevalence | E1, E1b | **Explains most instability** |
| 2 | Exposure/burden distributions | E2, E2b | **Construction defect identified** |
| 3 | Information content | E3, E4 | Null |
| 4 | Temporal horizon | E5 | Partial, no incremental value |
| 5 | Network structure | E5, E6 | **Self-conditioning artefact — retracted** |
| 6 | Model specification | E3, E4 | Null |
| 7 | Calibration | E3 | Ranking is degraded, not improved |
| 8 | Seed robustness | all | 20 independent worlds throughout |

---

## Q1 — Signal prevalence (E1, E1b)

Census over 20 independently generated worlds, frozen PV target:

| quantity | value |
|---|---|
| PV rate | 0.00031 (mean) |
| PV positives per world | 4 – 38 |
| **worlds with ≥1 TEST positive** | **9 / 20** |
| **median TEST positives per world** | **0** |
| total PV positives across 20 worlds | 393 |
| of which land in a test split | **42 (10.7 %)** |
| unique positive borrowers | 5.9 of 400 (mean) |
| unique positive JLGs | 5.3 of 80 (mean) |
| share of a world's positives in its top group | 0.385 |
| PV as a share of new-stress events | 0.180 |

**E1b — how much instability does this alone explain?** Null simulation, two models of
*identical* true quality differing only by idiosyncratic noise:

| test positives | median abs ΔPR | p99 abs ΔPR | P(abs Δ > 0.0587) |
|---|---|---|---|
| 4 | 0.0006 | 0.1221 | 2.0 % |
| 8 | 0.0008 | 0.0936 | 1.7 % |
| 200 | 0.0035 | 0.0153 | 0.0 % |

And the spread of PR-AUC itself for one fixed-quality model:

| test positives | mean PR | sd | **sd / mean** |
|---|---|---|---|
| 3 | 0.0052 | 0.0221 | **4.25** |
| 8 | 0.0075 | 0.0176 | **2.34** |
| 200 | 0.0510 | 0.0083 | 0.16 |

At the observed design point the standard deviation of the metric is **2–4× its own
mean**. Two equally good models exceed the frozen M2C headline lift (+0.0587) by chance
alone about 1.7 % of the time *per world*; across ~4 scoreable worlds that is ≈6.6 %.
**Sparsity alone is sufficient to produce the observed instability.**

---

## Q2 — Distributions, and the central finding (E2, E2b)

**Attribution quantities saturate.** Episode contribution: median 0.999, 97.0 % ≥ 0.30,
82.7 % ≥ 0.80, 62.9 % ≈ 1.00. Incremental/full burden median 0.999, zero-fraction 0.000.
The 0.30 threshold removes ~3 % of episodes — **it is not what makes PV rare.**

**The exposure feature is near-zero inflated and heavy tailed.** 12.5 % exactly zero,
89.4 % below 1e-3, skew 4.8, excess kurtosis 34.3, p99/median 481×, top-1 % of rows hold
21.6 % of total exposure, group-level Gini 0.797.

### The central finding: a good predictor multiplied by an anti-predictive one

Frozen construction: `exposure = peer_predicted_stress_mean × borrower_liability_share`
where `borrower_liability_share = own_buffer / group_buffer_total`.

Univariate rank-AUC against PV, **train + validation only**, 20 worlds:

| component | mean rank-AUC | worlds > 0.5 |
|---|---|---|
| `peer_predicted_stress_mean` (source stress) | **0.7154** | **18 / 20** |
| `borrower_liability_share` (the multiplier) | **0.0901** | **0 / 20** |
| `cash_buffer_mean_4w` (what the multiplier proxies) | 0.1070 | 0 / 20 |
| **product = frozen exposure** | **0.4482** | **8 / 20** |

Median liability share is 0.0047 on positives versus 0.2064 on negatives — a ~44× gap,
because positives are liquidity-poor and the share is a *relative wealth* measure.
Multiplying a directional predictor (0.72) by a strongly inverse one (0.09) collapses the
result to chance (0.45) and makes its sign unstable across worlds.

This is a **mechanistic explanation** for why frozen Model C never beats Model B, and it
reconciles with the earlier redundancy result (exposure is recoverable from B's features
at R² 0.9987): Model C's only unique column is a *degraded* version of a feature Model B
already holds.

---

## Q3 & Q6 — Information content and model class (E3, E4)

Pre-registered in `e3_preregistration.md` before running, predicting **H3-null**.
Variants: A, B, C (frozen), C1 (persistence/trend/volatility/duration), C2 (log1p,
winsorised, within-week percentile rank), C3 (source stress × *absolute* destination
fragility, replacing the defective multiplier), M0 (deterministic recent-delinquency).
Model classes: LightGBM (frozen config) and L2 logistic regression.

### A pre-registered prediction that was falsified

**P1 said monotone transforms are no-ops for an axis-aligned tree, so C2 must equal C
exactly.** Measured max |C2 − C| = **0.222**, not 0. P1 was wrong because I mislabelled
the transform set: `log1p` and winsorisation *are* monotone, but the **within-week
percentile rank is not** a global monotone transform — it reorders rows across weeks and
genuinely adds information. Recording this as a corrected prediction rather than quietly
dropping it.

### E3 raw result on the frozen PV target (9 scoreable worlds, 42 positives)

Two variants passed the pre-registered four-part rule: `logreg C2` (p = 0.0117, 8/9 wins)
and `logreg C3` (p = 0.0391, 8/9 wins).

**I do not report these as findings.** Three reasons, all checkable:

1. **Multiple comparisons.** Ten variant-vs-B tests were run. Bonferroni-corrected:
   C2 → p = 0.094, C3 → p = 0.313. Neither survives.
2. **Sanity failure.** Model A — a strict *subset* of B — also "beat" B (+0.058 lgbm,
   +0.110 logreg). A cannot carry more information than a superset of itself, so the
   comparison is noise-dominated.
3. **Confirmatory test (E4).** Re-run on `rt_new_stress`, a clearly labelled research
   target with 375 test positives across all 20 worlds:

| model | spec | mean Δ PR vs B | wins/losses | p (Bonferroni) |
|---|---|---|---|---|
| lgbm | C | −0.0045 | 7 / 7 | 1.000 |
| lgbm | C2 | +0.0154 | 8 / 9 | 1.000 |
| lgbm | C3 | +0.0055 | 8 / 6 | 1.000 |
| logreg | C | −0.0115 | 11 / 9 | 1.000 |
| logreg | **C2** | **−0.0182** | 8 / 12 | 1.000 |
| logreg | C3 | +0.0028 | 11 / 9 | 1.000 |

**C2's sign reverses** where power exists. Every variant is null. The A-vs-B sanity gap
also shrinks to ≈0 (+0.0133 lgbm, +0.0002 logreg) exactly as power rises — confirming the
PV-target anomalies were sampling error.

---

## Q4 — Temporal horizon (E5)

Univariate rank-AUC of `peer_predicted_stress_mean` against research horizon targets
(observed stress only, clearly separate from PV, which is untouched):

| target | mean positives | peer-stress AUC | worlds > 0.5 | frozen-exposure AUC |
|---|---|---|---|---|
| PV (frozen) | 16.7 | 0.7154 | 18/20 | 0.4482 |
| stress ≤ 1w | 18.4 | 0.5324 | 15/20 | 0.0788 |
| stress ≤ 4w | 22.6 | 0.6014 | 18/20 | 0.1552 |
| stress ≤ 8w | 30.4 | 0.6745 | 18/20 | 0.2926 |
| stress ≤ 12w | 38.8 | 0.7204 | 19/20 | 0.3891 |
| persistent 4-of-12w | 246.1 | **0.1326** | **0/20** | 0.1288 |

Peer stress is a **slow** signal: association rises monotonically with horizon. Persistent
stress is *inversely* associated (0.13, 0/20) — chronic stress appears driven by individual
factors, not peer conditions. The frozen exposure is **below chance at every horizon**, so
the horizon is not what breaks it; the construction is.

---

## Q5 — Network structure: a finding I retracted (E5 → E6)

E5 initially found that inside the high-group-exposure stratum, peer stress separated
positives with **AUC 0.89 in 20/20 worlds** — the most stable association anywhere in M2D.

E6 tested it for circularity. The stratum was defined using the same variable being
measured. Re-conditioning on **independent** variables (realised group DPD > 0, realised
group shortfall > 0):

| stratum | peer-stress AUC | worlds > 0.5 |
|---|---|---|
| unconditioned | 0.7154 | 18/20 |
| **self-conditioned** (peer stress high) | **0.8896** | 20/20 |
| independent (group DPD > 0) | **0.6035** | 14/20 |
| independent (group DPD = 0) | 0.4231 | 4/20 |

Under independent conditioning the association is **weaker (0.60), not stronger**. The
0.89/20-of-20 result was self-conditioning inflation. **Q5 is not supported and the E5
stratum claim is withdrawn.** Frozen exposure is below chance in every stratum (0.20–0.39).

---

## Q7 — Calibration (E3)

78 raw-vs-calibrated comparisons: mean Δ PR-AUC **−0.00013**, mean Δ ROC-AUC **−0.04663**,
ranking changed in **34/78** cases, mean collapse of **67.5 distinct score values**.

Platt scaling is monotone and therefore cannot reorder strictly distinct scores. The
observed ROC changes come from distinct raw scores collapsing into exact float ties, which
destroys ranking resolution. **Calibration degrades ranking here; it does not improve it.**
Calibration was left unchanged — altering it to improve a metric would violate the protocol.

---

## Q8 — Seed robustness

Every experiment ran on 20 independently generated worlds, never pooled for splitting.
Unscoreable worlds are recorded with status `unscoreable_no_test_positives`, never dropped;
a test enforces that all 20 worlds appear for the frozen target.

---

## What We Learned

1. **The frozen exposure feature is mis-constructed, and we can say exactly how.** It
   multiplies a genuinely predictive source-stress term (rank-AUC 0.715, 18/20 worlds) by a
   relative-wealth multiplier that is strongly *anti*-predictive (0.090, 0/20 worlds),
   collapsing the product to chance. This is the mechanistic answer to "why is propagation
   exposure weak", and it is reproducible across 20 worlds.
2. **Repairing the construction does not help,** because the useful component
   (`peer_predicted_stress_mean`) is already a Model B feature. The corrected variant C3 is
   null at every power level. Pre-registered prediction H3-null held.
3. **The evaluation population cannot support the comparison.** 9/20 worlds have any test
   positives, median 0, and PR-AUC has a coefficient of variation of 2.3–4.3 at these counts.
4. **Two apparent wins were false positives** and were killed by multiple-comparison
   correction plus a confirmatory test at 9× the positive count, where one reversed sign.
5. **Peer stress is a slow signal** — association strengthens with horizon out to 12 weeks —
   but never contributes incrementally over Model B.
6. **Calibration degrades ranking** through tie collapse.
7. **One of our own findings was an artefact.** The Q5 structure result was self-conditioning
   inflation and is retracted.

## What We Should NOT Claim

- **Not** that Model C is superior, or that any variant beat Model B. Every variant is null
  after correction and confirmation.
- **Not** that fixing the multiplier would make propagation-aware exposure valuable. We
  tested exactly that (C3) and it is null.
- **Not** that propagation-aware exposure contains no information *in principle*. We showed
  it contains none **incremental to Model B on this population**, which is a narrower claim.
- **Not** that propagation effects are concentrated in particular network structures. That
  finding was circular and is withdrawn.
- **Not** that the frozen M2C conclusion should change. M2D **corroborates** "evidence
  inconclusive" and explains the mechanism behind it.
- **No causal claims.** Peer stress associating with later borrower stress is an association
  in a synthetic world, not evidence that peers cause stress.
- **No generalisation** to real microfinance populations. Everything is specific to the
  current generator assumptions (400 borrowers, 5-member JLGs, 156 weeks, conservative slack).

## Recommended next research direction

Ordered by expected value, none implemented — all would require approval because they
change frozen components:

1. **Raise evaluation power before testing any further representation.** At 42 test positives
   across 20 worlds no representation can be evaluated. Options that do not touch the target:
   many more worlds, or pooled cross-world evaluation with world-level blocking. This is a
   precondition for every other item.
2. **Increase JLG size (5 → 10–15).** With 4 peers, mean/max/count/HHI of peer stress are
   near-collinear, so concentration structure cannot exist. Justified independently of any
   hoped-for result.
3. **Correct `borrower_liability_share`** to the contractual `liability_weight` actually used
   by `mechanics.apply_group_liability_coverage`, or rename it. It currently claims to be a
   liability share while measuring relative wealth. A naming/semantics defect worth fixing on
   its own merits — but note C3 already tested a corrected variant and it was null.
4. **Investigate why PV lands in test splits only 10.7 % of the time** — less than the 19 %
   the window length implies. Likely horizon truncation at the timeline end.

## Reproducing

```bash
python research/m2d/gen_world.py 42 101 202 303 404 505 606 707 808 909 1201 1302 1403 1504 1605 1706 1807 1908 2009 2110
python research/m2d/gen_lineage.py 42 101 202 303 404 505 606 707 808 909 1201 1302 1403 1504 1605 1706 1807 1908 2009 2110
python research/m2d/e1_prevalence.py
python research/m2d/e1b_power.py
python research/m2d/e2_distributions.py
python research/m2d/e2b_inversion.py
python research/m2d/e3_variants.py && python research/m2d/e3_analyse.py
python research/m2d/e4_confirmatory.py
python research/m2d/e5_horizon_structure.py
python research/m2d/e6_circularity.py
python -m pytest tests/
```
World caches are regenerable and gitignored.
