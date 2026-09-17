# M2D RESEARCH CONCLUSION

## Verdict

### EVIDENCE INSUFFICIENT — FURTHER DATA GENERATION / TARGET STUDY REQUIRED

## What was asked

"Can a better propagation-aware representation produce reproducible incremental predictive
information beyond the network-context baseline?" — not "make Model C beat Model B".

## What was done

20 independently generated synthetic worlds (never pooled for splitting): 10 M2C worlds as
RESEARCH VALIDATION, 10 fresh worlds (1201–2110) as RESEARCH HOLDOUT, scored once against a
pre-registered decision rule (`holdout_preregistration.md`, written before any holdout world
was scored). 360 logged experiment rows plus ablation, redundancy and class-imbalance studies.

## Result

**No candidate met the pre-registered criteria.** Primary candidate `B+P5_process` on the
holdout: mean PR-AUC delta **+0.0343** (PASS), wins on **6/10** worlds (FAIL, needed ≥7),
paired Wilcoxon **p = 0.3008** (FAIL, needed <0.05), 184 test positives (PASS).

The positive mean is carried by a single world. Leave-one-world-out:

| dropped world | mean PR delta |
|---|---|
| none (full holdout) | +0.0343 |
| **seed 1807** | **+0.0020** |

Seed 1807 alone contributes +0.3252. Excluding it, the effect is indistinguishable from zero —
**the same single-seed pathology that made the frozen M2C result inconclusive**, now reproduced
on independent worlds with a different candidate. This is the signature of noise, not signal.

On the frozen `propagation_vulnerability` target the holdout was scoreable on only 5/10 worlds
with 20 test positives; every candidate had a *negative* mean PR delta there, and on validation
even Model A "beat" Model B by +0.0617 — an impossible ordering that demonstrates the
comparison is noise-dominated at this sample size.

## Why (mechanism, not speculation)

1. **Redundancy.** The frozen exposure is exactly `peer_predicted_stress_mean × borrower_liability_share`
   (max error 0.0) and both factors are Model B features. RandomForest recovers it from B at R² 0.9987.
   Every alternative representation is likewise ≥0.97 recoverable, because a 5-member JLG gives each
   borrower only 4 peers — mean, max, count and HHI are near-collinear at that cardinality.
2. **Ablation** (`ablation_results.csv`, target rt_new_stress, 10 worlds): propagation features
   *alone* reach PR 0.0055 vs B's 0.2065; added *on top of* B they give −0.0005 PR; substituted
   *for* B's peer-context block they give +0.0121. They re-express information B already holds
   rather than adding any. (`ab4` and `ab5` are identical by construction — B minus its 8 peer
   columns is exactly feature set A.)
3. **Not class imbalance.** Under `low_min_child`, B improves markedly (PR 0.2065 → 0.2756) while
   the propagation increment goes *negative* (−0.0161). Extra learner capacity helps B's existing
   features and does not let propagation features contribute. Under `balanced_weight` the B→B+P5
   median delta stays 0.0000 in every regime. The limitation is informational, not an imbalance artifact.
4. **Target sparsity is upstream of attribution.** Only 0.19% of borrower-weeks are new-stress
   events at all; PV is 0.05%. Relaxing the network-attribution rule moves PV 31→36 (window) or
   31→32 (any contribution). The binding constraint is the event rate in the generator.

## Secondary observation (reported, not claimed)

Network *context* as a block does appear to carry information for new-stress prediction:
B − A median +0.0576 PR, 7/10 validation worlds, Wilcoxon p = 0.2324 — suggestive but
**not statistically significant at n=10**, and it concerns peer context generally, not
propagation-aware exposure specifically.

## Limits of this study

Results are specific to the synthetic generation assumptions (400 borrowers, 5-member JLGs,
156 weeks, conservative slack regime). No causal claim is made. Nothing here generalizes to
real microfinance populations. The observed deltas are reported as observations on these
worlds, not as properties of any model.

## Proposed next steps (NOT implemented — require explicit approval)

These would change the generator or target and were deliberately **not** carried out, since
doing so to make a desired result appear would invalidate the experiment:

1. **Increase JLG size** (5 → 10–15 members). The single highest-value change: concentration
   statistics cannot diverge from the mean with only 4 peers. This is the direct structural
   cause of finding 1, and is defensible independently of any hoped-for result.
2. **Raise propagation event frequency** so a 30-week test window holds ≥100 positives — via
   longer horizons, more worlds, or higher shock frequency. Required for *any* directional
   comparison, whatever the representation.
3. **Correct `borrower_liability_share`** to the contractual `liability_weight` used by
   `mechanics.apply_group_liability_coverage`, or rename the feature. It currently claims to
   be a liability share while measuring relative wealth. This is a naming/semantics defect
   worth fixing on its own merits.
4. **Pooled cross-world evaluation with world-level blocking**, to escape the per-world
   positive-count ceiling without changing the target.

## Product impact

**None.** No research candidate is proposed for integration. Frozen Model C remains the
production operational risk score. The frozen M2C conclusion — "Evidence remains
inconclusive" — is *corroborated* by this study, and the mechanism behind it is now explained.
