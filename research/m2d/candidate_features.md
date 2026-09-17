# M2D CANDIDATE PROPAGATION REPRESENTATIONS

All defined in `research/m2d/features_research.py`. Research only — `src/features_propagation.py` untouched.

**Provenance guarantee (all candidates):** peer stress uses the same out-of-fold Model-A
current-stress estimates as M2C; `liability_weight` is a static contractual parameter fixed at
origination (t=0); rolling windows look backwards only. No hidden lineage, no future outcomes,
no scenario labels. Every value is computable from information available at or before t.

| id | formula | why it should represent propagation | overlapping B features | expected failure mode | measured R² from B |
|---|---|---|---|---|---|
| **P0** (frozen M2C) | `peer_predicted_stress_mean × borrower_liability_share` | baseline under test | both factors ARE B features | no new information by construction | 0.9987 |
| **P1** liability channel | `Σ_{j≠i} s_j w_j / Σ_{j≠i} w_j`, w = contractual liability weight | routes stress through the channel the generator actually uses, instead of a wealth proxy | peer_predicted_stress_mean | w is uniform random and uncorrelated with stress, so the reweighting is near-neutral | 0.9884 |
| **P2** concentration | `max_{j≠i} s_j`, `#{j≠i : s_j > 0.5}`, `Σs_j²/(Σs_j)²` | one severe peer ≠ several mild peers; a mean cannot express this | peer_predicted_stress_mean, expected_peer_debt_burden | only 4 peers per JLG, so max/count/HHI collapse onto the mean | 0.9998 |
| **P3** persistence | `Σ_{k=0..3} 0.6^k · peer_mean_{t-k}`; velocity `peer_mean_t − peer_mean_{t-4}` | a peer stressed several weeks running drains the group pot cumulatively | peer_predicted_stress_mean, peer_dpd_mean_4w | B's 4w/12w rolling peer aggregates already encode duration | 0.9986 / 0.8008 |
| **P4** destination fragility | `buffer/(amount_due+expenses)`; `1/(1+adequacy)` | absolute capacity to absorb a liability call, which B lacks (it has only a relative share) | cash_buffer_mean_*, debt_burden_ratio | a ratio of two B features | 0.9996 / 0.9992 |
| **P5** process (primary) | `P1 × P3_decayed × P4_vulnerability` | the section-6 form: source stress × transmission channel × temporal persistence × destination vulnerability, as one interpretable scalar | peer_predicted_stress_mean | product of three near-redundant terms | 0.9738 |
| **P6** full | all ten research features | upper bound for this family | — | variance inflation from 10 correlated columns | — |

Also exposed: `r_liability_weight` (the raw contractual weight, R² 0.745 from B, |Spearman| to
its closest B feature only 0.067). It is the **only** genuinely new column in the family — but it
is an i.i.d. `uniform(0.1,1.0)` draw, so it is new *noise* rather than new signal unless it
modulates transmission strongly, which it does not at this group size.
