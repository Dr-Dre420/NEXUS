import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  FlaskConical, ShieldCheck, Database, BarChart3, Info, TriangleAlert, Scale, Users,
} from 'lucide-react';
import { api } from './lib/api';
import { count, indexValue, percent } from './lib/format';
import {
  PageHeader, Panel, SemanticBadge, DataRow, AsyncView, SkeletonPanel,
} from './lib/ui';

const num = (v, dp = 3) => (Number.isFinite(v) ? v.toFixed(dp) : '—');

export default function ModelImpactLab() {
  const [data, setData] = useState(null);
  const [assumptions, setAssumptions] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const nav = useNavigate();

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    Promise.all([api.evaluation(), api.assumptions().catch(() => null)])
      .then(([e, a]) => { setData(e); setAssumptions(a); })
      .catch(setError)
      .finally(() => setLoading(false));
  }, []);
  useEffect(load, [load]);

  const agg = data?.aggregate_results;
  const seedInfo = agg?.B_PR_AUC;

  // Per-seed scoreability is the heart of the honest story on this page.
  const scoreable = useMemo(() => {
    if (!data?.per_seed_results) return null;
    const rows = data.per_seed_results;
    const ok = rows.filter((r) => r.B_PR_AUC != null);
    return {
      total: rows.length,
      ok: ok.length,
      dropped: rows.filter((r) => r.B_PR_AUC == null).map((r) => r.world_seed),
      positives: ok.reduce((a, r) => a + (r.test_positive_count || 0), 0),
      rows,
    };
  }, [data]);

  const identical = useMemo(() => {
    const a = data?.prediction_identity_audit;
    if (!a) return null;
    return { same: a.filter((e) => e.predictions_identical).length, total: a.length };
  }, [data]);

  return (
    <>
      <PageHeader
        title="Model & Impact Lab"
        subtitle="Transparency panel for the frozen M2C evaluation. Every figure is read from the canonical evaluation artifact; nothing on this page is illustrative."
      >
        {data && <span className="badge badge-neutral">{data.analytics_version}</span>}
        {data?.evaluation_version && <span className="badge badge-neutral">eval {data.evaluation_version}</span>}
      </PageHeader>

      <AsyncView
        loading={loading}
        error={error}
        onRetry={load}
        skeleton={<div className="stack"><SkeletonPanel rows={2} /><div className="grid-2"><SkeletonPanel /><SkeletonPanel /></div></div>}
      >
        {data && (
          <div className="stack fade-in">
            {/* ---------------- 1. RESULT ---------------- */}
            <Panel title="Result" icon={Scale} semantic="predictive"
              aside={<span className="badge badge-warn">Evidence inconclusive</span>}>
              <p style={{ fontSize: 15, lineHeight: 1.6, color: 'var(--text)' }}>
                The evaluation does not establish that the propagation-aware exposure feature adds
                predictive value over the network-context baseline.
              </p>
              <div className="callout callout-warn" style={{ marginTop: 14 }}>
                <TriangleAlert size={15} aria-hidden="true" />
                <span>
                  {scoreable && (
                    <>
                      Only <strong>{scoreable.ok} of {scoreable.total}</strong> evaluation worlds produced a
                      test split containing any positive cases, giving <strong>{count(scoreable.positives)} test
                      positives in total</strong>. The remaining {scoreable.total - scoreable.ok} worlds
                      (seeds {scoreable.dropped.join(', ')}) yield undefined metrics and are excluded from
                      every aggregate below.
                    </>
                  )}
                </span>
              </div>
            </Panel>

            {/* ---------------- 2. WHY ---------------- */}
            <div className="grid-2">
              <Panel title="Why the comparison is inconclusive" icon={Info}>
                <div className="stack-sm">
                  <div className="callout">
                    <span>
                      <strong>1. Too few positive cases.</strong> The propagation-vulnerability target is
                      extremely rare in these synthetic worlds. A comparison resting on
                      {scoreable ? ` ${scoreable.positives} ` : ' a handful of '}
                      positives cannot separate a real effect from sampling noise.
                    </span>
                  </div>
                  <div className="callout">
                    <span>
                      <strong>2. The aggregate is concentrated.</strong> Where a difference does appear, it
                      comes from a small number of worlds rather than holding consistently across them.
                    </span>
                  </div>
                  {identical && (
                    <div className="callout">
                      <span>
                        <strong>3. The models often agree exactly.</strong> Model B and Model C produced
                        identical raw predictions on <strong>{identical.same} of {identical.total}</strong> worlds,
                        meaning the extra feature received no weight from the learner in those runs.
                      </span>
                    </div>
                  )}
                </div>
              </Panel>

              <Panel title="Model comparison" icon={BarChart3} semantic="predictive"
                aside={<SemanticBadge kind="predictive" />}
                note={seedInfo ? `Aggregated over ${seedInfo.n_seeds_contributing} of ${seedInfo.n_seeds_total} worlds — the ${seedInfo.n_seeds_dropped_no_test_positives} worlds without test positives are excluded.` : undefined}>
                <div className="table-scroll">
                  <table className="rank-table">
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th className="num">ROC-AUC</th>
                        <th className="num">PR-AUC</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(data.models || {}).map(([name, m]) => (
                        <tr key={name} style={{ cursor: 'default' }}>
                          <td>{name}</td>
                          <td className="num mono">{num(m.roc_auc)}</td>
                          <td className="num mono">{num(m.pr_auc)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="panel-note">
                  Model B is the network-context baseline ({data.B_feature_list?.length ?? '—'} features).
                  Model C adds the propagation-aware exposure feature ({data.C_feature_list?.length ?? '—'} features).
                  PR-AUC is the appropriate headline metric at this base rate; ROC-AUC is inflated by class imbalance.
                </p>
              </Panel>
            </div>

            {/* ---------------- 3. EVIDENCE ---------------- */}
            <Panel title="Evidence — per-world detail" icon={Database}
              note="One row per independently generated world. Worlds with no positive test cases cannot produce a metric and are shown as unscoreable rather than as a zero.">
              <div className="table-scroll">
                <table className="rank-table">
                  <thead>
                    <tr>
                      <th>World seed</th>
                      <th className="num">Test positives</th>
                      <th className="num">Model B PR-AUC</th>
                      <th className="num">Model C PR-AUC</th>
                      <th className="num hide-sm">Delta</th>
                      <th className="num">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {scoreable?.rows.map((r) => {
                      const ok = r.B_PR_AUC != null;
                      const d = r.C_raw_minus_B_PR_AUC;
                      return (
                        <tr key={r.world_seed} style={{ cursor: 'default' }}>
                          <td className="mono">{r.world_seed}</td>
                          <td className="num mono">{r.test_positive_count ?? 0}</td>
                          <td className="num mono">{ok ? num(r.B_PR_AUC, 4) : '—'}</td>
                          <td className="num mono">{ok ? num(r.C_raw_PR_AUC, 4) : '—'}</td>
                          <td className="num mono hide-sm">
                            {d == null ? '—' : (
                              <span className={`badge ${Math.abs(d) < 1e-9 ? 'badge-neutral' : d > 0 ? 'badge-ok' : 'badge-warn'}`}>
                                {d >= 0 ? '+' : ''}{num(d, 4)}
                              </span>
                            )}
                          </td>
                          <td className="num">
                            {ok ? <span className="badge badge-ok">Scoreable</span>
                                : <span className="badge badge-neutral">No test positives</span>}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Panel>

            <div className="grid-2">
              <Panel title="Dataset & evaluation protocol" icon={Database}>
                <DataRow label="Evaluation worlds" value={count(data.world_seeds?.length)} />
                <DataRow label="Episode-lineage rows" value={count(data.target_counts?.total_episodes)} />
                <DataRow label="Positive target events" value={count(data.target_counts?.pv_positive_events)} />
                <DataRow label="Contribution threshold" value={data.target_counts?.pv_threshold} />
                <DataRow label="Cross-world variation" value={data.per_seed_variation} />
                {data.temporal_safeguards && (
                  <>
                    <DataRow label="Purge gap" value={`${data.temporal_safeguards.purge_gap_weeks} weeks`} />
                    <DataRow label="Chronological split" value={data.temporal_safeguards.chronological_split ? 'Enforced' : 'No'} />
                    <DataRow label="Worlds pooled for splitting" value={data.temporal_safeguards.no_global_world_concatenation ? 'No — each world split independently' : 'Yes'} />
                  </>
                )}
              </Panel>

              <Panel title="Safeguards & calibration" icon={ShieldCheck}>
                {data.calibration_metadata && (
                  <>
                    <DataRow label="Calibrator" value={data.calibration_metadata.calibrator_type?.split(' (')[0]} />
                    <DataRow label="Fitted on" value={data.calibration_metadata.fitted_on} />
                    <DataRow label="Test labels used" value={data.calibration_metadata.test_labels_used ? 'Yes' : 'No'} />
                    <DataRow label="Frozen before test" value={data.calibration_metadata.frozen_before_test ? 'Yes' : 'No'} />
                  </>
                )}
                {data.leakage_checks && (
                  <>
                    <DataRow label="Hidden lineage in features" value={data.leakage_checks.hidden_lineage_excluded_from_features ? 'Excluded' : 'Present'} />
                    <DataRow label="Scenario labels in features" value={data.leakage_checks.scenario_labels_excluded_from_features ? 'Excluded' : 'Present'} />
                  </>
                )}
                <p className="panel-note">
                  {data.attribution_statistics && (
                    <>Attribution basis: {data.attribution_statistics.attribution_basis}. </>
                  )}
                  Hidden synthetic lineage is used for evaluation only and is never exposed through
                  the product API or any page in this application.
                </p>
              </Panel>
            </div>

            {/* ---------------- 4. LIMITATIONS ---------------- */}
            <Panel title="Limitations" icon={TriangleAlert} semantic="diagnostic">
              <div className="stack-sm">
                {(data.results || []).slice(1).map((r, i) => (
                  <div key={i} className="callout callout-diagnostic">
                    <Info size={14} aria-hidden="true" />
                    <span>{r}</span>
                  </div>
                ))}
                <div className="callout callout-warn">
                  <TriangleAlert size={14} aria-hidden="true" />
                  <span>
                    All results are produced on synthetic data generated under the current NEXUS
                    assumptions. They are specific to those assumptions, do not generalize to real
                    microfinance populations, and establish no causal relationship between network
                    structure and borrower stress.
                  </span>
                </div>
                {assumptions?.synthetic_data_disclaimer && (
                  <div className="callout">
                    <Info size={14} aria-hidden="true" />
                    <span>{assumptions.synthetic_data_disclaimer}</span>
                  </div>
                )}
              </div>
              <details style={{ marginTop: 14 }}>
                <summary style={{ cursor: 'pointer', fontSize: 12, color: 'var(--text-3)' }}>
                  Full evaluation conclusion (verbatim from the canonical artifact)
                </summary>
                <p className="panel-note" style={{ marginTop: 10 }}>
                  {data.final_scientific_conclusion}
                </p>
              </details>
            </Panel>

            <Panel>
              <div className="action-bar">
                <button className="btn" onClick={() => nav('/command-center')}>
                  <Users size={14} aria-hidden="true" /> Back to Portfolio
                </button>
              </div>
            </Panel>
          </div>
        )}
      </AsyncView>
    </>
  );
}
