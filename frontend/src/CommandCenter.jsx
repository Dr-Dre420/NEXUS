import React, { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity, Share2, TrendingUp, Users, Layers, Info } from 'lucide-react';
import { api } from './lib/api';
import { currency, count, indexValue, percent, probability, percentile } from './lib/format';
import {
  PageHeader, Panel, MetricCard, Bar, SemanticBadge, RiskTierBadge,
  AsyncView, SkeletonPanel,
} from './lib/ui';
import { useSelection } from './App';

export default function CommandCenter() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const nav = useNavigate();
  const { setBorrowerId, setGroupId } = useSelection();

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api.portfolioSummary()
      .then(setData)
      .catch(setError)
      .finally(() => setLoading(false));
  }, []);

  useEffect(load, [load]);

  const openBorrower = (id) => { setBorrowerId(id); nav('/borrower'); };
  const openGroup = (id) => { setGroupId(id); nav('/network'); };

  const risk = data?.operational_risk;
  const net = data?.network_evidence;
  const maxBand = data ? Math.max(...data.risk_distribution.map((b) => b.count), 1) : 1;

  return (
    <>
      <PageHeader
        title="Command Center"
        subtitle="Portfolio-wide operational risk and joint-liability network exposure, read from the frozen Seed 909 snapshot. Predictive model output and diagnostic network evidence are reported separately and never combined into a single score."
      >
        {data && (
          <>
            <span className="badge badge-neutral">Seed {data.world_seed}</span>
            <span className="badge badge-neutral">{data.analytics_version}</span>
            <span className="badge badge-neutral">As-of week {data.as_of_week}</span>
          </>
        )}
      </PageHeader>

      <AsyncView
        loading={loading}
        error={error}
        onRetry={load}
        skeleton={<div className="grid-4">{[0, 1, 2, 3].map((i) => <SkeletonPanel key={i} rows={1} />)}</div>}
      >
        {data && (
          <div className="stack fade-in">
            {/* ---------------- primary metrics ---------------- */}
            <div className="grid-4">
              <MetricCard
                label="Portfolio cohort"
                value={count(data.total_borrowers_in_cohort)}
                sub={`${count(net.total_groups)} joint-liability groups`}
                title="Borrowers present in the frozen snapshot at the as-of week."
              />
              <MetricCard
                label="Currently stressed"
                value={count(data.currently_stressed_borrowers)}
                sub={`${percent(data.currently_stressed_borrowers / data.total_borrowers_in_cohort, 1)} of cohort — observed state, not a prediction`}
              />
              <MetricCard
                label="Elevated risk band"
                value={count(risk.tier_counts.Elevated)}
                sub="Top 5% by calibrated Model C score"
                semantic="predictive"
                badge="predictive"
                title={risk.definition}
              />
              <MetricCard
                label="Network-exposed borrowers"
                value={count(net.borrowers_with_nonzero_exposure)}
                sub={`${count(net.groups_with_stressed_members)} groups contain a stressed member`}
                semantic="diagnostic"
                badge="diagnostic"
                title={net.definition}
              />
            </div>

            {/* ---------------- main visual area ---------------- */}
            <div className="grid-2">
              <Panel
                title="Operational risk distribution"
                icon={TrendingUp}
                semantic="predictive"
                aside={<SemanticBadge kind="predictive" />}
                note={risk.note}
              >
                <div className="stack-sm">
                  {data.risk_distribution.map((b) => (
                    <div className="dist-row" key={b.band}>
                      <span className="dist-row-label">p{b.band}</span>
                      <Bar value={b.count} max={maxBand} semantic="predictive" label={`${b.count} borrowers in percentile band ${b.band}`} />
                      <span className="dist-row-count">{count(b.count)}</span>
                    </div>
                  ))}
                </div>
                <div style={{ marginTop: 14 }}>
                  <div className="data-row">
                    <span className="data-row-label">Median score (probability)</span>
                    <span className="data-row-value">{probability(risk.median)}</span>
                  </div>
                  <div className="data-row">
                    <span className="data-row-label">Maximum score (probability)</span>
                    <span className="data-row-value">{probability(risk.max)}</span>
                  </div>
                  <div className="data-row">
                    <span className="data-row-label">Distinct score values in cohort</span>
                    <span className="data-row-value">{count(risk.distinct_values)}</span>
                  </div>
                </div>
                <div className="callout callout-warn" style={{ marginTop: 12 }}>
                  <Info size={14} aria-hidden="true" />
                  <span>
                    The calibrated score takes only {risk.distinct_values} distinct values across
                    the cohort, so percentile bands are coarse and some bands are empty. This is a
                    property of the frozen model at this base rate, reported rather than smoothed.
                  </span>
                </div>
              </Panel>

              <Panel
                title="Network exposure concentration"
                icon={Share2}
                semantic="diagnostic"
                aside={<SemanticBadge kind="diagnostic" />}
                note="Propagation exposure is a modeled diagnostic derived from joint-liability structure. It describes where transmission capacity concentrates — it does not predict default and does not assert causality between individuals."
              >
                <div className="grid-2" style={{ gap: 12 }}>
                  <div>
                    <span className="metric-label">Exposure (sum)</span>
                    <div className="metric-secondary">{indexValue(net.exposure_sum)}</div>
                    <div className="metric-sub">Unitless index, summed across cohort</div>
                  </div>
                  <div>
                    <span className="metric-label">Mean exposure</span>
                    <div className="metric-secondary">{indexValue(net.exposure_mean)}</div>
                    <div className="metric-sub">Median {indexValue(net.exposure_median)}</div>
                  </div>
                  <div>
                    <span className="metric-label">Top decile share</span>
                    <div className="metric-secondary">{percent(net.top_decile_share_of_exposure, 1)}</div>
                    <div className="metric-sub">Of total exposure index</div>
                  </div>
                  <div>
                    <span className="metric-label">Maximum exposure</span>
                    <div className="metric-secondary">{indexValue(net.exposure_max)}</div>
                    <div className="metric-sub">Highest single borrower</div>
                  </div>
                </div>

                <div className="panel-head" style={{ marginTop: 18, marginBottom: 10 }}>
                  <h3 className="panel-title"><Layers size={13} aria-hidden="true" /> Highest-exposure groups</h3>
                </div>
                <div className="table-scroll">
                  <table className="rank-table">
                    <thead>
                      <tr>
                        <th>Group</th>
                        <th className="num">Exposure (sum)</th>
                        <th className="num hide-sm">Stressed</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.top_exposure_groups.slice(0, 5).map((g) => (
                        <tr key={g.group_id} onClick={() => openGroup(g.group_id)} tabIndex={0}
                            onKeyDown={(e) => e.key === 'Enter' && openGroup(g.group_id)}>
                          <td><span className="link-cell">{g.group_id}</span></td>
                          <td className="num mono">{indexValue(g.aggregate_exposure)}</td>
                          <td className="num mono hide-sm">{g.stressed_members}/{g.member_count}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>
            </div>

            {/* ---------------- action area ---------------- */}
            <Panel
              title="Highest operational risk — open a borrower to investigate"
              icon={Users}
              aside={<SemanticBadge kind="predictive" />}
            >
              <div className="table-scroll">
                <table className="rank-table">
                  <thead>
                    <tr>
                      <th>Borrower</th>
                      <th>Group</th>
                      <th className="num">Risk score</th>
                      <th className="num hide-sm">Percentile</th>
                      <th className="num">Tier</th>
                      <th className="num hide-sm">Exposure</th>
                      <th className="num hide-sm">State</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.top_risk_borrowers.map((b) => (
                      <tr key={b.borrower_id} onClick={() => openBorrower(b.borrower_id)} tabIndex={0}
                          onKeyDown={(e) => e.key === 'Enter' && openBorrower(b.borrower_id)}>
                        <td><span className="link-cell">{b.borrower_id}</span></td>
                        <td className="mono text-dim">{b.group_id}</td>
                        <td className="num mono">{probability(b.operational_risk_score)}</td>
                        <td className="num mono hide-sm">{percentile(b.risk_percentile)}</td>
                        <td className="num"><RiskTierBadge tier={b.risk_tier} /></td>
                        <td className="num mono hide-sm">{indexValue(b.borrower_propagation_exposure)}</td>
                        <td className="num hide-sm">
                          {b.current_stress
                            ? <span className="badge badge-danger">Stressed</span>
                            : <span className="badge badge-ok">OK</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="panel-note">
                Risk score is a calibrated probability; percentile is a descriptive rank within
                this cohort of {count(data.total_borrowers_in_cohort)}. The two are different
                quantities and are never shown interchangeably. Select any row to open the full
                borrower dossier.
              </p>
            </Panel>
          </div>
        )}
      </AsyncView>
    </>
  );
}
