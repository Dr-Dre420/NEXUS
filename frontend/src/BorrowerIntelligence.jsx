import React, { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Search, Wallet, Gauge, Share2, Waves, SlidersHorizontal, Users, Info, UserSearch,
} from 'lucide-react';
import { api, canonicalBorrower } from './lib/api';
import { currency, probability, percentile, indexValue, percent, count } from './lib/format';
import {
  PageHeader, Panel, Metric, DataRow, SemanticBadge, RiskTierBadge, StressBadge,
  AsyncView, SkeletonPanel, EmptyState, Bar,
} from './lib/ui';
import { useSelection } from './App';

export default function BorrowerIntelligence() {
  const { borrowerId, setBorrowerId, setGroupId } = useSelection();
  const [query, setQuery] = useState(borrowerId || 'B10');
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [invalid, setInvalid] = useState(null);
  const nav = useNavigate();

  const load = useCallback((raw) => {
    const id = canonicalBorrower(raw);
    if (!id) {
      setInvalid('Enter a borrower id in the form B10 (or just the number 10).');
      return;
    }
    setInvalid(null);
    setLoading(true);
    setError(null);
    api.borrower(id)
      .then((d) => { setData(d); setBorrowerId(d.borrower_id); setQuery(d.borrower_id); })
      .catch((e) => { setError(e); setData(null); })
      .finally(() => setLoading(false));
  }, [setBorrowerId]);

  // Load whatever the rest of the product has selected.
  useEffect(() => { if (borrowerId) load(borrowerId); }, []); // eslint-disable-line

  const submit = (e) => { e.preventDefault(); load(query); };

  const fin = data?.financial_state;
  const risk = data?.operational_risk;
  const netv = data?.network_evidence;

  return (
    <>
      <PageHeader
        title="Borrower Intelligence"
        subtitle="Borrower dossier from the frozen snapshot: observed financial state, predictive operational risk, and diagnostic network evidence, kept visually distinct."
      >
        <form onSubmit={submit} className="row" role="search">
          <label htmlFor="borrower-id" className="sr-only">Borrower identifier</label>
          <input
            id="borrower-id"
            className="input input-mono"
            style={{ width: 150 }}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="B10"
            aria-invalid={invalid ? 'true' : 'false'}
            aria-describedby={invalid ? 'borrower-id-error' : undefined}
          />
          <button type="submit" className="btn btn-primary" disabled={loading}>
            <Search size={14} aria-hidden="true" />
            {loading ? 'Loading' : 'Inspect'}
          </button>
        </form>
      </PageHeader>

      {invalid && <p id="borrower-id-error" className="field-error" style={{ marginBottom: 12 }}>{invalid}</p>}

      <AsyncView
        loading={loading}
        error={error}
        onRetry={() => load(query)}
        empty={!data && !loading && !error}
        emptyProps={{
          icon: UserSearch,
          title: 'No borrower selected',
          message: 'Enter a borrower identifier such as B10, or pick one from the Command Center risk table.',
        }}
        skeleton={<div className="grid-2"><SkeletonPanel /><SkeletonPanel /></div>}
      >
        {data && (
          <div className="stack fade-in">
            {/* identity strip */}
            <Panel>
              <div className="row-between" style={{ flexWrap: 'wrap' }}>
                <div className="row-wrap">
                  <div>
                    <span className="metric-label">Borrower</span>
                    <div className="metric-secondary">{data.borrower_id}</div>
                  </div>
                  <div style={{ marginLeft: 22 }}>
                    <span className="metric-label">Joint-liability group</span>
                    <div className="metric-secondary">
                      <button className="link-cell" style={{ fontSize: 17 }}
                        onClick={() => { setGroupId(data.group_id); nav('/network'); }}>
                        {data.group_id}
                      </button>
                    </div>
                  </div>
                  <div style={{ marginLeft: 22 }}>
                    <span className="metric-label">Observed state</span>
                    <div style={{ marginTop: 7 }}><StressBadge stressed={fin.current_stress} /></div>
                  </div>
                </div>
                <div className="action-bar">
                  <button className="btn" onClick={() => { setGroupId(data.group_id); nav('/network'); }}>
                    <Share2 size={14} aria-hidden="true" /> View network
                  </button>
                  <button className="btn" onClick={() => { setBorrowerId(data.borrower_id); nav('/simulator'); }}>
                    <Waves size={14} aria-hidden="true" /> Simulate shock
                  </button>
                  <button className="btn" onClick={() => { setBorrowerId(data.borrower_id); nav('/intervene'); }}>
                    <SlidersHorizontal size={14} aria-hidden="true" /> Explore intervention
                  </button>
                </div>
              </div>
            </Panel>

            <div className="grid-2">
              {/* ---------------- financial state ---------------- */}
              <Panel title="Financial state" icon={Wallet}
                aside={<span className="badge badge-neutral">Observed</span>}
                note="Observed values from the frozen snapshot, averaged over the trailing 4 weeks where indicated. These are recorded facts about the synthetic world, not model output.">
                <div className="grid-2" style={{ gap: 14 }}>
                  <Metric label="Cash buffer (4w avg)" value={currency(fin.cash_buffer_mean_4w, true)} />
                  <Metric label="Weekly income (4w avg)" value={currency(fin.weekly_income_mean_4w, true)} />
                </div>
                <div style={{ marginTop: 16 }}>
                  <DataRow label="Weekly expenses (4w avg)" value={currency(fin.weekly_expenses_mean_4w, true)} />
                  <DataRow label="Amount due (4w avg)" value={currency(fin.amount_due_mean_4w, true)} />
                  <DataRow label="Principal remaining" value={currency(fin.principal_remaining, true)} />
                  <DataRow label="Debt burden ratio" value={fin.debt_burden_ratio.toFixed(4)}
                    hint="Amount due relative to income over the feature window." />
                  <DataRow label="Days past due (4w max)" value={count(fin.days_past_due_max_4w)} />
                  <DataRow label="Shortfall (4w avg)" value={currency(fin.shortfall_mean_4w, true)} />
                  <DataRow label="Buffer trend (4w)" value={currency(fin.buffer_trend_4w, true)}
                    hint="Change in cash buffer across the trailing window." />
                </div>
              </Panel>

              {/* ---------------- operational risk ---------------- */}
              <Panel title="Operational risk" icon={Gauge} semantic="predictive"
                aside={<SemanticBadge kind="predictive" />}
                note="This is the only predictive quantity on this page. It is the calibrated Model C probability of propagation vulnerability within the evaluation horizon. It is not a statement that this borrower will default.">
                <Metric
                  label="Operational Risk Score"
                  title="Calibrated Model C output for the selected borrower at the current prediction cutoff."
                  value={probability(risk.score)}
                  semantic="predictive"
                  sub={`Model B comparison baseline: ${probability(risk.baseline_score)}`}
                />
                <div style={{ marginTop: 16 }}>
                  <div className="row-between" style={{ marginBottom: 6 }}>
                    <span className="metric-label">Percentile rank in cohort</span>
                    <RiskTierBadge tier={risk.risk_tier} />
                  </div>
                  <Bar value={risk.risk_percentile} max={100} semantic="predictive"
                    label={`Percentile ${risk.risk_percentile.toFixed(1)} of 100`} />
                  <div className="metric-sub">
                    Rank {percentile(risk.risk_percentile)} of 100 among {count(risk.cohort_size)} borrowers.
                    A percentile is an ordering statistic — it is not a probability and the two are
                    not interchangeable.
                  </div>
                </div>
                <div className="callout callout-warn" style={{ marginTop: 14 }}>
                  <Info size={14} aria-hidden="true" />
                  <span>
                    Calibrated scores are tightly clustered at this base rate, so the percentile is a
                    coarse ordering. Read the probability as the quantity of record.
                  </span>
                </div>
              </Panel>
            </div>

            {/* ---------------- network evidence ---------------- */}
            <Panel title="Network / propagation evidence" icon={Share2} semantic="diagnostic"
              aside={<SemanticBadge kind="diagnostic" />}
              note="Diagnostic evidence only. These quantities describe the borrower's position in the joint-liability structure. They are not predictions, and they do not assert that any specific peer caused or will cause this borrower's stress.">
              <div className="grid-3">
                <Metric
                  label="Modeled propagation exposure"
                  value={indexValue(netv.borrower_propagation_exposure)}
                  semantic="diagnostic"
                  sub="Unitless index. Counterfactual construct, not an observed transfer."
                />
                <Metric
                  label="Peer predicted stress (mean)"
                  value={probability(netv.peer_stress_mean)}
                  semantic="diagnostic"
                  sub="Mean modeled stress probability across the other members of this group."
                />
                <Metric
                  label="Liability"
                  title="Relative share of eligible group liability represented by this connection."
                  value={percent(netv.borrower_liability_share, 2)}
                  semantic="diagnostic"
                  sub="This borrower's cash buffer as a share of the group total. A relative liquidity measure, not a contractual liability percentage."
                />
              </div>
              <div className="callout callout-diagnostic" style={{ marginTop: 16 }}>
                <Info size={14} aria-hidden="true" />
                <span>
                  Propagation exposure is derived from modeled peer stress and group liquidity
                  structure. It indicates where joint-liability transmission capacity exists — it
                  does not represent observed causality between named individuals, and no borrower-to-borrower
                  attribution is exposed anywhere in this product.
                </span>
              </div>
            </Panel>
          </div>
        )}
      </AsyncView>
    </>
  );
}
