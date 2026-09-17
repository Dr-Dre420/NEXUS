import React, { useCallback, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import { Waves, Play, Info, SlidersHorizontal, Users, TriangleAlert } from 'lucide-react';
import { api, canonicalBorrower, ApiError } from './lib/api';
import { currency, count } from './lib/format';
import { toSeries, summarise, SHOCK_TYPES, BOUNDS, validateScenario, moneyAxis, allZero } from './lib/scenario';
import {
  PageHeader, Panel, SemanticBadge, EmptyState, ErrorState, LoadingState, DataRow,
} from './lib/ui';
import { useSelection } from './App';

const tipStyle = {
  backgroundColor: '#121826', border: '1px solid #2a3547',
  borderRadius: 6, fontSize: 12,
};

export default function RippleSimulator() {
  const { borrowerId, setBorrowerId } = useSelection();
  const [form, setForm] = useState({
    borrowerId: borrowerId || 'B10',
    shockType: 'income_reduction',
    magnitude: 500,
    duration: 12,
  });
  const [errors, setErrors] = useState({});
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const nav = useNavigate();

  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }));

  const run = useCallback(async (e) => {
    e.preventDefault();
    const id = canonicalBorrower(form.borrowerId);
    const errs = validateScenario({ borrowerId: id, magnitude: form.magnitude, duration: form.duration });
    setErrors(errs);
    if (Object.keys(errs).length) return;

    setLoading(true);
    setError(null);
    try {
      const r = await api.simulate({
        borrower_id: id,
        shock_magnitude: Number(form.magnitude),
        shock_duration_weeks: Number(form.duration),
        shock_type: form.shockType,
      });
      setResult(r);
      setBorrowerId(id);
    } catch (err) {
      setError(err);
      setResult(null);          // never leave a stale scenario on screen
    } finally {
      setLoading(false);
    }
  }, [form, setBorrowerId]);

  const series = useMemo(() => toSeries(result?.trajectory), [result]);
  const summary = useMemo(() => summarise(series), [series]);
  const cashAxis = useMemo(() => moneyAxis(series, ['baselineCash', 'scenarioCash']), [series]);
  const peerAxis = useMemo(() => moneyAxis(series, ['peerBaselineCash', 'peerScenarioCash']), [series]);
  const sfAxis   = useMemo(() => moneyAxis(series, ['baselineShortfall', 'scenarioShortfall']), [series]);
  const noShortfall = useMemo(() => allZero(series, ['baselineShortfall', 'scenarioShortfall']), [series]);
  const shockHelp = SHOCK_TYPES.find((s) => s.value === form.shockType)?.help;

  return (
    <>
      <PageHeader
        title="Ripple Simulator"
        subtitle="Explore how a defined financial shock changes the modeled borrower and network state."
      >
        <SemanticBadge kind="counterfactual" />
      </PageHeader>

      <div className="workspace">
        {/* ------------------------ controls ------------------------ */}
        <Panel title="Scenario parameters" icon={SlidersHorizontal} semantic="counterfactual">
          <form onSubmit={run} className="stack-sm">
            <div className="field">
              <label className="field-label" htmlFor="sim-borrower">Target borrower</label>
              <input id="sim-borrower" className="input input-mono" value={form.borrowerId}
                onChange={set('borrowerId')} placeholder="B10"
                aria-invalid={errors.borrowerId ? 'true' : 'false'} />
              {errors.borrowerId && <span className="field-error">{errors.borrowerId}</span>}
            </div>

            <div className="field">
              <label className="field-label" htmlFor="sim-type">Shock type</label>
              <select id="sim-type" className="select" value={form.shockType} onChange={set('shockType')}>
                {SHOCK_TYPES.map((s) => <option key={s.value} value={s.value}>{s.label}</option>)}
              </select>
              <span className="field-hint">{shockHelp}</span>
            </div>

            <div className="field">
              <label className="field-label" htmlFor="sim-mag">Magnitude (₹)</label>
              <input id="sim-mag" className="input input-mono" type="number"
                min={BOUNDS.magnitude.min} max={BOUNDS.magnitude.max} step={BOUNDS.magnitude.step}
                value={form.magnitude} onChange={set('magnitude')}
                aria-invalid={errors.magnitude ? 'true' : 'false'} />
              {errors.magnitude
                ? <span className="field-error">{errors.magnitude}</span>
                : <span className="field-hint">0 – {BOUNDS.magnitude.max.toLocaleString('en-IN')}</span>}
            </div>

            <div className="field">
              <label className="field-label" htmlFor="sim-dur">Horizon (weeks)</label>
              <input id="sim-dur" className="input input-mono" type="number"
                min={BOUNDS.duration.min} max={BOUNDS.duration.max}
                value={form.duration} onChange={set('duration')}
                aria-invalid={errors.duration ? 'true' : 'false'} />
              {errors.duration
                ? <span className="field-error">{errors.duration}</span>
                : <span className="field-hint">{BOUNDS.duration.min} – {BOUNDS.duration.max} weeks</span>}
            </div>

            <button type="submit" className="btn btn-counterfactual btn-block" disabled={loading}
              style={{ marginTop: 6 }}>
              <Play size={14} aria-hidden="true" />
              {loading ? 'Running scenario' : 'Run scenario'}
            </button>
          </form>

          {error && (
            <div className="callout callout-danger" style={{ marginTop: 12 }} role="alert">
              <TriangleAlert size={14} aria-hidden="true" />
              <span>{error instanceof ApiError ? error.message : String(error)}</span>
            </div>
          )}

          <div className="callout callout-counterfactual" style={{ marginTop: 12 }}>
            <Info size={14} aria-hidden="true" />
            <span>
              Both arms run identical deterministic mechanics on independent deep copies of the
              frozen baseline. The baseline state is never mutated, and repeating a scenario with
              the same inputs returns the same result.
            </span>
          </div>
        </Panel>

        {/* ------------------------ results ------------------------ */}
        <div className="stack">
          {loading && <Panel><LoadingState label="Running deterministic scenario" /></Panel>}

          {!result && !loading && !error && (
            <Panel>
              <EmptyState icon={Waves} title="No scenario run yet"
                message="Choose a borrower, configure a shock and run the scenario to see the modeled trajectory against the unshocked baseline." />
            </Panel>
          )}

          {result && !loading && (
            <div className="stack fade-in">
              <Panel semantic="counterfactual">
                <div className="row-between" style={{ flexWrap: 'wrap', gap: 10 }}>
                  <div>
                    <span className="metric-label">Scenario</span>
                    <div className="metric-secondary">
                      {SHOCK_TYPES.find((s) => s.value === result.scenario_type)?.label || result.scenario_type}
                      {' · '}{result.borrower_id}
                    </div>
                  </div>
                  <SemanticBadge kind="counterfactual" />
                </div>
                <div className="callout callout-counterfactual" style={{ marginTop: 12 }}>
                  <Info size={14} aria-hidden="true" />
                  <span><strong>Scenario simulation — not a guaranteed forecast.</strong> Outputs are modeled downstream effects under strict deterministic assumptions.</span>
                </div>
              </Panel>

              {summary && (
                <div className="grid-3">
                  <div className="metric-card">
                    <span className="metric-label">Target cash buffer · week {summary.weeks}</span>
                    <div className="metric-primary is-counterfactual">{currency(summary.cash.scenario)}</div>
                    <div className="metric-sub">
                      Baseline {currency(summary.cash.baseline)} · delta {summary.cash.delta >= 0 ? '+' : ''}{currency(summary.cash.delta)}
                    </div>
                  </div>
                  <div className="metric-card" title="Cumulative downstream shortfall measured over the defined scenario horizon.">
                    <span className="metric-label">Downstream Burden</span>
                    <div className="metric-primary is-counterfactual">{currency(summary.shortfall.scenario)}</div>
                    <div className="metric-sub">Baseline {currency(summary.shortfall.baseline)}</div>
                  </div>
                  <div className="metric-card">
                    <span className="metric-label">Avg peer cash buffer · week {summary.weeks}</span>
                    <div className="metric-primary is-counterfactual">{currency(summary.peerCash.scenario)}</div>
                    <div className="metric-sub">
                      Across {count(summary.peerCount)} peers · baseline {currency(summary.peerCash.baseline)}
                    </div>
                  </div>
                </div>
              )}

              <Panel title="Target borrower — cash buffer trajectory" icon={Waves} semantic="counterfactual">
                <div className="chart-frame">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={series} margin={{ top: 6, right: 8, left: 4, bottom: 4 }}>
                      <defs>
                        <linearGradient id="gradBase" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#7c8aa5" stopOpacity={0.28} />
                          <stop offset="100%" stopColor="#7c8aa5" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="gradScen" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#a97bf5" stopOpacity={0.3} />
                          <stop offset="100%" stopColor="#a97bf5" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                      <XAxis dataKey="label" tickLine={false} axisLine={{ stroke: '#1c2434' }} />
                      <YAxis tickLine={false} axisLine={{ stroke: '#1c2434' }} tickFormatter={cashAxis} width={58} />
                      <Tooltip contentStyle={tipStyle} formatter={(v) => currency(v, true)}
                        labelFormatter={(l) => `Week ${String(l).replace('W', '')}`} />
                      <Legend iconType="plainline" wrapperStyle={{ fontSize: 11.5 }} />
                      <Area type="monotone" dataKey="baselineCash" name="Baseline (no shock)"
                        stroke="#7c8aa5" strokeWidth={1.8} fill="url(#gradBase)" />
                      <Area type="monotone" dataKey="scenarioCash" name="Scenario (with shock)"
                        stroke="#a97bf5" strokeWidth={2} fill="url(#gradScen)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <p className="chart-caption">
                  Cash buffer in ₹ for the target borrower each week. The scenario line diverges from
                  the baseline only through the modeled shock.
                </p>
              </Panel>

              <div className="grid-2">
                <Panel title="Downstream — average peer cash buffer" icon={Users} semantic="counterfactual">
                  <div className="chart-frame--sm chart-frame">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={series} margin={{ top: 6, right: 8, left: 4, bottom: 4 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                        <XAxis dataKey="label" tickLine={false} axisLine={{ stroke: '#1c2434' }} />
                        <YAxis tickLine={false} axisLine={{ stroke: '#1c2434' }} tickFormatter={peerAxis} width={58} />
                        <Tooltip contentStyle={tipStyle} formatter={(v) => currency(v, true)} />
                        <Legend iconType="plainline" wrapperStyle={{ fontSize: 11.5 }} />
                        <Line type="monotone" dataKey="peerBaselineCash" name="Baseline" stroke="#7c8aa5" strokeWidth={1.8} dot={false} />
                        <Line type="monotone" dataKey="peerScenarioCash" name="Scenario" stroke="#a97bf5" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <p className="chart-caption">
                    Under the modeled joint-liability mechanism, healthy members contribute cash toward
                    a member's shortfall, which draws down their own buffers.
                  </p>
                </Panel>

                <Panel title="Target shortfall per week" icon={TriangleAlert} semantic="counterfactual">
                  {noShortfall ? (
                    <EmptyState icon={TriangleAlert} title="No shortfall in either arm"
                      message={`Over this ${series.length}-week horizon the target borrower meets every scheduled repayment in both the baseline and the scenario, so there is nothing to plot.`} />
                  ) : (
                    <>
                      <div className="chart-frame--sm chart-frame">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={series} margin={{ top: 6, right: 8, left: 4, bottom: 4 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                            <XAxis dataKey="label" tickLine={false} axisLine={{ stroke: '#1c2434' }} />
                            <YAxis tickLine={false} axisLine={{ stroke: '#1c2434' }} tickFormatter={sfAxis} width={58} />
                            <Tooltip contentStyle={tipStyle} formatter={(v) => currency(v, true)} />
                            <Legend iconType="plainline" wrapperStyle={{ fontSize: 11.5 }} />
                            <Line type="monotone" dataKey="baselineShortfall" name="Baseline" stroke="#7c8aa5" strokeWidth={1.8} dot={false} />
                            <Line type="monotone" dataKey="scenarioShortfall" name="Scenario" stroke="#f0656f" strokeWidth={2} dot={false} />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                      <p className="chart-caption">Unpaid amount remaining after each weekly collection.</p>
                    </>
                  )}
                </Panel>
              </div>

              <Panel>
                <div className="action-bar">
                  <button className="btn" onClick={() => { setBorrowerId(result.borrower_id); nav('/intervene'); }}>
                    <SlidersHorizontal size={14} aria-hidden="true" /> Explore an intervention for {result.borrower_id}
                  </button>
                  <button className="btn" onClick={() => { setBorrowerId(result.borrower_id); nav('/borrower'); }}>
                    <Users size={14} aria-hidden="true" /> Back to dossier
                  </button>
                </div>
              </Panel>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
