import React, { useCallback, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts';
import {
  SlidersHorizontal, Play, Info, Users, Waves, TriangleAlert, ArrowRight,
} from 'lucide-react';
import { api, canonicalBorrower, ApiError } from './lib/api';
import { currency, count } from './lib/format';
import { toSeries, summarise, INTERVENTIONS, BOUNDS, validateScenario, moneyAxis, allZero } from './lib/scenario';
import {
  PageHeader, Panel, SemanticBadge, EmptyState, ErrorState, LoadingState,
} from './lib/ui';
import { useSelection } from './App';

const tipStyle = { backgroundColor: '#121826', border: '1px solid #2a3547', borderRadius: 6, fontSize: 12 };

function DeltaRow({ label, baseline, scenario, delta, betterWhenLower = true }) {
  const improved = betterWhenLower ? delta < 0 : delta > 0;
  const neutral = Math.abs(delta) < 1e-9;
  return (
    <tr>
      <td>{label}</td>
      <td className="num mono">{currency(baseline, true)}</td>
      <td className="num mono">{currency(scenario, true)}</td>
      <td className="num mono">
        <span className={`badge ${neutral ? 'badge-neutral' : improved ? 'badge-ok' : 'badge-warn'}`}>
          {delta >= 0 ? '+' : ''}{currency(delta, true)}
        </span>
      </td>
    </tr>
  );
}

export default function InterventionStudio() {
  const { borrowerId, setBorrowerId } = useSelection();
  const [form, setForm] = useState({
    borrowerId: borrowerId || 'B10',
    type: 'restructure',
    amount: 500,
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
    const errs = validateScenario({ borrowerId: id, magnitude: form.amount, duration: form.duration });
    setErrors(errs);
    if (Object.keys(errs).length) return;

    setLoading(true);
    setError(null);
    try {
      const r = await api.intervene({
        borrower_id: id,
        intervention_type: form.type,
        amount: Number(form.amount),
        duration_weeks: Number(form.duration),
      });
      setResult(r);
      setBorrowerId(id);
    } catch (err) {
      setError(err);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [form, setBorrowerId]);

  const series = useMemo(() => toSeries(result?.trajectory), [result]);
  const summary = useMemo(() => summarise(series), [series]);
  const dueAxis  = useMemo(() => moneyAxis(series, ['baselineDue', 'scenarioDue']), [series]);
  const cashAxis = useMemo(() => moneyAxis(series, ['baselineCash', 'scenarioCash']), [series]);
  const peerAxis = useMemo(() => moneyAxis(series, ['peerBaselineCash', 'peerScenarioCash']), [series]);
  const noDue    = useMemo(() => allZero(series, ['baselineDue', 'scenarioDue']), [series]);
  const active = INTERVENTIONS.find((i) => i.value === form.type);
  const resultLabel = INTERVENTIONS.find((i) => i.value === result?.scenario_type)?.label || result?.scenario_type;

  return (
    <>
      <PageHeader
        title="Intervention Studio"
        subtitle="Apply a supported intervention to one borrower and compare the modeled counterfactual against the untouched baseline over the same horizon."
      >
        <SemanticBadge kind="counterfactual" />
      </PageHeader>

      <div className="workspace">
        {/* ------------------------ controls ------------------------ */}
        <Panel title="Intervention parameters" icon={SlidersHorizontal} semantic="counterfactual">
          <form onSubmit={run} className="stack-sm">
            <div className="field">
              <label className="field-label" htmlFor="int-borrower">Target borrower</label>
              <input id="int-borrower" className="input input-mono" value={form.borrowerId}
                onChange={set('borrowerId')} placeholder="B10"
                aria-invalid={errors.borrowerId ? 'true' : 'false'} />
              {errors.borrowerId && <span className="field-error">{errors.borrowerId}</span>}
            </div>

            <div className="field">
              <label className="field-label" htmlFor="int-type">Intervention</label>
              <select id="int-type" className="select" value={form.type} onChange={set('type')}>
                {INTERVENTIONS.map((i) => <option key={i.value} value={i.value}>{i.label}</option>)}
              </select>
              <span className="field-hint">{active?.help}</span>
            </div>

            <div className="field">
              <label className="field-label" htmlFor="int-amount">Amount (₹)</label>
              <input id="int-amount" className="input input-mono" type="number"
                min={BOUNDS.magnitude.min} max={BOUNDS.magnitude.max} step={BOUNDS.magnitude.step}
                value={form.amount} onChange={set('amount')}
                aria-invalid={errors.magnitude ? 'true' : 'false'} />
              {errors.magnitude
                ? <span className="field-error">{errors.magnitude}</span>
                : <span className="field-hint">0 – {BOUNDS.magnitude.max.toLocaleString('en-IN')}</span>}
            </div>

            <div className="field">
              <label className="field-label" htmlFor="int-dur">Evaluation horizon (weeks)</label>
              <input id="int-dur" className="input input-mono" type="number"
                min={BOUNDS.duration.min} max={BOUNDS.duration.max}
                value={form.duration} onChange={set('duration')}
                aria-invalid={errors.duration ? 'true' : 'false'} />
              {errors.duration
                ? <span className="field-error">{errors.duration}</span>
                : <span className="field-hint">The intervention is applied once at week 0; this is how long the outcome is observed.</span>}
            </div>

            <button type="submit" className="btn btn-counterfactual btn-block" disabled={loading}
              style={{ marginTop: 6 }}>
              <Play size={14} aria-hidden="true" />
              {loading ? 'Evaluating' : 'Evaluate intervention'}
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
              The intervention is applied to an isolated deep copy of the frozen baseline. Global
              analytical state is never modified and identical inputs return identical results.
            </span>
          </div>
        </Panel>

        {/* ------------------------ results ------------------------ */}
        <div className="stack">
          {loading && <Panel><LoadingState label="Evaluating modeled counterfactual" /></Panel>}

          {!result && !loading && !error && (
            <Panel>
              <EmptyState icon={SlidersHorizontal} title="No intervention evaluated yet"
                message="Select a borrower and a supported intervention, then evaluate it to compare the modeled counterfactual against the baseline." />
            </Panel>
          )}

          {result && !loading && (
            <div className="stack fade-in">
              <Panel semantic="counterfactual">
                <div className="row-between" style={{ flexWrap: 'wrap', gap: 10 }}>
                  <div>
                    <span className="metric-label">Evaluated intervention</span>
                    <div className="metric-secondary">{resultLabel} · {result.borrower_id}</div>
                  </div>
                  <SemanticBadge kind="counterfactual" />
                </div>
                <div className="callout callout-counterfactual" style={{ marginTop: 12 }}>
                  <Info size={14} aria-hidden="true" />
                  <span>
                    <strong>Modeled counterfactual under stated assumptions.</strong> This
                    comparison describes what the deterministic model produces under these inputs. It does not
                    establish that the intervention would produce this outcome for a real borrower, and it
                    makes no claim about preventing default or guaranteeing recovery.
                  </span>
                </div>
              </Panel>

              {/* before / modeled / delta */}
              {summary && (
                <Panel title="Baseline vs modeled scenario" icon={ArrowRight} semantic="counterfactual"
                  note="Delta is modeled scenario minus baseline at the end of the horizon (cumulative for shortfall). A favourable direction is shown in green; it reflects the model's arithmetic, not a prediction about the real world.">
                  <table className="rank-table">
                    <thead>
                      <tr>
                        <th>Measure</th>
                        <th className="num">Baseline</th>
                        <th className="num">Modeled scenario</th>
                        <th className="num">Delta</th>
                      </tr>
                    </thead>
                    <tbody>
                      <DeltaRow label={`Target cash buffer · week ${summary.weeks}`}
                        baseline={summary.cash.baseline} scenario={summary.cash.scenario}
                        delta={summary.cash.delta} betterWhenLower={false} />
                      <DeltaRow label="Cumulative target shortfall"
                        baseline={summary.shortfall.baseline} scenario={summary.shortfall.scenario}
                        delta={summary.shortfall.delta} />
                      {summary.due && (
                        <DeltaRow label={`Amount due · week ${summary.weeks}`}
                          baseline={summary.due.baseline} scenario={summary.due.scenario}
                          delta={summary.due.delta} />
                      )}
                      <DeltaRow label={`Avg peer cash buffer · week ${summary.weeks}`}
                        baseline={summary.peerCash.baseline} scenario={summary.peerCash.scenario}
                        delta={summary.peerCash.delta} betterWhenLower={false} />
                    </tbody>
                  </table>
                </Panel>
              )}

              <Panel title="Target borrower — outstanding amount due" icon={SlidersHorizontal} semantic="counterfactual">
                {noDue ? (
                  <EmptyState icon={Info} title="No outstanding amount due in either arm"
                    message={`This borrower carries no unpaid instalment at any point in the ${series.length}-week horizon, in the baseline or under the intervention, so there is nothing to plot. Compare the cash-buffer panels below instead.`} />
                ) : (
                  <>
                  <div className="chart-frame">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={series} margin={{ top: 6, right: 8, left: 4, bottom: 4 }}>
                      <defs>
                        <linearGradient id="gradIntBase" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#7c8aa5" stopOpacity={0.28} />
                          <stop offset="100%" stopColor="#7c8aa5" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="gradIntScen" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#a97bf5" stopOpacity={0.3} />
                          <stop offset="100%" stopColor="#a97bf5" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                      <XAxis dataKey="label" tickLine={false} axisLine={{ stroke: '#1c2434' }} />
                      <YAxis tickLine={false} axisLine={{ stroke: '#1c2434' }} tickFormatter={dueAxis} width={58} />
                      <Tooltip contentStyle={tipStyle} formatter={(v) => currency(v, true)}
                        labelFormatter={(l) => `Week ${String(l).replace('W', '')}`} />
                      <Legend iconType="plainline" wrapperStyle={{ fontSize: 11.5 }} />
                      <Area type="monotone" dataKey="baselineDue" name="Baseline"
                        stroke="#7c8aa5" strokeWidth={1.8} fill="url(#gradIntBase)" />
                      <Area type="monotone" dataKey="scenarioDue" name="With intervention"
                        stroke="#a97bf5" strokeWidth={2} fill="url(#gradIntScen)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
                <p className="chart-caption">Outstanding amount due in ₹ after each weekly collection.</p>
                  </>
                )}
              </Panel>

              <div className="grid-2">
                <Panel title="Target cash buffer" icon={Waves} semantic="counterfactual">
                  <div className="chart-frame chart-frame--sm">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={series} margin={{ top: 6, right: 8, left: 4, bottom: 4 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                        <XAxis dataKey="label" tickLine={false} axisLine={{ stroke: '#1c2434' }} />
                        <YAxis tickLine={false} axisLine={{ stroke: '#1c2434' }} tickFormatter={cashAxis} width={58} />
                        <Tooltip contentStyle={tipStyle} formatter={(v) => currency(v, true)} />
                        <Legend iconType="plainline" wrapperStyle={{ fontSize: 11.5 }} />
                        <Line type="monotone" dataKey="baselineCash" name="Baseline" stroke="#7c8aa5" strokeWidth={1.8} dot={false} />
                        <Line type="monotone" dataKey="scenarioCash" name="With intervention" stroke="#a97bf5" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </Panel>

                <Panel title="Average peer cash buffer" icon={Users} semantic="counterfactual">
                  <div className="chart-frame chart-frame--sm">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={series} margin={{ top: 6, right: 8, left: 4, bottom: 4 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.1)" />
                        <XAxis dataKey="label" tickLine={false} axisLine={{ stroke: '#1c2434' }} />
                        <YAxis tickLine={false} axisLine={{ stroke: '#1c2434' }} tickFormatter={peerAxis} width={58} />
                        <Tooltip contentStyle={tipStyle} formatter={(v) => currency(v, true)} />
                        <Legend iconType="plainline" wrapperStyle={{ fontSize: 11.5 }} />
                        <Line type="monotone" dataKey="peerBaselineCash" name="Baseline" stroke="#7c8aa5" strokeWidth={1.8} dot={false} />
                        <Line type="monotone" dataKey="peerScenarioCash" name="With intervention" stroke="#a97bf5" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                  <p className="chart-caption">
                    Modeled effect on the rest of the joint-liability group over the same horizon.
                  </p>
                </Panel>
              </div>

              <Panel>
                <div className="action-bar">
                  <button className="btn" onClick={() => { setBorrowerId(result.borrower_id); nav('/simulator'); }}>
                    <Waves size={14} aria-hidden="true" /> Back to shock scenario
                  </button>
                  <button className="btn" onClick={() => { setBorrowerId(result.borrower_id); nav('/lab'); }}>
                    How is this evaluated?
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
