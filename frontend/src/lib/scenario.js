/* Shared scenario-trajectory shaping for the Ripple Simulator and Intervention
   Studio. Previously duplicated verbatim in both pages. */

export function toSeries(trajectory) {
  if (!Array.isArray(trajectory)) return [];
  return trajectory.map((w) => {
    const peers = Object.values(w.group_members || {});
    const n = peers.length || 1;
    const sum = (sel) => peers.reduce((a, m) => a + sel(m), 0);
    return {
      week: w.week,
      label: `W${w.week}`,
      baselineCash: w.target_borrower.baseline.cash_buffer,
      scenarioCash: w.target_borrower.scenario.cash_buffer,
      baselineShortfall: w.target_borrower.baseline.shortfall,
      scenarioShortfall: w.target_borrower.scenario.shortfall,
      baselineDue: w.target_borrower.baseline.amount_due ?? null,
      scenarioDue: w.target_borrower.scenario.amount_due ?? null,
      baselineCoverage: w.target_borrower.baseline.group_coverage_used ?? 0,
      scenarioCoverage: w.target_borrower.scenario.group_coverage_used ?? 0,
      peerBaselineCash: peers.length ? sum((m) => m.baseline.cash_buffer) / n : 0,
      peerScenarioCash: peers.length ? sum((m) => m.scenario.cash_buffer) / n : 0,
      peerBaselineShortfall: peers.length ? sum((m) => m.baseline.shortfall) / n : 0,
      peerScenarioShortfall: peers.length ? sum((m) => m.scenario.shortfall) / n : 0,
      peerCount: peers.length,
    };
  });
}

/** Terminal-week deltas, used for the before / modeled / delta summary. */
export function summarise(series) {
  if (!series.length) return null;
  const last = series[series.length - 1];
  const d = (a, b) => b - a;
  return {
    weeks: series.length,
    peerCount: last.peerCount,
    cash: { baseline: last.baselineCash, scenario: last.scenarioCash, delta: d(last.baselineCash, last.scenarioCash) },
    shortfall: {
      baseline: series.reduce((a, r) => a + r.baselineShortfall, 0),
      scenario: series.reduce((a, r) => a + r.scenarioShortfall, 0),
      get delta() { return this.scenario - this.baseline; },
    },
    peerCash: { baseline: last.peerBaselineCash, scenario: last.peerScenarioCash, delta: d(last.peerBaselineCash, last.peerScenarioCash) },
    due: last.baselineDue == null ? null
      : { baseline: last.baselineDue, scenario: last.scenarioDue, delta: d(last.baselineDue, last.scenarioDue) },
  };
}

export const SHOCK_TYPES = [
  { value: 'income_reduction', label: 'Income reduction (weekly)', help: 'Reduces the household weekly income by the stated amount.' },
  { value: 'expense_increase', label: 'Expense increase (weekly)', help: 'Increases the household weekly expenses by the stated amount.' },
  { value: 'cash_shock',       label: 'Immediate cash shock',      help: 'Removes the stated amount from the household cash buffer once, at week 0.' },
];

export const INTERVENTIONS = [
  {
    value: 'restructure',
    label: 'Restructure — reduce weekly instalment',
    help: 'Lowers the weekly instalment by the stated amount for the remainder of the horizon. The modeled mechanic reduces the instalment only; it does not re-amortise principal over a longer term.',
  },
  {
    value: 'cash_injection',
    label: 'Cash injection — increase buffer',
    help: 'Adds the stated amount to the household cash buffer once, at week 0.',
  },
  {
    value: 'payment_adjustment',
    label: 'Payment adjustment — reduce amount due',
    help: 'Reduces the outstanding amount due by the stated amount. The modeled mechanic adjusts amount due only; principal remaining is not written down.',
  },
];

export const BOUNDS = {
  magnitude: { min: 0, max: 100000, step: 50 },
  duration: { min: 1, max: 52, step: 1 },
};

export function validateScenario({ borrowerId, magnitude, duration }) {
  const errors = {};
  if (!borrowerId) errors.borrowerId = 'Enter a borrower id such as B10.';
  const m = Number(magnitude);
  if (!Number.isFinite(m)) errors.magnitude = 'Enter a number.';
  else if (m < BOUNDS.magnitude.min) errors.magnitude = 'Magnitude cannot be negative.';
  else if (m > BOUNDS.magnitude.max) errors.magnitude = `Maximum is ${BOUNDS.magnitude.max.toLocaleString('en-IN')}.`;
  const d = Number(duration);
  if (!Number.isInteger(d)) errors.duration = 'Enter a whole number of weeks.';
  else if (d < BOUNDS.duration.min || d > BOUNDS.duration.max) {
    errors.duration = `Choose between ${BOUNDS.duration.min} and ${BOUNDS.duration.max} weeks.`;
  }
  return errors;
}

/** Axis tick formatter that adapts to the magnitude actually present. */
export function moneyAxis(series, keys) {
  const vals = series.flatMap((r) => keys.map((k) => r[k]).filter(Number.isFinite));
  const max = vals.length ? Math.max(...vals.map(Math.abs)) : 0;
  if (max >= 10000) return (v) => `₹${Math.round(v / 1000)}k`;
  if (max >= 100)   return (v) => `₹${Math.round(v)}`;
  return (v) => `₹${v.toFixed(0)}`;
}

/** True when every point in every named series is (effectively) zero. */
export function allZero(series, keys) {
  return series.length > 0 && series.every((r) =>
    keys.every((k) => !Number.isFinite(r[k]) || Math.abs(r[k]) < 1e-9));
}
