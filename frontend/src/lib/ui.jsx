import React from 'react';
import { AlertTriangle, Inbox, Loader2, RefreshCw } from 'lucide-react';

/* --------------------------------------------------------------------------
   Semantic vocabulary.
   PREDICTIVE     = model output (calibrated Model C operational risk)
   DIAGNOSTIC     = derived evidence (network / propagation)
   COUNTERFACTUAL = simulated scenario (simulator / intervention)
   Every page uses the same three treatments so the distinction is learnable.
   ----------------------------------------------------------------------- */

export const SEMANTIC = {
  predictive: {
    cls: 'predictive',
    label: 'Predictive',
    badge: 'badge-predictive',
    help: 'Model output. Calibrated Model C probability.',
  },
  diagnostic: {
    cls: 'diagnostic',
    label: 'Diagnostic',
    badge: 'badge-diagnostic',
    help: 'Derived evidence about network structure. Not a prediction.',
  },
  counterfactual: {
    cls: 'counterfactual',
    label: 'Counterfactual',
    badge: 'badge-counterfactual',
    help: 'Modeled scenario under stated assumptions. Not a forecast.',
  },
};

export function SemanticBadge({ kind, children }) {
  const s = SEMANTIC[kind];
  if (!s) return null;
  return (
    <span className={`badge ${s.badge}`} title={s.help}>
      {children || s.label}
    </span>
  );
}

export function PageHeader({ title, subtitle, children }) {
  return (
    <header className="page-header">
      <div>
        <h1 className="page-title">{title}</h1>
        {subtitle && <p className="page-subtitle">{subtitle}</p>}
      </div>
      {children && <div className="page-header-aside">{children}</div>}
    </header>
  );
}

export function Panel({ title, icon: Icon, semantic, aside, children, note, className = '' }) {
  const sem = semantic ? `semantic-${semantic}` : '';
  return (
    <section className={`nexus-panel ${sem} ${className}`.trim()}>
      {(title || aside) && (
        <div className="panel-head">
          <h2 className="panel-title">
            {Icon && <Icon size={13} aria-hidden="true" />}
            {title}
          </h2>
          {aside}
        </div>
      )}
      {children}
      {note && <p className="panel-note">{note}</p>}
    </section>
  );
}

export function Metric({ label, value, sub, semantic, title }) {
  const tone = semantic ? `is-${semantic}` : '';
  return (
    <div title={title}>
      <span className="metric-label">{label}</span>
      <div className={`metric-primary ${tone}`.trim()}>{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  );
}

export function MetricCard({ label, value, sub, semantic, badge, title }) {
  return (
    <div className="metric-card" title={title}>
      <div className="row-between">
        <span className="metric-label">{label}</span>
        {badge && <SemanticBadge kind={badge} />}
      </div>
      <div className={`metric-primary ${semantic ? `is-${semantic}` : ''}`.trim()}>{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  );
}

export function DataRow({ label, value, hint }) {
  return (
    <div className="data-row">
      <span className="data-row-label" title={hint}>{label}</span>
      <span className="data-row-value">{value}</span>
    </div>
  );
}

/* ------------------------------ states --------------------------------- */

export function LoadingState({ label = 'Loading' }) {
  return (
    <div className="state-block" role="status" aria-live="polite">
      <Loader2 size={26} className="state-icon" aria-hidden="true" />
      <p className="state-title">{label}</p>
      <p className="state-message">Reading the frozen Seed 909 snapshot.</p>
    </div>
  );
}

export function SkeletonPanel({ rows = 3 }) {
  return (
    <div className="nexus-panel" aria-hidden="true">
      <div className="skeleton skeleton-line" style={{ width: '32%' }} />
      <div className="skeleton skeleton-metric" />
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="skeleton skeleton-line" style={{ width: `${88 - i * 13}%`, marginTop: 14 }} />
      ))}
    </div>
  );
}

export function EmptyState({ title, message, icon: Icon = Inbox, children }) {
  return (
    <div className="state-block">
      <Icon size={34} className="state-icon" aria-hidden="true" />
      <p className="state-title">{title}</p>
      {message && <p className="state-message">{message}</p>}
      {children}
    </div>
  );
}

export function ErrorState({ error, onRetry, title = 'Could not load this view' }) {
  const message =
    typeof error === 'string' ? error : error?.message || 'An unexpected error occurred.';
  return (
    <div className="state-block state-block--error" role="alert">
      <AlertTriangle size={30} className="state-icon" aria-hidden="true" />
      <p className="state-title">{title}</p>
      <p className="state-message">{message}</p>
      {onRetry && (
        <button type="button" className="btn btn-sm" onClick={onRetry}>
          <RefreshCw size={13} aria-hidden="true" /> Retry
        </button>
      )}
    </div>
  );
}

/** Renders the right state for an async view without repeating the branching. */
export function AsyncView({ loading, error, onRetry, empty, emptyProps, children, skeleton }) {
  if (loading) return skeleton ?? <LoadingState />;
  if (error) return <ErrorState error={error} onRetry={onRetry} />;
  if (empty) return <EmptyState {...emptyProps} />;
  return children;
}

/* ------------------------------ bars ----------------------------------- */

export function Bar({ value, max = 1, semantic = 'diagnostic', label }) {
  const pct = max > 0 ? Math.max(0, Math.min(100, (value / max) * 100)) : 0;
  return (
    <div
      className="bar-track"
      role="img"
      aria-label={label || `${pct.toFixed(1)} percent of maximum`}
    >
      <div className={`bar-fill is-${semantic}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

export function RiskTierBadge({ tier }) {
  const map = { Elevated: 'badge-danger', Watch: 'badge-warn', Standard: 'badge-neutral' };
  return <span className={`badge ${map[tier] || 'badge-neutral'}`}>{tier || 'Unknown'}</span>;
}

export function StressBadge({ stressed }) {
  return stressed
    ? <span className="badge badge-danger">Stressed</span>
    : <span className="badge badge-ok">Not stressed</span>;
}
