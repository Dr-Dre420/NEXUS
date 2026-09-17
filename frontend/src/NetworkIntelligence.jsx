import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Share2, Users, Info, Waves, SlidersHorizontal, MousePointerClick, Search } from 'lucide-react';
import { api, canonicalGroup } from './lib/api';
import { currency, probability, percentile, indexValue, percent, count } from './lib/format';
import {
  PageHeader, Panel, Metric, DataRow, SemanticBadge, RiskTierBadge, StressBadge,
  AsyncView, SkeletonPanel, EmptyState, ErrorState, LoadingState,
} from './lib/ui';
import { useSelection } from './App';

const VIEW = 380;          // square viewBox; scales responsively
const CENTER = VIEW / 2;
const RADIUS = 122;

export default function NetworkIntelligence() {
  const { groupId, setGroupId, setBorrowerId } = useSelection();
  const [index, setIndex] = useState(null);
  const [indexErr, setIndexErr] = useState(null);
  const [filter, setFilter] = useState('');

  const [group, setGroup] = useState(null);
  const [groupErr, setGroupErr] = useState(null);
  const [groupLoading, setGroupLoading] = useState(false);
  const [selected, setSelected] = useState(null);
  const nav = useNavigate();

  const loadIndex = useCallback(() => {
    setIndexErr(null);
    api.network().then(setIndex).catch(setIndexErr);
  }, []);
  useEffect(loadIndex, [loadIndex]);

  const loadGroup = useCallback((raw) => {
    const id = canonicalGroup(raw);
    if (!id) return;
    setGroupId(id);
    setGroupLoading(true);
    setGroupErr(null);
    setSelected(null);
    api.group(id)
      .then(setGroup)
      .catch((e) => { setGroupErr(e); setGroup(null); })
      .finally(() => setGroupLoading(false));
  }, [setGroupId]);

  useEffect(() => { if (groupId) loadGroup(groupId); }, []); // eslint-disable-line

  const groups = useMemo(() => {
    if (!index) return [];
    const q = filter.trim().toUpperCase();
    const rows = q ? index.groups.filter((g) => g.group_id.includes(q)) : index.groups;
    return [...rows].sort((a, b) => b.aggregate_exposure - a.aggregate_exposure);
  }, [index, filter]);

  const maxExposure = useMemo(
    () => (index ? Math.max(...index.groups.map((g) => g.aggregate_exposure), 1e-9) : 1),
    [index],
  );

  /* -------- JLG star topology: one group hub, members on a ring -------- */
  const layout = useMemo(() => {
    if (!group?.nodes) return null;
    const hub = group.nodes.find((n) => n.type === 'group');
    const members = group.nodes.filter((n) => n.type !== 'group');
    const placed = members.map((n, i) => {
      const a = (i / members.length) * 2 * Math.PI - Math.PI / 2;
      return { ...n, x: CENTER + RADIUS * Math.cos(a), y: CENTER + RADIUS * Math.sin(a) };
    });
    return { hub: hub ? { ...hub, x: CENTER, y: CENTER } : null, members: placed };
  }, [group]);

  const maxMemberExposure = useMemo(
    () => (layout ? Math.max(...layout.members.map((m) => m.borrower_propagation_exposure), 1e-9) : 1),
    [layout],
  );

  return (
    <>
      <PageHeader
        title="Network Intelligence"
        subtitle="Joint-liability group structure from the frozen snapshot. Borrowers are connected through their JLG, which is the modeled transmission mechanism — no borrower-to-borrower bilateral relationships are constructed or implied."
      >
        {group && <span className="badge badge-neutral">{group.group_id}</span>}
        <SemanticBadge kind="diagnostic" />
      </PageHeader>

      <div className="split-index">
        {/* ------------------------- index ------------------------- */}
        <Panel title="JLG index" icon={Share2}
          aside={index ? <span className="tiny text-muted">{count(groups.length)} groups</span> : null}>
          <div className="field" style={{ marginBottom: 10 }}>
            <label htmlFor="group-filter" className="sr-only">Filter groups</label>
            <input id="group-filter" className="input input-mono" placeholder="Filter e.g. G2"
              value={filter} onChange={(e) => setFilter(e.target.value)} />
          </div>

          {indexErr && <ErrorState error={indexErr} onRetry={loadIndex} title="Could not load the group index" />}
          {!index && !indexErr && <div className="skeleton skeleton-line" style={{ height: 180 }} />}

          {index && groups.length === 0 && (
            <EmptyState icon={Search} title="No groups match" message={`Nothing matches "${filter}".`} />
          )}

          {index && groups.length > 0 && (
            <div className="index-list">
              {groups.map((g) => (
                <button
                  key={g.group_id}
                  type="button"
                  className={`index-item ${group?.group_id === g.group_id ? 'is-selected' : ''}`}
                  onClick={() => loadGroup(g.group_id)}
                  aria-pressed={group?.group_id === g.group_id}
                >
                  <div className="row-between">
                    <span className="index-item-title">{g.group_id}</span>
                    {g.stressed_members > 0
                      ? <span className="badge badge-danger">{g.stressed_members} stressed</span>
                      : <span className="badge badge-neutral">{g.member_count} members</span>}
                  </div>
                  <div className="index-item-meta">
                    <span>Exposure (sum)</span>
                    <span className="mono">{indexValue(g.aggregate_exposure)}</span>
                  </div>
                  <div className="bar-track" style={{ marginTop: 6, height: 3 }}>
                    <div className="bar-fill is-diagnostic"
                      style={{ width: `${(g.aggregate_exposure / maxExposure) * 100}%` }} />
                  </div>
                </button>
              ))}
            </div>
          )}
        </Panel>

        {/* ------------------------- detail ------------------------- */}
        <div className="stack">
          {!group && !groupLoading && !groupErr && (
            <Panel>
              <EmptyState icon={MousePointerClick} title="Select a joint-liability group"
                message="Choose a group from the index to see its structure, member risk and diagnostic network evidence." />
            </Panel>
          )}
          {groupLoading && <Panel><LoadingState label="Loading group" /></Panel>}
          {groupErr && <Panel><ErrorState error={groupErr} onRetry={() => loadGroup(groupId)} title="Could not load that group" /></Panel>}

          {group && !groupLoading && (
            <div className="stack fade-in">
              <div className="grid-4">
                <div className="metric-card">
                  <span className="metric-label">Members</span>
                  <div className="metric-primary">{count(group.member_count)}</div>
                </div>
                <div className="metric-card">
                  <span className="metric-label">Stressed members</span>
                  <div className="metric-primary">{count(group.stressed_members)}</div>
                  <div className="metric-sub">Observed state</div>
                </div>
                <div className="metric-card">
                  <span className="metric-label">Exposure (sum)</span>
                  <div className="metric-primary is-diagnostic">{indexValue(group.aggregate_group_exposure)}</div>
                  <div className="metric-sub">Mean {indexValue(group.mean_group_exposure)}</div>
                </div>
                <div className="metric-card">
                  <span className="metric-label">Group cash buffer</span>
                  <div className="metric-primary">{currency(group.group_buffer_total)}</div>
                  <div className="metric-sub">Sum of member 4w averages</div>
                </div>
              </div>

              <Panel title={`JLG topology — ${group.group_id}`} icon={Share2} semantic="diagnostic"
                aside={<SemanticBadge kind="diagnostic" />}
                note="Every edge is a borrower's joint-liability relationship to the group, which is the only transmission channel modeled. Node size reflects modeled propagation exposure; node colour reflects observed stress state.">
                <div className="graph-stage">
                  <span className="graph-stage-label">Branch → Centre → JLG → Borrowers</span>
                  {layout && (
                    <svg viewBox={`0 0 ${VIEW} ${VIEW}`}
                      style={{ width: '100%', maxWidth: 420, height: 'auto' }}
                      role="img"
                      aria-label={`Joint liability group ${group.group_id} with ${group.member_count} members connected to a central group node`}>
                      {/* edges */}
                      {layout.hub && layout.members.map((m) => {
                        const isSel = selected?.id === m.id;
                        return (
                          <line key={`e-${m.id}`}
                            x1={layout.hub.x} y1={layout.hub.y} x2={m.x} y2={m.y}
                            stroke={isSel ? 'var(--diagnostic)' : 'rgba(148,163,184,0.22)'}
                            strokeWidth={isSel ? 2 : 1.2} />
                        );
                      })}
                      {/* hub */}
                      {layout.hub && (
                        <g>
                          <circle cx={layout.hub.x} cy={layout.hub.y} r="30"
                            fill="var(--bg-panel-2)" stroke="var(--line-strong)" strokeWidth="1.5" />
                          <text x={layout.hub.x} y={layout.hub.y} textAnchor="middle" dy="0.35em"
                            className="graph-node-label" fill="var(--text-2)">{group.group_id}</text>
                        </g>
                      )}
                      {/* members */}
                      {layout.members.map((m) => {
                        const isSel = selected?.id === m.id;
                        const r = 14 + 8 * Math.sqrt(m.borrower_propagation_exposure / maxMemberExposure || 0);
                        return (
                          <g key={m.id} className="graph-node" tabIndex={0} role="button"
                            aria-label={`Borrower ${m.id}, ${m.current_stress ? 'stressed' : 'not stressed'}, risk tier ${m.risk_tier}`}
                            onClick={() => setSelected(m)}
                            onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), setSelected(m))}>
                            {isSel && <circle cx={m.x} cy={m.y} r={r + 6} fill="none" stroke="var(--diagnostic)" strokeWidth="1.5" opacity="0.55" />}
                            <circle cx={m.x} cy={m.y} r={r}
                              fill={m.current_stress ? 'rgba(240,101,111,0.9)' : 'rgba(63,191,143,0.85)'}
                              stroke={isSel ? 'var(--diagnostic)' : 'rgba(255,255,255,0.18)'}
                              strokeWidth={isSel ? 2.5 : 1} />
                            <text x={m.x} y={m.y} textAnchor="middle" dy="0.35em"
                              className="graph-node-label" fill="#08111e">{m.id.replace('B', '')}</text>
                          </g>
                        );
                      })}
                    </svg>
                  )}
                  <div className="graph-legend">
                    <span className="graph-legend-key">
                      <i className="graph-legend-dot" style={{ background: 'rgba(63,191,143,0.85)' }} /> Not stressed
                    </span>
                    <span className="graph-legend-key">
                      <i className="graph-legend-dot" style={{ background: 'rgba(240,101,111,0.9)' }} /> Stressed
                    </span>
                    <span className="graph-legend-key">Node size = modeled exposure</span>
                  </div>
                </div>
              </Panel>

              {/* member inspector */}
              <Panel title={selected ? `Member detail — ${selected.id}` : 'Member detail'} icon={Users}
                aside={selected && <StressBadge stressed={selected.current_stress} />}>
                {!selected ? (
                  <EmptyState icon={MousePointerClick} title="Select a borrower node"
                    message="Choose any borrower in the topology above to inspect their predictive risk and diagnostic network evidence." />
                ) : (
                  <div className="stack fade-in">
                    <div className="grid-2">
                      <div className="nexus-panel nexus-panel--inset semantic-predictive">
                        <div className="row-between" style={{ marginBottom: 8 }}>
                          <span className="panel-title">Predictive</span>
                          <SemanticBadge kind="predictive" />
                        </div>
                        <Metric label="Calibrated Model C probability"
                          value={probability(selected.operational_risk_score)} semantic="predictive"
                          sub={`Percentile rank ${percentile(selected.risk_percentile)} · tier ${selected.risk_tier}`} />
                      </div>
                      <div className="nexus-panel nexus-panel--inset semantic-diagnostic">
                        <div className="row-between" style={{ marginBottom: 8 }}>
                          <span className="panel-title">Diagnostic</span>
                          <SemanticBadge kind="diagnostic" />
                        </div>
                        <Metric label="Modeled propagation exposure"
                          value={indexValue(selected.borrower_propagation_exposure)} semantic="diagnostic"
                          sub="Unitless index derived from joint-liability structure" />
                      </div>
                    </div>

                    <div>
                      <DataRow label="Cash buffer (4w avg)" value={currency(selected.cash_buffer_mean_4w, true)} />
                      <DataRow label="Weekly income (4w avg)" value={currency(selected.weekly_income_mean_4w, true)} />
                      <DataRow label="Group buffer share" value={percent(selected.liability_share, 2)}
                        hint="Share of the group's total cash buffer held by this borrower." />
                    </div>

                    <div className="action-bar">
                      <button className="btn" onClick={() => { setBorrowerId(selected.id); nav('/borrower'); }}>
                        <Users size={14} aria-hidden="true" /> Open dossier
                      </button>
                      <button className="btn" onClick={() => { setBorrowerId(selected.id); nav('/simulator'); }}>
                        <Waves size={14} aria-hidden="true" /> Simulate shock
                      </button>
                      <button className="btn" onClick={() => { setBorrowerId(selected.id); nav('/intervene'); }}>
                        <SlidersHorizontal size={14} aria-hidden="true" /> Explore intervention
                      </button>
                    </div>

                    <div className="callout callout-diagnostic">
                      <Info size={14} aria-hidden="true" />
                      <span>
                        Propagation exposure is a modeled scenario metric derived from the joint-liability
                        structure. It does not represent observed causality between specific individuals,
                        and no source-to-destination attribution is exposed in this product.
                      </span>
                    </div>
                  </div>
                )}
              </Panel>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
