import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { BrowserRouter, Routes, Route, NavLink, Navigate } from 'react-router-dom';
import { Activity, Users, Share2, Waves, SlidersHorizontal, FlaskConical } from 'lucide-react';
import './index.css';

import { api } from './lib/api';
import CommandCenter from './CommandCenter';
import BorrowerIntelligence from './BorrowerIntelligence';
import NetworkIntelligence from './NetworkIntelligence';
import RippleSimulator from './RippleSimulator';
import InterventionStudio from './InterventionStudio';
import ModelImpactLab from './ModelImpactLab';

/* Shared selection so the six pages behave as one product: picking a borrower
   in the Command Center carries through to Borrower, Network, Simulator and
   Intervention Studio instead of each page starting from nothing. */
const SelectionContext = createContext(null);
export const useSelection = () => useContext(SelectionContext);

const NAV = [
  { to: '/command-center', label: 'Command Center',    icon: Activity },
  { to: '/borrower',       label: 'Borrower Intelligence', icon: Users },
  { to: '/network',        label: 'Network Intelligence',  icon: Share2 },
  { to: '/simulator',      label: 'Ripple Simulator',      icon: Waves },
  { to: '/intervene',      label: 'Intervention Studio',   icon: SlidersHorizontal },
  { to: '/lab',            label: 'Model & Impact Lab',    icon: FlaskConical },
];

function Provenance() {
  const [a, setA] = useState(null);
  useEffect(() => { api.assumptions().then(setA).catch(() => setA(null)); }, []);
  if (!a) return null;
  return (
    <div className="rail-provenance">
      <dl>
        <dt>World seed</dt><dd>{a.world_seed}</dd>
        <dt>Analytics</dt><dd>{a.analytics_version}</dd>
        <dt>As-of week</dt><dd>{a.as_of_week}</dd>
      </dl>
      <p className="rail-provenance-note">
        Synthetic dataset. Scenario and model outputs are not forecasts and do not
        generalize to real microfinance populations.
      </p>
    </div>
  );
}

function Shell({ children }) {
  return (
    <div className="app">
      <aside className="rail">
        <div className="rail-brand">
          <div className="rail-brand-mark">NEXUS</div>
          <div className="rail-brand-sub">Financial Resilience Intelligence</div>
        </div>
        <nav className="rail-nav" aria-label="Primary">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) => `rail-link ${isActive ? 'is-active' : ''}`}
            >
              <Icon size={16} aria-hidden="true" />
              {label}
            </NavLink>
          ))}
        </nav>
        <Provenance />
      </aside>
      <main className="main">{children}</main>
    </div>
  );
}

export default function App() {
  const [borrowerId, setBorrowerId] = useState('B10');
  const [groupId, setGroupId] = useState(null);

  const selection = useMemo(
    () => ({ borrowerId, setBorrowerId, groupId, setGroupId }),
    [borrowerId, groupId],
  );

  return (
    <SelectionContext.Provider value={selection}>
      <BrowserRouter>
        <Shell>
          <Routes>
            <Route path="/" element={<Navigate to="/command-center" replace />} />
            <Route path="/command-center" element={<CommandCenter />} />
            <Route path="/borrower" element={<BorrowerIntelligence />} />
            <Route path="/network" element={<NetworkIntelligence />} />
            <Route path="/simulator" element={<RippleSimulator />} />
            <Route path="/intervene" element={<InterventionStudio />} />
            <Route path="/lab" element={<ModelImpactLab />} />
            <Route path="*" element={<Navigate to="/command-center" replace />} />
          </Routes>
        </Shell>
      </BrowserRouter>
    </SelectionContext.Provider>
  );
}
