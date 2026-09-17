import React from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink, Navigate } from 'react-router-dom';
import { Activity, Users, Share2, Waves, ActivitySquare, TestTube } from 'lucide-react';
import './index.css';

import CommandCenter from './CommandCenter';
import BorrowerIntelligence from './BorrowerIntelligence';
import NetworkIntelligence from './NetworkIntelligence';
import RippleSimulator from './RippleSimulator';
import InterventionStudio from './InterventionStudio';
import ModelImpactLab from './ModelImpactLab';

// No placeholder components remaining


function App() {
  return (
    <Router>
      <div className="app-container">
        {/* Sidebar Navigation */}
        <aside className="sidebar">
          <div className="sidebar-brand">
            NEXUS
          </div>
          <nav className="sidebar-nav">
            <NavLink to="/command-center" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <Activity size={18} />
              Command Center
            </NavLink>
            <NavLink to="/borrower" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <Users size={18} />
              Borrower Intelligence
            </NavLink>
            <NavLink to="/network" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <Share2 size={18} />
              Network Intelligence
            </NavLink>
            <NavLink to="/simulator" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <Waves size={18} />
              Ripple Simulator
            </NavLink>
            <NavLink to="/intervene" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <ActivitySquare size={18} />
              Intervention Studio
            </NavLink>
            <NavLink to="/lab" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
              <TestTube size={18} />
              Model & Impact Lab
            </NavLink>
          </nav>
        </aside>

        {/* Main Content Area */}
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Navigate to="/command-center" replace />} />
            <Route path="/command-center" element={<CommandCenter />} />
            <Route path="/borrower" element={<BorrowerIntelligence />} />
            <Route path="/network" element={<NetworkIntelligence />} />
            <Route path="/simulator" element={<RippleSimulator />} />
            <Route path="/intervene" element={<InterventionStudio />} />
            <Route path="/lab" element={<ModelImpactLab />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
