import React, { useState, useEffect } from 'react';

const CommandCenter = () => {
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/portfolio/summary')
      .then(res => {
        if (!res.ok) throw new Error('API Error');
        return res.json();
      })
      .then(data => setSummary(data))
      .catch(err => setError(err.message));
  }, []);

  if (error) return <div className="glass-panel text-red-400">Error loading portfolio: {error}</div>;
  if (!summary) return <div className="glass-panel">Loading Command Center...</div>;

  return (
    <div className="grid grid-cols-2 gap-4">
      <div className="glass-panel border-l-4 border-l-blue-500 col-span-2">
        <h2 className="glass-panel-header text-blue-400">Diagnostic Network Evidence</h2>
        <div className="grid grid-cols-2 gap-4 mt-4">
          <div>
            <div className="metric-label text-blue-400">Network-Exposed Borrowers</div>
            <div className="metric-value">{summary.network_exposed_borrowers}</div>
          </div>
          <div>
            <div className="metric-label text-blue-400">Aggregate Exposure Volume</div>
            <div className="metric-value">{summary.aggregate_network_exposure_index.toFixed(2)}</div>
          </div>
        </div>
        <div className="mt-4">
          <span className="badge badge-diagnostic">Diagnostic / Counterfactual</span>
        </div>
      </div>
      
      <div className="glass-panel col-span-2">
        <h3 className="glass-panel-header">Portfolio Status As-Of Week {summary.as_of_week}</h3>
        <p>Total Eligible Borrowers: {summary.total_eligible_borrowers}</p>
        <p className="mt-4 text-sm text-gray-400">Click a borrower or group in the sidebar to drill down into specific exposure mechanisms.</p>
      </div>
    </div>
  );
};

export default CommandCenter;
