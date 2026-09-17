import React, { useState } from 'react';

const BorrowerIntelligence = () => {
  const [borrowerId, setBorrowerId] = useState('');
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchBorrower = async (e) => {
    e.preventDefault();
    if (!borrowerId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`http://127.0.0.1:8000/borrower/${borrowerId}`);
      if (!res.ok) {
        if (res.status === 404) throw new Error("Borrower not found or not eligible at current week");
        throw new Error("API Error");
      }
      const json = await res.json();
      setData(json);
    } catch (err) {
      setError(err.message);
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="glass-panel flex items-center justify-between">
        <h2 className="m-0">Borrower Intelligence</h2>
        <form onSubmit={fetchBorrower} className="flex gap-2">
          <input 
            type="number" 
            placeholder="Borrower ID (e.g. 1)" 
            value={borrowerId}
            onChange={(e) => setBorrowerId(e.target.value)}
            className="px-3 py-1 bg-black/30 border border-white/10 rounded text-white"
          />
          <button type="submit" className="btn btn-primary" disabled={loading}>
            {loading ? 'Searching...' : 'Inspect'}
          </button>
        </form>
      </div>

      {error && (
        <div className="glass-panel text-red-400">
          <p>{error}</p>
        </div>
      )}

      {!data && !error && !loading && (
        <div className="glass-panel text-center py-8 text-gray-400">
          <p>Enter a Borrower ID to view risk and network evidence.</p>
        </div>
      )}

      {data && (
        <div className="grid grid-cols-2 gap-4">
          <div className="glass-panel">
            <h3 className="glass-panel-header text-emerald-400">Financial State</h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="metric-label">Cash Buffer (4w)</div>
                <div className="metric-value">${data.financial_state.cash_buffer_mean_4w.toFixed(2)}</div>
              </div>
              <div>
                <div className="metric-label">Avg Income (4w)</div>
                <div className="metric-value">${data.financial_state.weekly_income_mean_4w.toFixed(2)}</div>
              </div>
              <div>
                <div className="metric-label">Avg Expenses (4w)</div>
                <div className="metric-value">${data.financial_state.weekly_expenses_mean_4w.toFixed(2)}</div>
              </div>
              <div>
                <div className="metric-label">Current Status</div>
                <div className="mt-2">
                  {data.financial_state.current_stress ? 
                    <span className="badge badge-risk-high">Stressed</span> : 
                    <span className="badge badge-predictive">Healthy</span>}
                </div>
              </div>
            </div>
          </div>

          <div className="glass-panel">
            <h3 className="glass-panel-header text-red-400 flex justify-between">
              Operational Risk
              <span className="badge badge-predictive">Predictive</span>
            </h3>
            <div>
              <div className="metric-label">Calibrated Model C Score</div>
              <div className="metric-value text-red-400">
                {(data.operational_risk.score * 100).toFixed(2)}%
              </div>
              <p className="mt-2 text-sm text-gray-400">Baseline Context Score (Model B): {(data.operational_risk.baseline_score * 100).toFixed(2)}%</p>
            </div>
          </div>

          <div className="glass-panel col-span-2 border-blue-500/30">
            <h3 className="glass-panel-header text-blue-400 flex justify-between">
              Network / Propagation Evidence
              <span className="badge badge-diagnostic">Diagnostic</span>
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="metric-label text-blue-400">Modeled Propagation Exposure</div>
                <div className="metric-value">{(data.network_evidence.borrower_propagation_exposure * 100).toFixed(2)}%</div>
                <p className="text-xs text-blue-300 mt-1">Counterfactual exposure (not a causal guarantee)</p>
              </div>
              <div>
                <div className="metric-label text-blue-400">Peer Stress Mean (4w)</div>
                <div className="metric-value">{(data.network_evidence.peer_stress_mean * 100).toFixed(2)}%</div>
              </div>
            </div>
            
            <div className="mt-6 flex justify-end">
               <button className="btn btn-primary bg-blue-600 border-blue-500">
                 View Group {data.group_id} Network Context
               </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BorrowerIntelligence;
