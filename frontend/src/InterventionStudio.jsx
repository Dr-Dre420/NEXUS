import React, { useState } from 'react';
import { Activity, Settings2, Play, Info, AlertTriangle } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';

const InterventionStudio = () => {
  const [borrowerId, setBorrowerId] = useState('B1');
  const [interventionType, setInterventionType] = useState('restructure');
  const [amount, setAmount] = useState(500);
  const [duration, setDuration] = useState(12);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const runIntervention = async (e) => {
    e.preventDefault();
    if (!borrowerId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('http://127.0.0.1:8000/intervene', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          borrower_id: borrowerId,
          amount: Number(amount),
          duration_weeks: Number(duration),
          intervention_type: interventionType
        })
      });
      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(errorData.detail || "Intervention failed");
      }
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const transformChartData = (trajectory) => {
    return trajectory.map(week => {
      // Aggregate peer downstream impacts
      let peerBaselineShortfall = 0;
      let peerScenarioShortfall = 0;
      let peerScenarioCash = 0;
      let peerBaselineCash = 0;
      let peerCount = 0;

      Object.values(week.group_members).forEach(member => {
        peerBaselineShortfall += member.baseline.shortfall;
        peerScenarioShortfall += member.scenario.shortfall;
        peerBaselineCash += member.baseline.cash_buffer;
        peerScenarioCash += member.scenario.cash_buffer;
        peerCount++;
      });

      return {
        name: `Week ${week.week}`,
        baselineCash: week.target_borrower.baseline.cash_buffer,
        scenarioCash: week.target_borrower.scenario.cash_buffer,
        baselineShortfall: week.target_borrower.baseline.shortfall,
        scenarioShortfall: week.target_borrower.scenario.shortfall,
        baselineAmountDue: week.target_borrower.baseline.amount_due,
        scenarioAmountDue: week.target_borrower.scenario.amount_due,
        peerScenarioCash: peerCount > 0 ? peerScenarioCash / peerCount : 0,
        peerBaselineCash: peerCount > 0 ? peerBaselineCash / peerCount : 0,
      };
    });
  };

  return (
    <div className="flex flex-col gap-4 h-full">
      <div className="glass-panel flex items-center justify-between">
        <h2 className="flex items-center gap-2 m-0">
          <Activity size={20} className="text-purple-400" />
          Intervention Studio
        </h2>
      </div>

      <div className="flex gap-4 flex-1 overflow-hidden">
        {/* Left: Configuration Panel */}
        <div className="w-1/3 glass-panel overflow-y-auto">
          <h3 className="glass-panel-header text-gray-300 flex items-center gap-2">
            <Settings2 size={16} /> Strategy Parameters
          </h3>
          <form onSubmit={runIntervention} className="space-y-4">
            <div>
              <label className="block text-xs uppercase tracking-wider text-gray-400 mb-1">Target Borrower ID</label>
              <input 
                type="text" 
                value={borrowerId}
                onChange={(e) => setBorrowerId(e.target.value)}
                className="w-full px-3 py-2 bg-black/30 border border-white/10 rounded text-white"
                placeholder="e.g. B1"
                required
              />
            </div>
            <div>
              <label className="block text-xs uppercase tracking-wider text-gray-400 mb-1">Intervention Type</label>
              <select 
                value={interventionType}
                onChange={(e) => setInterventionType(e.target.value)}
                className="w-full px-3 py-2 bg-black/30 border border-white/10 rounded text-white"
              >
                <option value="restructure">Restructure (Reduce Weekly Instalment)</option>
                <option value="cash_injection">Cash Injection (Increase Buffer)</option>
                <option value="payment_adjustment">Payment Adjustment (Reduce Principal/Amount Due)</option>
              </select>
            </div>
            <div>
              <label className="block text-xs uppercase tracking-wider text-gray-400 mb-1">Magnitude ($)</label>
              <input 
                type="number" 
                value={amount}
                onChange={(e) => setAmount(e.target.value)}
                className="w-full px-3 py-2 bg-black/30 border border-white/10 rounded text-white"
                min="0"
                required
              />
            </div>
            <div>
              <label className="block text-xs uppercase tracking-wider text-gray-400 mb-1">Evaluation Duration (Weeks)</label>
              <input 
                type="number" 
                value={duration}
                onChange={(e) => setDuration(e.target.value)}
                className="w-full px-3 py-2 bg-black/30 border border-white/10 rounded text-white"
                min="1"
                max="52"
                required
              />
            </div>
            
            <button 
              type="submit" 
              className="w-full btn btn-primary bg-purple-600 border-purple-500 flex items-center justify-center gap-2 mt-4 py-3"
              disabled={loading}
            >
              {loading ? 'Evaluating...' : <><Play size={16} /> Evaluate Intervention</>}
            </button>
          </form>

          {error && (
            <div className="mt-4 p-3 bg-red-900/20 border border-red-500/30 text-red-400 text-sm rounded">
              {error}
            </div>
          )}
          
          <div className="mt-6 p-3 bg-purple-900/10 border border-purple-500/20 text-purple-300 text-xs rounded flex gap-2">
            <Info size={16} className="flex-shrink-0 mt-0.5" />
            <p>Applies a localized structural change and computes the resulting trajectory against the frozen baseline. Avoids modifying global analytical states.</p>
          </div>
        </div>

        {/* Right: Results Panel */}
        <div className="w-2/3 flex flex-col gap-4 overflow-y-auto pr-2">
          {!result && !loading && !error && (
            <div className="glass-panel flex-1 flex flex-col items-center justify-center text-gray-400 h-full">
              <Activity size={48} className="opacity-20 mb-4" />
              <p>Configure an intervention to observe counterfactual downstream effects.</p>
            </div>
          )}
          
          {loading && (
            <div className="glass-panel flex-1 flex items-center justify-center text-gray-400 h-full">
              Evaluating intervention trajectory...
            </div>
          )}

          {result && (
            <>
              <div className="glass-panel bg-gradient-to-br from-black/40 to-purple-900/10 border-purple-500/20">
                <div className="flex justify-between items-start mb-2">
                  <p className="text-sm text-purple-400 font-semibold flex items-center gap-2">
                    <Activity size={16} /> Result: {result.scenario_type}
                  </p>
                  <span className="text-[10px] uppercase bg-purple-500/20 px-2 py-0.5 rounded text-purple-300">
                    Modeled Counterfactual
                  </span>
                </div>
                <p className="text-xs text-gray-300 font-semibold">{result.disclaimer}</p>
                <div className="mt-2 text-xs text-yellow-500/80 flex items-center gap-1">
                  <AlertTriangle size={12} />
                  Note: This intervention demonstrates a mathematical counterfactual under stated assumptions. It does not claim causality or guarantee recovery in the real world.
                </div>
              </div>

              <div className="grid grid-cols-1 gap-4">
                {/* Target Borrower Amount Due */}
                <div className="glass-panel" style={{ height: '250px' }}>
                  <h3 className="glass-panel-header">Target Borrower: Financial Burden Trajectory (Amount Due)</h3>
                  <ResponsiveContainer width="100%" height="85%">
                    <AreaChart data={transformChartData(result.trajectory)} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                      <defs>
                        <linearGradient id="colorBaseline" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#94a3b8" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#94a3b8" stopOpacity={0}/>
                        </linearGradient>
                        <linearGradient id="colorScenario" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#c084fc" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#c084fc" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} tickLine={false} />
                      <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} tickFormatter={(val) => `$${val}`} />
                      <RechartsTooltip 
                        contentStyle={{ backgroundColor: '#1e293b', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '4px' }}
                        itemStyle={{ color: '#fff' }}
                      />
                      <Legend iconType="circle" wrapperStyle={{ fontSize: '12px' }}/>
                      <Area type="monotone" dataKey="baselineAmountDue" name="Baseline Burden" stroke="#94a3b8" fillOpacity={1} fill="url(#colorBaseline)" />
                      <Area type="monotone" dataKey="scenarioAmountDue" name="Intervention Burden" stroke="#c084fc" fillOpacity={1} fill="url(#colorScenario)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                {/* Downstream Effects: Peer Cash Buffer Drain */}
                <div className="glass-panel" style={{ height: '250px' }}>
                  <h3 className="glass-panel-header text-purple-400">Downstream Ripple Mitigation: Avg Peer Liquidity Preservation</h3>
                  <ResponsiveContainer width="100%" height="85%">
                    <LineChart data={transformChartData(result.trajectory)} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                      <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} tickLine={false} />
                      <YAxis stroke="#94a3b8" fontSize={12} tickLine={false} tickFormatter={(val) => `$${val}`} />
                      <RechartsTooltip 
                        contentStyle={{ backgroundColor: '#1e293b', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '4px' }}
                      />
                      <Legend iconType="circle" wrapperStyle={{ fontSize: '12px' }}/>
                      <Line type="monotone" dataKey="peerBaselineCash" name="Baseline Avg Peer Cash" stroke="#94a3b8" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="peerScenarioCash" name="Intervention Avg Peer Cash" stroke="#a855f7" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default InterventionStudio;
