import React, { useState, useEffect } from 'react';
import { Microscope, Database, ShieldCheck, AlertCircle, BarChart3, Share2 } from 'lucide-react';

const ModelImpactLab = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/evaluation')
      .then(res => res.json())
      .then(res => {
        setData(res);
        setLoading(false);
      })
      .catch(err => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  if (loading) return <div className="glass-panel text-gray-400 p-8">Loading frozen M2C evaluation artifacts...</div>;
  if (error) return <div className="glass-panel text-red-400 p-8">Error: {error}</div>;
  if (!data) return null;

  return (
    <div className="h-full overflow-y-auto space-y-6 pr-2">
      <div className="glass-panel bg-gradient-to-r from-teal-900/20 to-blue-900/20 border-teal-500/20">
        <h2 className="flex items-center gap-2 m-0 text-xl text-teal-400">
          <Microscope size={24} />
          Model & Impact Lab (M2C-FROZEN)
        </h2>
        <p className="text-gray-400 text-sm mt-2">
          Transparency report for the final M2C analytical evaluation run across {data.target_counts.total_episodes.toLocaleString()} independent synthetic episodes.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-6">
        
        {/* A. MODEL COMPARISON */}
        <div className="glass-panel">
          <h3 className="glass-panel-header flex items-center gap-2">
            <BarChart3 size={18} /> Model Comparison
          </h3>
          <div className="space-y-3 mt-4">
            {Object.entries(data.models).map(([modelName, metrics]) => (
              <div key={modelName} className="flex justify-between items-center p-2 rounded bg-black/30 border border-white/5">
                <span className={`font-semibold ${modelName.includes('Model C') ? 'text-teal-400' : 'text-gray-300'}`}>{modelName}</span>
                <div className="flex gap-4 text-xs font-mono">
                  <span>ROC-AUC: {metrics.roc_auc.toFixed(3)}</span>
                  <span>PR-AUC: {metrics.pr_auc.toFixed(3)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* B & C. PROPAGATION EXPERIMENT & M2C RESULT */}
        <div className="glass-panel border-l-4 border-l-teal-500">
          <h3 className="glass-panel-header flex items-center gap-2">
            <Share2 size={18} /> Propagation Experiment Result
          </h3>
          <div className="mt-4 p-4 bg-teal-900/10 rounded border border-teal-500/20 text-teal-100">
            <h4 className="text-sm font-bold text-teal-400 mb-2 uppercase tracking-wider">Primary M2C Conclusion</h4>
            <p className="text-sm font-semibold">{data.results[0]}</p>
          </div>
          <div className="mt-4 space-y-2 text-sm text-gray-400">
            <p className="flex gap-2"><AlertCircle size={16} className="text-yellow-500 flex-shrink-0 mt-0.5" /> {data.results[1]}</p>
            <p className="flex gap-2"><AlertCircle size={16} className="text-blue-400 flex-shrink-0 mt-0.5" /> {data.results[2]}</p>
          </div>
        </div>

        {/* D. DATASET / EVALUATION */}
        <div className="glass-panel">
          <h3 className="glass-panel-header flex items-center gap-2">
            <Database size={18} /> Dataset & Evaluation Parameters
          </h3>
          <div className="grid grid-cols-2 gap-4 mt-4 text-sm">
            <div>
              <p className="text-gray-500 text-xs uppercase tracking-wider">Total Episodes</p>
              <p className="text-white font-mono">{data.target_counts.total_episodes.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs uppercase tracking-wider">PV Positive Events</p>
              <p className="text-white font-mono">{data.target_counts.pv_positive_events.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs uppercase tracking-wider">PV Threshold</p>
              <p className="text-white font-mono">{data.target_counts.pv_threshold}</p>
            </div>
            <div>
              <p className="text-gray-500 text-xs uppercase tracking-wider">Seed Variation</p>
              <p className="text-white font-mono">{data.per_seed_variation}</p>
            </div>
          </div>
        </div>

        {/* E & F. ATTRIBUTION & SAFEGUARDS */}
        <div className="glass-panel">
          <h3 className="glass-panel-header flex items-center gap-2">
            <ShieldCheck size={18} /> Safeguards & Attribution
          </h3>
          <div className="space-y-4 mt-4">
            <div className="bg-black/20 p-3 rounded">
              <h4 className="text-xs text-gray-400 uppercase tracking-wider mb-1">Attribution Basis</h4>
              <p className="text-sm text-gray-200">{data.contribution_statistics.attribution_basis}</p>
              <p className="text-xs text-teal-400 mt-1">Mean Episode Contribution: {data.contribution_statistics.mean_episode_network_contribution.toFixed(2)}</p>
            </div>
            
            <div className="space-y-2">
              <h4 className="text-xs text-gray-400 uppercase tracking-wider mb-1">Methodological Safeguards</h4>
              {Object.entries(data.safeguards).map(([key, val]) => (
                <div key={key} className="text-sm">
                  <span className="text-gray-300 font-semibold">{key.replace(/_/g, ' ')}:</span> <span className="text-gray-500">{val}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default ModelImpactLab;
