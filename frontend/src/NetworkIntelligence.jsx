import React, { useState, useEffect } from 'react';
import { Share2, Users, AlertCircle, ShieldAlert, Activity } from 'lucide-react';

const NetworkIntelligence = () => {
  const [network, setNetwork] = useState(null);
  const [loadingList, setLoadingList] = useState(false);
  const [errorList, setErrorList] = useState(null);

  const [selectedGroup, setSelectedGroup] = useState(null);
  const [groupData, setGroupData] = useState(null);
  const [loadingGroup, setLoadingGroup] = useState(false);
  const [errorGroup, setErrorGroup] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);

  useEffect(() => {
    fetchNetwork();
  }, []);

  const fetchNetwork = async () => {
    setLoadingList(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/network');
      if (!res.ok) throw new Error('Failed to load network index');
      const data = await res.json();
      setNetwork(data);
    } catch (err) {
      setErrorList(err.message);
    } finally {
      setLoadingList(false);
    }
  };

  const selectGroup = async (groupId) => {
    setSelectedGroup(groupId);
    setLoadingGroup(true);
    setErrorGroup(null);
    setGroupData(null);
    setSelectedNode(null);

    try {
      const res = await fetch(`http://127.0.0.1:8000/group/${groupId}`);
      if (!res.ok) throw new Error('Failed to load group details');
      const data = await res.json();
      setGroupData(data);
    } catch (err) {
      setErrorGroup(err.message);
    } finally {
      setLoadingGroup(false);
    }
  };

  // Helper to draw circular graph
  const renderGraph = () => {
    if (!groupData || !groupData.nodes) return null;

    const radius = 100;
    const center = { x: 150, y: 150 };
    
    const groupNode = groupData.nodes.find(n => n.type === 'group');
    const borrowerNodes = groupData.nodes.filter(n => n.type !== 'group');
    const nodesCount = borrowerNodes.length;
    
    // Calculate positions
    const positionedBorrowers = borrowerNodes.map((node, i) => {
      const angle = (i / nodesCount) * 2 * Math.PI - Math.PI / 2;
      return {
        ...node,
        x: center.x + radius * Math.cos(angle),
        y: center.y + radius * Math.sin(angle)
      };
    });

    const positionedNodes = groupNode ? [{...groupNode, x: center.x, y: center.y}, ...positionedBorrowers] : positionedBorrowers;

    return (
      <svg width="300" height="300" className="mx-auto overflow-visible">
        {/* Draw Edges */}
        {groupData.edges.map((edge, i) => {
          const source = positionedNodes.find(n => n.id === edge.source);
          const target = positionedNodes.find(n => n.id === edge.target);
          if (!source || !target) return null;
          return (
            <line
              key={`edge-${i}`}
              x1={source.x}
              y1={source.y}
              x2={target.x}
              y2={target.y}
              stroke="rgba(255, 255, 255, 0.1)"
              strokeWidth="2"
            />
          );
        })}
        
        {/* Draw Nodes */}
        {positionedNodes.map(node => {
          const isStressed = node.current_stress;
          const isSelected = selectedNode?.id === node.id;
          return (
            <g 
              key={node.id} 
              transform={`translate(${node.x}, ${node.y})`}
              onClick={() => setSelectedNode(node)}
              className="cursor-pointer transition-transform hover:scale-110"
            >
              <circle
                r={node.type === 'group' ? "24" : "16"}
                fill={node.type === 'group' ? '#1e293b' : (isStressed ? '#ef4444' : '#10b981')}
                stroke={isSelected ? '#3b82f6' : (node.type === 'group' ? '#475569' : 'rgba(255,255,255,0.2)')}
                strokeWidth={isSelected ? "3" : (node.type === 'group' ? "2" : "1")}
              />
              <text 
                textAnchor="middle" 
                dy=".3em" 
                fontSize={node.type === 'group' ? "12" : "10"} 
                fill="#fff" 
                className="font-bold pointer-events-none"
              >
                {node.type === 'group' ? node.id : node.id.replace('B', '')}
              </text>
            </g>
          );
        })}
      </svg>
    );
  };

  return (
    <div className="flex h-full gap-4">
      {/* Sidebar: Group List */}
      <div className="w-1/3 glass-panel flex flex-col overflow-hidden h-[calc(100vh-80px)]">
        <h2 className="flex items-center gap-2 mb-4">
          <Share2 size={20} className="text-blue-400" />
          JLG Network Index
        </h2>
        
        {loadingList && <p className="text-gray-400">Loading network...</p>}
        {errorList && <p className="text-red-400">{errorList}</p>}
        
        <div className="flex-1 overflow-y-auto space-y-2 pr-2">
          {network?.groups.map(group => (
            <div 
              key={group.group_id}
              onClick={() => selectGroup(group.group_id)}
              className={`p-3 rounded border cursor-pointer transition-colors ${
                selectedGroup === group.group_id 
                  ? 'bg-blue-900/40 border-blue-500' 
                  : 'bg-black/20 border-white/10 hover:bg-white/5'
              }`}
            >
              <div className="flex justify-between items-center mb-1">
                <strong className="text-white">Group {group.group_id}</strong>
                <span className="text-xs px-2 py-1 bg-white/10 rounded">
                  {group.member_count} members
                </span>
              </div>
              <div className="text-sm text-gray-400 flex justify-between">
                <span>Stressed: {group.stressed_members}</span>
                <span>Exposure: {(group.aggregate_exposure * 100).toFixed(1)}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Content: Group Details */}
      <div className="w-2/3 flex flex-col gap-4 h-[calc(100vh-80px)]">
        {!selectedGroup ? (
          <div className="glass-panel flex-1 flex items-center justify-center text-gray-400">
            Select a group from the index to view its network structure.
          </div>
        ) : (
          <>
            {loadingGroup ? (
              <div className="glass-panel flex-1 flex items-center justify-center">Loading group data...</div>
            ) : errorGroup ? (
              <div className="glass-panel flex-1 text-red-400 flex items-center justify-center">{errorGroup}</div>
            ) : groupData ? (
              <>
                <div className="glass-panel h-1/2 flex items-center justify-center relative">
                  <h3 className="absolute top-4 left-4 text-white/50 text-sm tracking-wider uppercase">JLG Topology</h3>
                  {renderGraph()}
                </div>
                
                <div className="glass-panel h-1/2 overflow-y-auto">
                  {!selectedNode ? (
                    <div className="h-full flex flex-col items-center justify-center text-gray-400">
                      <Users size={48} className="mb-4 opacity-20" />
                      <p>Click a borrower node to inspect risk and network evidence.</p>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="flex justify-between items-center pb-2 border-b border-white/10">
                        <h3 className="text-xl">Borrower {selectedNode.id}</h3>
                        {selectedNode.current_stress ? (
                          <span className="badge badge-risk-high flex items-center gap-1"><AlertCircle size={14}/> Stressed</span>
                        ) : (
                          <span className="badge badge-predictive flex items-center gap-1"><Activity size={14}/> Healthy</span>
                        )}
                      </div>
                      
                      <div className="grid grid-cols-2 gap-4">
                        {/* Predictive Block */}
                        <div className="bg-red-900/10 border border-red-500/20 p-4 rounded">
                          <h4 className="text-red-400 text-sm font-semibold mb-3 flex justify-between items-center">
                            Predictive
                            <span className="text-[10px] uppercase bg-red-500/20 px-2 py-0.5 rounded">Model C</span>
                          </h4>
                          <div className="text-2xl text-white mb-1">{(selectedNode.operational_risk_score * 100).toFixed(1)}%</div>
                          <p className="text-xs text-gray-400">Calibrated Operational Risk Score</p>
                        </div>
                        
                        {/* Diagnostic Block */}
                        <div className="bg-blue-900/10 border border-blue-500/20 p-4 rounded">
                          <h4 className="text-blue-400 text-sm font-semibold mb-3 flex justify-between items-center">
                            Diagnostic
                            <span className="text-[10px] uppercase bg-blue-500/20 px-2 py-0.5 rounded">Network Evidence</span>
                          </h4>
                          <div className="text-2xl text-white mb-1">{(selectedNode.borrower_propagation_exposure * 100).toFixed(1)}%</div>
                          <p className="text-xs text-gray-400">Modeled Counterfactual Propagation Exposure</p>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-4 mt-2">
                        <div>
                          <p className="text-xs text-gray-500 uppercase tracking-wider">Cash Buffer (4w)</p>
                          <p className="text-white">${selectedNode.cash_buffer_mean_4w.toFixed(2)}</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-500 uppercase tracking-wider">Liability Share</p>
                          <p className="text-white">{(selectedNode.liability_share * 100).toFixed(1)}%</p>
                        </div>
                      </div>
                      
                      <div className="mt-4 p-3 bg-black/40 rounded flex gap-3 text-sm text-gray-400 items-start">
                        <ShieldAlert size={16} className="text-yellow-500 flex-shrink-0 mt-0.5" />
                        <p><strong>Note:</strong> Propagation exposure is a modeled scenario metric derived from joint liability structure. It does not represent observed causality between specific individuals.</p>
                      </div>
                    </div>
                  )}
                </div>
              </>
            ) : null}
          </>
        )}
      </div>
    </div>
  );
};

export default NetworkIntelligence;
