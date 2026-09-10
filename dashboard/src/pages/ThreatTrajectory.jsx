import React, { useState, useEffect } from 'react';
import { fetchForecast } from '../services/apiService';
import { Target, AlertTriangle, ShieldOff, Database } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

export default function ThreatTrajectory() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [scenario, setScenario] = useState('wednesday');

  useEffect(() => {
    setLoading(true);
    fetchForecast(scenario, 5).then(res => {
      setData(res);
      setLoading(false);
    }).catch(console.error);
  }, [scenario]);

  const trajectoryData = data?.forecast_windows.map(w => ({
    Window: `t+${w.window_index}`,
    Probability: w.probability,
    Risk: w.risk_level
  })) || [];

  return (
    <div>
      <h1 className="text-h1" style={{ color: 'var(--accent-cyan)' }}>Threat Trajectory & Attack Progression</h1>
      <p className="text-body mb-6">Visualizing the recursive K-step future state progression.</p>
      
      <div className="nexus-card mb-6 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Database size={20} color="var(--accent-blue)" />
          <div>
            <div className="text-xs text-muted mb-1">DATASET SELECTION</div>
            <select 
              className="bg-bg-tertiary border border-border-subtle text-text-primary rounded p-2 outline-none focus:border-accent-cyan"
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
              style={{ minWidth: '400px' }}
            >
              <option value="wednesday">Wednesday 14-02-2018 - Brute Force Validation</option>
              <option value="friday">Friday 16-02-2018 - DoS Unseen Day</option>
              <option value="march1">Thursday 01-03-2018 - Infiltration Unseen Day</option>
            </select>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center p-12 text-muted">Projecting future states via FastAPI...</div>
      ) : (
        <>
          <div className="nexus-card mb-8">
            <h2 className="text-h2 mb-6 flex items-center gap-2"><Target size={20} /> Future Forecast Progression</h2>
            
            <div style={{ display: 'flex', alignItems: 'center', position: 'relative', padding: '2rem 0' }}>
              <div style={{ position: 'absolute', top: '50%', left: '0', right: '0', height: '2px', backgroundColor: 'var(--border-subtle)', zIndex: 0 }}></div>
              
              <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', position: 'relative', zIndex: 1 }}>
                  <div style={{ 
                      width: '40px', height: '40px', borderRadius: '50%', 
                      backgroundColor: 'var(--bg-tertiary)', border: '2px solid var(--accent-cyan)',
                      display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem'
                    }}>
                    <ShieldOff size={18} color="var(--accent-cyan)" />
                  </div>
                  <div className="text-sm font-semibold">Current S(t)</div>
                  <div className={`text-xs mt-1 text-risk-${(data?.current_risk_level || 'low').toLowerCase()}`}>{(data?.current_probability * 100).toFixed(1)}%</div>
              </div>

              {trajectoryData.map((stage, idx) => (
                <div key={idx} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', position: 'relative', zIndex: 1 }}>
                  <div 
                    style={{ 
                      width: '40px', height: '40px', borderRadius: '50%', 
                      backgroundColor: stage.Risk === 'High' ? 'var(--risk-high)' : (stage.Risk === 'Medium' ? 'var(--risk-medium)' : 'var(--bg-tertiary)'),
                      border: `2px solid ${stage.Risk === 'Low' ? 'var(--border-subtle)' : 'var(--bg-primary)'}`,
                      display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1rem',
                      boxShadow: stage.Risk === 'High' ? '0 0 15px rgba(239,68,68,0.5)' : 'none'
                    }}
                  >
                    {stage.Risk === 'High' && <AlertTriangle size={18} color="#fff" />}
                  </div>
                  <div className="text-sm font-semibold">{stage.Window}</div>
                  <div className={`text-xs mt-1 text-risk-${stage.Risk.toLowerCase()}`}>{(stage.Probability * 100).toFixed(1)}%</div>
                </div>
              ))}
            </div>
          </div>

          <div className="nexus-card" style={{ height: '300px' }}>
            <h2 className="text-h2 mb-4">Risk Extrapolation Trend</h2>
            <ResponsiveContainer width="100%" height="85%">
              <AreaChart data={[{ Window: 'Current', Probability: data?.current_probability }, ...trajectoryData]}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
                <XAxis dataKey="Window" stroke="var(--text-muted)" />
                <YAxis stroke="var(--text-muted)" domain={[0, 1]} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
                <Tooltip contentStyle={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)' }} formatter={(value) => [`${(value * 100).toFixed(2)}%`, 'Attack Probability']} />
                <ReferenceLine y={0.75} stroke="var(--risk-high)" strokeDasharray="3 3" />
                <ReferenceLine y={0.50} stroke="var(--risk-medium)" strokeDasharray="3 3" />
                <Area type="monotone" dataKey="Probability" stroke="var(--accent-purple)" fill="var(--accent-purple)" fillOpacity={0.2} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}
