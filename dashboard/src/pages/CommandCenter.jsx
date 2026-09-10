import React, { useState, useEffect } from 'react';
import { fetchForecast, fetchModelStatus } from '../services/apiService';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Activity, ShieldAlert, Zap, Server, ActivitySquare } from 'lucide-react';

export default function CommandCenter() {
  const [data, setData] = useState(null);
  const [modelStatus, setModelStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetchForecast('wednesday', 10),
      fetchModelStatus()
    ]).then(([forecastRes, statusRes]) => {
      setData(forecastRes);
      setModelStatus(statusRes);
      setLoading(false);
    }).catch(console.error);
  }, []);

  if (loading) return <div className="text-body flex items-center justify-center h-full">Initializing Command Center via FastAPI...</div>;

  const currentRisk = data?.current_risk_level || 'LOW';
  
  // Format timeline data for the chart from forecast_windows
  const timelineData = data?.forecast_windows.map(w => ({
    Window: `t+${w.window_index}`,
    Attack_Probability: w.probability
  })) || [];
  
  return (
    <div>
      <div className="flex justify-between items-center mb-8">
        <div>
          <h1 className="text-h1" style={{ color: 'var(--accent-cyan)' }}>Predictive Cyber Defense Command Center</h1>
          <p className="text-body">Observe network state. Forecast attacker progression. Act before compromise.</p>
        </div>
        <div className={`badge badge-${currentRisk.toLowerCase()}`} style={{ fontSize: '1rem', padding: '0.5rem 1rem' }}>
          SYSTEM RISK: {currentRisk}
        </div>
      </div>

      {/* Top Stats Cards */}
      <div className="grid grid-cols-4 gap-6 mb-6">
        <div className="nexus-card">
          <div className="flex items-center gap-2 text-muted mb-2"><ShieldAlert size={16} /> Peak Forecast Risk</div>
          <div className="text-h1" style={{ color: 'var(--risk-high)' }}>{((data?.summary?.peak_forecast_risk_probability || 0) * 100).toFixed(1)}%</div>
          <div className="text-body mt-2">Maximum probability in forecast horizon.</div>
        </div>
        <div className="nexus-card">
          <div className="flex items-center gap-2 text-muted mb-2"><Activity size={16} /> Analyzed Sequences</div>
          <div className="text-h1">5,000</div>
          <div className="text-body mt-2">Processed in offline analysis scenario.</div>
        </div>
        <div className="nexus-card">
          <div className="flex items-center gap-2 text-muted mb-2"><Zap size={16} /> Predicted Attack Windows</div>
          <div className="text-h1" style={{ color: 'var(--risk-medium)' }}>{data?.summary?.predicted_attack_count ?? 0}</div>
          <div className="text-body mt-2">Future forecast windows at High/Critical risk.</div>
        </div>
        <div className="nexus-card">
          <div className="flex items-center gap-2 text-muted mb-2"><Server size={16} /> AI World Model</div>
          <div className="text-h1" style={{ color: 'var(--accent-blue)' }}>ONLINE</div>
          <div className="text-body mt-2">LSTM Temporal Sequence Modeling Active.</div>
        </div>
      </div>

      {/* Main Forecast Area */}
      <div className="grid grid-cols-3 gap-6">
        <div className="nexus-card" style={{ gridColumn: 'span 2' }}>
          <h2 className="text-h2 mb-4">Threat Forecast Timeline (K-Steps)</h2>
          <div style={{ height: '300px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={timelineData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
                <XAxis dataKey="Window" stroke="var(--text-muted)" tick={{ fill: 'var(--text-muted)', fontSize: 12 }} />
                <YAxis stroke="var(--text-muted)" domain={[0, 1]} tick={{ fill: 'var(--text-muted)', fontSize: 12 }} tickFormatter={(val) => `${(val*100).toFixed(0)}%`} />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)', borderRadius: '4px' }}
                  labelStyle={{ color: 'var(--text-muted)' }}
                  itemStyle={{ color: 'var(--text-primary)' }}
                  formatter={(value) => [`${(value * 100).toFixed(2)}%`, 'Attack Probability']}
                />
                <ReferenceLine y={0.75} stroke="var(--risk-high)" strokeDasharray="3 3" />
                <ReferenceLine y={0.50} stroke="var(--risk-medium)" strokeDasharray="3 3" />
                <ReferenceLine y={0.25} stroke="var(--accent-blue)" strokeDasharray="3 3" />
                <Line type="monotone" dataKey="Attack_Probability" name="Attack Probability" stroke="var(--accent-cyan)" dot={{ r: 4, fill: 'var(--bg-secondary)' }} strokeWidth={2} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="flex flex-col gap-6">
          <div className="nexus-card flex-1">
            <h2 className="text-h2 mb-4">Prediction Summary</h2>
            <div className="space-y-4">
              <div>
                <div className="text-body mb-1">Average Forecast Risk</div>
                <div className="flex items-center justify-between">
                  <div style={{ width: '80%', height: '8px', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{ width: `${(data?.summary?.average_forecast_risk_probability || 0) * 100}%`, height: '100%', backgroundColor: 'var(--accent-blue)' }}></div>
                  </div>
                  <span className="text-mono">{((data?.summary?.average_forecast_risk_probability || 0) * 100).toFixed(1)}%</span>
                </div>
              </div>
              <div className="mt-4">
                <div className="text-body mb-1">Current Selected Scenario</div>
                <div className="badge badge-medium" style={{ display: 'block', textAlign: 'center', padding: '0.5rem' }}>Wednesday (Brute Force Validation)</div>
              </div>
              <div className="mt-4">
                <div className="text-body mb-2">Model Status</div>
                <div className="flex items-center gap-2"><ActivitySquare size={14} color="var(--risk-low)"/> <span className="text-mono">Backend: FastAPI</span></div>
                <div className="flex items-center gap-2 mt-1"><ActivitySquare size={14} color="var(--risk-low)"/> <span className="text-mono">Model: {modelStatus?.model_name || 'Loading'}</span></div>
                <div className="flex items-center gap-2 mt-1"><ActivitySquare size={14} color="var(--risk-low)"/> <span className="text-mono">Sequence Length: {modelStatus?.sequence_length || 10}</span></div>
              </div>
              <div className="mt-4 p-3 border border-border-subtle rounded bg-bg-secondary">
                <div className="text-xs text-muted">EARLY WARNING SYSTEM</div>
                <div className="text-sm font-semibold mt-1" style={{ color: data?.summary?.earliest_high_risk_window ? 'var(--risk-high)' : 'var(--text-primary)' }}>
                  {data?.summary?.warning_message || 'Stable'}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
