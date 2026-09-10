import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { fetchForecast } from '../services/apiService';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Area, AreaChart } from 'recharts';
import { ShieldAlert, TrendingUp, AlertTriangle, Database } from 'lucide-react';

export default function AttackForecast() {
  const location = useLocation();
  const uploadedData = location.state?.uploadedData;

  const [data, setData] = useState(uploadedData || null);
  const [loading, setLoading] = useState(!uploadedData);
  const [scenario, setScenario] = useState(uploadedData ? 'custom' : 'wednesday');

  useEffect(() => {
    if (scenario === 'custom') {
      if (uploadedData) {
        setData(uploadedData);
        setLoading(false);
      }
      return;
    }

    setLoading(true);
    fetchForecast(scenario, 15).then(res => { // longer forecast for AreaChart
      setData(res);
      setLoading(false);
    }).catch(console.error);
  }, [scenario, uploadedData]);

  const trajectoryData = data?.forecast_windows.map(w => ({
    Window: `t+${w.window_index}`,
    Attack_Probability: w.probability
  })) || [];
  
  if (data) {
      trajectoryData.unshift({
          Window: 'Current',
          Attack_Probability: data.current_probability
      });
  }

  return (
    <div>
      <h1 className="text-h1" style={{ color: 'var(--accent-cyan)' }}>Attack Forecast Engine</h1>
      <p className="text-body mb-6">The model learns temporal network-state dynamics and estimates future attack risk from evolving traffic patterns.</p>
      
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
              {uploadedData && <option value="custom">Custom Uploaded CSV</option>}
              <option value="wednesday">Wednesday 14-02-2018 - Brute Force Validation</option>
              <option value="friday">Friday 16-02-2018 - DoS Unseen Day</option>
              <option value="march1">Thursday 01-03-2018 - Infiltration Unseen Day</option>
            </select>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center p-12 text-muted">Loading forecast engine via FastAPI...</div>
      ) : (
        <>
          <div className="grid grid-cols-4 gap-6 mb-6">
            <div className="nexus-card">
              <div className="text-body text-muted mb-1">Forecast Horizon</div>
              <div className="text-h2">15 Steps</div>
            </div>
            <div className="nexus-card">
              <div className="text-body text-muted mb-1">Active Model</div>
              <div className="text-h2 text-accent-blue">LSTM World Model</div>
            </div>
            <div className="nexus-card">
              <div className="text-body text-muted mb-1">Peak Probability</div>
              <div className="text-h2">{(data?.summary?.peak_forecast_risk_probability * 100).toFixed(1)}%</div>
            </div>
            <div className="nexus-card">
              <div className="text-body text-muted mb-1">Current Risk Level</div>
              <div className={`badge badge-${(data?.current_risk_level || 'low').toLowerCase()}`}>{data?.current_risk_level} RISK</div>
            </div>
          </div>

          <div className="nexus-card mb-6" style={{ height: '450px' }}>
            <h2 className="text-h2 mb-4 flex items-center gap-2">
              <TrendingUp size={20} color="var(--accent-cyan)"/> 
              Recursive K-Step Probability Timeline
            </h2>
            <ResponsiveContainer width="100%" height="85%">
              <AreaChart data={trajectoryData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorProb" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--risk-high)" stopOpacity={0.8}/>
                    <stop offset="50%" stopColor="var(--risk-medium)" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="var(--accent-cyan)" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
                <XAxis dataKey="Window" stroke="var(--text-muted)" tick={{ fill: 'var(--text-muted)', fontSize: 12 }} />
                <YAxis stroke="var(--text-muted)" domain={[0, 1]} tick={{ fill: 'var(--text-muted)', fontSize: 12 }} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)', borderRadius: '4px' }}
                  labelStyle={{ color: 'var(--text-muted)' }}
                  itemStyle={{ color: 'var(--text-primary)' }}
                  formatter={(value) => [`${(value * 100).toFixed(2)}%`, 'Attack Probability']}
                />
                <ReferenceLine y={0.75} stroke="var(--risk-high)" strokeDasharray="3 3" label={{ position: 'insideTopLeft', value: 'Critical Risk Threshold (0.75)', fill: 'var(--risk-high)', fontSize: 12 }} />
                <ReferenceLine y={0.50} stroke="var(--risk-medium)" strokeDasharray="3 3" label={{ position: 'insideTopLeft', value: 'High Risk Threshold (0.50)', fill: 'var(--risk-medium)', fontSize: 12 }} />
                <ReferenceLine y={0.25} stroke="var(--accent-blue)" strokeDasharray="3 3" label={{ position: 'insideTopLeft', value: 'Medium Risk Threshold (0.25)', fill: 'var(--accent-blue)', fontSize: 12 }} />
                <Area type="monotone" dataKey="Attack_Probability" stroke="var(--accent-cyan)" fillOpacity={1} fill="url(#colorProb)" isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-2 gap-6">
            <div className="nexus-card">
              <h2 className="text-h2 mb-2 flex items-center gap-2"><ShieldAlert size={18} /> Temporal Interpretation</h2>
              <p className="text-body mb-4">
                The visualization displays the LSTM's probabilistic risk assessment as network sequences are recursively forecasted. 
                Rapid spikes indicate sudden deviations in flow behavior (e.g. DoS bursts), while gradual inclines suggest mounting reconnaissance or infiltration activities.
              </p>
              <div className="flex gap-4">
                <div className="flex items-center gap-2"><div style={{width: 12, height: 12, backgroundColor: 'var(--risk-low)', borderRadius: '50%'}}></div> <span className="text-body">Stable</span></div>
                <div className="flex items-center gap-2"><div style={{width: 12, height: 12, backgroundColor: 'var(--risk-medium)', borderRadius: '50%'}}></div> <span className="text-body">Risk Increasing</span></div>
                <div className="flex items-center gap-2"><div style={{width: 12, height: 12, backgroundColor: 'var(--risk-high)', borderRadius: '50%'}}></div> <span className="text-body">Critical State</span></div>
              </div>
            </div>
            
            <div className="nexus-card">
              <h2 className="text-h2 mb-2 flex items-center gap-2"><AlertTriangle size={18} /> Analyst Decision Support</h2>
              <p className="text-body mb-4">
                Forecasted states consistently hovering above the 0.50 threshold warrant automated mitigation scaling, while predictions breaching 0.75 indicate high-confidence active attacks requiring immediate intervention.
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
