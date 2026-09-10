import React, { useState, useEffect } from 'react';
import { fetchForecast } from '../services/apiService';
import { Network, Activity, ArrowRightLeft, Database } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function NetworkState() {
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

  const formatFeatureValue = (key, val) => {
    if (val === undefined || val === null) return 'N/A';
    if (key.includes('Port') || key.includes('Protocol') || key.includes('Cnt')) return Math.round(val);
    if (val > 1000) return val.toLocaleString(undefined, { maximumFractionDigits: 0 });
    return val.toFixed(2);
  };

  const rawFeatures = data?.raw_features ? Object.entries(data.raw_features).map(([k, v]) => ({
    name: k,
    value: formatFeatureValue(k, v),
    type: 'Feature'
  })) : [];

  // Use real backend forecasted data for Flow Pkts/s
  const realTrafficData = React.useMemo(() => {
    if (!data) return [];
    
    // Add current state
    const sequence = [
      { time: 't=0', flowPktsPerSec: data.raw_features?.['Flow Pkts/s'] || 0 }
    ];
    
    // Add future forecasted states
    if (data.future_states && Array.isArray(data.future_states)) {
      data.future_states.forEach((state, i) => {
        sequence.push({
          time: `t+${i + 1}`,
          flowPktsPerSec: state['Flow Pkts/s'] || 0
        });
      });
    }
    return sequence;
  }, [data]);

  return (
    <div>
      <h1 className="text-h1" style={{ color: 'var(--accent-cyan)' }}>Evolving Network State</h1>
      <p className="text-body mb-6">Representing historical network traffic as evolving states S(t) → S(t+1) → future risk. <span className="text-muted text-xs ml-2">(Offline Analysis)</span></p>
      
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
        <div className="flex justify-center p-12 text-muted">Extracting State Features via FastAPI...</div>
      ) : (
        <>
          <div className="grid grid-cols-3 gap-6 mb-8">
            <div className="nexus-card" style={{ gridColumn: 'span 2' }}>
              <h2 className="text-h2 mb-4 flex items-center gap-2"><Network size={20} /> Current State Vector S(t)</h2>
              <div className="grid grid-cols-3 gap-3">
                {rawFeatures.map(f => (
                  <div key={f.name} className="p-3 border rounded border-border-subtle bg-bg-tertiary">
                    <div className="text-xs text-muted mb-1">{f.name.split(' ').pop()}</div>
                    <div className="text-sm font-semibold text-ellipsis overflow-hidden whitespace-nowrap">{f.name}</div>
                    <div className="text-mono mt-2" style={{ color: 'var(--accent-blue)' }}>{f.value}</div>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="nexus-card flex flex-col justify-center items-center relative">
              <h2 className="text-h2 mb-4 self-start">State Transition Risk</h2>
              
              <div className="w-full max-w-[200px] border border-border-subtle bg-bg-tertiary rounded p-4 text-center mb-2">
                <div className="text-xs text-muted">Current S(t)</div>
                <div className={`text-lg text-risk-${data?.current_risk_level?.toLowerCase()}`}>{data?.current_risk_level || 'LOW'} Risk</div>
              </div>
              
              <ArrowRightLeft size={24} className="text-muted my-2 rotate-90" />
              
              <div className="w-full max-w-[200px] border border-accent-cyan bg-bg-tertiary rounded p-4 text-center mt-2 shadow-[0_0_15px_rgba(0,240,255,0.2)]">
                <div className="text-xs text-accent-cyan">Forecast S(t+K)</div>
                <div className={`text-lg text-risk-${data?.summary?.peak_forecast_risk_level?.toLowerCase()}`}>{data?.summary?.peak_forecast_risk_level || 'LOW'} Peak</div>
              </div>
            </div>
          </div>

          <div className="nexus-card" style={{ height: '300px' }}>
            <h2 className="text-h2 mb-4 flex items-center gap-2"><Activity size={20} /> Forecasted Flow Packet Rate (Flow Pkts/s)</h2>
            <ResponsiveContainer width="100%" height="85%">
              <AreaChart data={realTrafficData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
                <XAxis dataKey="time" stroke="var(--text-muted)" />
                <YAxis stroke="var(--text-muted)" />
                <Tooltip contentStyle={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)' }} />
                <Area type="monotone" dataKey="flowPktsPerSec" stroke="var(--accent-purple)" fill="var(--accent-purple)" fillOpacity={0.2} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </>
      )}
    </div>
  );
}
