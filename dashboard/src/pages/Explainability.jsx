import React, { useState, useEffect } from 'react';
import { fetchForecast } from '../services/apiService';
import { ShieldAlert, ChevronRight, Target, Database } from 'lucide-react';

export default function Explainability() {
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

  const maxImportance = data?.explainability?.length > 0 ? Math.max(...data.explainability.map(f => f.importance)) : 1;

  return (
    <div>
      <h1 className="text-h1" style={{ color: 'var(--accent-cyan)' }}>Explainable Threat Intelligence</h1>
      <p className="text-body mb-6">Understanding the temporal network features driving the LSTM's risk forecast using Gradient Attribution.</p>
      
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

      <div className="nexus-card mb-8 border-l-4 border-l-risk-medium bg-bg-tertiary">
        <h3 className="text-h2 mb-2 flex items-center gap-2"><Target size={18} /> Model Explainability (Input x Gradient)</h3>
        <p className="text-body">
          These features represent the true, mathematically calculated attributions driving the LSTM's current forecast. The values are computed using Input x Gradient attribution across the entire temporal sequence (10 timesteps) via PyTorch's autograd engine.
        </p>
      </div>

      {loading ? (
        <div className="flex justify-center p-12 text-muted">Calculating Gradient Attribution via FastAPI...</div>
      ) : (
        <div className="grid grid-cols-2 gap-6">
          {data?.explainability?.slice(0, 10).map((feat, i) => {
            // Normalize to 1-10 scale for UI bars
            const strength = Math.ceil((feat.importance / maxImportance) * 10) || 1;
            
            return (
              <div key={i} className="nexus-card">
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <h3 className="text-h2 mb-1 flex items-center gap-2 text-accent-blue"><ChevronRight size={18} /> {feat.feature}</h3>
                    <div className="text-xs text-muted font-mono">DIRECTION: {feat.direction}</div>
                  </div>
                  <div className="flex gap-1">
                    {Array.from({length: 10}).map((_, j) => (
                      <div key={j} style={{
                        width: '6px', height: '20px', borderRadius: '2px',
                        backgroundColor: j < strength ? (j > 7 ? 'var(--risk-high)' : (j > 4 ? 'var(--risk-medium)' : 'var(--risk-low)')) : 'var(--bg-tertiary)'
                      }}></div>
                    ))}
                  </div>
                </div>
                <p className="text-body text-sm font-mono mt-2 bg-bg-secondary p-2 rounded">
                  Raw Attribution Score: {feat.raw_attribution.toExponential(2)}
                </p>
                <p className="text-body mt-2">{feat.explanation}</p>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
