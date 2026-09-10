import React, { useState, useEffect } from 'react';
import { fetchForecast } from '../services/apiService';
import { Database, ShieldAlert, CheckCircle, Crosshair, Target } from 'lucide-react';

export default function AttackAnalysis() {
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

  return (
    <div>
      <h1 className="text-h1" style={{ color: 'var(--accent-cyan)' }}>Attack Intelligence & MITRE Analysis</h1>
      <p className="text-body mb-6">Investigate contextual attack mapping and rule-based stage identification.</p>
      
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
        <div className="text-right">
          <div className="text-xs text-muted mb-1">ACTIVE SCENARIO ID</div>
          <div className="text-mono text-accent-cyan">CIC-IDS2018_{scenario.toUpperCase()}</div>
        </div>
      </div>

      {loading ? (
        <div className="flex justify-center p-12 text-muted">Loading scenario data via FastAPI...</div>
      ) : (
        <>
          <div className="grid grid-cols-3 gap-6 mb-6">
            <div className="nexus-card">
              <div className="text-body text-muted mb-1">Current State Prediction</div>
              <div className="text-h2" style={{ color: data?.current_probability > 0.5 ? 'var(--risk-high)' : 'var(--risk-low)' }}>
                {((data?.current_probability || 0) * 100).toFixed(2)}%
              </div>
            </div>
            <div className="nexus-card">
              <div className="text-body text-muted mb-1">Risk Level</div>
              <div className="text-h2">
                <span className={`badge badge-${(data?.current_risk_level || 'LOW').toLowerCase()}`} style={{ fontSize: '1.25rem', padding: '0.25rem 1rem' }}>
                  {data?.current_risk_level || 'LOW'}
                </span>
              </div>
            </div>
            <div className="nexus-card">
              <div className="text-body text-muted mb-1">MITRE Context Hints</div>
              <div className="text-h2 text-accent-cyan">{data?.mitre_context?.length || 0} Rule Matches</div>
            </div>
          </div>

          <h2 className="text-h2 mb-4 mt-8 flex items-center gap-2"><Target size={20} /> MITRE ATT&CK Contextual Mapping</h2>
          <div className="grid grid-cols-1 gap-4">
            {data?.mitre_context && data.mitre_context.length > 0 ? (
              data.mitre_context.map((ctx, idx) => (
                <div key={idx} className="nexus-card flex flex-col gap-2" style={{ borderLeft: `4px solid ${ctx.confidence === 'High' ? 'var(--risk-high)' : (ctx.confidence === 'Medium' ? 'var(--risk-medium)' : 'var(--accent-blue)')}` }}>
                  <div className="flex justify-between items-center">
                    <h3 className="text-h3">{ctx.stage}</h3>
                    <span className="badge badge-medium">Confidence: {ctx.confidence}</span>
                  </div>
                  <div className="grid grid-cols-2 gap-4 mt-2">
                    <div>
                      <div className="text-xs text-muted mb-1 flex items-center gap-1"><Crosshair size={12}/> Network Evidence</div>
                      <div className="text-body text-mono bg-bg-tertiary p-2 rounded border border-border-subtle">{ctx.evidence}</div>
                    </div>
                    <div>
                      <div className="text-xs text-muted mb-1 flex items-center gap-1"><CheckCircle size={12}/> Reasoning</div>
                      <div className="text-body text-muted">{ctx.reason}</div>
                    </div>
                  </div>
                  {ctx.disclaimer && (
                    <div className="text-xs text-muted mt-2 italic bg-bg-tertiary p-2 rounded">
                      <ShieldAlert size={12} className="inline mr-1"/> {ctx.disclaimer}
                    </div>
                  )}
                </div>
              ))
            ) : (
              <div className="nexus-card text-center text-muted py-8">
                No distinct MITRE ATT&CK heuristics detected in the current flow context.
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
