import React, { useState, useEffect } from 'react';
import { fetchBenchmark } from '../services/apiService';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Target, AlertTriangle, ShieldCheck } from 'lucide-react';

export default function Benchmark() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchBenchmark().then(res => {
      setData(res);
      setLoading(false);
    }).catch(console.error);
  }, []);

  if (loading) return <div className="text-body flex items-center justify-center h-full">Loading Benchmark Data...</div>;

  const benchmarkData = [
    {
      scenario: 'Wednesday (Validation - Best Threshold: 0.25)',
      Accuracy: parseFloat(data?.lstm_results?.Wednesday_validation?.Accuracy),
      Precision: parseFloat(data?.lstm_results?.Wednesday_validation?.Precision),
      Recall: parseFloat(data?.lstm_results?.Wednesday_validation?.Recall),
      F1: parseFloat(data?.lstm_results?.Wednesday_validation?.F1),
      ROC_AUC: parseFloat(data?.lstm_results?.Wednesday_validation?.ROC_AUC)
    },
    {
      scenario: 'Friday (Unseen DoS)',
      Accuracy: parseFloat(data?.lstm_results?.Friday_unseen_DoS?.Accuracy),
      Precision: parseFloat(data?.lstm_results?.Friday_unseen_DoS?.Precision),
      Recall: parseFloat(data?.lstm_results?.Friday_unseen_DoS?.Recall),
      F1: parseFloat(data?.lstm_results?.Friday_unseen_DoS?.F1),
      ROC_AUC: parseFloat(data?.lstm_results?.Friday_unseen_DoS?.ROC_AUC)
    },
    {
      scenario: 'March 1 (Unseen Infiltration)',
      Accuracy: parseFloat(data?.lstm_results?.March_1_unseen_Infiltration?.Accuracy),
      Precision: parseFloat(data?.lstm_results?.March_1_unseen_Infiltration?.Precision),
      Recall: parseFloat(data?.lstm_results?.March_1_unseen_Infiltration?.Recall),
      F1: parseFloat(data?.lstm_results?.March_1_unseen_Infiltration?.F1),
      ROC_AUC: parseFloat(data?.lstm_results?.March_1_unseen_Infiltration?.ROC_AUC)
    }
  ];

  return (
    <div>
      <h1 className="text-h1" style={{ color: 'var(--accent-cyan)' }}>Model Benchmark & Generalization</h1>
      <p className="text-body mb-6">Scientific evaluation of the AI World Model against unseen attack scenarios.</p>
      
      <div className="nexus-card mb-6 border-l-4 border-l-risk-medium bg-bg-tertiary">
        <h3 className="text-h2 mb-2 flex items-center gap-2"><AlertTriangle size={18} /> Generalization Challenge</h3>
        <p className="text-body">
          {data?.limitations || 'Loading limitations...'}
        </p>
      </div>

      <div className="nexus-card mb-8" style={{ height: '400px' }}>
        <h2 className="text-h2 mb-4 flex items-center gap-2"><Target size={20} /> Cross-Day Generalization Performance</h2>
        <ResponsiveContainer width="100%" height="85%">
          <BarChart data={benchmarkData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border-subtle)" vertical={false} />
            <XAxis dataKey="scenario" stroke="var(--text-muted)" />
            <YAxis stroke="var(--text-muted)" domain={[0, 100]} tickFormatter={(val) => `${val}%`} />
            <Tooltip contentStyle={{ backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--border-subtle)' }} />
            <Legend />
            <Bar dataKey="Accuracy" fill="var(--text-muted)" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Recall" fill="var(--accent-cyan)" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Precision" fill="var(--accent-purple)" radius={[4, 4, 0, 0]} />
            <Bar dataKey="ROC_AUC" name="ROC AUC" fill="var(--accent-blue)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="nexus-card">
        <h2 className="text-h2 mb-4">Detailed Metrics Table</h2>
        <div className="overflow-x-auto">
          <table style={{ width: '100%', textAlign: 'left', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border-subtle)' }}>
                <th className="p-3 text-muted font-medium">Scenario</th>
                <th className="p-3 text-muted font-medium">Accuracy</th>
                <th className="p-3 text-muted font-medium">Precision</th>
                <th className="p-3 text-muted font-medium">Recall</th>
                <th className="p-3 text-muted font-medium">F1 Score</th>
                <th className="p-3 text-muted font-medium">FPR</th>
                <th className="p-3 text-muted font-medium">ROC-AUC</th>
              </tr>
            </thead>
            <tbody>
              {benchmarkData.map((row, i) => (
                <tr key={i} style={{ borderBottom: '1px solid rgba(42, 52, 74, 0.5)' }}>
                  <td className="p-3 font-semibold text-text-primary">{row.scenario}</td>
                  <td className="p-3">{row.Accuracy}%</td>
                  <td className="p-3">{row.Precision}%</td>
                  <td className={`p-3 font-bold ${row.Recall < 20 ? 'text-risk-high' : 'text-risk-medium'}`}>{row.Recall}%</td>
                  <td className="p-3">{row.F1}%</td>
                  <td className="p-3">{row.scenario.includes('Friday') ? data.lstm_results.Friday_unseen_DoS.FPR : (row.scenario.includes('Wednesday') ? data.lstm_results.Wednesday_validation.FPR : data.lstm_results.March_1_unseen_Infiltration.FPR)}</td>
                  <td className="p-3 text-accent-cyan">{row.ROC_AUC}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      <div className="nexus-card mt-6">
        <h2 className="text-h2 mb-4 flex items-center gap-2"><ShieldCheck size={20} /> Logistic Regression Baseline (Wednesday Validation)</h2>
        <div className="flex gap-4">
            <div className="text-body"><span className="text-muted">Accuracy:</span> {data?.logistic_regression_baseline?.Accuracy}</div>
            <div className="text-body"><span className="text-muted">Precision:</span> {data?.logistic_regression_baseline?.Precision}</div>
            <div className="text-body"><span className="text-muted">Recall:</span> {data?.logistic_regression_baseline?.Recall}</div>
            <div className="text-body"><span className="text-muted">ROC-AUC:</span> {data?.logistic_regression_baseline?.ROC_AUC}</div>
        </div>
      </div>
    </div>
  );
}
