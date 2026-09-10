import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Play, Loader, ShieldAlert, Cpu, UploadCloud } from 'lucide-react';
import { uploadCsv } from '../services/apiService';

export default function DemoMode() {
  const [running, setRunning] = useState(false);
  const [file, setFile] = useState(null);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleRunForecast = async () => {
    if (!file) {
      setError('Please select a CSV file first.');
      return;
    }
    setRunning(true);
    setError('');
    
    try {
      const res = await uploadCsv(file, 15);
      navigate('/forecast', { state: { uploadedData: res } });
    } catch (err) {
      setError('Failed to upload CSV. Please try again.');
      setRunning(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-[80vh]">
      <div className="text-center mb-8">
        <h1 className="text-h1 mb-2" style={{ color: 'var(--accent-cyan)', fontSize: '2.5rem' }}>ByteStorm- Forecast</h1>
        <p className="text-body" style={{ fontSize: '1.25rem' }}>Offline Custom CSV Forecasting</p>
      </div>
      
      <div className="nexus-card" style={{ width: '600px', maxWidth: '100%', padding: '3rem' }}>
        <h2 className="text-h2 mb-8 text-center">Predictive Cyber Defense Console</h2>
        
        <div className="space-y-6 mb-8">
          <div className="flex items-center gap-4 text-muted">
            <Cpu size={24} />
            <div>
              <div className="font-semibold text-text-primary">1. Model Initialization</div>
              <div className="text-sm">Load LSTM weights and temporal scaler...</div>
            </div>
          </div>
          <div className="flex items-center gap-4 text-muted">
            <ShieldAlert size={24} />
            <div>
              <div className="font-semibold text-text-primary">2. Traffic Ingestion</div>
              <div className="text-sm">Upload custom offline CSV file...</div>
            </div>
          </div>
          <div className="flex items-center gap-4 text-muted">
            <Play size={24} />
            <div>
              <div className="font-semibold text-text-primary">3. Forecasting</div>
              <div className="text-sm">Compute attack probability and threat trajectory...</div>
            </div>
          </div>
        </div>

        <div className="mb-6">
          <label className="block text-body mb-2 text-text-primary font-semibold flex items-center gap-2">
            <UploadCloud size={18} />
            Select Custom CSV File
          </label>
          <input 
            type="file" 
            accept=".csv" 
            onChange={(e) => setFile(e.target.files[0])}
            className="block w-full text-sm text-slate-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-full file:border-0
              file:text-sm file:font-semibold
              file:bg-[var(--bg-tertiary)] file:text-[var(--accent-cyan)]
              hover:file:bg-[var(--border-subtle)] focus:outline-none"
            disabled={running}
          />
          {file && <div className="mt-2 text-sm text-accent-cyan">Selected: {file.name}</div>}
          {error && <div className="mt-2 text-sm text-[var(--risk-high)]">{error}</div>}
        </div>

        {running ? (
          <div className="flex flex-col items-center justify-center py-4">
            <Loader className="animate-spin mb-4" size={32} color="var(--accent-cyan)" />
            <div className="text-body mb-2 text-accent-cyan animate-pulse">Running Offline Forecast...</div>
          </div>
        ) : (
          <div className="flex justify-center mt-8">
            <button 
              className="nexus-btn" 
              style={{ fontSize: '1.125rem', padding: '1rem 3rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}
              onClick={handleRunForecast}
            >
              <Play size={20} />
              Run Offline Forecast
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
