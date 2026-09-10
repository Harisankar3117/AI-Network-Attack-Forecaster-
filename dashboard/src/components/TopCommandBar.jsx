import React from 'react';
import { ShieldCheck, WifiOff, Database } from 'lucide-react';

export function TopCommandBar({ scenario }) {
  return (
    <div className="top-bar">
      <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--risk-low)' }}>
          <ShieldCheck size={18} />
          <span style={{ fontSize: '0.875rem', fontWeight: 500 }}>System Secure</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--text-muted)' }}>
          <WifiOff size={18} />
          <span style={{ fontSize: '0.875rem' }}>Offline Mode (Local Analysis)</span>
        </div>
      </div>
      
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Database size={16} color="var(--accent-blue)" />
          <span className="text-body">Dataset: <strong>{scenario || 'CIC-IDS2018'}</strong></span>
        </div>
        <div style={{ width: '1px', height: '24px', backgroundColor: 'var(--border-subtle)' }}></div>
        <div className="text-mono" style={{ color: 'var(--text-muted)' }}>
          Last Analysis: {new Date().toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
}
