import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, TrendingUp, Network, Crosshair, BarChart2, ShieldAlert, Zap, Presentation } from 'lucide-react';

export function Sidebar() {
  const navItems = [
    { to: '/', icon: <LayoutDashboard size={20} />, label: 'Command Center' },
    { to: '/forecast', icon: <TrendingUp size={20} />, label: 'Attack Forecast' },
    { to: '/threat-trajectory', icon: <Crosshair size={20} />, label: 'Threat Trajectory' },
    { to: '/network-state', icon: <Network size={20} />, label: 'Network State' },
    { to: '/attack-analysis', icon: <BarChart2 size={20} />, label: 'Attack Analysis' },
    { to: '/explainability', icon: <ShieldAlert size={20} />, label: 'Explainability' },
    { to: '/benchmark', icon: <Zap size={20} />, label: 'Model Benchmark' },
    { to: '/demo', icon: <Presentation size={20} />, label: 'Demo Mode' },
  ];

  return (
    <div className="sidebar" style={{ paddingTop: '1.5rem' }}>
      <div style={{ padding: '0 1.5rem', marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 700, color: 'var(--accent-cyan)', letterSpacing: '0.05em', margin: 0 }}>
          ByteStorm<span style={{ color: 'var(--text-primary)' }}>- Forecast</span>
        </h2>
        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
          AI Predictive Defense Console
        </div>
      </div>
      
      <nav style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', padding: '0 1rem' }}>
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            style={({ isActive }) => ({
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
              padding: '0.75rem 1rem',
              borderRadius: '6px',
              textDecoration: 'none',
              color: isActive ? 'var(--accent-cyan)' : 'var(--text-secondary)',
              backgroundColor: isActive ? 'rgba(0, 240, 255, 0.1)' : 'transparent',
              fontWeight: isActive ? 600 : 500,
              transition: 'all 0.2s ease',
            })}
          >
            {item.icon}
            {item.label}
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
