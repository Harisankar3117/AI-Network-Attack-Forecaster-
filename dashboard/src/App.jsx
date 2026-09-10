import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout } from './components/Layout';
import CommandCenter from './pages/CommandCenter';
import AttackForecast from './pages/AttackForecast';
import ThreatTrajectory from './pages/ThreatTrajectory';
import NetworkState from './pages/NetworkState';
import AttackAnalysis from './pages/AttackAnalysis';
import Explainability from './pages/Explainability';
import Benchmark from './pages/Benchmark';
import DemoMode from './pages/DemoMode';
import { ForecastProvider } from './context/ForecastContext';
import './App.css';

function App() {
  return (
    <ForecastProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<CommandCenter />} />
            <Route path="forecast" element={<AttackForecast />} />
            <Route path="threat-trajectory" element={<ThreatTrajectory />} />
            <Route path="network-state" element={<NetworkState />} />
            <Route path="attack-analysis" element={<AttackAnalysis />} />
            <Route path="explainability" element={<Explainability />} />
            <Route path="benchmark" element={<Benchmark />} />
            <Route path="demo" element={<DemoMode />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ForecastProvider>
  );
}

export default App;
