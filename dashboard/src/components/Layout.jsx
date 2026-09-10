import React from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { TopCommandBar } from './TopCommandBar';

export function Layout() {
  return (
    <div className="app-container">
      <Sidebar />
      <div className="main-content">
        <TopCommandBar scenario="CIC-IDS2018" />
        <div className="page-container">
          <Outlet />
        </div>
      </div>
    </div>
  );
}
