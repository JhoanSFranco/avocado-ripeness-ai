import React from 'react';

const STATUS = {
  null: { label: 'Checking', modifier: 'checking' },
  true: { label: 'Online', modifier: 'online' },
  false: { label: 'Offline', modifier: 'offline' },
};

export default function Header({ apiOnline }) {
  const status = STATUS[String(apiOnline)];

  return (
    <header className="header">
      <div className="header__brand">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#2ecc71" strokeWidth="2" aria-hidden="true">
          <path d="M12 2C8 2 6 7 6 12a6 6 0 0 0 12 0c0-5-2-10-6-10z" />
          <circle cx="12" cy="14" r="2.5" />
        </svg>
        <div>
          <h1 className="header__title">Avocado Ripeness AI</h1>
          <p className="header__subtitle">Hass avocado ripeness and shelf-life estimation</p>
        </div>
      </div>
      <div className="header__status" role="status">
        <span className={`status-dot status-dot--${status.modifier}`} />
        <span>{status.label}</span>
      </div>
    </header>
  );
}
