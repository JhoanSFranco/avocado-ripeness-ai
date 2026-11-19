import React from 'react';

export default function Spinner({ title, subtitle }) {
  return (
    <div className="spinner" role="status" aria-live="polite">
      <div className="spinner__circle" />
      <p className="spinner__title">{title}</p>
      {subtitle && <p className="spinner__subtitle">{subtitle}</p>}
    </div>
  );
}
