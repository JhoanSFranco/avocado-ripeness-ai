import React from 'react';
import { formatPercent } from '../utils/format';

/** Horizontal bar with a label and percentage (used for confidence and top candidates). */
export default function ProbabilityBar({ label, probability, color, large = false }) {
  return (
    <div className={`prob-bar${large ? ' prob-bar--large' : ''}`}>
      <span className="prob-bar__label">{label}</span>
      <div className="prob-bar__track">
        <div className="prob-bar__fill" style={{ width: formatPercent(probability), background: color }} />
      </div>
      <span className="prob-bar__value" style={{ color }}>
        {formatPercent(probability)}
      </span>
    </div>
  );
}
