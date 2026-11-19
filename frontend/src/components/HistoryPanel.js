import React from 'react';
import { CONFIDENCE_COLORS } from '../constants';
import { formatPercent } from '../utils/format';

/** Recent analyses; clicking one shows its result again. */
export default function HistoryPanel({ history, onSelect }) {
  if (!history.length) return null;

  return (
    <section className="history">
      <h4 className="section-title">Recent analyses</h4>
      <ul className="history__list">
        {history.map((item) => {
          const { prediction } = item;
          return (
            <li key={item.image_id}>
              <button type="button" className="history__item" onClick={() => onSelect(item)}>
                <img src={item.previewUrl} alt="" className="history__thumb" />
                <span className="history__info">
                  <span className="history__label">{prediction.label}</span>
                  <span className="history__meta">
                    {prediction.images_used} photo{prediction.images_used === 1 ? '' : 's'} combined
                  </span>
                </span>
                <span className="history__confidence" style={{ color: CONFIDENCE_COLORS[prediction.confidence_level] }}>
                  {formatPercent(prediction.confidence)}
                </span>
              </button>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
