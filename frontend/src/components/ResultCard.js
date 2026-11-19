import React from 'react';
import { CONFIDENCE_COLORS, CONFIDENCE_LABELS } from '../constants';
import { formatDuration } from '../utils/format';
import FeedbackPanel from './FeedbackPanel';
import InsightPanel from './InsightPanel';
import ProbabilityBar from './ProbabilityBar';

function Stat({ label, value }) {
  return (
    <div className="stat">
      <span className="stat__label">{label}</span>
      <span className="stat__value">{value}</span>
    </div>
  );
}

function Candidates({ candidates, highlight }) {
  return (
    <section>
      <h4 className="section-title">Most likely stages</h4>
      {candidates.map((candidate, index) => (
        <ProbabilityBar
          key={candidate.stage}
          label={candidate.label}
          probability={candidate.probability}
          color={index === 0 ? highlight : '#5a7a62'}
        />
      ))}
    </section>
  );
}

/** Shows the (session-fused) prediction, the insight and the feedback controls. */
export default function ResultCard({ result }) {
  const { prediction, insight } = result;
  const color = CONFIDENCE_COLORS[prediction.confidence_level];

  // Not reliable enough to act on: show candidates and ask for another photo.
  if (!prediction.reliable) {
    return (
      <article className="card" style={{ '--accent': CONFIDENCE_COLORS.low }}>
        <span className="badge">Insufficient confidence</span>
        <p className="text text--lead">{prediction.message}</p>
        {prediction.warning && <div className="warning">{prediction.warning}</div>}
        <Candidates candidates={prediction.top_candidates} highlight={CONFIDENCE_COLORS.medium} />
      </article>
    );
  }

  return (
    <article className="card" style={{ '--accent': color }}>
      <div className="card__header">
        <div>
          <h2 className="card__title">{prediction.label}</h2>
          <p className="card__subtitle">
            Stage {prediction.stage} of 5 · {prediction.dataset_name}
          </p>
        </div>
        <span className="badge">{CONFIDENCE_LABELS[prediction.confidence_level]}</span>
      </div>

      <ProbabilityBar label="Confidence" probability={prediction.confidence} color={color} large />

      <p className="text text--lead">{prediction.message}</p>
      {prediction.warning && <div className="warning">{prediction.warning}</div>}

      <div className="stats">
        <Stat label="Time until overripe" value={formatDuration(prediction.days_until_overripe)} />
        <Stat label="Uncertainty (days)" value={`± ${prediction.days_uncertainty.toFixed(1)}`} />
        <Stat label="Class uncertainty" value={prediction.class_uncertainty.toFixed(4)} />
        <Stat label="Photos combined" value={prediction.images_used} />
      </div>

      <hr className="divider" />
      <InsightPanel insight={insight} />
      {insight && <hr className="divider" />}

      <Candidates candidates={prediction.top_candidates} highlight={color} />

      <hr className="divider" />
      <FeedbackPanel sessionId={result.session_id} imageId={result.image_id} predictedStage={prediction.stage} />
    </article>
  );
}
