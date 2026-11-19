import React, { useState } from 'react';
import { sendFeedback } from '../api/client';
import { STAGES } from '../constants';

/** "Was this right?" If not, the user can tell us the correct stage. */
export default function FeedbackPanel({ sessionId, imageId, predictedStage }) {
  const [state, setState] = useState('idle'); // idle | choosing | sending | sent | error
  const [correctStage, setCorrectStage] = useState('');

  const submit = async (isCorrect, stage = null) => {
    setState('sending');
    try {
      await sendFeedback({
        session_id: sessionId,
        image_id: imageId,
        predicted_stage: predictedStage,
        is_correct: isCorrect,
        correct_stage: stage,
      });
      setState('sent');
    } catch {
      setState('error');
    }
  };

  if (state === 'sent') return <p className="feedback__done">Thanks! Your feedback was recorded.</p>;

  if (state === 'choosing') {
    return (
      <div className="feedback">
        <label className="feedback__label" htmlFor="correct-stage">
          What is the correct stage?
        </label>
        <div className="feedback__row">
          <select
            id="correct-stage"
            className="select"
            value={correctStage}
            onChange={(event) => setCorrectStage(event.target.value)}
          >
            <option value="">I don't know</option>
            {STAGES.filter((s) => s.stage !== predictedStage).map((s) => (
              <option key={s.stage} value={s.stage}>
                {s.stage}. {s.label}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="button button--primary"
            onClick={() => submit(false, correctStage ? Number(correctStage) : null)}
          >
            Send
          </button>
        </div>
      </div>
    );
  }

  const sending = state === 'sending';
  return (
    <div className="feedback">
      <p className="feedback__label">Was this prediction correct?</p>
      <div className="feedback__row">
        <button type="button" className="button button--outline button--grow" disabled={sending} onClick={() => submit(true)}>
          {sending ? 'Sending...' : 'Correct'}
        </button>
        <button
          type="button"
          className="button button--danger button--grow"
          disabled={sending}
          onClick={() => setState('choosing')}
        >
          Incorrect
        </button>
      </div>
      {state === 'error' && <p className="feedback__error">Feedback could not be sent. Please try again.</p>}
    </div>
  );
}
