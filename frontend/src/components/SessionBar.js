import React from 'react';

/** Explains the "one session = one avocado" model and lets the user reset it. */
export default function SessionBar({ photos, onNewAvocado }) {
  return (
    <div className="session-bar">
      <div>
        <p className="session-bar__title">
          Current avocado · {photos} photo{photos === 1 ? '' : 's'}
        </p>
        <p className="session-bar__hint">
          Photos of the same avocado are combined for a more reliable result.
        </p>
      </div>
      {photos > 0 && (
        <button type="button" className="button button--small button--ghost" onClick={onNewAvocado}>
          New avocado
        </button>
      )}
    </div>
  );
}
